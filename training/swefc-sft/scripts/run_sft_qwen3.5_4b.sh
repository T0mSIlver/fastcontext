#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="/root/slime/"
source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

EXP_ROOT="/mnt/local/exp"
EXP_CKPT=${EXP_ROOT}/ckpt/swefc_sft_3.5_4b_slime/
EXP_HF_CKPT=${EXP_ROOT}/ckpt/swefc_sft_3.5_4b_hf/
mkdir -p ${EXP_CKPT}
mkdir -p ${EXP_HF_CKPT}

CKPT_ARGS=(
   --hf-checkpoint /mnt/local/models/Qwen3.5-4B/
   --ref-load /mnt/local/models/Qwen3.5-4B_torch_dist/
   --save ${EXP_CKPT}
   --save-interval 1000
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /mnt/local/datasets/swefc_sft.jsonl
   --input-key messages
   --tool-key tools
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 64
   --global-batch-size 64
   --disable-rollout-trim-samples

   --loss-type sft_loss
   --loss-mask-type qwen3_5

   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# 8 GPU shared: TP=4, DP=8/4/2=1
# max-seq-len: 128K
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

# --- WANDB CONFIGURATION---
WANDB_KEY=${WANDB_KEY:-""}
if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project swefc
      --wandb-group qwen3.5-4B-sft-v3
      --wandb-key ${WANDB_KEY}
   )
else
   WANDB_ARGS=()
fi


MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_DIR}/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}


echo "Training script completed."

## convert to hf model
echo "Converting to HF model..."
ITER=$(ls ${EXP_CKPT} | grep "iter_" | sort | tail -n 1)
CKPT_SAVE_DIR="${EXP_HF_CKPT}/$ITER"
mkdir -p ${CKPT_SAVE_DIR}
ls -all -h ${EXP_CKPT}/$ITER

PYTHONPATH=/root/Megatron-LM python /root/slime/tools/convert_torch_dist_to_hf.py \
  --input-dir "${EXP_CKPT}/$ITER" \
  --output-dir ${CKPT_SAVE_DIR} \
  --origin-hf-dir "/mnt/local/models/Qwen3.5-4B" \
  --force

echo "Converted HF model saved to ${CKPT_SAVE_DIR}"
ls -all -h ${CKPT_SAVE_DIR}
echo "Conversion completed."