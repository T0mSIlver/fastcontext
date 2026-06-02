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

NUM_NODES=2
SSH_PORT=2222

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# 3.5-35B-A3B and 3.6-35B-A3B share the same model configuration
source "scripts/models/qwen3.5-35B-A3B.sh"

EXP_ROOT="/mnt/local/exp"
EXP_CKPT=${EXP_ROOT}/ckpt/swefc_sft_qwen3.6_35b_slime/
EXP_HF_CKPT=${EXP_ROOT}/ckpt/swefc_sft_qwen3.6_35b_hf/
mkdir -p ${EXP_CKPT}
mkdir -p ${EXP_HF_CKPT}

CKPT_ARGS=(
   --hf-checkpoint /mnt/local/models/Qwen3.6-35B-A3B/
   --ref-load /mnt/local/models/Qwen3.6-35B-A3B_torch_dist/
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

# 8 GPU shared: TP=2, DP=8/2/4=1, PP=1, CP=4, EP=8,
# max-seq-len: 128K
PERF_ARGS=(
   --tensor-model-parallel-size 2 #  num_query_groups (2) must be a multiple of tensor_model_parallel_size
   --sequence-parallel
   --pipeline-model-parallel-size 2
   # --decoder-last-pipeline-num-layers 30
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --use-distributed-optimizer
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# --- WANDB CONFIGURATION---
WANDB_KEY=${WANDB_KEY:-""}
if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project swefc
      --wandb-group qwen3.6-35B-sft-v3
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

   # use deepep for megatron
   # --moe-token-dispatcher-type flex
   # --moe-enable-deepep
)

# launch the master node of ray in container
MASTER_IP=$(hostname -I | awk '{print $1}')
export MASTER_ADDR=${MASTER_ADDR:-"${MASTER_IP}"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# This allows you to execute the training entirely from master node:
for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
  if [[ "$WORKER_IP" == "$MASTER_IP" ]]; then
    continue
  fi
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh -p ${SSH_PORT} root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265" &
done
wait

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes ${NUM_NODES} \
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
