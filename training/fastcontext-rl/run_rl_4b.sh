#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
ps aux | grep ray | grep -v grep | awk '{print $2}' | xargs kill -9
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:False"}


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="/root/slime/"
source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"
MEGATRON_DIR="/root/Megatron-LM/"
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%F-%H%M%S)}
echo "SCRIPT_DIR: $SCRIPT_DIR"

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-2}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-$((NUM_NODES * ROLLOUT_GPUS_PER_NODE))}


HF_CKPT=${HF_CKPT:-/mnt/local/models/fastcontext_sft_qwen3.5_4b_hf_iter_0000137}
REF_LOAD=${REF_LOAD:-${HF_CKPT}_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-/mnt/local/exp/ckpt/fastcontext-rl_${RUN_TIMESTAMP}}
mkdir -p "${SAVE_CKPT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval 10
  # --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
  --prompt-data /mnt/local/datasets/fastcontext_rl_training.jsonl
  --input-key messages
  --label-key label
  --metadata-key metadata
  --rollout-shuffle
  --reward-key score
  --num-rollout 1000
  --rollout-batch-size 4
  --n-samples-per-prompt 8
  --global-batch-size 32
  --rollout-max-response-len 2048
  --rollout-max-context-len 65536
  --rollout-temperature 1
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_fastcontext.generate
  --custom-rm-path generate_with_fastcontext.reward_func
)

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
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
  --log-probs-chunk-size 256
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

EVAL_ARGS=()


SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-context-length 128000
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

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

WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project fastcontext
    --wandb-group fastcontext-rl-qwen3p5-4b
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi


# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"no_proxy\": \"${no_proxy}\"
  }
}"


# --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}" \
# --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \

ray job submit --address="http://${MASTER_ADDR}:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -u "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes "${NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]} \
    ${EVAL_ARGS[@]}


# ray job logs --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
# RAY_LOG_EXIT=$?
# RAY_STATUS_OUTPUT=$(ray job status --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
# echo "${RAY_STATUS_OUTPUT}"

echo "Training script completed."
