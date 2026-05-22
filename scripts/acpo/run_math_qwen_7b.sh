#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

METHOD=${METHOD:-acpo}
REGIME=${REGIME:-offpolicy}
N_GPUS=${N_GPUS:-4}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B}
DATA_DIR=${DATA_DIR:-${HOME}/data/open_reasoner_zero_nochat}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
PROJECT_NAME=${PROJECT_NAME:-acpo}
LOGGER=${LOGGER:-"['console']"}
START_RAY=${START_RAY:-1}
PRELOAD_MODEL=${PRELOAD_MODEL:-0}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.3}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONUNBUFFERED=1

case "${REGIME}" in
  near_onpolicy|nonp|bsz2x|2x)
    REGIME_NAME=near_onpolicy
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
    PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
    ;;
  offpolicy|offp|bsz16x|16x)
    REGIME_NAME=offpolicy
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
    PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
    ;;
  *)
    echo "Unknown REGIME=${REGIME}. Use near_onpolicy or offpolicy." >&2
    exit 1
    ;;
esac

COMMON_ARGS=(
  algorithm.adv_estimator=grpo
  "data.train_files=${DATA_DIR}/train.parquet"
  "data.val_files=${VAL_FILES:-[\"${DATA_DIR}/test1.parquet\",\"${DATA_DIR}/test2.parquet\",\"${DATA_DIR}/test_amc.parquet\",\"${DATA_DIR}/test_minerva.parquet\",\"${DATA_DIR}/test_aime2025.parquet\"]}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  data.filter_overlong_prompts=True
  actor_rollout_ref.rollout.val_kwargs.temperature=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.n=8
  data.max_prompt_length=1024
  data.max_response_length=3072
  data.apply_chat_template=False
  "actor_rollout_ref.model.path=${BASE_MODEL}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.optim.lr=1e-6
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  critic.optim.lr=1e-6
  algorithm.kl_ctrl.kl_coef=0.0
  "trainer.logger=${LOGGER}"
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  trainer.val_before_train=True
  trainer.default_hdfs_dir=null
  "trainer.n_gpus_per_node=${N_GPUS}"
  trainer.nnodes=1
  trainer.save_freq=50
  trainer.test_freq=50
  "trainer.project_name=${PROJECT_NAME}"
  trainer.total_epochs=15
  actor_rollout_ref.rollout.n=10
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
)

METHOD_ARGS=()
case "${METHOD}" in
  acpo)
    METHOD_NAME=acpo
    METHOD_ARGS=(
      actor_rollout_ref.actor.use_variance_adaptive_clip=True
      "actor_rollout_ref.actor.variance_alpha=${VARIANCE_ALPHA:-3.0}"
      "actor_rollout_ref.actor.variance_base_clip=${VARIANCE_BASE_CLIP:-0.2}"
      "actor_rollout_ref.actor.variance_clip_min=${VARIANCE_CLIP_MIN:-0.0}"
      "actor_rollout_ref.actor.variance_clip_max=${VARIANCE_CLIP_MAX:-3.0}"
      "actor_rollout_ref.actor.variance_num_bins=${VARIANCE_NUM_BINS:-5}"
    )
    ;;
  dapo|baseline)
    METHOD_NAME=dapo
    ;;
  cispo)
    METHOD_NAME=cispo
    METHOD_ARGS=(
      actor_rollout_ref.actor.use_cispo=True
      "actor_rollout_ref.actor.clip_ratio_is_high=${CISPO_CLIP_RATIO_IS_HIGH:-0.45}"
      "actor_rollout_ref.actor.clip_ratio_is_low=${CISPO_CLIP_RATIO_IS_LOW:-1.0}"
    )
    ;;
  ar_lopti)
    METHOD_NAME=ar_lopti
    COMMON_ARGS=("${COMMON_ARGS[@]/actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}/actor_rollout_ref.actor.clip_ratio_high=0.24}")
    METHOD_ARGS=(
      algorithm.use_ar_lopti=True
      algorithm.ar_alpha=0.3
      algorithm.ar_tau=0.7
      algorithm.ar_neg_adv_weight=1.0
      algorithm.use_lopti=True
      algorithm.lopti_prob_threshold=0.5
    )
    ;;
  entropy_top)
    METHOD_NAME=entropy_top
    METHOD_ARGS=(
      actor_rollout_ref.actor.use_entropy_top_k=True
      actor_rollout_ref.actor.use_entropy_bottom_k=False
      actor_rollout_ref.actor.entropy_k_ratio=0.2
    )
    ;;
  entropy_bottom)
    METHOD_NAME=entropy_bottom
    METHOD_ARGS=(
      actor_rollout_ref.actor.use_entropy_top_k=False
      actor_rollout_ref.actor.use_entropy_bottom_k=True
      actor_rollout_ref.actor.entropy_k_ratio=0.8
    )
    ;;
  *)
    echo "Unknown METHOD=${METHOD}. Use acpo, dapo, cispo, ar_lopti, entropy_top, or entropy_bottom." >&2
    exit 1
    ;;
esac

EXP_NAME=${EXP_NAME:-qwen2.5-7b_${METHOD_NAME}_${REGIME_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXP_NAME}}
mkdir -p "${OUTPUT_DIR}"

if [ "${START_RAY}" = "1" ]; then
  ray stop || true
  env VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND}" RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 ray start --head --dashboard-host=0.0.0.0
fi

if [ "${PRELOAD_MODEL}" = "1" ]; then
  python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${BASE_MODEL}');"
fi

cd "${REPO_ROOT}"
python3 -m verl.trainer.main_ppo \
  "${COMMON_ARGS[@]}" \
  "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints" \
  "trainer.experiment_name=${EXP_NAME}" \
  "${METHOD_ARGS[@]}" \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/training_process.log"
