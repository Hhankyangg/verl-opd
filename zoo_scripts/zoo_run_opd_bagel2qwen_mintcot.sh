#!/usr/bin/env bash
set -euo pipefail
set -x

NNODES="${1:-2}"
NGPU="${2:-8}"

EXTRA_ARGS=()
if (( $# > 3 )); then
    EXTRA_ARGS=("${@:4}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_ROOT="${REPO_ROOT}"
OPD_REPO_ROOT="${OPD_REPO_ROOT:-${REPO_ROOT}}"

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache/opd_qwen2_5_vl_7b}"
DATA_DIR="${DATA_DIR:-${OPD_REPO_ROOT}/projects/opd/data_parquet/mint_cot_dataset}"
TRAIN_DATA_FILE="${TRAIN_DATA_FILE:-${DATA_DIR}/train.parquet}"
VAL_DATA_FILE="${VAL_DATA_FILE:-${DATA_DIR}/val_sample100.parquet}"

PROJECT_NAME="verl_opd"
EXPERIMENT_NAME="opd_bagel_2_qwen2_5_vl_7b_trial2_mintcot"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
LOGGER='["console","wandb"]'
PYTHON_BIN="${PYTHON_BIN:-python3}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export BAGEL_CODEBASE="${BAGEL_CODEBASE:-}"

mkdir -p \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${VLLM_CACHE_ROOT}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${CACHE_ROOT}/rlhf" \
    "${DEFAULT_LOCAL_DIR}"

STUDENT_MODEL="${STUDENT_MODEL:-}"
TEACHER_MODEL="${TEACHER_MODEL:-}"

if [[ -z "${BAGEL_CODEBASE}" || ! -d "${BAGEL_CODEBASE}" ]]; then
    echo "ERROR: Set BAGEL_CODEBASE to the local BAGEL repository root." >&2
    exit 1
fi
if [[ -z "${STUDENT_MODEL}" ]]; then
    echo "ERROR: Set STUDENT_MODEL to the Qwen2.5-VL student model path or HF repo id." >&2
    exit 1
fi
if [[ -z "${TEACHER_MODEL}" ]]; then
    echo "ERROR: Set TEACHER_MODEL to the BAGEL teacher checkpoint directory." >&2
    exit 1
fi

if [[ "${STUDENT_MODEL}" == "${TEACHER_MODEL}" ]]; then
    echo "WARNING: STUDENT_MODEL == TEACHER_MODEL. This is only suitable for smoke testing." >&2
fi

N_ROLL="${N_ROLL:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((32 * NNODES))}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESP_LEN="${MAX_RESP_LEN:-4096}"
LR="${LR:-5e-6}"

ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.50}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-12288}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-12288}"
TEST_FREQ="${TEST_FREQ:-20}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-10}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${DEFAULT_LOCAL_DIR}}"
TOTAL_STEPS="${TOTAL_STEPS:-800}"

if [[ ! -f "${TRAIN_DATA_FILE}" || ! -f "${VAL_DATA_FILE}" ]]; then
    echo "Expected MINT-CoT parquet files not found:" >&2
    echo "  TRAIN_DATA_FILE=${TRAIN_DATA_FILE}" >&2
    echo "  VAL_DATA_FILE=${VAL_DATA_FILE}" >&2
    echo "Prepare them with:" >&2
    echo "  PYTHON_BIN=${PYTHON_BIN} bash ${REPO_ROOT}/projects/opd/prepare_data/run_mint_cot_prep.sh" >&2
    exit 1
fi

cd "${VERL_ROOT}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=no_op \
    data.train_files="${TRAIN_DATA_FILE}" \
    data.val_files="${VAL_DATA_FILE}" \
    ++data.cache_dir="${CACHE_ROOT}/rlhf" \
    data.image_key=images \
    data.return_raw_chat=True \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESP_LEN}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.model_type=language_model \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.95]" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=importance_sampling \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    ++actor_rollout_ref.actor.loss_scale_factor="${MAX_RESP_LEN}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    ++actor_rollout_ref.rollout.limit_images=1 \
    actor_rollout_ref.rollout.n="${N_ROLL}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    ++actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
    ++actor_rollout_ref.ref.model.model_type=bagel \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.forward_only=True \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=1.0 \
    reward.custom_reward_function.path="${REPO_ROOT}/zoo_scripts/opd_reward.py" \
    reward.custom_reward_function.name=dummy_compute_score \
    reward.reward_manager.name=naive \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.n_gpus_per_node="${NGPU}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=50 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.val_before_train=True \
    trainer.use_v1=False \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    "${EXTRA_ARGS[@]}"
