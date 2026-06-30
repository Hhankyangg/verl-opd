#!/usr/bin/env bash
set -euo pipefail
set -x

NNODES="${1:-2}"
NGPU="${2:-8}"

EXTRA_ARGS=()
USE_VLLM_TEACHER_REF="${USE_VLLM_TEACHER_REF:-False}"
if (( $# > 3 )); then
    EXTRA_ARGS=("${@:4}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_ROOT="${REPO_ROOT}"
OPD_REPO_ROOT="${OPD_REPO_ROOT:-${REPO_ROOT}}"

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache/opd_bagel_student_qwen32b_teacher}"
DATA_DIR="${DATA_DIR:-${OPD_REPO_ROOT}/projects/opd/data_parquet/mint_cot_dataset}"
TRAIN_DATA_FILE="${TRAIN_DATA_FILE:-${DATA_FILE:-${DATA_DIR}/train.parquet}}"
VAL_DATA_FILE="${VAL_DATA_FILE:-${DATA_DIR}/val_sample100.parquet}"

PROJECT_NAME="verl_opd"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd_qwen2_5_vl_32b_2_bagel_7b_mintcot}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
LOGGER='["console","wandb"]'
PYTHON_BIN="${PYTHON_BIN:-python3}"
SAVE_INITIAL_CHECKPOINT="${SAVE_INITIAL_CHECKPOINT:-False}"
SKIP_VLLM_TEACHER_TIMEOUT_STEP="${SKIP_VLLM_TEACHER_TIMEOUT_STEP:-True}"
SAVE_FREQ="${SAVE_FREQ:-25}"
ACTOR_CHECKPOINT_SAVE_CONTENTS="${ACTOR_CHECKPOINT_SAVE_CONTENTS:-['model','extra']}"
ACTOR_CHECKPOINT_LOAD_CONTENTS="${ACTOR_CHECKPOINT_LOAD_CONTENTS:-['model','optimizer','extra']}"
NCCL_TIMEOUT_S="${NCCL_TIMEOUT_S:-${NCCL_TIMEOUT:-1800}}"
POST_CHECKPOINT_SLEEP_S="${POST_CHECKPOINT_SLEEP_S:-30}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export BAGEL_CODEBASE="${BAGEL_CODEBASE:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT_S}"

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
VLLM_TEACHER_BASE_URL="${VLLM_TEACHER_BASE_URL:-}"
VLLM_TEACHER_PROTOCOL="${VLLM_TEACHER_PROTOCOL:-kraken_distill}"
VLLM_TEACHER_MODEL="${VLLM_TEACHER_MODEL:-${TEACHER_MODEL}}"
VLLM_TEACHER_API_KEY="${VLLM_TEACHER_API_KEY:-}"
VLLM_TEACHER_TIMEOUT_S="${VLLM_TEACHER_TIMEOUT_S:-3600.0}"
VLLM_TEACHER_MAX_CONCURRENT_REQUESTS="${VLLM_TEACHER_MAX_CONCURRENT_REQUESTS:-1}"
VLLM_TEACHER_MAX_RETRIES="${VLLM_TEACHER_MAX_RETRIES:-0}"
VLLM_TEACHER_RETRY_BACKOFF_S="${VLLM_TEACHER_RETRY_BACKOFF_S:-60.0}"
VLLM_TEACHER_TEMPERATURE="${VLLM_TEACHER_TEMPERATURE:-1.0}"
VLLM_TEACHER_MISSING_TOKEN_LOGPROB="${VLLM_TEACHER_MISSING_TOKEN_LOGPROB:-}"
VLLM_TEACHER_MICRO_BATCH_SIZE="${VLLM_TEACHER_MICRO_BATCH_SIZE:-1}"
VLLM_TEACHER_RESTART_URL="${VLLM_TEACHER_RESTART_URL:-}"
VLLM_TEACHER_RESTART_TOKEN="${VLLM_TEACHER_RESTART_TOKEN:-restart}"
VLLM_TEACHER_RESTART_TIMEOUT_S="${VLLM_TEACHER_RESTART_TIMEOUT_S:-5.0}"
VLLM_TEACHER_ARGS=()

if [[ -z "${BAGEL_CODEBASE}" || ! -d "${BAGEL_CODEBASE}" ]]; then
    echo "ERROR: Set BAGEL_CODEBASE to the local BAGEL repository root." >&2
    exit 1
fi
if [[ -z "${STUDENT_MODEL}" ]]; then
    echo "ERROR: Set STUDENT_MODEL to the BAGEL student checkpoint directory." >&2
    exit 1
fi
if [[ -z "${TEACHER_MODEL}" ]]; then
    echo "ERROR: Set TEACHER_MODEL to the Qwen2.5-VL teacher model path or HF repo id." >&2
    exit 1
fi

if [[ "${USE_VLLM_TEACHER_REF}" == "True" && -z "${VLLM_TEACHER_BASE_URL}" ]]; then
    if [[ "${VLLM_TEACHER_PROTOCOL}" == "kraken"* || "${VLLM_TEACHER_PROTOCOL}" == "distill" ]]; then
        VLLM_TEACHER_BASE_URL="http://127.0.0.1:8001"
    else
        VLLM_TEACHER_BASE_URL="http://127.0.0.1:8000"
    fi
fi

if [[ "${USE_VLLM_TEACHER_REF}" == "True" ]]; then
    echo "ERROR: USE_VLLM_TEACHER_REF=True is not wired in this migrated verl-opd path yet." >&2
    echo "Use the default FSDP ref path, or port vllm_teacher_ref into engine_workers before enabling it." >&2
    exit 1
fi

if [[ "${USE_VLLM_TEACHER_REF}" == "True" && -n "${VLLM_TEACHER_BASE_URL}" ]]; then
    VLLM_TEACHER_ARGS=(
        "actor_rollout_ref.ref.vllm_teacher.protocol=${VLLM_TEACHER_PROTOCOL}"
        "actor_rollout_ref.ref.vllm_teacher.base_url=${VLLM_TEACHER_BASE_URL}"
        "actor_rollout_ref.ref.vllm_teacher.model=${VLLM_TEACHER_MODEL}"
        "actor_rollout_ref.ref.vllm_teacher.request_timeout_s=${VLLM_TEACHER_TIMEOUT_S}"
        "actor_rollout_ref.ref.vllm_teacher.max_concurrent_requests=${VLLM_TEACHER_MAX_CONCURRENT_REQUESTS}"
        "actor_rollout_ref.ref.vllm_teacher.max_retries=${VLLM_TEACHER_MAX_RETRIES}"
        "actor_rollout_ref.ref.vllm_teacher.retry_backoff_s=${VLLM_TEACHER_RETRY_BACKOFF_S}"
        "actor_rollout_ref.ref.vllm_teacher.temperature=${VLLM_TEACHER_TEMPERATURE}"
        "actor_rollout_ref.ref.vllm_teacher.micro_batch_size=${VLLM_TEACHER_MICRO_BATCH_SIZE}"
    )
    if [[ -n "${VLLM_TEACHER_MISSING_TOKEN_LOGPROB}" ]]; then
        VLLM_TEACHER_ARGS+=("actor_rollout_ref.ref.vllm_teacher.missing_token_logprob=${VLLM_TEACHER_MISSING_TOKEN_LOGPROB}")
    fi
    if [[ -n "${VLLM_TEACHER_RESTART_URL}" ]]; then
        VLLM_TEACHER_ARGS+=(
            "actor_rollout_ref.ref.vllm_teacher.restart_url=${VLLM_TEACHER_RESTART_URL}"
            "actor_rollout_ref.ref.vllm_teacher.restart_timeout_s=${VLLM_TEACHER_RESTART_TIMEOUT_S}"
        )
    fi
    if [[ -n "${VLLM_TEACHER_RESTART_TOKEN}" ]]; then
        VLLM_TEACHER_ARGS+=("actor_rollout_ref.ref.vllm_teacher.restart_token=${VLLM_TEACHER_RESTART_TOKEN}")
    fi
    if [[ -n "${VLLM_TEACHER_API_KEY}" ]]; then
        VLLM_TEACHER_ARGS+=("actor_rollout_ref.ref.vllm_teacher.api_key=${VLLM_TEACHER_API_KEY}")
    fi
fi

if [[ "${STUDENT_MODEL}" == "${TEACHER_MODEL}" ]]; then
    echo "WARNING: STUDENT_MODEL == TEACHER_MODEL. This is only suitable for smoke testing." >&2
fi

# Conservative defaults for 32B ref to reduce OOM risk.
N_ROLL="${N_ROLL:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((8 * NNODES))}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_RESP_LEN="${MAX_RESP_LEN:-1024}"
LR="${LR:-5e-6}"

REF_LOGPROB_MB="${REF_LOGPROB_MB:-1}"
ROLLOUT_LOGPROB_MB="${ROLLOUT_LOGPROB_MB:-1}"
REF_MAX_TOKEN_LEN_PER_GPU="${REF_MAX_TOKEN_LEN_PER_GPU:-4096}"
TEST_FREQ="${TEST_FREQ:-50}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-4}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${DEFAULT_LOCAL_DIR}}"

TOTAL_STEPS="${TOTAL_STEPS:-3200}"

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
    actor_rollout_ref.model.model_type=bagel \
    actor_rollout_ref.model.use_remove_padding=False \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CHECKPOINT_SAVE_CONTENTS}" \
    ++actor_rollout_ref.actor.checkpoint.load_contents="${ACTOR_CHECKPOINT_LOAD_CONTENTS}" \
    actor_rollout_ref.nccl_timeout="${NCCL_TIMEOUT_S}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MB}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$((NNODES * NGPU))" \
    actor_rollout_ref.rollout.name=bagel \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.agent.default_agent_loop=bagel_single_turn_agent \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n="${N_ROLL}" \
    ++actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
    ++actor_rollout_ref.ref.model.model_type=language_model \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_MAX_TOKEN_LEN_PER_GPU}" \
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
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.use_v1=False \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    "${VLLM_TEACHER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
