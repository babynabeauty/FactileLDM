#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-data/grasp_pipette_and_press_button_106ep}"
DATA_ASSET_ID="${DATA_ASSET_ID:-$(basename "$DATA_REPO")}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
RUN_TAG="${RUN_TAG:-106ep_20k_$(date +%m%d_%H%M)}"

PYTHON="${PYTHON:-env/.venv/bin/python}"
WEIGHT_PATH="${WEIGHT_PATH:-checkpoints/pi0_base/params}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_datasets_cache}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p logs "$HF_DATASETS_CACHE" "$HF_HOME"

if [[ ! -f "$DATA_REPO/meta/info.json" ]]; then
  cat >&2 <<EOF
ERROR: Local LeRobot dataset not found: $DATA_REPO
Expected file: $DATA_REPO/meta/info.json

Pass an existing dataset path relative to FactileLDM, for example:
  bash scripts/run_tactile_abc_experiments.sh data/grasp_pipette_and_press_button_106ep

This script intentionally refuses missing paths to avoid falling back to Hugging Face.
EOF
  exit 2
fi

CONFIG_A="pi0_xhand_full_finetune"
CONFIG_B="pi0_xhand_tactile_obs_ae_full_finetune"
CONFIG_C="pi0_xhand_tactile_structured_single_ae"
CONFIG_D="pi0_xhand_tactile_structured_dual_ae"

EXP_A="A_no_tactile_${RUN_TAG}"
EXP_B="B_current_tactile_${RUN_TAG}"
EXP_C="C_structured_single_ae_${RUN_TAG}"
EXP_D="D_structured_dual_ae_${RUN_TAG}"

declare -A JOB_NAME
declare -A JOB_GPU

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

wait_and_report() {
  local pid="$1"
  local label="${JOB_NAME[$pid]}"
  local status

  set +e
  wait "$pid"
  status=$?
  set -e

  if (( status == 0 )); then
    log "${label} completed successfully."
    return 0
  fi

  log "ERROR: ${label} failed with exit code ${status}."
  return "$status"
}

wait_all_or_exit() {
  local status=0
  local pid

  for pid in "$@"; do
    wait_and_report "$pid" || status=$?
  done

  if (( status != 0 )); then
    log "One or more jobs failed. Check logs/."
    exit "$status"
  fi
}

compute_norm_stats() {
  local config="$1"
  local gpu_id="$2"
  local log_file="logs/norm_${config}_${RUN_TAG}.log"

  log "Computing normalization stats: config=${config}, GPU=${gpu_id}, asset=${DATA_ASSET_ID}"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    "$PYTHON" scripts/compute_norm_stats.py \
      --config-name "$config" \
      --repo-id "$DATA_REPO" \
      --asset-id "$DATA_ASSET_ID" \
    > "$log_file" 2>&1
}

launch_norm_stats() {
  local config="$1"
  local gpu_id="$2"

  compute_norm_stats "$config" "$gpu_id" &
  local pid=$!
  JOB_NAME["$pid"]="norm_${config}"
  JOB_GPU["$pid"]="$gpu_id"
  log "Normalization started: config=${config}, GPU=${gpu_id}, PID=${pid}, log=logs/norm_${config}_${RUN_TAG}.log"
  LAUNCHED_PID="$pid"
}

launch_train() {
  local label="$1"
  local config="$2"
  local exp_name="$3"
  local gpu_ids="$4"
  local log_file="logs/${exp_name}.log"

  log "Starting ${label}: config=${config}, GPUs=${gpu_ids}, batch=${GLOBAL_BATCH_SIZE}, steps=${TRAIN_STEPS}"
  setsid nohup env \
    HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    HF_HOME="$HF_HOME" \
    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PYTHON" scripts/train.py "$config" \
      --exp-name "$exp_name" \
      --data.repo-id "$DATA_REPO" \
      --data.assets.asset-id "$DATA_ASSET_ID" \
      --num-train-steps "$TRAIN_STEPS" \
      --batch-size "$GLOBAL_BATCH_SIZE" \
      --fsdp-devices "$FSDP_DEVICES" \
      --num-workers "$NUM_WORKERS" \
      --save-interval "$SAVE_INTERVAL" \
      --keep-period "$KEEP_PERIOD" \
      --no-wandb-enabled \
      --overwrite \
      --weight-loader.params-path "$WEIGHT_PATH" \
    > "$log_file" 2>&1 &

  local pid=$!
  JOB_NAME["$pid"]="$label"
  JOB_GPU["$pid"]="$gpu_ids"
  log "${label} started: PID=${pid}, GPUs=${gpu_ids}, log=${log_file}"
  LAUNCHED_PID="$pid"
}

log "Dataset repo: ${DATA_REPO}"
log "Asset id: ${DATA_ASSET_ID}"
log "Python: ${PYTHON}"
log "Weight path: ${WEIGHT_PATH}"
log "Train steps: ${TRAIN_STEPS}, batch: ${GLOBAL_BATCH_SIZE}, fsdp devices: ${FSDP_DEVICES}"
log "Save interval: ${SAVE_INTERVAL}, keep period: ${KEEP_PERIOD}, run tag: ${RUN_TAG}"

log "Stage 1/3: normalization A/B/C/D in parallel on GPUs 0/1/2/3"
launch_norm_stats "$CONFIG_A" "0"
PID_NORM_A="$LAUNCHED_PID"
launch_norm_stats "$CONFIG_B" "1"
PID_NORM_B="$LAUNCHED_PID"
launch_norm_stats "$CONFIG_C" "2"
PID_NORM_C="$LAUNCHED_PID"
launch_norm_stats "$CONFIG_D" "3"
PID_NORM_D="$LAUNCHED_PID"
wait_all_or_exit "$PID_NORM_A" "$PID_NORM_B" "$PID_NORM_C" "$PID_NORM_D"
log "All normalization stats completed successfully."

log "Stage 2/3: launch A on GPUs 0-3 and B on GPUs 4-7"
launch_train "A" "$CONFIG_A" "$EXP_A" "0,1,2,3"
PID_A="$LAUNCHED_PID"
launch_train "B" "$CONFIG_B" "$EXP_B" "4,5,6,7"
PID_B="$LAUNCHED_PID"
wait_all_or_exit "$PID_A" "$PID_B"
log "A and B completed successfully."

log "Stage 3/3: launch C on GPUs 0-3 and D on GPUs 4-7"
launch_train "C" "$CONFIG_C" "$EXP_C" "0,1,2,3"
PID_C="$LAUNCHED_PID"
launch_train "D" "$CONFIG_D" "$EXP_D" "4,5,6,7"
PID_D="$LAUNCHED_PID"
wait_all_or_exit "$PID_C" "$PID_D"

log "All A/B/C/D experiments completed successfully."
