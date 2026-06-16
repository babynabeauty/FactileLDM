#!/usr/bin/env bash
"""

setsid nohup env \
  TRAIN_STEPS=5000 \
  GLOBAL_BATCH_SIZE=4 \
  NUM_WORKERS=0 \
  RUN_TAG=59ep_5k_0616 \
  bash scripts/run_tactile_abc_experiments.sh \
    grasp_pipette_and_press_button_0616_59ep \
  > logs/abc_scheduler_59ep_5k_0616.log 2>&1 &

echo "Scheduler PID=$!"

"""
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-data/grasp_pipette_and_press_button_0616_59ep}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
RUN_TAG="${RUN_TAG:-60ep_5k_$(date +%m%d_%H%M)}"

PYTHON="${PYTHON:-env/.venv/bin/python}"
WEIGHT_PATH="${WEIGHT_PATH:-checkpoints/pi0_base/params}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_datasets_cache}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p logs "$HF_DATASETS_CACHE" "$HF_HOME"

CONFIG_A="pi0_xhand_full_finetune"
CONFIG_B="pi0_xhand_tactile_obs_ae_full_finetune"
CONFIG_C="pi0_xhand_tactile_structured_single_ae"

EXP_A="A_no_tactile_${RUN_TAG}"
EXP_B="B_current_tactile_${RUN_TAG}"
EXP_C="C_structured_single_ae_${RUN_TAG}"

declare -A JOB_NAME
declare -A JOB_GPU

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

compute_norm_stats() {
  local config="$1"
  local gpu_id="$2"
  local log_file="logs/norm_${config}_${RUN_TAG}.log"

  log "Computing normalization stats: ${config} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    "$PYTHON" scripts/compute_norm_stats.py \
      --config-name "$config" \
      --repo-id "$DATA_REPO" \
    > "$log_file" 2>&1
}

launch_norm_stats() {
  local config="$1"
  local gpu_id="$2"

  compute_norm_stats "$config" "$gpu_id" &
  local pid=$!
  JOB_NAME["$pid"]="norm_${config}"
  log "Normalization stats started: config=${config}, GPU=${gpu_id}, PID=${pid}, log=logs/norm_${config}_${RUN_TAG}.log"
  LAUNCHED_PID="$pid"
}

wait_for_norm_stats() {
  local status=0
  local pid

  for pid in "$@"; do
    if wait_and_report "$pid"; then
      :
    else
      status=$?
    fi
  done

  if (( status != 0 )); then
    log "One or more normalization jobs failed. Training will not start."
    exit "$status"
  fi

  log "All normalization stats completed successfully."
}

launch_train() {
  local label="$1"
  local config="$2"
  local exp_name="$3"
  local gpu_ids="$4"
  local log_file="logs/${exp_name}.log"

  log "Starting ${label}: config=${config}, GPUs=${gpu_ids}, global_batch=${GLOBAL_BATCH_SIZE}"
  setsid nohup env \
    HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    HF_HOME="$HF_HOME" \
    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PYTHON" scripts/train.py "$config" \
      --exp-name "$exp_name" \
      --data.repo-id "$DATA_REPO" \
      --num-train-steps "$TRAIN_STEPS" \
      --batch-size "$GLOBAL_BATCH_SIZE" \
      --fsdp-devices 4 \
      --num-workers "$NUM_WORKERS" \
      --save-interval "$SAVE_INTERVAL" \
      --keep-period "$SAVE_INTERVAL" \
      --no-wandb-enabled \
      --overwrite \
      --weight-loader.params-path "$WEIGHT_PATH" \
    > "$log_file" 2>&1 &

  local pid=$!
  JOB_NAME["$pid"]="$label"
  JOB_GPU["$pid"]="$gpu_ids"
  log "${label} started: PID=${pid}, log=${log_file}"
  LAUNCHED_PID="$pid"
}

wait_for_first() {
  local pid_a="$1"
  local pid_b="$2"

  while true; do
    if ! kill -0 "$pid_a" 2>/dev/null; then
      FIRST_PID="$pid_a"
      OTHER_PID="$pid_b"
      return
    fi
    if ! kill -0 "$pid_b" 2>/dev/null; then
      FIRST_PID="$pid_b"
      OTHER_PID="$pid_a"
      return
    fi
    sleep 30
  done
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

log "Dataset: ${DATA_REPO}"
log "Stage 1/2: normalization A/B/C in parallel on GPUs 1/2/3"
launch_norm_stats "$CONFIG_A" "1"
PID_NORM_A="$LAUNCHED_PID"

launch_norm_stats "$CONFIG_B" "2"
PID_NORM_B="$LAUNCHED_PID"

launch_norm_stats "$CONFIG_C" "3"
PID_NORM_C="$LAUNCHED_PID"

wait_for_norm_stats "$PID_NORM_A" "$PID_NORM_B" "$PID_NORM_C"

log "Stage 2/2: launch A on GPUs 0-3 and B on GPUs 4-7"
launch_train "A" "$CONFIG_A" "$EXP_A" "0,1,2,3"
PID_A="$LAUNCHED_PID"
launch_train "B" "$CONFIG_B" "$EXP_B" "4,5,6,7"
PID_B="$LAUNCHED_PID"

wait_for_first "$PID_A" "$PID_B"
FIRST_GPU="${JOB_GPU[$FIRST_PID]}"

if ! wait_and_report "$FIRST_PID"; then
  log "A/B first completed job failed; C will not start."
  wait_and_report "$OTHER_PID" || true
  exit 1
fi

log "${JOB_NAME[$FIRST_PID]} released GPUs ${FIRST_GPU}; starting C on those GPUs."
launch_train "C" "$CONFIG_C" "$EXP_C" "$FIRST_GPU"
PID_C="$LAUNCHED_PID"

remaining_status=0
wait_and_report "$OTHER_PID" || remaining_status=$?
wait_and_report "$PID_C" || remaining_status=$?

if (( remaining_status != 0 )); then
  log "One or more experiments failed. Check logs/."
  exit "$remaining_status"
fi

log "All A/B/C experiments completed successfully."
