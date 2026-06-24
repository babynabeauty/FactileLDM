#!/usr/bin/env bash

# Shared two-slot scheduler for 8-GPU training servers.
# The caller must define JOB_LABELS, JOB_CONFIGS, and JOB_ASSET_DIRS.

run_four_gpu_training_queue() {
  if (( ${#JOB_LABELS[@]} == 0 )); then
    echo "ERROR: No training jobs were configured." >&2
    return 2
  fi
  if (( ${#JOB_LABELS[@]} != ${#JOB_CONFIGS[@]} || ${#JOB_LABELS[@]} != ${#JOB_ASSET_DIRS[@]} )); then
    echo "ERROR: JOB_LABELS, JOB_CONFIGS, and JOB_ASSET_DIRS must have the same length." >&2
    return 2
  fi

  local data_repo="${1:-data/task1_2_206ep}"
  local data_asset_id="${DATA_ASSET_ID:-$(basename "$data_repo")}" 
  local train_steps="${TRAIN_STEPS:-20000}"
  local global_batch_size="${GLOBAL_BATCH_SIZE:-8}"
  local fsdp_devices="${FSDP_DEVICES:-4}"
  local num_workers="${NUM_WORKERS:-2}"
  local save_interval="${SAVE_INTERVAL:-5000}"
  local keep_period="${KEEP_PERIOD:-5000}"
  local run_tag="${RUN_TAG:-task1_2_206ep_20k_$(date +%m%d_%H%M%S)}"
  local python_bin="${PYTHON:-env/.venv/bin/python}"
  local weight_path="${WEIGHT_PATH:-checkpoints/pi0_base/params}"
  local -a gpu_slots=("0,1,2,3" "4,5,6,7")

  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PWD}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/.hf_datasets_cache}"
  export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false

  mkdir -p logs "$HF_DATASETS_CACHE" "$HF_HOME"

  if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: Python is not executable: $python_bin" >&2
    return 2
  fi
  if [[ ! -f "$data_repo/meta/info.json" ]]; then
    echo "ERROR: Dataset not found: $data_repo/meta/info.json" >&2
    return 2
  fi
  if [[ ! -e "$weight_path" ]]; then
    echo "ERROR: Base checkpoint not found: $weight_path" >&2
    return 2
  fi

  local asset_dir
  for asset_dir in "${JOB_ASSET_DIRS[@]}"; do
    if [[ ! -f "$asset_dir/$data_asset_id/norm_stats.json" ]]; then
      echo "ERROR: Norm stats not found: $asset_dir/$data_asset_id/norm_stats.json" >&2
      return 2
    fi
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi -L | wc -l)"
    if (( gpu_count < 8 )); then
      echo "ERROR: This scheduler requires 8 visible GPUs, found $gpu_count." >&2
      return 2
    fi
  fi

  log_queue() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
  }

  declare -A pid_to_slot=()
  declare -A pid_to_label=()
  local -a active_pids=()
  local next_job=0
  local failed=0

  remove_active_pid() {
    local target_pid="$1"
    local -a remaining=()
    local pid
    for pid in "${active_pids[@]}"; do
      if [[ "$pid" != "$target_pid" ]]; then
        remaining+=("$pid")
      fi
    done
    active_pids=("${remaining[@]}")
  }

  local waited_pid=""
  local waited_status=0
  wait_for_one_job() {
    local pid
    local status

    # Bash wait -n ignores jobs that finished before wait -n was called. Reap
    # one such child explicitly before waiting for a newly completed child.
    for pid in "${active_pids[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        set +e
        wait "$pid"
        status=$?
        set -e
        waited_pid="$pid"
        waited_status="$status"
        return 0
      fi
    done

    waited_pid=""
    set +e
    wait -n -p waited_pid "${active_pids[@]}" 2>/dev/null
    status=$?
    set -e
    if [[ -n "$waited_pid" ]]; then
      waited_status="$status"
      return 0
    fi

    # All children may have exited between kill -0 and wait -n. Their exit
    # statuses are still available through an explicit wait.
    for pid in "${active_pids[@]}"; do
      set +e
      wait "$pid"
      status=$?
      set -e
      if (( status != 127 )); then
        waited_pid="$pid"
        waited_status="$status"
        return 0
      fi
    done
    return 1
  }

  terminate_active_jobs() {
    local pid
    if (( ${#active_pids[@]} == 0 )); then
      return
    fi
    log_queue "Stopping active training jobs..."
    for pid in "${active_pids[@]}"; do
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
  }
  trap 'terminate_active_jobs; exit 130' INT TERM

  launch_job() {
    local job_index="$1"
    local slot_index="$2"
    local label="${JOB_LABELS[$job_index]}"
    local config="${JOB_CONFIGS[$job_index]}"
    local assets_dir="${JOB_ASSET_DIRS[$job_index]}"
    local gpu_ids="${gpu_slots[$slot_index]}"
    local exp_name="${label}_${run_tag}"
    local log_file="logs/${exp_name}.log"

    log_queue "Starting ${label}: config=${config}, GPUs=${gpu_ids}, assets=${assets_dir}"
    setsid --wait env \
      HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
      HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
      HF_HOME="$HF_HOME" \
      HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
      CUDA_VISIBLE_DEVICES="$gpu_ids" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "$python_bin" scripts/train.py "$config" \
        --exp-name "$exp_name" \
        --data.repo-id "$data_repo" \
        --data.assets.asset-id "$data_asset_id" \
        --data.assets.assets-dir "$assets_dir" \
        --num-train-steps "$train_steps" \
        --batch-size "$global_batch_size" \
        --fsdp-devices "$fsdp_devices" \
        --num-workers "$num_workers" \
        --save-interval "$save_interval" \
        --keep-period "$keep_period" \
        --no-wandb-enabled \
        --overwrite \
        --weight-loader.params-path "$weight_path" \
      > "$log_file" 2>&1 &

    local pid=$!
    pid_to_slot["$pid"]="$slot_index"
    pid_to_label["$pid"]="$label"
    active_pids+=("$pid")
    log_queue "Started ${label}: PID=${pid}, log=${log_file}"
  }

  log_queue "Dataset: $data_repo"
  log_queue "Asset ID: $data_asset_id"
  log_queue "Jobs: ${#JOB_LABELS[@]}, steps=${train_steps}, batch=${global_batch_size}, FSDP=${fsdp_devices}"
  log_queue "Two GPU slots: ${gpu_slots[0]} and ${gpu_slots[1]}"

  local slot_index
  for slot_index in 0 1; do
    if (( next_job < ${#JOB_LABELS[@]} )); then
      launch_job "$next_job" "$slot_index"
      ((next_job += 1))
    fi
  done

  while (( ${#active_pids[@]} > 0 )); do
    if ! wait_for_one_job; then
      log_queue "ERROR: wait returned without a child PID."
      failed=1
      break
    fi
    local finished_pid="$waited_pid"
    local status="$waited_status"

    local freed_slot="${pid_to_slot[$finished_pid]}"
    local finished_label="${pid_to_label[$finished_pid]}"
    remove_active_pid "$finished_pid"
    unset 'pid_to_slot[$finished_pid]' 'pid_to_label[$finished_pid]'

    if (( status == 0 )); then
      log_queue "Completed ${finished_label} successfully."
    else
      log_queue "ERROR: ${finished_label} failed with exit code ${status}. No queued jobs will be started."
      failed=1
    fi

    if (( failed == 0 && next_job < ${#JOB_LABELS[@]} )); then
      launch_job "$next_job" "$freed_slot"
      ((next_job += 1))
    fi
  done

  trap - INT TERM
  if (( failed != 0 )); then
    terminate_active_jobs
    return 1
  fi

  log_queue "All queued experiments completed successfully."
}
