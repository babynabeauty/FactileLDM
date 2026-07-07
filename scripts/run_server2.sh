#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RAW_TACTILE_ASSETS_DIR="${RAW_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"

export TRAIN_STEPS="${TRAIN_STEPS:-60000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export FSDP_DEVICES="${FSDP_DEVICES:-4}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export KEEP_PERIOD="${KEEP_PERIOD:-10000}"
export GPU_SLOTS="${GPU_SLOTS:-0,1,2,3,4,5,6,7}"

JOB_LABELS=(
  "D_patch_f4_h16_async"
  "E_patch_f8_h16_async"
  "F_raw_f4_h16_async"
)
JOB_CONFIGS=(
  "pi0_xhand_dual_patch_f4_h16_async"
  "pi0_xhand_dual_patch_f8_h16_async"
  "pi0_xhand_dual_raw_f4_h16_async"
)
JOB_ASSET_DIRS=(
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
)

source scripts/four_gpu_training_queue.sh
run_four_gpu_training_queue "${1:-data/task12345-2}"
