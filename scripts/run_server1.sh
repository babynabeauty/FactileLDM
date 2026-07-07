#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RAW_TACTILE_ASSETS_DIR="${RAW_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"

JOB_LABELS=(
  "A_raw_f4_h16"
  "B_raw_f4_h16_async"
  "C_raw_f8_h16_async"
)
JOB_CONFIGS=(
  "pi0_xhand_dual_raw_f4_h16"
  "pi0_xhand_dual_raw_f4_h16_async"
  "pi0_xhand_dual_raw_f8_h16_async"
)
JOB_ASSET_DIRS=(
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
)

source scripts/four_gpu_training_queue.sh
run_four_gpu_training_queue "${1:-data/task12345-2}"
