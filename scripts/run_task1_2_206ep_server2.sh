#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RAW_TACTILE_ASSETS_DIR="${RAW_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_raw_dual_ae}"

JOB_LABELS=(
  "E_structured_raw_dual_ae"
  "F_structured_raw_single_ae"
  "G_structured_raw_dual_ae_arm_future_hand_mask"
)
JOB_CONFIGS=(
  "pi0_xhand_tactile_structured_raw_dual_ae"
  "pi0_xhand_tactile_structured_raw_single_ae"
  "pi0_xhand_tactile_structured_raw_dual_ae_arm_future_hand_mask"
)
JOB_ASSET_DIRS=(
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
  "$RAW_TACTILE_ASSETS_DIR"
)

source scripts/four_gpu_training_queue.sh
run_four_gpu_training_queue "${1:-data/task1_2_206ep}"
