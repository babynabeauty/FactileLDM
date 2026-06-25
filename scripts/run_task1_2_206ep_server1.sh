#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CALC_TACTILE_ASSETS_DIR="${CALC_TACTILE_ASSETS_DIR:-assets/pi0_xhand_tactile_structured_dual_ae}"
NO_TACTILE_ASSETS_DIR="${NO_TACTILE_ASSETS_DIR:-$CALC_TACTILE_ASSETS_DIR}"

JOB_LABELS=(
  "A_no_tactile"
  "B_structured_dual_ae"
  "C_structured_single_ae"
  "D_structured_dual_ae_arm_future_hand_mask"
)
JOB_CONFIGS=(
  "pi0_xhand_full_finetune"
  "pi0_xhand_tactile_structured_dual_ae"
  "pi0_xhand_tactile_structured_single_ae"
  "pi0_xhand_tactile_structured_dual_ae_arm_future_hand_mask"
)
JOB_ASSET_DIRS=(
  "$NO_TACTILE_ASSETS_DIR"
  "$CALC_TACTILE_ASSETS_DIR"
  "$CALC_TACTILE_ASSETS_DIR"
  "$CALC_TACTILE_ASSETS_DIR"
)

source scripts/four_gpu_training_queue.sh
run_four_gpu_training_queue "${1:-data/task1_2_206ep}"
