#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_REPO="${1:-data/task12345-2}"
ASSET_ID="${ASSET_ID:-$(basename "$DATA_REPO")}"
PI0_BASE_PARAMS="${PI0_BASE_PARAMS:-checkpoints/pi0_base/params}"
PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-/workspace/mnt/sqzhang26/gaoyuxuan/openpi/.cache/openpi-assets/checkpoints/pi05_base/params}"
PI05_ASSETS_DIR="${PI05_ASSETS_DIR:-assets/pi05_xhand_full_finetune_h16}"
PI0_TACTILE_ASSETS_DIR="${PI0_TACTILE_ASSETS_DIR:-assets/pi0_xhand_state_tactile_finetune_h16}"
PI05_TACTILE_ASSETS_DIR="${PI05_TACTILE_ASSETS_DIR:-assets/pi05_xhand_state_tactile_finetune_h16}"

export TRAIN_STEPS="${TRAIN_STEPS:-50000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export FSDP_DEVICES="${FSDP_DEVICES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export KEEP_PERIOD="${KEEP_PERIOD:-10000}"
# Two four-GPU slots by default: run the first two jobs concurrently, then
# launch the third as soon as either slot is released.
export GPU_SLOTS="${GPU_SLOTS:-0,1,2,3;4,5,6,7}"

JOB_LABELS=(
  "A_pi05_full_h16"
  "B_pi0_raw_state_tactile_h16"
  "C_pi05_raw_state_tactile_h16"
)
JOB_CONFIGS=(
  "pi05_xhand_full_finetune_h16"
  "pi0_xhand_state_tactile_finetune_h16"
  "pi05_xhand_state_tactile_finetune_h16"
)
JOB_ASSET_DIRS=(
  "$PI05_ASSETS_DIR"
  "$PI0_TACTILE_ASSETS_DIR"
  "$PI05_TACTILE_ASSETS_DIR"
)
JOB_WEIGHT_PATHS=(
  "$PI05_BASE_PARAMS"
  "$PI0_BASE_PARAMS"
  "$PI05_BASE_PARAMS"
)

echo "[run_server1] Stage 1/2: normalization for ${DATA_REPO} (asset=${ASSET_ID})"
NORM_CONFIGS="pi05_xhand_full_finetune_h16 pi0_xhand_state_tactile_finetune_h16 pi05_xhand_state_tactile_finetune_h16" \
ASSET_ID="$ASSET_ID" \
bash scripts/run_xhand_norm_stats_parallel.sh "$DATA_REPO"

echo "[run_server1] Stage 2/2: queued training for pi05/pi0 tactile baselines"
source scripts/four_gpu_training_queue.sh
DATA_ASSET_ID="$ASSET_ID" run_four_gpu_training_queue "$DATA_REPO"
