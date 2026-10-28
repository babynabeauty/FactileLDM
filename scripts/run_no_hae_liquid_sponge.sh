#!/usr/bin/env bash
# Normalize task_liquid_sponge and train the two no-HAE policy stages.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/env/.venv/bin/python"
DATA="$ROOT/data/task_liquid_sponge"
ASSET_ID="task_liquid_sponge"
CONFIG_A="pi0_xhand_dual_patch_pretrained_f4_h16_async_no_hae_freeze"
CONFIG_B="pi0_xhand_dual_patch_pretrained_f4_h16_async_no_hae"
ASSET_DIR="$ROOT/assets/$CONFIG_B"
NORM_FILE="$ASSET_DIR/$ASSET_ID/norm_stats.json"
ENCODER_PARAMS="${ENCODER_PARAMS:-$ROOT/checkpoints/xhand_patch_tactile_encoder_pretrain/xhand_patch_tactile_encoder_pretrain_task12345_2_patch_pretrain_fast_0709/19999/params}"
BASE_PARAMS="${BASE_PARAMS:-/workspace/mnt/sqzhang26/hf_weight/pi0_base/params}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
STAGE2A_STEPS=5000
STAGE2B_STEPS=50000
RUN_TAG="${RUN_TAG:-task_liquid_sponge_no_hae_$(date +%Y%m%d_%H%M%S)}"
EXP_A="${CONFIG_A}_${RUN_TAG}"
EXP_B="${CONFIG_B}_${RUN_TAG}"
CHECKPOINT_A="$ROOT/checkpoints/$CONFIG_A/$EXP_A"
CHECKPOINT_B="$ROOT/checkpoints/$CONFIG_B/$EXP_B"
LOG_A="$ROOT/logs/${EXP_A}.log"
LOG_B="$ROOT/logs/${EXP_B}.log"

stage="${1:-all}"
case "$stage" in all|norm|train) ;; *) echo "Usage: $0 [all|norm|train]" >&2; exit 2;; esac
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$DATA/meta/info.json" ]] || { echo "Missing dataset: $DATA" >&2; exit 1; }
export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_LEROBOT_HOME="$ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/.hf_datasets_cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$GPUS"

# Fail before creating output if the server checkout or environment is not ready.
"$PYTHON" -B -c 'import importlib.util
try:
    spec = importlib.util.find_spec("lerobot.common.datasets.lerobot_dataset")
except ModuleNotFoundError:
    spec = None
if spec is None:
    raise SystemExit("Missing lerobot.common.datasets.lerobot_dataset in FactileLDM env/.venv; use the training-compatible LeRobot installation.")
from openpi.training import config
for name in ("pi0_xhand_dual_patch_pretrained_f4_h16_async_no_hae_freeze", "pi0_xhand_dual_patch_pretrained_f4_h16_async_no_hae"):
    config.get_config(name)
print("Environment and no-HAE configurations: OK")'

if [[ "$stage" == all || "$stage" == train ]]; then
    [[ -d "$BASE_PARAMS" ]] || { echo "Missing pi0 base params: $BASE_PARAMS. Set BASE_PARAMS to the original base checkpoint /params." >&2; exit 1; }
    [[ -d "$ENCODER_PARAMS" ]] || { echo "Missing Stage 1 encoder params: $ENCODER_PARAMS" >&2; exit 1; }
    [[ "$STAGE2A_STEPS" =~ ^[1-9][0-9]*$ && "$STAGE2B_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "Stage step counts must be positive integers." >&2; exit 1; }
    for path in "$CHECKPOINT_A" "$CHECKPOINT_B" "$LOG_A" "$LOG_B"; do
        [[ ! -e "$path" ]] || { echo "Output already exists: $path. Set a new RUN_TAG." >&2; exit 1; }
    done
fi

if [[ "$stage" == all || "$stage" == norm ]]; then
    [[ ! -e "$NORM_FILE" ]] || { echo "Normalization file already exists: $NORM_FILE. Use train to reuse it." >&2; exit 1; }
    "$PYTHON" -B scripts/compute_norm_stats.py --config-name "$CONFIG_B" \
        --repo-id "$DATA" --asset-id "$ASSET_ID" --batch-size "$BATCH_SIZE" --num-workers 0
fi

if [[ "$stage" == all || "$stage" == train ]]; then
    [[ -f "$NORM_FILE" ]] || { echo "Missing normalization file: $NORM_FILE. Run norm first." >&2; exit 1; }
    mkdir -p "$ROOT/logs"
    echo "Stage 2A: $EXP_A"
    "$PYTHON" -B scripts/train.py "$CONFIG_A" \
        --exp-name "$EXP_A" --data.repo-id "data/$ASSET_ID" \
        --data.assets.asset-id "$ASSET_ID" --data.assets.assets-dir "$ASSET_DIR" \
        --weight-loader.pi0-params-path "$BASE_PARAMS" --weight-loader.encoder-params-path "$ENCODER_PARAMS" \
        --num-train-steps "$STAGE2A_STEPS" --batch-size "$BATCH_SIZE" --fsdp-devices "$FSDP_DEVICES" \
        --num-workers "$NUM_WORKERS" --save-interval 5000 --keep-period 10000 --no-wandb-enabled \
        2>&1 | tee -a "$LOG_A"
    STAGE2A_PARAMS="$CHECKPOINT_A/$((STAGE2A_STEPS - 1))/params"
    [[ -d "$STAGE2A_PARAMS" ]] || { echo "Missing Stage 2A final params: $STAGE2A_PARAMS" >&2; exit 1; }
    echo "Stage 2B: $EXP_B"
    "$PYTHON" -B scripts/train.py "$CONFIG_B" \
        --exp-name "$EXP_B" --data.repo-id "data/$ASSET_ID" \
        --data.assets.asset-id "$ASSET_ID" --data.assets.assets-dir "$ASSET_DIR" \
        --weight-loader.pi0-params-path "$STAGE2A_PARAMS" --weight-loader.encoder-params-path "$ENCODER_PARAMS" \
        --num-train-steps "$STAGE2B_STEPS" --batch-size "$BATCH_SIZE" --fsdp-devices "$FSDP_DEVICES" \
        --num-workers "$NUM_WORKERS" --save-interval 39999 --keep-period 10000 --no-wandb-enabled \
        2>&1 | tee -a "$LOG_B"
    echo "Final checkpoint: $CHECKPOINT_B/$((STAGE2B_STEPS - 1))"
fi
