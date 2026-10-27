#!/usr/bin/env bash
# Run on the training server using its existing FactileLDM environment.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/mnt/sqzhang26/FactileLDM}"
cd "$REPO_ROOT"
# Use this checkout and its own environment only; an inherited PYTHONPATH or
# PYTHON_BIN can otherwise pull packages from a different openpi checkout.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
unset PYTHONHOME
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Extraction is single-GPU; do not reserve all 16 A800s for a readout experiment.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="$REPO_ROOT/env/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: FactileLDM Python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi
"$PYTHON_BIN" -c 'import importlib.util, sys; spec = importlib.util.find_spec("flax"); print("Probe Python:", sys.executable, "\nEnvironment:", sys.prefix, "\nFlax:", spec.origin if spec else "not installed", flush=True)'

POLICY_PARAMS="${POLICY_PARAMS:-$REPO_ROOT/checkpoints/pi0_xhand_dual_patch_pretrained_f4_h16_async/pi0_xhand_dual_patch_pretrained_f4_h16_async_task12345_2_patch_pretrain_fast_0709/59999}"
CONFIG_NAME="${CONFIG_NAME:-pi0_xhand_dual_patch_pretrained_f4_h16_async}"
TRAIN_ROOT="${TRAIN_ROOT:-$REPO_ROOT/data/task12345-2}"
TEST_ROOT="${TEST_ROOT:-$REPO_ROOT/data/task-test-collect}"
TEST_TASK_SCOPE="${TEST_TASK_SCOPE:-common}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/frozen_fae_decoder_59999_${TEST_TASK_SCOPE}_tasks}"
CACHE_DIR="${CACHE_DIR:-$OUTPUT_ROOT/cache}"
SOURCE="${SOURCE:-fae}"
DECODER_TYPE="${DECODER_TYPE:-linear}"
PROBE_DIR="${PROBE_DIR:-$OUTPUT_ROOT/${SOURCE}_${DECODER_TYPE}}"
SEED="${SEED:-42}"

# Explicit native-sensor label definition, NOT a claim that these units are N.
# Change only using training/validation data, then use a fresh OUTPUT_ROOT/cache.
CONTACT_THRESHOLD="${CONTACT_THRESHOLD:-1.0}"
CONTACT_TEMPERATURE="${CONTACT_TEMPERATURE:-0.5}"
FORCE_UNIT="${FORCE_UNIT:-dataset_native}"

run_probe() {
    "$PYTHON_BIN" scripts/run_future_patch_probe.py "$@"
}

stage="${1:-all}"
case "$stage" in
    all|prepare|extract|train|evaluate) ;;
    *) echo "Usage: $0 [all|prepare|extract|train|evaluate]" >&2; exit 2 ;;
esac

if [[ "$stage" == all || "$stage" == prepare ]]; then
    run_probe prepare --train-root "$TRAIN_ROOT" --test-root "$TEST_ROOT" \
        --manifest "$OUTPUT_ROOT/splits.json" --val-ratio "${VAL_RATIO:-0.1}" --seed "$SEED" \
        --test-task-scope "$TEST_TASK_SCOPE"
fi

if [[ "$stage" == all || "$stage" == extract ]]; then
    extract_args=(extract --config-name "$CONFIG_NAME" --policy-params "$POLICY_PARAMS"
        --manifest "$OUTPUT_ROOT/splits.json" --cache-dir "$CACHE_DIR"
        --contact-threshold "$CONTACT_THRESHOLD" --contact-temperature "$CONTACT_TEMPERATURE"
        --force-unit "$FORCE_UNIT" --batch-size "${EXTRACT_BATCH_SIZE:-2}"
        --stride "${CACHE_STRIDE:-4}" --num-workers "${NUM_WORKERS:-0}"
        --seed "$SEED" --resume)
    if [[ -n "${NORM_STATS:-}" ]]; then
        extract_args+=(--norm-stats "$NORM_STATS")
    fi
    run_probe "${extract_args[@]}"
fi

if [[ "$stage" == all || "$stage" == train ]]; then
    run_probe train --cache-dir "$CACHE_DIR" --output-dir "$PROBE_DIR" \
        --source "$SOURCE" --decoder-type "$DECODER_TYPE" --epochs "${EPOCHS:-20}" \
        --batch-size "${PROBE_BATCH_SIZE:-128}" --learning-rate "${LEARNING_RATE:-0.001}" --seed "$SEED"
fi

if [[ "$stage" == all || "$stage" == evaluate ]]; then
    run_probe evaluate --cache-dir "$CACHE_DIR" --probe-checkpoint "$PROBE_DIR/best_probe.npz" \
        --output-dir "$PROBE_DIR" --batch-size "${PROBE_BATCH_SIZE:-128}" \
        --bootstrap-samples "${BOOTSTRAP_SAMPLES:-200}" --seed "$SEED"
fi
