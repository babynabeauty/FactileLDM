#!/usr/bin/env bash
set -euo pipefail
CURRENT_PID=1249044
ROOT=/data/workspace/zhangshiqi/forceWAM
WAIT_LOG="$ROOT/logs/pi0_xhand_tactile_direct_force_waiter.log"
TRAIN_LOG="$ROOT/logs/pi0_xhand_tactile_direct_force_vlm_lora_action_full.log"

{
  echo "[$(date '+%F %T')] waiting for current force-only training pid=${CURRENT_PID}"
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep 300
  done
  echo "[$(date '+%F %T')] pid=${CURRENT_PID} finished; waiting 60s for CUDA cleanup"
  sleep 60
  echo "[$(date '+%F %T')] launching direct-force single-student-AE experiment"
} >> "$WAIT_LOG" 2>&1

cd "$ROOT"
exec setsid env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  HF_HOME=/data/shared_workspace/zhangshiqi/hf \
  CUDA_VISIBLE_DEVICES=7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /data/workspace/zhangshiqi/forceWAM/env/.venv/bin/python \
  /data/workspace/zhangshiqi/forceWAM/scripts/train.py pi0_xhand_tactile_direct_force_vlm_lora_action_full \
    --exp-name pi0_xhand_tactile_direct_force_vlm_lora_action_full \
    --num-train-steps 30000 \
    --batch-size 1 \
    --num-workers 0 \
    --save-interval 5000 \
    --keep-period 10000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path /data/shared_workspace/zhangshiqi/hf/pi0_base_jax/pi0_base/params \
    --data.assets.assets-dir /data/workspace/zhangshiqi/forceWAM/assets/pi0_xhand_tactile_flow_vlm_lora_action_full \
  >> "$TRAIN_LOG" 2>&1
