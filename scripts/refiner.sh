cd /workspace/mnt/sqzhang26/FactileLDM
mkdir -p logs

DATA_REPO=data/task1_2_206ep
ASSET_ID=$(basename "$DATA_REPO")
RUN_TAG=206ep_20k_0626

setsid nohup bash -c '
set -e

env \
  HF_LEROBOT_HOME=. \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_raw_dual_ae_arm_future_hand_mask_tactile_refine \
    --exp-name raw_tactile_refine_'"$RUN_TAG"' \
    --data.repo-id '"$DATA_REPO"' \
    --data.assets.asset-id '"$ASSET_ID"' \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_raw_dual_ae \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 1 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/raw_tactile_refine_'"$RUN_TAG"'.log 2>&1

env \
  HF_LEROBOT_HOME=. \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_adaptive_patch_raw_dual_ae_arm_future_hand_mask_tactile_refine \
    --exp-name adaptive_patch_raw_tactile_refine_'"$RUN_TAG"' \
    --data.repo-id '"$DATA_REPO"' \
    --data.assets.asset-id '"$ASSET_ID"' \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_raw_dual_ae \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 1 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/adaptive_patch_raw_tactile_refine_'"$RUN_TAG"'.log 2>&1
' > logs/refine_two_stage_4567_${RUN_TAG}.scheduler.log 2>&1 &