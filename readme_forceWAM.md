# ForceWAM / FactileLDM 实验命令

## 每次运行前先设置

```bash
cd /workspace/mnt/sqzhang26/FactileLDM
source env/.venv/bin/activate

export PROJECT_ROOT=/workspace/mnt/sqzhang26/FactileLDM
export HF_LEROBOT_HOME="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=.hf_datasets_cache

DATA_REPO="data/grasp_pipette_and_press_button_106ep"
ASSET_ID="$(basename "$DATA_REPO")"
mkdir -p logs
```

说明：
- 统一使用 `HF_LEROBOT_HOME=/workspace/mnt/sqzhang26/FactileLDM`
- 统一使用 `DATA_REPO=data/xxx`
- 不要再使用 `HF_LEROBOT_HOME=/workspace/mnt/sqzhang26/FactileLDM/data`，否则容易变成 `data/data/xxx`

## 归一化

### 5x3 structured dual AE

```bash
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_tactile_structured_dual_ae \
  --repo-id "$DATA_REPO" \
  --asset-id "$ASSET_ID"
```

### 5x120x3 raw structured dual AE

```bash
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_tactile_structured_raw_dual_ae \
  --repo-id "$DATA_REPO" \
  --asset-id "$ASSET_ID"
```

### structured single AE

```bash
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_tactile_structured_single_ae \
  --repo-id "$DATA_REPO" \
  --asset-id "$ASSET_ID"
```

## 推荐训练命令

### A. pi0 no tactile

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_full_finetune \
    --exp-name pi0_xhand_full_finetune_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/pi0_xhand_full_finetune_106ep_20k.log 2>&1 &
```

### B. current 5x3 tactile observation tokens

如果复用旧 obs-AE 归一化文件，保留 `--data.assets.assets-dir`。

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_obs_ae_full_finetune \
    --exp-name pi0_xhand_tactile_obs_ae_full_finetune_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir assets/pi0_xhand_tactile_flow_full_finetune \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/pi0_xhand_tactile_obs_ae_full_finetune_106ep_20k.log 2>&1 &
```

### C. structured single AE

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_single_ae \
    --exp-name structured_single_ae_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_single_ae_106ep_20k.log 2>&1 &
```

### D. structured dual AE 5x3

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_dual_ae \
    --exp-name structured_dual_ae_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_dual_ae_106ep_20k.log 2>&1 &
```

### E. structured dual AE 5x3 + arm/future/hand mask

这里复用 `pi0_xhand_tactile_structured_dual_ae` 的归一化文件，所以保留 `--data.assets.assets-dir`。

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_dual_ae_arm_future_hand_mask \
    --exp-name structured_dual_ae_arm_future_hand_mask_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_dual_ae \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_dual_ae_arm_future_hand_mask_106ep_20k.log 2>&1 &
```

### F. structured raw dual AE 5x120x3

raw 点阵触觉需要单独归一化，不要复用 5x3 的 assets。

```bash
setsid nohup env \
  HF_LEROBOT_HOME="$PROJECT_ROOT" \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_raw_dual_ae \
    --exp-name pi0_xhand_tactile_structured_raw_dual_ae_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/pi0_xhand_tactile_structured_raw_dual_ae_106ep_20k.log 2>&1 &
```

## 历史内容归档

下面是之前的命令和杂项记录，保留备用；日常跑实验优先复制上面的推荐命令。

```bash
cd /workspace/mnt/sqzhang26/FactileLDM
source env/.venv/bin/activate
export PROJECT_ROOT=/workspace/mnt/sqzhang26/FactileLDM
export HF_LEROBOT_HOME="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=.hf_datasets_cache
DATA_REPO="data/grasp_pipette_and_press_button_106ep"
ASSET_ID="$(basename "$DATA_REPO")"
```

# 合并数据集
python scripts/merge_lerobot_v21_datasets.py \
  --sources \
    /workspace/mnt/sqzhang26/FactileLDM/data/47ep \
    /workspace/mnt/sqzhang26/FactileLDM/data/grasp_pipette_and_press_button_0616_59ep \
  --output /workspace/mnt/sqzhang26/FactileLDM/data/grasp_pipette_and_press_button_106ep \
  --overwrite


# 计算光流图像

```bash
    CUDA_VISIBLE_DEVICES=0 \
env/.venv/bin/python scripts/compute_lerobot_future_flow_video.py \
  --repo-id data/grasp_pipette_and_press_button_0616_59ep_flow \
  --output-dir ./flow_videos \
  --future-step 32 \
  --overwrite
后台挂起
  setsid nohup env CUDA_VISIBLE_DEVICES=4 \
  /data/workspace/zhangshiqi/forceWAM/env/.venv/bin/python \
  /data/workspace/zhangshiqi/forceWAM/scripts/compute_lerobot_future_flow_video.py \
  --repo-id /data/workspace/zhangshiqi/forceWAM/grasp_pipette_and_press_button_26ep \
  --output-dir /data/workspace/zhangshiqi/forceWAM/flow_videos \
  --future-step 32 \
  > flow_videos/compute_future_flow.log 2>&1 &
```

# 将光流视频添加到 LeRobot 数据集
```bash
  env/.venv/bin/python scripts/add_flow_videos_to_lerobot.py \
  --repo-id grasp_pipette_and_press_0614_7ep \
  --flow-videos-dir flow_videos/videos \
  --map \
    cam_front=observation.future_flow.cam_front \
    cam_right=observation.future_flow.cam_right \
  --overwrite
```


# 计算3D偏移点
```bash
DATA=grasp_pipette_and_press_button
OUT=outputs/front_scene_flow_grasp_pipette_sam3_tracked_npz

mkdir -p "$OUT"
for EP in $(seq 0 48); do
  echo "===== episode ${EP} ====="
  CUDA_VISIBLE_DEVICES=4 env/.venv/bin/python scripts/visualize_front_scene_flow_episode_sam3_tracked_object.py \
    --repo-id "$DATA" \
    --episode-index "$EP" \
    --future-step 32 \
    --stride 3 \
    --max-depth 2.5 \
    --pair-step 1 \
    --overlay-step 1000000 \
    --sam3-checkpoint /data/shared_workspace/zhangshiqi/hf/SAM/sam3/sam3.pt \
    --sam3-device cuda \
    --sam3-confidence-threshold 0.35 \
    --object-points '238,215;242,250;246,275' \
    --object-negative-points '249,326;259,361;248,301' \
    --save-flow-npz \
    --skip-ply \
    --output-dir "$OUT"
done 2>&1 | tee "$OUT/batch.log"
```


# 归一化
# 根据config修改config-name 需要在config里面修改repo-id
DATA_REPO=data/grasp_pipette_and_press_button_106ep
ASSET_ID=$(basename "$DATA_REPO")
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_tactile_structured_raw_dual_ae \
  --repo-id "$DATA_REPO" \
  --asset-id "$ASSET_ID"


# pi0训练
mkdir -p /data/workspace/zhangshiqi/forceWAM/logs
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /data/workspace/zhangshiqi/forceWAM/env/.venv/bin/python \
  /data/workspace/zhangshiqi/forceWAM/scripts/train.py pi0_xhand_full_finetune \
    --exp-name pi0_xhand_full_finetune_20k_4gpu \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_full_finetune_20k_4gpu.log 2>&1 &




# 1AE + current 5x3 tactile observation tokens 
setsid nohup env \
  HF_LEROBOT_HOME=. \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python \
  scripts/train.py pi0_xhand_tactile_obs_ae_full_finetune \
    --exp-name pi0_xhand_tactile_obs_ae_full_finetune_26ep_0614 \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path /workspace/mnt/sqzhang26/hf_weight/pi0_base/params \
    --data.assets.assets-dir assets/pi0_xhand_tactile_flow_full_finetune \
  > logs/pi0_xhand_tactile_obs_ae_full_finetune_26ep_0615.log 2>&1 &

# 1AE + tactile
setsid nohup env \
  HF_LEROBOT_HOME=. \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python \
  scripts/train.py pi0_xhand_tactile_structured_single_ae \
    --exp-name structured_single_ae_60ep_20k_0616 \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_single_ae_60ep_20k_0616.log 2>&1 &

# 2AE full finetune + tactile + 2D flow
setsid nohup env \
  HF_LEROBOT_HOME=. \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /data/workspace/zhangshiqi/forceWAM/env/.venv/bin/python \
  /data/workspace/zhangshiqi/forceWAM/scripts/train.py pi0_xhand_tactile_flow_full_finetune \
    --exp-name pi0_xhand_tactile_flow_full_finetune \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
    --model.flow-vae-name /data/shared_workspace/zhangshiqi/hf/models--stabilityai--sdxl-vae \
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_flow_full_finetune_20k_4gpu.log 2>&1 &


# 2AE full finetune + tactile + rawTactile
setsid nohup env \
  HF_LEROBOT_HOME=. \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_raw_dual_ae \
    --exp-name pi0_xhand_tactile_structured_raw_dual_ae_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/pi0_xhand_tactile_structured_raw_dual_ae_106ep_20k.log 2>&1 &

# 2AE full finetune + tactile  + mask
setsid nohup env \
  HF_LEROBOT_HOME=. \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py \
    pi0_xhand_tactile_structured_dual_ae_arm_future_hand_mask \
    --exp-name structured_dual_ae_arm_future_hand_mask_106ep_20k \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --data.assets.assets-dir assets/pi0_xhand_tactile_structured_dual_ae \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_dual_ae_arm_future_hand_mask_106ep_20k.log 2>&1 &

# 2AE full finetune + tactile + 3D displacement
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py pi0_xhand_tactile_3dflow_full_finetune \
    --exp-name pi0_xhand_tactile_3dflow_full_finetune \
    --num-train-steps 20000 \
    --batch-size 8 \
    --fsdp-devices 4 \
    --num-workers 2 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
    --data.assets.assets-dir /data/workspace/zhangshiqi/forceWAM/assets/pi0_xhand_tactile_flow_full_finetune
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_3dflow_full_finetune_20k_4gpu.log 2>&1 &

# config总结：

仿FLARE setting
scripts/run_action_aware_stage1_stage2.sh
future_tactile_encoder_pretrain_flare_dit
pi0_xhand_tactile_action_aware_flare_single_ae

pi0_xhand_tactile_structured_raw_dual_ae
pi0_xhand_tactile_structured_raw_single_ae

# 模型推理
```bash
CUDA_VISIBLE_DEVICES=0 env/.venv/bin/python scripts/serve_policy.py --port=8990 policy:checkpoint \
--policy.config=pi0_xhand_tactile_flow_full_finetune \
--policy.dir=checkpoints/pi0_xhand_tactile_flow_full_finetune/pi0_xhand_tactile_flow_full_finetune/29999

CUDA_VISIBLE_DEVICES=0 env/.venv/bin/python scripts/serve_policy.py --port=8990 policy:checkpoint \
--policy.config=pi0_xhand_tactile_3dflow_full_finetune \
--policy.dir=FactileLDM/checkpoints/pi0_xhand_tactile_3dflow_full_finetune/pi0_xhand_tactile_3dflow_full_finetune/29999

CUDA_VISIBLE_DEVICES=0 env/.venv/bin/python scripts/serve_policy.py --port=8990 policy:checkpoint \
--policy.config=pi0_xhand_full_finetune \
--policy.dir=checkpoints/pi0_xhand_full_finetune/pi0_xhand_full_finetune_30k_2gpu/29999

```

# 压缩
tar -czvf data.tar.gz grasp_pipette_and_press_button_26ep_26ep/
# 解压
tar -xzvf 文件名.tar.gz

pip install awscli

aws configure
AWS Access Key ID [None]:
AWS Secret Access Key [None]:
Default region name [None]: huhehaote-1
Default output format [None]: json

aws s3 ls s3://sqzhang26-2   --endpoint-url https://eos-huhehaote-1.cmecloud.cn

#下载文件
aws s3 cp s3://sqzhang26-2/data.tar.gz data.tar.gz \
  --endpoint-url https://eos-huhehaote-1.cmecloud.cn

#下载整个目录
aws s3 cp s3://sqzhang26-2/path/to/folder ./folder \
  --recursive \
  --endpoint-url https://eos-huhehaote-1.cmecloud.cn

# 大文件上传  mac支持\
s3cmd put /Users/babyna/Downloads/grasp_pipette_and_press_w_force_w_depth_0615_good_tactile_26ep.zip \
  s3://sqzhang26-2/grasp_pipette_and_press_w_force_w_depth_0615_good_tactile_26ep.zip
