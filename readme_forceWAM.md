每次运行前先进入项目目录并设置环境变量：
```bash
cd FactileLDM
source env/.venv/bin/activate
export HF_LEROBOT_HOME=/workspace/mnt/sqzhang26/FactileLDM
export HF_HUB_OFFLINE=1
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

# 将光流视频添加到 LeRobot 数据集
```bash
  env/.venv/bin/python scripts/add_flow_videos_to_lerobot.py \
  --repo-id grasp_pipette_and_press_0614_7ep \
  --flow-videos-dir flow_videos/videos \
  --map \
    cam_front=observation.future_flow.cam_front \
    cam_right=observation.future_flow.cam_right \
  --overwrite


# 计算3D偏移点
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


# 归一化
# 根据config修改config-name 需要在config里面修改repo-id
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_tactile_structured_single_ae \
  --repo-id data/grasp_pipette_and_press_button_0616_59ep \ 
  --asset-id grasp_pipette_and_press_button_0616_59ep 
 
  #--repo-id  实际数据集路径，相对于 FactileLDM
  #--asset-id  归一化 stats 保存/读取的名字


export HF_LEROBOT_HOME=/workspace/mnt/sqzhang26/FactileLDM/data
export HF_HUB_OFFLINE=1
# pi0训练
mkdir -p /data/workspace/zhangshiqi/forceWAM/logs
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  CUDA_VISIBLE_DEVICES=4,5 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /data/workspace/zhangshiqi/forceWAM/env/.venv/bin/python \
  /data/workspace/zhangshiqi/forceWAM/scripts/train.py pi0_xhand_full_finetune \
    --exp-name pi0_xhand_full_finetune_30k_2gpu \
    --num-train-steps 30000 \
    --batch-size 2 \
    --num-workers 0 \
    --save-interval 5000 \
    --keep-period 10000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_full_finetune_30k_2gpu.log 2>&1 &

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
    --batch-size 4 \
    --num-workers 0 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path /workspace/mnt/sqzhang26/hf_weight/pi0_base/params \
    --data.assets.assets-dir assets/pi0_xhand_tactile_flow_full_finetune \
  > logs/pi0_xhand_tactile_obs_ae_full_finetune_26ep_0615.log 2>&1 &

# 1AE + tactile

DATA_REPO="data/grasp_pipette_and_press_button_0616_59ep"
ASSET_ID="$(basename "$DATA_REPO")"

setsid nohup env \
  HF_LEROBOT_HOME=. \
  HF_DATASETS_CACHE=.hf_datasets_cache \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python \
  scripts/train.py pi0_xhand_tactile_structured_single_ae \
    --exp-name structured_single_ae_60ep_5k_0616 \
    --data.repo-id "$DATA_REPO" \
    --data.assets.asset-id "$ASSET_ID" \
    --num-train-steps 5000 \
    --batch-size 4 \
    --fsdp-devices 4 \
    --num-workers 0 \
    --save-interval 1000 \
    --keep-period 1000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_single_ae_60ep_5k_0616.log 2>&1 &

# 2AE full finetune + tactile
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  CUDA_VISIBLE_DEVICES=6,7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /data/workspace/zhangshiqi/forceWAM/env/.venv/bin/python \
  /data/workspace/zhangshiqi/forceWAM/scripts/train.py pi0_xhand_tactile_noflow_full_finetune \
    --exp-name pi0_xhand_tactile_noflow_full_finetune \
    --num-train-steps 30000 \
    --batch-size 4 \
    --num-workers 0 \
    --save-interval 5000 \
    --keep-period 10000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
    > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_noflow_full_finetune_30k_2gpu.log 2>&1 &

# 2AE full finetune + tactile + 2D flow
setsid nohup env \
  HF_LEROBOT_HOME=. \
  CUDA_VISIBLE_DEVICES=7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /data/workspace/zhangshiqi/forceWAM/env/.venv/bin/python \
  /data/workspace/zhangshiqi/forceWAM/scripts/train.py pi0_xhand_tactile_flow_full_finetune \
    --exp-name pi0_xhand_tactile_flow_full_finetune \
    --num-train-steps 30000 \
    --batch-size 1 \
    --num-workers 0 \
    --save-interval 5000 \
    --keep-period 10000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
    --model.flow-vae-name /data/shared_workspace/zhangshiqi/hf/models--stabilityai--sdxl-vae \
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_flow_full_finetune_30k_1gpu.log 2>&1 &

# 2AE full finetune + tactile 
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
    --batch-size 4 \
    --fsdp-devices 4 \
    --num-workers 0 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
  > logs/structured_dual_ae_arm_future_hand_mask_106ep_20k.log 2>&1 &

# 2AE full finetune + tactile + 3D displacement
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  CUDA_VISIBLE_DEVICES=7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py pi0_xhand_tactile_3dflow_full_finetune \
    --exp-name pi0_xhand_tactile_3dflow_full_finetune \
    --num-train-steps 30000 \
    --batch-size 1 \
    --num-workers 0 \
    --save-interval 5000 \
    --keep-period 10000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
    --data.assets.assets-dir /data/workspace/zhangshiqi/forceWAM/assets/pi0_xhand_tactile_flow_full_finetune
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_3dflow_full_finetune_30k_1gpu.log 2>&1 &

# 2AE full finetune + raw tactile + 2D flow
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  CUDA_VISIBLE_DEVICES=7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train.py pi0_xhand_tactile_grid_flow \
    --exp-name pi0_xhand_tactile_grid_flow \
    --num-train-steps 30000 \
    --batch-size 1 \
    --num-workers 0 \
    --save-interval 5000 \
    --keep-period 10000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path checkpoints/pi0_base/params \
    --data.assets.assets-dir /data/workspace/zhangshiqi/forceWAM/assets/pi0_xhand_tactile_flow_full_finetune \
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_grid_flow.log 2>&1 &



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



 rsync -av --progress zhangshiqi@211.86.155.48:/data/workspace/zhangshiqi/forceWAM/checkpoints/pi0_xhand_tactile_forceonly_full_finetune/pi0_xhand_tactile_forceonly_full_finetune/29999 //home/sai/zsq/FactileLDM/checkpoints/pi0_xhand_tactile_forceonly_full_finetune/pi0_xhand_tactile_forceonly_full_finetune

# 压缩
tar -czvf data.tar.gz grasp_pipette_and_press_button_26ep_26ep/
# 解压
tar -xzvf 文件名.tar.gz

pip install awscli

aws configure
AWS Access Key ID [None]: URulYpw0hpArLO6SL8Z4ydO61GKN
AWS Secret Access Key [None]: DW3uAAS84SG7pTM3KWs85O3w2dL6GC
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

# 大文件上传  mac支持
s3cmd put /Users/babyna/Downloads/grasp_pipette_and_press_w_force_w_depth_0615_good_tactile_26ep.zip \
  s3://sqzhang26-2/grasp_pipette_and_press_w_force_w_depth_0615_good_tactile_26ep.zip