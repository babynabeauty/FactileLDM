训练 ForceWAM 模型
本机环境
代码路径：`/data/workspace/zhangshiqi/forceWAM`
已配置的 Python：`3.11.13`
已配置的 uv 环境：`/data/workspace/zhangshiqi/forceWAM/env/.venv`

每次运行前先进入项目目录并设置环境变量：
```bash
cd /data/workspace/zhangshiqi/forceWAM
export UV_PROJECT_ENVIRONMENT=/data/workspace/zhangshiqi/forceWAM/env/.venv
export UV_CACHE_DIR=/data/workspace/zhangshiqi/forceWAM/.uv-cache
export UV_PYTHON_INSTALL_DIR=/data/workspace/zhangshiqi/forceWAM/.uv-python
```

环境已经按 Python 3.11 安装完成，包含 base openpi 依赖、`rlds` 依赖组、`lerobot`、`dlimp`、`tensorflow==2.15.0`、`torch==2.7.1` 和 `jax==0.5.3`。

验证命令：
```bash
env/.venv/bin/python -c "import openpi, lerobot, dlimp, tensorflow, torch, jax; print('ok')"
uv run --no-sync scripts/train.py --help
```

注意：本机当前 `nvidia-smi` 无法连接 NVIDIA driver，因此环境能导入，但训练需要在 NVIDIA driver/GPU 可见的节点上运行。

训练流程
环境激活：
```bash
cd /data/workspace/zhangshiqi/forceWAM
export UV_PROJECT_ENVIRONMENT=/data/workspace/zhangshiqi/forceWAM/env/.venv
export UV_CACHE_DIR=/data/workspace/zhangshiqi/forceWAM/.uv-cache
export UV_PYTHON_INSTALL_DIR=/data/workspace/zhangshiqi/forceWAM/.uv-python
```
在开始模型训练前，假定已经得到了一个 LeRobot v2.0 格式的数据集。
假定数据集的 id 为 `<repo_id>`，以下为 repo id 为 `llly/vga_0120` 为例。
1.  数据预处理
(1) 为 LeRobot 数据集添加`observation.effort` 字段
```bash
uv run --no-sync python scripts/add_force.py --repo_id llly/vga_0120
```
这一步完成后，可以通过检查 LeRobot 数据集目录下的 meta/info.json 来查看是否成功。如果里面多了一个 "observation.effort"，则表明成功为 LeRobot 数据集添加了 6 维力。
(2) 计算光流图像
这一步可能需要非常长的时间，因为需要计算每一帧的光流。
```bash
    CUDA_VISIBLE_DEVICES=4 \
env/.venv/bin/python scripts/compute_lerobot_future_flow_video.py \
  --repo-id /data/shared_workspace/zhangshiqi/dataset/tactile_xhand_ur7e/grasp_pipette_and_press_button \
  --output-dir ./flow_videos \
  --future-step 32 \
  --overwrite
后台挂起
  setsid nohup env CUDA_VISIBLE_DEVICES=4 \
  /data/workspace/zhangshiqi/forceWAM/env/.venv/bin/python \
  /data/workspace/zhangshiqi/forceWAM/scripts/compute_lerobot_future_flow_video.py \
  --repo-id /data/workspace/zhangshiqi/forceWAM/grasp_pipette_and_press_button \
  --output-dir /data/workspace/zhangshiqi/forceWAM/flow_videos \
  --future-step 32 \
  > /data/workspace/zhangshiqi/forceWAM/flow_videos/compute_future_flow.log 2>&1 &

# 处理3D点云
DATA=/data/workspace/zhangshiqi/forceWAM/grasp_pipette_and_press_button
OUT=/data/workspace/zhangshiqi/forceWAM/outputs/front_scene_flow_grasp_pipette_sam3_tracked_npz

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
参数说明：
- repo-id：LeRobot 数据集的 id
- output-dir：预测的光流输出的路径
- future-step：用未来第几帧的图像来计算光流，future-step = 32 代表用未来第 32 帧的图像和当前帧图像计算一个光流
(3) 将光流视频添加到 LeRobot 数据集
```bash
  env/.venv/bin/python scripts/add_flow_videos_to_lerobot.py \
  --repo-id /data/workspace/zhangshiqi/forceWAM/grasp_pipette_and_press_button \
  --flow-videos-dir /data/workspace/zhangshiqi/forceWAM/flow_videos/videos \
  --map \
    cam_front=observation.future_flow.cam_front \
    cam_right=observation.future_flow.cam_right \
  --overwrite

```
2.  模型训练
编辑 src/openpi/training/config.py 文件，修改 pi0_latent_flow_noise 配置，主要是修改数据集的 repo id。
TrainConfig(
        name="pi0_latent_flow_noise",  # 训练配置的名称
        model=pi0_config.Pi0LatentFlowConfig(
            action_horizon=32,  # action chunk 的大小
            effort_type=EffortType.MOT,
            effort_dim=6,  # 6-axis force sensor
            # new parms
            force_input_frames=10,
            distill_layer_indices=(8, 12, 16),
            future_force_align_loss_weight=0.5, # 蒸馏损失的权重
            future_flow_align_loss_weight=0.5,
            student_future_query_noise_scale_max=0.3,
            student_future_query_noise_start_ratio=0.3,
            student_future_query_noise_end_ratio=0.7,
            use_future_rgb_instead_of_flow = False # 是否使用rgb替代光流，设置为 False，使用未来光流；设置为True，使用未来RGB图像
        ),
        data=LeRobotOptimalFlowDataConfig(
            repo_id="llly/all_0409_stage_flow", # 修改为你的lerobot 数据集 id
            effort_history=tuple(list((4 * i - 36 for i in range(10))) + list(range(1, 33))),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False, # Yuanluo actions are absolute
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        num_train_steps=30_000, # Default to 30k steps, adjust as needed
        # num_workers=8,
        batch_size=16,
        save_interval=15000,
        keep_period=15000,
        ema_decay = None # 节省显存
    ),
开始模型训练：

source /workspace/mnt/sqzhang26/FactileLDM/env/.venv/bin/activate

# 归一化，根据config修改config-name 需要在config里面修改repo-id
mkdir -p /data/workspace/zhangshiqi/forceWAM/.hf_datasets_cache
HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
HF_HOME=/data/shared_workspace/zhangshiqi/hf \
HF_DATASETS_CACHE=/data/workspace/zhangshiqi/forceWAM/.hf_datasets_cache \
CUDA_VISIBLE_DEVICES=7 \
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_tactile_flow_full_finetune 
  
  <!-- --max-frames 10000 -->

CUDA_VISIBLE_DEVICES="5" \
HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
HF_HOME=/data/shared_workspace/zhangshiqi/hf \
HF_DATASETS_CACHE=/data/workspace/zhangshiqi/forceWAM/.hf_datasets_cache \
env/.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_full_finetune \
  --max-frames 10000

# pi0
mkdir -p /data/workspace/zhangshiqi/forceWAM/logs

setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  HF_HOME=/data/shared_workspace/zhangshiqi/hf \
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
    --weight-loader.params-path /data/shared_workspace/zhangshiqi/hf/pi0_base_jax/pi0_base/params \
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_full_finetune_30k_2gpu.log 2>&1 &

# pi0 + current 5x3 tactile observation tokens + single action expert
cd /workspace/mnt/sqzhang26/FactileLDM
mkdir -p .hf_cache .hf_datasets_cache logs

setsid nohup env \
  HF_LEROBOT_HOME=/workspace/mnt/sqzhang26/FactileLDM \
  HF_HOME=/workspace/mnt/sqzhang26/hf_weight \
  HF_DATASETS_CACHE=/workspace/mnt/sqzhang26/FactileLDM/.hf_datasets_cache \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /workspace/mnt/sqzhang26/FactileLDM/env/.venv/bin/python \
  /workspace/mnt/sqzhang26/FactileLDM/scripts/train.py pi0_xhand_tactile_obs_ae_full_finetune \
    --exp-name pi0_xhand_tactile_obs_ae_full_finetune_26ep_0614 \
    --num-train-steps 20000 \
    --batch-size 4 \
    --num-workers 0 \
    --save-interval 5000 \
    --keep-period 5000 \
    --no-wandb-enabled \
    --overwrite \
    --weight-loader.params-path /workspace/mnt/sqzhang26/hf_weight/pi0_base/params \
    --data.assets.assets-dir /workspace/mnt/sqzhang26/FactileLDM/assets/pi0_xhand_tactile_flow_full_finetune \
  > /workspace/mnt/sqzhang26/FactileLDM/logs/pi0_xhand_tactile_obs_ae_full_finetune_26ep_0614.log 2>&1 &

# pi0 + full VLM + full image encoder + 2AE full finetune + tactile + 2D flow
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  HF_HOME=/data/shared_workspace/zhangshiqi/hf \
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
    --weight-loader.params-path /data/shared_workspace/zhangshiqi/hf/pi0_base_jax/pi0_base/params \
    --model.flow-vae-name /data/shared_workspace/zhangshiqi/hf/models--stabilityai--sdxl-vae \
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_flow_full_finetune_30k_1gpu.log 2>&1 &

# pi0 + full VLM + full image encoder + 2AE full finetune + tactile + no flow
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  HF_HOME=/data/shared_workspace/zhangshiqi/hf \
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
    --weight-loader.params-path /data/shared_workspace/zhangshiqi/hf/pi0_base_jax/pi0_base/params \
    > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_noflow_full_finetune_30k_2gpu.log 2>&1 &

# pi0 + full VLM + full image encoder + 2AE full finetune + tactile + 3D displacement
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  HF_HOME=/data/shared_workspace/zhangshiqi/hf \
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
    --weight-loader.params-path /data/shared_workspace/zhangshiqi/hf/pi0_base_jax/pi0_base/params \
    --data.assets.assets-dir /data/workspace/zhangshiqi/forceWAM/assets/pi0_xhand_tactile_flow_full_finetune
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_3dflow_full_finetune_30k_1gpu.log 2>&1 &

# pi0 + full VLM + full image encoder + 2AE full finetune + raw tactile + 2D flow
setsid nohup env \
  HF_LEROBOT_HOME=/data/workspace/zhangshiqi/forceWAM \
  HF_HOME=/data/shared_workspace/zhangshiqi/hf \
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
    --weight-loader.params-path /data/shared_workspace/zhangshiqi/hf/pi0_base_jax/pi0_base/params \
    --data.assets.assets-dir /data/workspace/zhangshiqi/forceWAM/assets/pi0_xhand_tactile_flow_full_finetune \
  > /data/workspace/zhangshiqi/forceWAM/logs/pi0_xhand_tactile_grid_flow.log 2>&1 &


单卡训练：
```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-sync scripts/train.py pi0_latent_flow_noise --exp-name=pi0_latent_flow_noise --overwrite
```
双卡训练：
```bash
CUDA_VISIBLE_DEVICES=5,6 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --no-sync scripts/train.py pi0_latent_flow_noise --exp-name=pi0_latent_flow_noise  --fsdp-devices 2   --overwrite
```
3. 模型推理
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
