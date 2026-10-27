# 冻结 FAE 的未来接触／力 decoder：服务器运行

## 已接通的流程

`prepare → extract → train → evaluate`。只有 `extract` 加载主模型；后两步只读缓存，不加载、更不更新 FAE。新入口不调用旧的 `train_future_tactile_token_probe.py`。

已按本次实验配置启动脚本 `scripts/run_frozen_fae_decoder.sh`：

- 配置：`pi0_xhand_dual_patch_pretrained_f4_h16_async`。
- 主模型：`checkpoints/pi0_xhand_dual_patch_pretrained_f4_h16_async/pi0_xhand_dual_patch_pretrained_f4_h16_async_task12345_2_patch_pretrain_fast_0709/59999`。
- decoder 训练池：`data/task12345-2`，用户提供元数据为 559 个 episodes、5 个任务。
- 测试数据源：`data/task-test-collect`，用户提供元数据为 64 个 episodes、5 个任务；当前默认只评估其中与训练池共有的四个任务，实际测试 episode 数由筛选后清单决定。
- 按任务分层从训练池留出约 10% 的 episodes 验证；不使用测试集选择 epoch、分类阈值或缩放统计。实际计数由服务器 metadata 决定。

测试数据必须也未参与主模型训练。两个库的 `info.json` 即使都写 `splits.train`，在本实验中角色仍由 `splits.json` 决定，不会自动合并。

两个库的任务名称不要求逐字相同。训练／验证按训练库原始名称分层，测试按测试库原始名称分组并计算宏平均；名称并集只用来分配指标分组 ID，不作为模型输入，也不自动猜测同义任务。`prepare` 会输出各 split 的任务名称，便于检查测试组是否对应预期任务集合。

本次日志确认：训练有 Pipette Press、测试有 Cabinet Retrieval，其余四项相同。因此启动脚本默认 `TEST_TASK_SCOPE=common`，仅测试 Cob Grasp、Bottle Grasp、Button Press、Sponge Wipe，保留训练／验证池的五项任务。排除的测试 episode ID 记录在 `splits.json`。这是按完全相同的原始任务标签取交集，不猜测同义描述。Python 入口 `prepare --test-task-scope all` 可保留所有测试任务。

提取阶段直接读取本地 LeRobot v2.1 的 Parquet 和每 episode 的 MP4，使用 PyArrow／OpenCV，不依赖 LeRobot 包的 `common.datasets` 或 `datasets` 命名空间，也不改动原策略训练 loader。只解码当前 RGB 帧并按 BGR→RGB 转换；状态窗口的首尾补齐均返回掩码。实际帧号、时间戳、视频 FPS 和长度必须与 metadata 一致。主模型仍使用原 config 的输入变换及归一化统计。

## 首次运行

将改动同步到服务器，在原训练环境中运行：

```bash
cd /workspace/mnt/sqzhang26/FactileLDM
# 脚本直接使用本仓库 env/.venv/bin/python。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_frozen_fae_decoder.sh all
```

脚本默认使用用户指定的服务器目录。可用 `REPO_ROOT`、`POLICY_PARAMS`、`CONFIG_NAME` 覆盖。Python 固定使用 `$REPO_ROOT/env/.venv/bin/python`，并清除外部 `PYTHONPATH` 和 `PYTHONHOME`，启动时显示实际解释器、`sys.prefix` 和 Flax 路径。checkpoint 参数可以指向 `59999` 或其 `params` 子目录。

归一化统计默认从该 checkpoint 的 `assets/*/norm_stats.json` 找唯一文件；若不存在或有多个，明确指定原主模型训练时的文件，不能重新用测试数据计算：

```bash
NORM_STATS=/path/to/original/norm_stats.json \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_frozen_fae_decoder.sh all
```

标签在策略归一化前生成。默认接触定义为 patch 内任一 taxel 的三轴向量模长大于 **1.0 原生传感器单位**，加权力的 sigmoid 温度为 **0.5 同单位**；这是显式的标签定义，不代表已标定为牛顿。应结合训练池的真实读数确认阈值。需要修改时：

```bash
CONTACT_THRESHOLD=1.0 CONTACT_TEMPERATURE=0.5 FORCE_UNIT=dataset_native \
OUTPUT_ROOT=outputs/frozen_fae_decoder_native_v1 \
bash scripts/run_frozen_fae_decoder.sh all
```

修改 checkpoint、标签定义、数据划分、采样 stride 或归一化统计后必须换缓存目录，代码会拒绝混用。`FORCE_UNIT` 只记录单位，**不会**执行传感器标定或单位转换。

`CONTACT_THRESHOLD` 与 `CONTACT_TEMPERATURE` 只在 `extract` 生成真实未来标签及 persistence 当前值时使用，不改动已训练 FAE 的内部触觉编码。前者定义 patch 接触标签，二者共同定义局部力的 sigmoid 加权平均；后续 decoder 用这些固定标签计算损失和指标。F1 的预测概率阈值是另一参数，在 `train` 末尾用验证集选择，和这里的温度无关。

## 分阶段运行及资源

```bash
bash scripts/run_frozen_fae_decoder.sh prepare
bash scripts/run_frozen_fae_decoder.sh extract
bash scripts/run_frozen_fae_decoder.sh train
bash scripts/run_frozen_fae_decoder.sh evaluate
```

主模型提取默认单 GPU、batch 2、每 4 帧取一个 anchor，按原配置递推 offsets 0/4/8/12，十步 Euler，fixed/refined 共用初始噪声。默认缓存 float16 特征，标签为 float32，建议预留几十 GB 磁盘。`EXTRACT_BATCH_SIZE`、`CACHE_STRIDE`、`NUM_WORKERS` 可调；采样规则在看测试结果之前锁定。

缓存按固定大小分片，评估是一次有限遍历，没有无限 loader 或 drop-last。每个样本保留 episode、frame、task；根据原始 episode 长度及 padding mask 排除尾帧。`extract --resume` 跳过已完成的整个 split；中断的 split 会从头重算并原子覆盖其分片，不混用另一套参数。

默认训练 20 个 decoder epochs，batch 128、Adam 1e-3；这是初始配置，不是已经调优的结果。验证损失选择 checkpoint，再在验证集固定接触 F1 的阈值。主结果使用共享线性双头（82,000 参数），同一个 decoder 平衡读取 fixed/refined 与各 offset。力按 **decoder 训练集活动 patch 的逐轴 RMS** 缩放后拟合，报告时还原至原生单位；缩放不影响接触标签。

新增模型使用新的 `PROBE_DIR`，不会覆盖已有 best checkpoint：

```bash
# 可选的非线性 probe，仍冻结主模型，共用已有缓存。
DECODER_TYPE=mlp bash scripts/run_frozen_fae_decoder.sh train
DECODER_TYPE=mlp bash scripts/run_frozen_fae_decoder.sh evaluate

# 当前/历史触觉表征的独立 probe；带 segment/offset 信息。
SOURCE=history bash scripts/run_frozen_fae_decoder.sh train
SOURCE=history bash scripts/run_frozen_fae_decoder.sh evaluate
```

history 基线合并每指 pooled-history/current 两个 tokens，附加 segment one-hot 和当前 offset；输入比 FAE probe 宽，结果会记录其参数量，不能宣称两者等参数。每种表示独立训练 probe；同一 FAE 的 fixed/refined 共用 probe。

## 结果及解释

输出默认在 `outputs/frozen_fae_decoder_59999_common_tasks/`，与之前全任务尝试使用的目录分开：

- `splits.json`：实际数据路径、元数据摘要及 episode 划分。
- `cache/protocol.json`：主模型/config、checkpoint 文件清单摘要、归一化文件 SHA、标签阈值和单位、采样参数。
- `cache/{train,val,test}/index.json` 和分片：有限、可追溯的冻结特征与真实标签。
- `fae_linear/best_probe.npz` / `.json`：仅 decoder 参数、验证选定阈值、force scale、参数量和来源。
- `fae_linear/training_history.json`、`validation_metrics.json`。
- `fae_linear/test_metrics.json`、`test_metrics.csv`：完整测试结果及主指标的置信区间。

主指标为 **contact AP** 与 **active-force RMSE**；辅助报告 F1、active-force MAE、inactive-force RMSE。AP 使用按分数并列组计算的 non-interpolated average precision，不与梯形积分 AUPRC 混写。F1 累计 TP/FP/FN，不平均 batch F1。力误差按真实接触标签选评估区域，**不使用真值接触门控预测力**。

报告分任务、分 offset、以及“任一 patch 相对当前状态发生接触变化的未来帧”子集；子集包含这些帧的全部 patch，不等于只评估发生变化的 patch。总结果先各任务计算再宏平均；没有正例的任务 AP 或没有活动区域的力指标为 null，并记录宏平均有效任务数。`all_future` 汇总多个预测时距的合法预测实例，相同未来物理帧可能对应不同 anchor/offset 的不同预测。

默认进行 200 次任务内 episode bootstrap，fixed/refined/persistence 使用相同抽样，报告全未来宏平均及 `refined − fixed` 的 95% 区间。误差指标的负差值表示 refined 更好。置信区间反映固定策略/readout 下的轨迹变化，不代表训练 seed 不确定性。

该实验是使用示教观测的离线回放：证明可解码性不等于证明策略利用这些信息改善闭环控制。新脚本不增加 HAE 输入，不向 FAE 输入真实动作或尚未观测的触觉，不将 decoder 接入部署动作回路。

## 本机验证范围

可在 CPU 的独立测试环境运行：

```bash
PYTHONPATH=src:. JAX_PLATFORMS=cpu python -m pytest \
  --confcutdir=src/openpi/models -q \
  src/openpi/models/future_patch_probe_test.py \
  scripts/extract_future_patch_probe_test.py \
  src/openpi/shared/future_patch_probe_utils_test.py \
  src/openpi/shared/future_patch_probe_data_test.py \
  scripts/run_future_patch_probe_test.py
```

包含原有 decoder/因果切片测试、episode 划分和共同任务筛选、严格参数校验、AP 并列分数、活动力误差、配对 episode CI，以及小型合成特征缓存上的 decoder 训练/保存/重载/测试集评估。新增真实 Parquet/MP4 测试覆盖视频当前帧、颜色通道、跨 episode 读取、时间戳和末尾掩码。它们不替代服务器上的真实 checkpoint、数据视频及 GPU 提取验证。
