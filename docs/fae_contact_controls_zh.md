# 冻结 FAE 的接触信息读出：补充对照

目的：检验预测 latent 是否包含与真实未来接触状态相关、可被已有小读出头读出的信息。
不是比较 fixed/refined 谁更好，也不要求精确重建连续力。两种 latent 都如实报告。

## 输入和计算

- 使用已经完成的 train/val/test 特征缓存及 `best_probe.npz`、同名 JSON。
- 不训练主模型或读出头，不加载主策略 checkpoint，不解码视频。
- 仅在测试集运行一次冻结小读出头，支持 `NUM_DEVICES=4`。fixed 只解码一次，refined 按缓存更新时刻解码。
- 后续打乱通过重排小头输出实现。对于此逐样本、确定性的冻结读出头，这与重排 latent 后再次解码等价，避免重复 GPU 推理。
- Bootstrap 和排序/AP 在 CPU 上执行，多张 GPU 不会加速该阶段。日志分别打印 PREDICT、BOOTSTRAP、SHUFFLE 进度。
- 不需要新增依赖，使用服务器原有 `env/.venv/bin/python`。无需 uv 或安装操作。

## 对照与指标

1. **prior**：训练集 task × finger × patch 接触频率，Laplace 平滑 alpha=1。使用 offset=0 的有效未来标签，每个缓存窗口内只计一次；不同窗口仍有重叠，与原实验一致。此基线额外获得任务身份，因此能检验任务接触先验的解释。概率不依赖测试标签，F1 阈值只在验证集网格 0.05..0.95 上按任务宏平均选择。
2. **fixed/refined**：使用原 best_probe，沿用原验证集阈值，不重新根据测试集选择 checkpoint 或阈值。
3. **fixed_shuffled/refined_shuffled**：默认 20 次随机重配。同任务、同更新时刻，必须来自另一条 episode，供体有效范围必须覆盖受体评价范围。先均匀选合格的其他 episode，再均匀选其合格 anchor，允许重复。标签、当前触觉和掩码始终属于受体。两种模式共用供体映射。不足两条合格轨迹会报错，不静默删样本。
4. **persistence**：当前接触状态延续到未来，帮助判断未来预测是否超越状态延续。

随机重配不是严格的一一置换，不生成置换检验 p 值。每次先算完整指标，再汇总指标均值、标准差和范围；不先平均预测分数来构造另一个平滑基线。

报告 AP（按相同分数分组的精确 average precision）、F1、precision、recall、接触正例比例、样本数及宏平均有效任务数。主报告保留原始 logits 排序。AP 的参照不是固定 0.5；无正例 AP 记为 null，不能当成零。

## 评价范围

- `all_future`：所有有效未来帧；严格排除已观测帧和 episode 末尾 padding。
- `offset_N`：从 chunk 起点算第 N 帧更新时刻。
- `lead_N`：相对本次更新，再过 N 个触觉帧的预测，**不同于 offset**。
- `contact_change_frames`：未来相对当前存在任意 patch 接触变化的帧，保留该帧全部 patch。
- `changed_patches`：只保留发生变化的 patch。
- `onset_candidates`：当前未接触的 patch，预测未来是否接触；包含一直未接触的负例。
- `release_candidates`：当前接触的 patch，预测未来是否解除；包含一直接触的负例。取反 logits，阈值取 `1 - contact_threshold`，不使用测试集调参。

change 两个子集按未来结果筛选，只是诊断，不能代替全测试集结果。onset/release 候选集合仅依赖当前观测，便于查看接触变化的预测能力。

## 置信区间

`all_future`、`contact_change_frames`、`onset_candidates`、`release_candidates` 默认各做 200 次任务分层、配对的 episode bootstrap。所有方法在每次抽样中使用相同受体 episode 权重；同时给出真实读出相对 prior、persistence、对应 shuffled 的 AP/F1 差值区间。

**shuffled 的置信区间固定使用预先指定的第 0 次供体映射**（seed，不挑最好或最差），条件于该映射及已训练模型。其他 19 次的均值、标准差和范围另列，不能当作置信区间、独立轨迹或训练 seed。区间不覆盖主模型/读出头重新训练的不确定性。供体轨迹的独立重采样也不包含在该条件区间中。

## 在服务器后台运行已有实验的补充评估

```bash
cd /workspace/mnt/sqzhang26/FactileLDM
git pull --rebase --autostash origin main
```

确认拉取成功后运行，复用已完成的单卡缓存也可以四卡解码：

```bash
mkdir -p logs
RUN_TAG=$(date +%Y%m%d_%H%M%S)
PROBE_DIR=outputs/frozen_fae_decoder_59999_common_tasks_pyav/fae_linear_20ep_20261027_211316
CONTROLS_DIR="$PROBE_DIR/contact_controls_${RUN_TAG}"
LOG_FILE="logs/fae_contact_controls_${RUN_TAG}.log"

nohup env \
  OUTPUT_ROOT=outputs/frozen_fae_decoder_59999_common_tasks_pyav \
  PROBE_DIR="$PROBE_DIR" CONTROLS_DIR="$CONTROLS_DIR" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_DEVICES=4 \
  PROBE_BATCH_SIZE=128 SHUFFLE_REPEATS=20 BOOTSTRAP_SAMPLES=200 \
  PYTHONUNBUFFERED=1 \
  bash scripts/run_frozen_fae_decoder.sh controls \
  > "$LOG_FILE" 2>&1 < /dev/null &

RUN_PID=$!
echo "$RUN_PID" > "${LOG_FILE%.log}.pid"
echo "PID: $RUN_PID"
echo "日志: $LOG_FILE"
echo "结果: $CONTROLS_DIR"
tail -f "$LOG_FILE"
```

`Ctrl+C` 退出 tail，不会终止后台评估。单卡只需改 `CUDA_VISIBLE_DEVICES=0 NUM_DEVICES=1`。
`controls` 是独立阶段，不要改成 `all`；它不会重训或重新提取主模型特征。目录已存在则拒绝覆盖。

## 输出和完成标志

- `contact_controls.json`：完整指标、配对差异区间、参数和实验边界说明。
- `contact_controls.csv`：每任务/宏平均 × 方法 × scope × 重复编号；`repeat=-1` 是真实配对及确定性基线，随机对照为 0..19。
- `shuffle_summary.json`：各 scope 的真实指标及打乱均值/标准差/范围/差值。
- `contact_prior.json`：训练集先验及验证集选出的阈值。
- `contact_predictions.npz`：标签、索引、掩码及固定读出头原始 logits；不包含大 latent。
- `donors_000.npy` 等：供体全局 anchor 行号，形状 `[anchors, offsets]`，与 predictions 的行顺序对应。无有效未来的行不参与评分。
- `run.json`：checkpoint 哈希、缓存协议及索引指纹，完成时 `status=complete`。日志最后为 `CONTACT CONTROLS COMPLETE`。

原 `best_probe.npz`、训练 history 和 `test_metrics.json` 均不改动。

## 验证与解释边界

不依赖 GPU/策略模型的合成测试：

```bash
PYTHONPATH=src:. env/.venv/bin/python -m unittest scripts.eval_fae_contact_controls_test -v
```

代码提交前仅运行了小型 NumPy 合成测试及 Python/Bash 语法检查；真实缓存、Flax/JAX 解码和四卡执行仍需在服务器验证。

真实配对优于先验及同任务跨轨迹重配，支持样本对应的未来接触信息可读出；它不能单独证明策略因果使用该信息，也不能排除当前触觉、动作或任务阶段与未来接触的相关性。没有超过 persistence 不等于完全没有信息，但不能声称超越当前状态延续的预测能力。测试集仅确认对读出头留出，不能由该实验直接推断原主模型未见过这些轨迹。
