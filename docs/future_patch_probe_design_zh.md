# 未来触觉 patch 解码诊断：接入设计与评估协议

> 接入更新：现已新增 `scripts/run_future_patch_probe.py` 与 `scripts/run_frozen_fae_decoder.sh`，连接严格权重恢复、数据划分、特征缓存、readout 训练及测试指标。具体命令、559/64 条数据协议和已实现边界见 [服务器运行说明](frozen_fae_decoder_run_zh.md)。真实服务器实验结果仍待运行。

## 1. 实验问题与实现边界

本实验回答：冻结的 ReTouch 部署态 FAE latent，是否包含可以读出的真实未来接触状态与局部三轴力信息？仅拟合 HAE 隐状态不能回答这一问题。成功解码支持“包含信息”，不单独证明策略利用这些信息改善控制；NZE3 要求的 generic recurrent latent 因果消融仍是另一项实验。

本改动提供 decoder、损失/时间掩码、部署特征提取组件以及完整的缓存、训练和评估入口，不修改原策略，也不把旧检查点再训练成触觉预测器。尚无真实 checkpoint 与数据的运行结果；下文结果表仍为待执行协议。

代码核对基于仓库提交 `7ea50c280bce78c159ea560758272abd4f3511f6`，维度参照 `pi0_xhand_dual_patch_f4_h16_async`。实际运行必须使用与待评估 checkpoint 配套的配置，不能仅凭配置名称认定论文模型已经恢复。

## 2. 从哪里接 decoder

使用 Pi0LatentFlow 的 sample_actions_cached_vlm_async_ae_slow/fast 接口返回的 final_future_hidden。它来自最后一次 Euler 网络调用中的 async_layer_hiddens[-1]，实际层为 distill_layer_indices[-1]，不是自动取网络末层。配置值 12 为零基索引，须与论文层编号口径核对并记录。不要改用训练路径中含真实动作构造的 noisy actions 来抽取特征。

目标配置为 4 个时间段、每段 4 帧、5 根手指、隐藏宽度 1024，因此输入为 [B,20,1024]，token 顺序为 segment-major、finger-minor。接入前严格核验 checkpoint/config、所有策略参数恢复、token 数与顺序；加载不完整即报错，禁止静默保留随机参数。冻结整套策略并使用推理模式，优化器只包含 probe 参数。

主实验接 projector 前的 FAE latent。decoder 只能读取这些 latent，不能额外读取当前触觉、机器人状态、动作、图像或任务标签；这些信息只可作为单独基线输入。HAE latent 可另训参考 probe，不能因 HAE probe 直接应用于 FAE 效果差，就认定 FAE 没有信息。

## 3. decoder 与逐帧目标

组件：src/openpi/models/future_patch_probe.py。

主结果使用共享线性头；辅助结果使用 1024→128→80 的小 MLP。每个 token 输出其对应四帧的 5 个 patch 接触 logits（20 维）和三轴力（60 维），共 80 维。线性头含偏置共 82,000 参数；单隐藏层 MLP 含偏置共 141,520 参数。所有 token 共享权重，不混合未来 token，也不平均四帧。

FuturePatchProbe 的输出为 contact_logits [B,16,5,5] 和 force [B,16,5,5,3]。中间先恢复 segment、finger、segment 内帧三个轴，再重排到完整时间轴；禁止把 token 轴直接当作未来帧轴。

标签来自真实未来 taxel，按固定区域映射确定性计算：patch 接触由固定阈值定义，力为接触加权三维均值，非接触 patch 的目标置零。阈值、温度、坐标系须明确。建议先在原生或标定单位生成标签，再用 probe 训练集统计缩放力；不要先中心化力向量后再判断接触。策略输入的 normalized units 不等于牛顿。复用现有目标函数时须确保输入与阈值同单位，并在报告中声明。

用于 rebuttal 的物理接触主结果应显式传入 `label_effort`，以原生/标定触觉计算接触标签；默认归一化标签用于调试或归一化传感量解码，不能直接等同于物理接触判定。区域解码仅验证这 25 个区域上的信息，不代表完整恢复 600 个 taxel 的细节。

损失采用接触 BCE 与力 Smooth-L1，活动/非活动 patch 分开归一，避免大量零力主导。测试 active-force 指标可以用真实接触标签选子集，但不能用真实接触标签门控 decoder 的预测。

## 4. 时间对齐与离线回放

提取器 scripts/extract_future_patch_probe.py 的 extract_probe_batch 接受已严格加载的 model、rng、observation 和必填 valid_future_frames。默认十步 Euler；label_effort 可提供与 observation.effort 同形状的原生/标定触觉整窗，此时必须显式给出 label_contact_threshold、label_contact_temperature。未提供时，标签使用策略归一化单位，不能标为 N。

future_frame_offsets 为 1…16。offset=4 时仅评估 5…16，offset=8 时评估 9…16，offset=12 时评估 13…16；make_strict_future_mask 同时排除尾部填充帧。必须核对数据时间戳确实对应这些偏移，不得把当前帧或重复补齐的尾帧当成未来真值。

先在 offset 0 调 slow 接口；后续 fast 调用仅可使用截至当前 offset 的触觉历史，语义/本体状态遵循原部署缓存规则。refined 保留上一调用预测，fixed 复用 offset 0 预测；两者比较同一绝对未来帧。保持相同初始动作噪声与推理设置。真实未来触觉只能在时间推进后成为合法历史或用于构造标签，不能提前输入模型。

这是使用示教轨迹观测进行条件化的离线回放。预测动作不会驱动环境产生后续观测，因此本实验不是机器人闭环 rollout，也不能证明预测了自身动作的真实接触后果。

## 5. 划分与训练顺序

按 episode 从训练池拆出 probe train/validation，独立测试库仅用于最终 test。本次指定 `task12345-2`（559 条、5 任务）和 `task-test-collect`（64 条、5 任务），不得将此次结果描述成论文原先的 800/100 条划分。固定 episode 清单后才切窗口，禁止相邻窗口跨集合；已有 test 结果不用于选 decoder、阈值或正则。保存 checkpoint 标识、完整配置、数据清单、时间偏移、标签单位和随机种子。

同一 probe 在训练集上平衡读取 fixed/refined 和各 offset，验证集选择 checkpoint，随后冻结它同时评估两种模式。这样避免只训练于 refined 再测试 fixed 带来的额外分布偏置。线性为预先指定主结果，MLP 为补充，不按 test 成绩挑选。

以下伪代码说明组件关系；实际的 manifest、缓存、训练及 episode 汇总已由 `scripts/run_future_patch_probe.py` 接入，运行命令见上述服务器说明：

~~~python
# 先严格恢复部署策略；策略不进入 probe 的梯度函数或优化器。
records = extract_probe_batch(
    model, rng, observation,
    valid_future_frames=valid_future_frames,
    num_steps=10,
    # 可选 label_effort 及同单位的显式阈值、温度。
)
# records: refined_hidden/fixed_hidden [B,O,20,1024]
# target_contact [B,16,5,5], target_force [B,16,5,5,3]
# valid_mask [B,O,16], offsets [O], future_frame_offsets [16]
# 还包括 history_tokens、current_contact、current_force，供独立基线使用。
save_cache_with_episode_and_chunk_ids(records)

probe = FuturePatchProbe(
    input_dim=1024, num_segments=4, num_fingers=5,
    steps_per_segment=4, num_patches=5,
    decoder_type="linear", hidden_dim=128, rngs=nnx.Rngs(seed),
)
for batch in finite_training_schedule:
    # 对模式与 offset 平衡采样，仅取对应 offset 的严格未来 mask。
    z, y_contact, y_force, mask = sample_cached_probe_batch(batch)
    pred = probe(jax.lax.stop_gradient(z))
    loss, stats = future_patch_probe_loss(
        pred, y_contact, y_force, valid_mask=mask,
    )
    update_probe_only(loss)
# 在有限且不重复的 validation/test manifest 上累计预测与标签。
# 所有指标在全集累计后计算；按 episode 重采样估计置信区间。
~~~

## 6. 对照、指标和报告

至少包含当前物理量保持不变的 persistence、当前/历史触觉表征 probe、固定预测及在线更新预测。加入只使用任务/时间先验的空特征基线，检查是否仅学到接触发生时机。历史基线应允许按预测 horizon 输出，或提供 lead-time 信息；简单重复当前 token 再接共享头，会使所有 segment 输出同样模式。报告其输入和参数量，不宣称这种基线严格等容量。

接触主指标采用 AP（non-interpolated average precision），不与梯形积分 AUPRC 混写；辅助报告全数据累计 TP/FP/FN 后的 F1，阈值在 validation 确定并锁定，不平均 batch F1。力报告活动 patch 的每坐标 RMSE/MAE，并注明单位。当前运行入口报告各 offset 和任务、全未来及接触变化帧子集，最终汇总按任务等权宏平均；逐预测提前量的细分可另行扩展。

额外报告“未来接触状态相对当前 offset 改变”的子集，并给样本数；其真值只用于评估分层，不输入 decoder。对成对的 fixed/refined 差值按 episode bootstrap，保留同一轨迹内全部相关窗口。评估器必须在清单末尾停止；无限循环 loader、重复 episode 或把每帧当独立样本均会夸大证据。

| 表征/基线 | 接触 AP ↑ | 全局 F1 ↑ | 活动力 RMSE ↓（注明单位） | 接触变化子集 | 95% episode CI |
| --- | --- | --- | --- | --- | --- |
| 任务/时间先验或空特征 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 当前物理量 persistence | 待测 | 待测 | 待测 | 待测 | 待测 |
| 当前/历史触觉 probe | 待测 | 待测 | 待测 | 待测 | 待测 |
| Fixed FAE，共享线性 probe | 待测 | 待测 | 待测 | 待测 | 待测 |
| Refined FAE，共享线性 probe | 待测 | 待测 | 待测 | 待测 | 待测 |
| Refined FAE，小 MLP（辅助） | 待测 | 待测 | 待测 | 待测 | 待测 |
| HAE 独立 probe（可选参考） | 待测 | 待测 | 待测 | 待测 | 待测 |

## 7. 与已有 probe 的区别

旧 train_future_tactile_token_probe.py 的冻结方向正确，但校验的是单段、5 个 pooled 5×3 触觉 token，不能直接作为当前 20-token 模型的实验。还需处理部署 latent 与训练前向分布不同、checkpoint 部分恢复的兜底、默认未强制 split、评估 loader 可能重复迭代和 batch 平均 F1 等问题。新组件不证明这些真实训练问题已全部完成修复；运行前应逐项核验。

## 8. 已完成的组件验证

已在 CPU、JAX 0.5.3、Flax 0.10.2 下通过 22 项单元测试。验证内容包括 token/帧重排、82,000 参数计数、输入冻结与 decoder 参数梯度、严格未来和尾帧掩码、无效 NaN 隔离、损失归一化，以及模拟策略接口中的时间切片和未来数据隔离。模拟策略测试不等于真实 Pi0LatentFlow/checkpoint 集成测试，更不代表已取得触觉预测实验结果。

~~~bash
PYTHONPATH=src:. JAX_PLATFORMS=cpu python -m pytest \
  --confcutdir=src/openpi/models -q \
  src/openpi/models/future_patch_probe_test.py \
  scripts/extract_future_patch_probe_test.py
~~~

`--confcutdir` 避开仓库父级 conftest 的 GPU 管理依赖。新增运行入口与 CPU 工具还通过 episode 划分、指标计算及合成缓存端到端测试；这些测试的命令和服务器验证边界见运行说明。
