# TouchWorld-inspired 适配基线：实验边界

本目录对应的模型是独立实现的 **TouchWorld-inspired 适配基线**，不是 TouchWorld 作者的完整模型，也不是其公开 checkpoint。它保留“未来视触觉目标预测—目标条件动作—执行期间触觉反馈修正”这一研究问题，但模型骨干、目标表示、训练数据和规划机制均与 TouchWorld 不同。

TouchWorld 原方法包含子任务规划器、Wan2.2 触觉世界模型、目标条件策略以及触觉残差控制器；我们尚未核实到原作者完整控制栈及训练权重。这个适配版不复用原版完整系统，也不应在实验表中直接标为 `TouchWorld`。建议使用 `TouchWorld-inspired (adapted)`，同时披露其训练数据、模型和接口变更。

## 目标与数据

使用 XHand 原始 5×120×3 触觉、同一时刻 RGB、机器人状态、任务标识和动作。任务由固定 task ID 表示，而不是原论文的子任务规划器或自然语言模型；因此本版本不能检验对未见语言指令或未见任务的泛化。模型预测未来视觉与触觉的潜在目标，动作策略只使用模型预测出的目标；真实未来观测仅用于训练预测器的监督，不得在验证或部署时作为策略输入。触觉快速更新根据当前及过去实际观测修正当前计划尚未执行的动作，不接收未来触觉或未来动作。

数据按完整 episode 分离训练、验证及测试；测试 episode 不能参与模型训练、归一化、任务参数选择或 checkpoint 选择。动作、状态、触觉与图像的时间对齐和单位需由真实数据元信息核实。触觉中的三个值不能直接视为已标定的牛顿力，也不能通过填零伪造腕部力/力矩。

## 训练与评估的科学边界

- 先训练观测编码，再固定目标编码器；之后训练预测器与目标条件策略。目标条件策略必须用预测目标，而不能直接用示范未来目标。
- 训练纠正器时，冻结目标预测与 nominal 策略，用模型实际生成的 nominal chunk 构造训练输入；在随机执行位置，纠正器只接收该时刻及以前的触觉历史。未来示范动作仅用作纠正监督。
- 相同七任务、相同训练数据及测试环境下，与 ReTouch 和其余 baseline 使用一致的轨迹划分、任务重置、评估次数及部分得分规则。记录不同骨干、参数量、训练资源和预训练数据，不能据此把差异全部归因于未来触觉预测。
- 单纯的离线 action MSE 或 teacher-forced 视频重建不能替代真实机器人闭环任务评价。不能在没有真实训练或实机实验时填写结果，也不能以 synthetic smoke test 宣称实验完成。
- 8 张 A800 提供计算资源，但是否能训练、耗时多少仍取决于显存、配置和数据；需在实际机器上检查。当前自动化环境不能访问用户的 GPU 和原始数据。

## 当前原型的限制

- 当前视觉、触觉编码器和动作策略都是小型网络、从零训练；没有加载论文所用的 Wan 世界模型、原版权重或 ReTouch 的预训练视觉语言模型。这种差异会影响公平比较；结果不能证明优于 TouchWorld，也不能将性能差归因于单个机制。
- 任务条件是训练数据中的整数 task ID。训练集、验证集、测试集及部署端必须使用同一任务语义到 ID 的映射；系统不能自动将新语言指令映射为 ID。
- 当前纠正器默认只修正 12 维手部动作，保留 6 维机械臂动作；原 TouchWorld 同时覆盖腕部和部分手部动作。可配置的动作语义必须与数据和控制器对齐。
- 首版训练入口只支持单进程 CPU 或单张 CUDA；明确不支持八卡并行。数据、时序、推理协议以及单卡验证通过后才能实现和验证多卡训练。
- 有模型级 `predict_nominal`、`refine` 接口，但尚未提供到机器人客户端的闭环控制适配、真实闭环测试和统计结果。示范轨迹重放使用的是示范采集时的触觉，并非模型执行偏差动作后测到的触觉。
- 内置 LeRobot v2.1 视频读取将缩放后的 RGB 缓存到磁盘，状态和触觉仍按轨迹加载到内存；应在真实数据上先测内存、缓存空间及吞吐。

## 单卡原型入口

准备训练池和独立测试集的 episode ID 文件，文件内容为 JSON 整数列表；验证集会从训练池中划分。训练入口要求 PyTorch、NumPy、PyArrow 和 OpenCV，且暂时只读取一个 RGB 相机。下面的路径均需替换为实际服务器上的路径，不能直接把示例当作已有训练结果：

```bash
PYTHONPATH=src python scripts/train_touchworld_lite.py \
  --repo-dir /path/to/training_lerobot \
  --test-repo-dir /path/to/heldout_lerobot \
  --train-episode-ids /path/to/train_episode_ids.json \
  --test-episode-ids /path/to/test_episode_ids.json \
  --image-key observation.images.cam_front \
  --output-dir /path/to/new_output_dir \
  --device cuda:0 --bf16
```

原始状态解析默认拒绝猜测未知布局。如果已经根据真实采集协议确认状态各项的位置与本仓库 XHand legacy 布局一致，再显式添加 `--allow-legacy-raw-layout`；这个开关不是标定或校验。若不存在内置 v2.1 视频布局，可通过 `--image-loader module:function` 提供读取回调。命令开始前应确认训练和测试的任务 ID 映射、坐标与单位、真实视频时间戳。具体参数见 `--help`。

输出包含 `run_manifest.json`、每阶段验证集最佳权重、`last.pt` 和 `model_final.pt`；验证 loss 是按样本加权的 batch 内归一化指标，不是整个数据集有效元素的全局 MSE。当前无真实数据训练、GPU 或机器人闭环结果。验证集选模型、测试集只用于最终评估；如需实机运行，仍需实现并验证独立机器人控制适配器。

论文及代码参考：

- TouchWorld 论文：https://arxiv.org/abs/2607.07287
- 作者关联的触觉世界模型组件：https://github.com/flyingGH/TouchWorld-Tactile_World_Model （尚未核实为完整官方策略发布）
