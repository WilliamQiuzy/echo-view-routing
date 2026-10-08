# Fold-output correction implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 EV9V 上比较：同一个时序模型使用视觉编码器的样本内预测或按 session 留出的预测训练，是否改变识别与时间分段表现。

**Architecture:** 三个已有折编码器全部冻结。T-in 与 T-out 都使用五类概率作为输入，训练同结构 ASRF；P-out 是不读邻帧的线性概率校准对照。主评估统一接冻结 B0 v3，另外固定报告三个 30-epoch 折编码器的辅助评估，不选择其中最好的编码器。

**Tech Stack:** 现有 Python/PyTorch、冻结的官方 ASRF、NumPy、项目的 10 Hz 特征缓存和评估代码。

---

## 负责人决定与生效范围

本文件是 **2026-10-08 的完整执行版本**。它替代需要翻找聊天记录的 T-in/T-out/P-out 摘要，不需要再索取“上一份任务书”。这是新实验的前瞻性设计，不修改已完成的建库、Q0/Q1 或解码实验。

四个缺失决定现在明确如下：

| 项目 | 决定 |
|---|---|
| ASRF 输入 | 五类 softmax 概率，`[T, 5]`；不读 512 维特征，不读原始 logits，不取 log、不温度校准、不做额外归一化 |
| T-in 的生成器 | 对折 k 的录像，使用已经训练过该折的模型 `(k + 1) % 3`；不是 B0 v3 |
| 训练序列 | 直接使用 session-fold 建库轮已经固定的每折 500 条完整构造流，共 1,500 条，加原生录像窗口；不是旧 L1+ASRF 的跨折训练流 |
| 选点 | 固定 50 epoch，取最终权重；不读开发半区选 epoch，不早停，不使用 calibration 可行性选点规则 |

这份版本还增加一个有明确原因的辅助检查：本地 `docs/results/2026-10-02_model_development/README.md` 记录 B0 v3 选择了 epoch 0，而三个折模型均为 epoch 29（完成 30 epoch）。因此保留原定 B0 v3 主评估，同时报告折模型输入下的表现，以分开观察主系统迁移和相近训练配方下的表现。辅助结果不能替代失败的主结果，也不能成为选择部署编码器的依据。无需新增编码器训练。

“epoch 0”是第一轮训练结束后的权重，并非未训练；采样有放回，也不能说每个录像恰好训练了一次。

## 实验问题及不可作出的结论

假设：下游时序模型在视觉模型的样本内预测上训练，可能很少看到部署时需要修正的错误。留出预测可能提供更合适的训练输入。本轮建库只证明 B0 的错误足够分散，尚未证明这个假设。

本轮使用所有原生窗口和构造流的参考标签，不再筛选 X-Y-X，也不训练“删/留”候选分类器。原来不符合二分类条件的序列照常进入训练。

按 session 留出不等于按患者留出；构造流的切面变化是人工拼接；原生录像继承公开录像标签，不是专家逐帧真值。结果只支持本次 B0 输入、ASRF 配方和构造任务，不能直接推广到 L1 或真实连续扫查。

## 文件边界

只新增本轮文件，复用已有模块。建议固定名称：

- `configs/experiments/fold_output_correction_v1.yaml`：本文件所有模型、数据、统计和决策规则。
- `scripts/prepare_fold_output_correction.py`：缓存身份、配对输入、训练序列和批次清单。
- `scripts/train_fold_output_correction.py`：一个 arm/seed 的训练、检查点和恢复。
- `scripts/report_fold_output_correction.py`：固定权重评估、配对统计和英文报告。
- `tests/test_fold_output_correction.py`：有实质风险的配对、训练和统计检查。
- `docs/results/2026-10-08_fold_output_correction/`：英文 README、preparation、逐种子表、汇总、训练曲线。

复用与核对：

- `configs/experiments/session_fold_library_v1.yaml` 和 `scripts/session_fold_library.py`：折、原生对比度、构造流与模型清单。
- `echo_routing/temporal/dev_heads.py::ASRFHead`：官方结构、损失与解码。
- `scripts/train_l1_asrf.py`：训练配方及配对批次/初始化检查。**不得继承其 calibration 选点逻辑，也不得保留其写死的 512 输入检查。**
- `echo_routing/temporal/baselines/b3_js_divergence.py`：固定 B3。
- `scripts/report_l1_asrf.py`：现有评估和错误归因的实现参考；不得引入 L1 缓存或旧训练流。

不修改旧实验配置、报告或模型，不操作原有暂存理论稿。含录像 ID 的清单和预测只留在服务器。不要推送。

## Task 1：冻结数据及实验身份

**Files:** Create 本轮 YAML、prepare 脚本和对应测试。

- [ ] 记录协议摘要、session 折映射摘要、三个折模型完整 checkpoint 摘要及训练清单摘要。模型 k 的训练集应恰好是另两折。
- [ ] 读取建库轮已有的每折 500 条流和原生对比度选择，不重新抽流。不使用筛出来的候选列表作为训练集。
- [ ] 原生窗口每 epoch 共 1,500 个，从完整训练原生库按现有 `TrainData.epoch_batches` 的窗口抽取规则生成，最长 256 个样本；原生录像的 gamma 版本沿用建库轮固定选择。短录像使用有效样本 mask，不重复填充成有效样本。
- [ ] 注意现有 `TrainData.epoch_batches` 的读取语句写死了 `orig`。新入口须显式按固定 native-variant 清单查缓存；只复用它的抽窗和排序规则，不能静默丢掉原生 gamma。用一段确定选择 gamma 的合成记录验证实际读入值及批次摘要。
- [ ] 三组逐批共享源录像、帧索引、时间、gamma、标签、序列长度和语义边界。每条训练流只使用一个折的录像，因此同一条流的输入必须全部来自同一个编码器，不能在拼接点换编码器。
- [ ] 类别列顺序取冻结 ontology，确认三个模型与 B0 v3 一致；不得从字母排序猜测。
- [ ] 配对数据规则：对折 k 的每个样本，T-out/P-out 使用模型 k；T-in 使用模型 `(k+1)%3`。用原始训练清单验证“未见/已见”，不要只相信目录名。
- [ ] T-in 需要补提取三个折模型对对应已见折的概率（orig/gamma090），保持现有 224 letterbox、数字帧顺序和 10 Hz 采样；写新缓存，不能覆盖 T-out 缓存。
- [ ] 不对三折概率做融合。五类概率的类别坐标统一，所以可以组成训练批次；512 维特征不进入本轮。
- [ ] 输出样本内/留出概率的逐样本错误率、录像级混淆、置信度，按折分别报告。这只是输入描述，不用于选择哪个折模型或筛选录像。
- [ ] 所有训练脚本只加载原训练录像。为禁止 validation/test 进入训练、编码器归属错误和单条流切换编码器写拒绝测试。
- [ ] 本轮配置、代码与上述规则在产生新开发集指标前提交。准备文件只放计数与摘要。

配对语义伪代码（实现时验证每一个来源样本）：

```python
k = session_to_fold[session_id]
out_model = models[k]
in_model = models[(k + 1) % 3]
assert session_id not in out_model.training_sessions
assert session_id in in_model.training_sessions
x_out = read_prob(out_model, clip, frame_indices, variant)
x_in = read_prob(in_model, clip, frame_indices, variant)
assert x_out.shape == x_in.shape == (len(frame_indices), 5)
```

注意：T-in/T-out 的不同输入由不同已训练模型产生。该设计比较的是两种输入生成程序，不把结果解释成在固定同一组权重时开关“记忆”的因果效应。

## Task 2：固定三个模型及训练预算

**Files:** Create train 脚本，extend 本轮 YAML 和测试。

### T-in 与 T-out

官方 ASRF 固定到当前已使用的提交 `9623f1e8d9a1171333a4eeb65d190997b6c44a95`：

```yaml
model:
  in_channel: 5
  n_classes: 5
  n_features: 64
  n_layers: 10
  n_stages: 4
  n_stages_asb: 4
  n_stages_brb: 4
loss:
  classification: official_weighted_CE_plus_GSTMSE
  lambda_b: 0.1
  boundary: official_BoundaryRegressionLoss
decode:
  method: official_refinement_with_boundary
  boundary_threshold: 0.5
```

使用 `ASRFHead` 当前官方损失的精确实现，不额外改 smoothing 权重；其 GSTMSE 的 similarity 输入也使用本组五类概率。该实验因此检验完整输入生成程序（包括官方损失依赖输入的固定加权），不能单独归因于某一个损失项。语义边界由参考类别变化给出，同类录像拼接不是正边界；起始/填充规则沿用官方 adapter。

### P-out

不读邻帧的 **log-probability 线性校准器**，参数共 30 个：

```python
logits_t = W @ log(clamp(p_t, min=1e-6)) + b
q_t = softmax(logits_t)
```

`W=I_5, b=0` 初始化，所以开始时输出等于只做数值截断后的原概率。P-out 接口仍是同一五类概率；取 log 仅是这个简单校准器的显式模型定义，不更改 ASRF 输入。它使用与 T-out 相同的逐样本类别权重和有效样本 mask，只训练类别 CE，不使用边界或时间平滑损失。这里把之前简写的“线性分类器”精确化，避免默认实现另起一种模型。

### 三组共用预算

```yaml
seeds: [0, 1, 2]
epochs: 50
streams_per_epoch: 1500
native_windows_per_epoch: 1500
streams_per_batch: 4
natives_per_batch: 4
updates_per_epoch: 375
total_updates: 18750
optimizer: AdamW
lr: 0.0003
weight_decay: 0.0001
grad_clip: 1.0
warmup_epochs: 2
schedule: cosine_after_warmup
precision: fp32
TF32: false
checkpoint: final
early_stopping: false
validation_during_training: false
```

- [ ] T-in/T-out 同种子读取同一个初始化文件、同一个每步批次清单；恢复 dropout RNG。P-out 的恒等初始化固定，种子仍控制批次和其他允许的随机性。
- [ ] 1,500 条流每 epoch 各用一次；与 1,500 个原生窗口组成 375 批，每批 4+4。每 epoch 排序及原生窗口抽取由共同种子清单确定。
- [ ] 类别权重与 boundary pos_weight 按现有 ASRF 官方 adapter，从共同 epoch-0 标签计算一次后冻结；三组类别权重一致。打印数值及摘要。
- [ ] 编码器没有梯度、不更新。主任务输出是完整逐时刻类别与区间，不只输出候选删留分数。
- [ ] 同种子短运行可复现；T-in/T-out 标签摘要一致；P-out 改邻帧不改变目标输出；恒等初始化输出与输入的差异只来自规定的 clamp。
- [ ] 为 ASRF 的 5 通道输入跑一次真实形状冒烟测试，检查有限损失、有效梯度和 mask；不得靠补零到 512 绕过接口检查。
- [ ] 记录训练损失及分项曲线，不用开发集曲线选 epoch。不因曲线或 seed 0 结果追加 epoch、种子、超参。
- [ ] 代码和配置审查完成、配对检查通过后按 `seed 0: T-in,T-out,P-out; seed 1: ...; seed 2: ...` 启动，共九次训练。检查通过已有授权，不再请求一次“是否开训”的确认。

## Task 3：固定评估输入与解码

**Files:** Create report 脚本；无需新训练。

### 主评估

- 固定 `models_frozen/b0_frame_encoder/v3_s0` 为部署编码器。恢复服务器连接后核对其完整摘要、checkpoint epoch、类序及缓存；摘要前缀应与旧记录 `beb36d80df31` 相符。
- 三组均在同一个 B0 v3 输出上评估，使用现有协议的 calibration/selection 原生录像、两段流和多段流。**主指标是多段流 segment F1@50**；两段流只作补充。
- T-in/T-out 使用上述官方 ASRF 解码。P-out 接下面数值完全固定的 B3。
- 固定 B3 直接读取 `q`，使用 `half_window=9, threshold=0.31007912515352554, nms_radius=5, require_label_change=true`。区间标签为该区间平均概率 argmax，沿用当前 `labels_from_boundaries`。这是 registry 中已存在的 audited B3 数值；不是分位数 0.995，也不是重新计算分位数。
- 同时运行未经校准的 `B0 v3 + 同一固定 B3`。这一行必须与历史最强 `E0+B3` 分开，不能假定两者设置或分数相同。
- 原始逐时刻 argmax 与完整系统结果同时报告，避免把固定 JS 阈值与概率尺度的相互作用当成识别机制。P-out 较差不能证明所有逐帧校准方法无效。
- 重现旧最强 E0+B3、E1+ASRF、B7+B3 在同一评估协议上的存档行，沿用它们各自既有设置，不重新选参或训练。无法复现应报告输入/版本问题，不能静默换行或改目标。

### 固定辅助评估（新增，不能选优）

- 三个冻结的 30-epoch 折编码器分别提取现有两个开发半区的概率。它们都未在这些开发录像上训练，但开发半区本身已被项目反复使用，仍不是新的独立验证。
- 每个编码器分别接每个 T-in/T-out/P-out 最终模型及同一固定 B3 基线。完整报告三个编码器的表，不投票、不集成、不选择最好者。
- 只用于说明输入生成配方与部署输入的差异，不能凭辅助结果把主 B0 v3 评估判成功。三编码器乘三种子也不是九套独立病例。
- 不对这三折再次直接做二级模型留折验证并称其严格独立：生成其他折训练输入的编码器可能已经用过待评估折的标签。主评估继续使用原开发半区。

两部分均无需查看 test，也不使用 Pediatric。

## Task 4：指标、统计与固定判定表

### 指标和误差表

每个系统、种子、半区报告：多段流 segment F1@50，边界 precision/recall（0.5 s 容差），原生录像 macro-F1、每类召回，原生录像额外 label switches，构造流误切和同类拼接误切。

错误修复/新增错误至少给：A5C→A4C、A4C→A5C；参考段由未匹配变匹配或相反；参考切换找回或丢失；冻结视觉头判对而时序输出判错及反向。提供来源录像和 session 去重计数。不要求重新训练候选 Q 模型，不给候选设置 95% 保留阈值。

### 什么叫“优于”

1. **主差值：** 同一部署编码器、同一半区、同一 seed 的 `T-out − T-in` 多段流 segment F1@50。
2. **次差值：** `T-out − P-out+B3`，以及 `T-out − 未校准B0 v3+固定B3`。这是相对本次具体简单对照，不是对所有校准器的排除。
3. **区间：** 沿用当前配对 bootstrap，2,000 次、种子 20261002；原生录像按 session，构造流按整条流，在两个系统中用完全相同的抽样索引。每个训练种子分别计算。构造流复用录像的依赖未完全建模，区间是近似、探索性的；训练种子共享病例，不拼成三倍样本量。
4. **一致正向：** 某一个半区的三个种子差值的 95% 区间下端全部严格大于 0。全部上端小于 0 为一致负向。其余为“未检出一致差异”，保留点估计和区间；不能称等价。
5. **两半区支持：** 两个半区各自满足一致正向。不得合并半区后替代此规则；只在一个半区成立就明确写“仅一半区支持”。

### 系统门槛（不用于挑 epoch）

固定最终模型直接接受检查：每 seed、每半区边界召回 >= 0.95，原生 macro-F1 >= 0.973（selection）/0.980（calibration）。未达标照样报告实际数值，不能回退到另一个 epoch 或解码参数。

最终目标沿用之前的多段流 segment F1@50 三种子均值：selection >= `0.944409687184662`，calibration >= `0.9821421709894332`，并同时满足上述逐 seed 约束。另列 calibration 以较晚的 `0.9738935784747763` 为参考再加 0.01 的敏感性检查，不取代原主目标。

判定与行动：

| 观察 | 可写结论 | 行动 |
|---|---|---|
| 主比较未在两个半区都一致正向 | 本次未得到两半区均支持；逐半区说明正向/负向/不确定 | 本轮结束，不自动搜索 |
| 主比较两半区均一致正向，但未同时胜过本次两个简单对照 | 留出预测训练相对 T-in 有收益，未证明优于简单对照 | 本轮结束，不称新模型贡献 |
| 两半区均胜过 T-in 和两个简单对照，但系统约束未满足 | 有比较收益，仍不可按当前标准使用 | 本轮结束，明确未满足的约束 |
| 上述比较与系统约束都满足，但未达既定最强配置目标 | B0 下的局部成功；最终目标未完成 | 交负责人决定是否另立向更强编码器迁移的实验 |
| 比较、约束和最终目标均满足 | 本开发协议下达到目标；仍需独立验证 | 冻结模型，提交结果；test 是否开启另行决定 |

辅助评估单列，不触发额外训练，不反转主表结论。任何情况都不能直接声称 L1 被修好或真实连续扫查有效。

## Task 5：修复记录、执行顺序与服务器不可用

- [ ] `1122a68` 已修复旧续训的模型/优化器绑定。本轮不重复实现；阅读共用恢复函数并在时序训练恢复测试中确认 optimizer 参数对象属于正在 forward 的模型。
- [ ] 连接恢复后只查可能走过旧 `train(..., resume=...)` 的历史 B0 运行。核对日志、命令/manifest 与被选 checkpoint；不能由 epoch 编号非零或缺少命令记录就断言用了/没用续训。证据不足写“未确认”。
- [ ] 若发现实际部署的 B0 v3 权重/缓存身份不符或其历史评分不可复现，暂停主系统评估并报告具体原因，不另选部署权重。如果只是未选中的后续 checkpoint 受影响，记录影响范围，不全盘重跑。
- [ ] 服务器不可连接时，先完成本地配置、脚本和纯合成数据测试；明确尚未完成 GPU 冒烟检查、配对真实输入检查和训练。不循环密集重试 SSH，不通过覆盖旧缓存绕过连接问题。
- [ ] 本地验证命令：`python -m pytest tests/test_fold_output_correction.py tests/test_session_fold_library.py -q`。使用可用项目环境；不为运行测试擅自升级依赖。报告实际命令和结果。
- [ ] 服务器恢复后在其项目 `.venv` 下运行相同测试、真实输入准备与一小段训练冒烟检查，输出每模型预计时长/显存，然后启动冻结队列。没有收到开始日志前不能说训练已开始。
- [ ] 只在九次训练完成及最终权重身份固定后运行开发集评估。每次运行包含 arm/seed/输入摘要；已有目录身份不符或未指定恢复时拒绝覆盖。
- [ ] 冻结六个 ASRF 和三个 P-out，英文 README 写明事先规则、一次辅助评估增加的原因、实际偏离及有限性。原始录像/ID/逐样本结果留服务器；仓库只存代码、配置、聚合结果。

推荐实现为 prepare/train/report 三个命令入口，具体 CLI 在实现时写入 YAML 的 commands 记录与英文 README；它们可以采用现有仓库的 parser 习惯，但不能改变本文件的实验语义。提交前仅 stage 本轮明确路径，保留其余脏文件和暂存理论稿。

## 实施验收检查表

- [ ] 三个缺失细节（概率、cyclic 已见折模型、当前三折内构造流）都由测试验证，没有用第一种猜测替代。
- [ ] T-in/T-out 真正只改变输入生成程序；同种子结构、初始化、标签、顺序、预算、解码一致。
- [ ] 没有无意读开发集挑 epoch，没有静默默认到旧的 calibration 可行性规则。
- [ ] 本轮固定 B3 与历史最强 B3 行分开，所有数值写明，无新阈值/窗口搜索。
- [ ] P-out 校准器数学定义、初始化和输出已验证；不把普通校准误报为时间机制。
- [ ] 主评估与三个辅助编码器结果严格分开，没有按结果选择编码器。
- [ ] 支持/不支持/不确定与最终目标达成分别报告；阴性结果不自动触发另一次扩展。
