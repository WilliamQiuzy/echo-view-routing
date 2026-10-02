# 心超切面时序模型训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本轮交给 Claude 执行；不需要再询问是否继续模型研发。

**Goal:** 开发并训练一个可输出逐时刻切面类别的模型，在统一评估条件下超过已有最强基线；先取得模型权重和验证集结果，再决定论文贡献。

**Architecture:** 第一版使用视觉编码器、局部时序分支、学习变化边界的分支，以及由预测边界约束的上下文注意力。检验的假设是：按语义变化控制跨帧信息聚合，能减少连续误判造成的内部切分，同时避免跨真实标签变化过度平滑。这个假设尚未得到验证，结构组合也不预先声称新颖。

**Tech Stack:** 现有 PyTorch、版本化特征缓存、EV9V family5_v1、官方 MS-TCN / ASFormer、ASRF，沿用项目训练服务器。

## 当前决定和交付范围

- 2026-10-02 用户明确要求继续模型研发，暂缓论文定稿。此决定取代研究计划历史记录中“停止追加训练、以审计稿收尾”的工作安排。
- 保留修复、审计结果、冻结权重和文稿作为历史材料；本轮不修改论文正文，不做投稿或发表操作。
- 本文件是给 Claude 的实施与训练任务书，不表示已实现、已发出训练任务或已产生收益。
- 主任务沿用现有离线、5 类切面分类与时序分段；10 Hz，正确时间顺序。新模型输入为当前视频流，不获得来源文件边界、标签、会话 ID 或拼接配方。
- 原生录像只有整段标签；人工拼接流只有来源标签定义的代理参考。不要将模型的门控值称为临床质量、证据充分性或心动相位，也不需要 AI/专家逐帧伪标注。
- 本轮需要实际训练新模型，不能以修改报告、再列计划或只跑基线作为完成。第一轮至少交付新模型 checkpoint、训练曲线、验证集预测和消融表。

## 已有起点和比较对象

以 `docs/paper/supplement/S1_Table_true_order_all_models.md` 的统一真实顺序表为历史起点：

| 方法 | 原生片段 macro-F1 | 多段流 segment F1@50 | 多段流 boundary recall | 原生内部切分/段 |
|---|---:|---:|---:|---:|
| 审计 B3 | 0.930 | 0.871 | 0.926 | 0.052 |
| MS-TCN v2 | 0.947 | 0.824 | 0.960 | 0.147 |
| ASFormer v2 | 0.956 | 0.856 | 0.942 | 0.139 |
| STFM 正确顺序滑窗 | 0.960 | 0.859 | 0.879 | 0.126 |
| EchoViewCLIP stage 1 滑窗 | 0.972 | 0.867 | 0.953 | 0.098 |

这些数字来自已使用过的测试集，仅用于解释需要超过的历史水平，不用于本轮选模型。其他 B0、B1、HMM、EchoPrime 及其后处理仍列入完整结果表。不能只超过旧 B0 或某一个较弱种子的 MS-TCN。

两类比较分别报告：

1. **同一视觉编码器的机制比较：** 所有时序头读取同一份特征、同一训练流、同一验证流程，确定收益是否来自新时序模块。
2. **完整系统比较：** 最终新模型对 EchoViewCLIP、STFM、ASFormer、MS-TCN 和调好的简单方法，包含全部视觉计算成本。换更强编码器的收益不能全部归给新时序模块。

ASRF 已经联合学习类别与边界，必须作为最接近的直接对照；ASFormer 官方仓库也包含与 ASRF 结合的路线。不能把“加一个边界头”本身写成新方法贡献。

## Task 1 — 固定本轮数据、指标与首批训练入口

**Files:**
- Create: `configs/experiments/model_development_v1.yaml`
- Create: `scripts/model_development_protocol.py`
- Create: `tests/test_model_development_protocol.py`
- Read/reuse: `echo_routing/compose/banks.py`, `scripts/true_order_joint.py`, `configs/labels.yaml`

- [ ] 读取服务器现有环境、GPU 占用和已冻结权重路径，记录实际 commit、环境及空闲显存；不要把旧的运行时间当成本轮完成时间。
- [ ] 固定训练、验证来源清单与配方摘要。保持官方 train/validation/test 划分；本轮候选筛选只看 validation。旧 test 已参与研究决策，最终重测仍是探索性验证；重新拼接旧录像不产生独立测试集。
- [ ] 验证集按会话划分为约各半的 checkpoint-selection 和 operating-point-calibration 两部分，固定 seed=20261002，仅依据标签/会话元数据保证两部分均覆盖 5 类，保存最终清单，不看候选模型得分。若无法得到覆盖全部类别的会话拆分，则第一轮统一用全部 validation 做探索性选择与校准，在 manifest 中明确记录此例外，不再声称两步数据独立；不使用 test 调参。
- [ ] `model_development_protocol.py` 写出运行目录内的 `protocol.json`，包括两部分的 clip/session 清单、各自 native/pair/multi banks 和 digest。新生成的验证流必须只从本部分的会话构造，不能把一条跨两部分的流按流 ID 硬分。保留历史统一验证 bank 仅作描述表，不参与 checkpoint/阈值选择。
- [ ] 原生片段分类统一由逐时刻预测多数投票得到；多段流统一 segment F1@50。两段流只作次要指标，不作为胜负依据。
- [ ] 预先保存每类召回、混淆矩阵、多段流 boundary F1/recall（0.5 秒）、内部切分/段及/分钟、同类拼接切分率、参数量、峰值显存和完整系统时延的输出格式。
- [ ] 检查训练入口实际设置 Python / NumPy / PyTorch / CUDA 与 worker 的随机数种子，记录初始化和前两个 epoch 的样本次序摘要。
- [ ] 给训练采样补一个有实际 batch 输出的检查：`EpochFrameDataset.resample()` 在 `persistent_workers=True` 时是否真正改变 worker 读到的样本。若没有，只做必要修复（例如关闭 persistent workers），让全部本轮训练共同使用修复。不要仅检查主进程 samples 列表。

本任务在训练启动前完成，不扩展为新一轮全面审计。

## Task 2 — 重训正确采样的 B0，并建立可共用的特征

**Files:**
- Reuse/必要修复: `scripts/train_frame_encoder.py`, `echo_routing/features/train_encoder.py`, `echo_routing/features/dataset.py`
- Reuse: `scripts/extract_features.py`, `echo_routing/features/cache.py`
- Create: `configs/experiments/b0_frame_encoder_v3.yaml`
- Test: `tests/test_frame_training_sampling.py`

- [ ] 当前 B0 是修复前训练的固定权重。新建 v3，不覆盖它；第一轮只训练 encoder seed 0。
- [ ] ResNet-18 / ImageNet 初始化，224 像素，16 帧/段训练采样；AdamW，lr=3e-4，weight decay=1e-4，batch=64，最多 30 epochs，patience=6。保持既有采样/类别加权设定，除非 Task 1 检出明确实现错误。
- [ ] 用 checkpoint-selection 部分的片段 macro-F1 选 encoder；确保统计使用与最终评估相同的多数投票，不把现有 `evaluate()` 的平均概率聚合混写成多数投票。最终评估仍用 10 Hz 全部有效样本。
- [ ] 给本轮各训练入口增加 `--protocol`，按显式 clip 清单过滤验证数据；现有脚本默认使用全部 validation，不能原样调用后声称已隔离 calibration。v3 训练的验证 loader 用 10 Hz 固定顺序样本，投票规则与最终评估一致；旧入口默认行为保持兼容。
- [ ] 抽样、权重、标签映射和解码配置全部写入 manifest；cache key 必须包含新 checkpoint digest，不能混入旧缓存。
- [ ] 提取 train/validation 特征及概率；暂不运行 test。对一个短片段验证缓存与在线输出一致。

现有命令可作为入口，新增配置必须先创建：

```bash
python scripts/train_frame_encoder.py -c configs/experiments/b0_frame_encoder_v3.yaml --set seed=0 --protocol ACTUAL_PROTOCOL_JSON
python scripts/extract_features.py --ckpt ACTUAL_V3_CHECKPOINT --splits train,validation
```

`--protocol` 是本轮要新增的参数；`ACTUAL_PROTOCOL_JSON` 从 Task 1 取得。`ACTUAL_V3_CHECKPOINT` 从本次运行的 `best_checkpoint.txt` 取得，不沿用历史路径。

## Task 3 — 实现并训练第一个新模型

**Files:**
- Create: `echo_routing/temporal/models/boundary_context.py`
- Create: `echo_routing/temporal/losses/boundary_context.py`
- Create: `scripts/train_boundary_context.py`
- Create: `configs/experiments/boundary_context_v1.yaml`
- Create: `tests/test_boundary_context.py`
- Reuse: `echo_routing/temporal/models/mstcn.py` 的 mask/时序层约定、现有 bank/render/cache 接口

### 明确的第一版网络

输入 `features[B,D,T]` 与 `mask[B,1,T]`。输出 `class_logits[B,5,T]`、`boundary_logits[B,1,T]`、`local_logits[B,5,T]`。边界索引 t 表示 t-1 与 t 之间，t=0 不计损失。

1. 逐帧线性投影到 128 维。
2. 局部分支：3 层残差 dilated Conv1d，kernel=3，dilation=1/2/4，dropout=0.2；得到局部表示和分类 logits。
3. 边界分支：MLP 读取相邻局部表示及其差值，输出每处边界概率 b。边界预测在上下文模块之前计算，避免循环定义。
4. 上下文分支：4-head local attention，每时刻看左右各 20 个样本（离线设定，总跨度约 4 秒）；相对位置偏置可学习。
5. 在标准 attention score 上增加从 t 到 j 路径的 `sum(log(1-b[k]))`，以预测变化边界抑制跨边界聚合；对自身位置该项为 0。b 以 1e-4/1-1e-4 截断，仅屏蔽 padding，不读真值边界。该项系数第一版固定为 1。
6. 学习一个 scalar gate g_t，将局部和上下文 logits 混合：`logits=(1-g)*local_logits+g*context_logits`。g 从两分支表示预测，作为模型内部权重，不解释为临床质量标签。

边界概率、路径权重和 g 都由训练学习。推理不允许教师强制、真值切面、来源片段 ID 或源录像上下文。

### 损失及监督来源

`L = L_stream + 0.25*L_native_frame + 0.2*L_local + 0.2*L_boundary + 0.2*L_native_clip`。

- `L_stream`：人工拼接流逐样本类别 CE；目标继承来源整段标签，明确这是代理监督。
- `L_native_frame`：原生录像的弱逐帧 CE，权重较低；不能声称每帧标签经专家确认。
- `L_local`：局部分支类别 CE，使用相同 valid mask，避免上下文把视觉分类能力完全替代。
- `L_boundary`：相邻来源标签不同的拼接为 1，同类拼接为 0；片段内部和原生录像内部是弱负例（权重 0.1），不是临床真值。每条序列计算 `sum(BCE_positive)/max(n_positive,1) + sum(w_negative*BCE_negative)/max(n_negative,1)`，再对序列取均值；强负例权重为 1，弱负例为 0.1。分母是未加权的有效位置数量，不能用权重之和，否则全弱负例序列的 0.1 会被抵消。无正例时正项为 0，t=0 和 padding 不进入计数。不得将所有文件连接都标为正例。
- `L_native_clip`：原生录像平均分类概率对整段标签的负对数似然；这是训练损失，最终评估聚合仍按协议使用多数投票。
- 各损失先按样本/序列归一化，再按上述系数求和；padding 不贡献损失。第一轮不同时加相位、蒸馏、对比学习等额外机制。

### 数据和训练

- 每个 epoch 使用固定的 1,500 条训练拼接流；额外加入等量原生片段训练窗口，按实际 optimizer updates 记录预算。窗口最大 256 samples，不足时 padding；长录像均匀取窗。
- 第一版 AdamW，lr=3e-4，weight decay=1e-4，batch=8（显存不足用累积保持等效 batch），gradient clip=1，cosine schedule，2 epochs warmup，最多 50 epochs，patience=8。
- 每 epoch 保存训练各项 loss、验证类指标、边界预测正例率、gate 均值、显存和耗时；保存 best/last checkpoint。仅依据 checkpoint-selection 数据选择 checkpoint。
- 第一轮所有时序模型共用 v3 encoder seed 0 的冻结特征；不先联合微调视觉 backbone，以便分辨时序头的增益。

### 必须先通过的功能检查

- [ ] 输出 shape、padding 不影响有效位置、T=1 无 NaN；所有实际 batch 样本的 loss 有限。
- [ ] 同类拼接边界目标为 0、异类为 1，t=0/padding 不参与；对已构造小例子手工核对标签。
- [ ] 相同预测、相同数量的全弱负例，其边界损失应为全强负例的 0.1 倍；用测试确认权重未被归一化抵消。
- [ ] 把边界概率设为接近 0 时，在数值容差内接近普通局部 attention；设为接近 1 时跨该边界的 attention 权重显著下降。这里只是单元测试，评估时不注入真实边界。
- [ ] 同一模型输入完全相同、只改配方/文件名元数据时预测不变。
- [ ] 16 条训练流的小样本过拟合检查，确认分类和边界损失能下降；再开始全量 seed 0。测试通过不能代替训练结果。

新增脚本实现后，应提供并实际执行这类入口；现在这些入口尚未实现：

```bash
python scripts/train_boundary_context.py -c configs/experiments/boundary_context_v1.yaml --set seed=0 --feature-manifest ACTUAL_V3_MANIFEST --protocol ACTUAL_PROTOCOL_JSON
```

## Task 4 — 有限的对照和消融，不把改训练配方的收益算成结构收益

**Files:**
- Create: `scripts/run_asrf_official.py`（固定上游 commit、显式 seed，新增副本，不改已有官方仓库）
- Create: `scripts/model_development_queue.py`
- Create: `scripts/report_model_development.py`
- Modify: 新模型配置，以明确开关控制消融

第一轮核心矩阵，每行 seed 0，共用同一冻结 v3 特征：

| 行 | 训练内容 | 回答什么 |
|---|---|---|
| M0 | MS-TCN，任务适配训练 | 当前公平的学习型起点 |
| M1 | ASRF，任务适配训练 | 已有类别/边界联合方法能否解决问题 |
| M2 | 新模型普通局部 attention；去掉边界损失和路径项，保留混合 gate | 双分支与容量本身的效果 |
| M3 | M2 + 边界辅助损失，attention 仍不读取边界 | 仅增加监督的效果 |
| M4 | 完整新模型，attention 读取预测边界 | 预测边界控制聚合的增量 |

给 M0/M1 相同的原生+拼接训练数据、优化更新预算、checkpoint 选择规则；适配部分与官方原配方结果分开标记。M0/M1 自有训练损失保留，不为迁就新模型而删除其关键损失。原配方冻结模型仍在系统对照表中。

- [ ] 同时对同一编码器概率调 B1/HMM/B3：参数网格与选择规则预先写进配置；只用 calibration 部分。不得只调新模型和 B3、再把未调的 HMM 当最强对手。
- [ ] 每个学习型方法相同的最大候选预算：第一轮先共同 lr=3e-4；必要时每行最多再试 lr=1e-3，一次参数变更，全部试验都入表。不能只给本方法无限搜索。
- [ ] 若 M4 在验证集有增量，再训练一个取消最终学习混合 gate 的消融（固定 g=0.5）；区分路径约束与额外融合参数的贡献。
- [ ] 连续干扰增强暂为第二步：对 M1 与 M4 同时增加训练期连续 3–8 样本的特征遮挡（每流概率 0.3，每源片段遮挡不超过 20%，距已知异类拼接至少 5 样本）。评估主表保持自然、未扰动输入。若提升只来自增强，应归因于训练策略，不能归给新结构。
- [ ] 不把随机打乱帧作为主训练增强，不重新启用已无增益证据的相位匹配。

## Task 5 — 更强视觉输入与三种子确认

这部分在 M4 至少相对同编码器的强对照显示有用趋势后执行；若未出现趋势，先报告实际失败并改一个有证据的问题，不能自动堆新模块。

- [ ] 第一强视觉候选使用已有 EchoViewCLIP stage 1 权重的 visual encoder（来自训练 split 的版本），提取逐帧投影特征，并在训练 split 上学习 5 类线性头。保留原始视频 TML 分类器作为完整系统基线；逐帧适配器不等于复现了原模型。
- [ ] 新增 `scripts/extract_echoviewclip_features.py` 和独立 cache manifest，记录所用视觉层、预处理、checkpoint 与特征维数；核验只用当前流实际可见帧。接口不兼容时写清具体问题，继续已有 v3 编码器模型训练，不等待无限环境迁移。
- [ ] 同一强特征上运行 M0/M1/M4 及调好的 B3，不能只升级 M4 的编码器。ASFormer 在最终候选编码器上重训为直接对照；保留现有官方配置单次结果与新适配结果的区别。
- [ ] 若强视觉冻结特征有益，最后只尝试一次受控联合微调：M4 与最强可训练对照均解冻相同的视觉末块、相同预算；backbone lr=1e-5，head lr=1e-4，最多 10 epochs。使用真实视频窗端到端重新算特征，不能把冻结缓存当成梯度可回传的输入。
- [ ] 配方定下后重复 seed 0/1/2；先固定 encoder、改变时序头 seed，以区分训练方差；再对最终模型与最强同编码器对照做 3 个配对完整训练 seed。报告两种方差，不混成一组。
- [ ] 最终完整系统表包含 MS-TCN、ASFormer、STFM、EchoViewCLIP、EchoPrime 与调好的简单基线。最接近的竞争者需同样做 3 seeds 才能主张稳定超过；有单次模型时明确标注，不挑其弱 seed。
- [ ] 当前是闭集分类与分段任务：EchoViewCLIP 的 stage 1 是对应分类基线；其 stage 2 拒识与质量分支不参与同一闭集指标。若后来加入拒识/覆盖率目标，必须比较其完整相应阶段，不能拿 stage 1 代替完整拒识方法。

## 如何选择模型和判定进展

### 候选筛选（只用 validation）

- 第一轮 checkpoint 选择：在 native clip macro-F1 不低于该编码器的逐帧多数投票基线 0.5 个百分点的 checkpoint 中，选多段流 segment F1@50 最高的；差距小于 0.2 个百分点时选原生内部切分更少的。若没有合格 checkpoint，保留 segment F1 最高的用于失败诊断，明确标为分类约束不达标。
- 主比较的解码参数选择规则对所有方法一致：在 calibration 部分满足同一 native-F1 下限（相对各自编码器的逐帧投票基线最多下降 0.5 个百分点）且 boundary recall >=0.95 的参数中，选多段流 segment F1@50 最高的；差距小于 0.2 个百分点时选内部切分更少的。包括 B1/HMM/B3 在内，不能以仅最小化切分的设置代表其最佳分段结果。若无合格设置，保留完整指标并标为不达标。
- 单独报告 recall/stability 工作点：在冻结 checkpoint 上，从满足相同分类下限与 recall >=0.95 的参数中选择内部切分最少的。此工作点与主比较分别命名，不能交叉选取各自最好的指标拼成一行。
- 新模型直接逐时刻 argmax 是默认解码；其分类变化就是评估边界，辅助边界头不能单独用来报一个与分类分段不一致的更高 recall。若新增解码参数，必须先固定网格并遵守同一 calibration 规则。达不到召回目标的候选明确标为不达标，不通过重新定义边界或压低召回来宣称稳定性获胜。
- 第一轮是否值得扩展：M4 相比 M0/M1/M2/M3 和同编码器调好的简单方法，多段流 segment F1@50 至少有 0.5 个百分点验证增量，native macro-F1 相对最强同编码器对照下降不超过 0.5 个百分点，并通过上述 recall 要求。边界细粒度校准不能在 checkpoint-selection 部分反复调。
- 若只提高分类或只减少切分但未达到门槛，作为局部收益报告并针对失败分析；不把单轮候选未过门槛等同于整个项目结束。

### 最终研发目标（预先写出，不保证一定做到）

1. 多段流 segment F1@50 高于全部预先列明基线；以超过最强基线至少 1 个百分点为实用目标。
2. 原生片段 macro-F1 高于最强完整系统基线；目前历史参考为 EchoViewCLIP 的 0.972，但最终比较用同轮输出。若仅不劣而未超过，则只报告分段提升，不写“所有方面都更好”。
3. 在边界召回不下降超过 1 个百分点且点估计达到 0.95 的条件下，减少原生内部切分；输出整个权衡，而不是挑一个低召回工作点。
4. 三个训练 seed 同向，并提供配对差异。旧测试的探索性显著性不能证明独立泛化；native 用会话重采样，构造流的复用依赖限制继续保留。正式跨数据集论证另需新协议和适用的数据，不能用已失败的 Pediatric 直接迁移假装完成。

上述是明确的多目标研发目标；达到其中一项不能改写为全部达成。最初的 baseline 修复/重训即使提高分数，也不算新方法的收益。

## Task 6 — 第一轮结果必须交付什么

**Files:**
- Create: `docs/results/2026-10-02_model_development/README.md`
- Create: 对应运行目录里的 `run_manifest.json`, `validation_metrics.json`, `training_curves.png`, checkpoint/prediction digest

- [ ] 在同一张表列出 M0–M4 的训练 seed、epoch、encoder、native macro-F1、各类召回、多段 segment F1@50、boundary recall/F1、内部切分、同类拼接切分率、成本。
- [ ] 给出新模型相对最强基线的实际差值；分开说明更强 encoder、辅助边界损失、边界控制 attention、最终 gate 各自带来多少。
- [ ] 分析 A4C/A5C、PLAX/PSAX 混淆以及连续错误段：新模型修好了什么，也破坏了什么；以验证集统计为主，不挑几张好看的例子替代统计。
- [ ] 第一批必须有实际训练曲线与 checkpoint；区分完成、运行中、失败和未启动。GPU 队列或 164 个旧测试通过不等于新模型完成。
- [ ] 记录一次短 profile 后再给真实训练耗时预估；按一个 GPU 的可用资源排队，先让 M4 seed 0 和直接对照出结果，不把所有长基线串在新模型前面。
- [ ] 只提交本轮直接相关文件。现有暂存的 `docs/theory/*`、审阅 Word 和其他无关文件保持原状；无需代用户署名或创建发表版本。

## 第一批执行顺序

1. 当天先完成 Task 1 的必要检查与配置，开始 v3 B0 训练，同时实现 M4 和测试。
2. 缓存就绪后先运行 M0 与 M4 的 smoke/full seed 0；随后 M1/M2/M3 和简单基线校准。不要只排基线而不训练 M4。
3. 输出第一轮完整验证表，然后按已写门槛决定强编码器和三种子扩展。论文包装保持暂缓。

## 参考与新意边界

- ASRF 官方论文：[Alleviating Over-segmentation Errors by Detecting Action Boundaries](https://arxiv.org/abs/2007.06866)；[官方代码](https://github.com/yiskw713/asrf)。类别/边界双分支已有先例；本候选需要证明预测边界控制上下文的具体增量，且仍需进一步近邻文献比对才能讨论新意。
- [ASFormer 官方代码](https://github.com/ChinaYi/ASFormer)包含与 ASRF 组合的路线；不能省略这一近邻比较。
- [EchoViewCLIP 官方代码](https://github.com/xmed-lab/EchoViewCLIP)区分 standard-view classification、OOD detector 与 quality-control 阶段。本轮不扩大到后两项任务。

先把可训练的新模型和真实提升做出来。是否足以形成方法论文，依据训练结果、消融和近邻方法比较再决定。
