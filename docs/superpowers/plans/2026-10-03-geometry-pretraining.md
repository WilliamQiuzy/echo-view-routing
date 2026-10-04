# 几何监督预训练（最小方案）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 交给 Claude 执行。正式训练（Task 6 之后）须逐条满足启动条件，并经项目负责人批准；在此之前只写代码、做检查。

**Goal:** 只回答一个问题：加入几何监督，能否比使用同样数据、同样预算的普通预训练，更好地保留 A4C/A5C 的差异，同时减少录像内部的错误变化？只在真实二维 EV9V 录像上判定，最终对照 B3 和 ASRF。三维角度预测更准、同一平面输出更稳定，都只算机制证据。

**Architecture:** 从三维体积切出几何关系已知的二维切片对，分三类：同一平面、两个时间（只有心脏在动）；同一时间、两个倾角（只有平面在动）；两者都变。在同一个 E1 编码器上比较三组。
- G0：原始 E1。
- G1：E1 加普通自监督。用 InfoNCE，同一平面的两个时间互为正样本，相当于同一段录像的两帧。
- G2：G1 再加倾角差 d_alpha 的回归。

G1 和 G2 用同一个切片对库、同样的顺序、增强、可训练参数和预算，只差几何项的权重（0 或 1）。之后在真实二维上先比较 A4C/A5C 分类，再用同一套下游评估（B3、ASRF）比较分段。

**Tech Stack:**
- 切片管线：`echo_routing/geometry`，只依赖 numpy 和 scipy。
- EchoSlicer 环境：只用于数据验证和分类器核查。
- PyTorch 主环境：训练和特征提取，TF32 关闭。
- 下游评估：现有的 probe、B3、ASRF 队列与配对 bootstrap。
- 预注册配置：`configs/experiments/geometry_pretraining_v1.yaml`。

## 决定与证据边界

- **数据。** CETUS 申请已由项目负责人发出，正在等待回复；本计划不等它。现有动态三维数据是 EchoSlicer 的 29 个体积（4 人）和 STACOM 2011 的 15 位志愿者（各一个体积），共 19 人。CETUS 获批后只增加数据量，判定规则不变。
- **标签。** 不给任何切片打 A4C/A5C 标签。EchoSlicer 的分类器引导搜索和 EchoPrime 分数只用来核查切片是否像标准切面，不能当真值。切片对的监督只有构造出来的几何关系。
- **机制证据。** 三维角度更准、同一平面在不同时间的表示更稳定，只能作为机制证据。只有在真实二维上同时做到两点才值得扩大规模：保住 A4C/A5C 的类别差异，并改善分类或分段结果。
- **对比度捷径。** 伪影对照显示，同一体积内对比度随倾角系统变化。最近邻结果只说明这种关系不依赖线性插值平均就能出现，不能完全排除平滑或采集过程的影响。本方案不为此另开调查，把三项对照放进训练：
  - 两张切片共用固定掩膜，只取该 ψ 下所有倾角都可见的区域；
  - 每张切片在掩膜内标准化到同一组均值和标准差；
  - 一个"只用对比度"的基线：用每张切片的均值、标准差、锐度和相对锐度预测 d_alpha，在保留人上评分。若它与 G2 的几何头差不多准，就说明 G2 可能走了捷径，每个 G2 结果旁都要报告这一点。
- **样本量。** 按具体实验确定，不以"凑到 60 人"为门槛。真实二维评估的样本由现有两个半区固定。三维侧按人保留 EchoSlicer 的 C 和 STACOM 的 v14、v15、v16，只用于机制检查和对比度基线，不用于选模型；训练按固定预算，取最终权重。
- **测试集。** test 继续封存。selection 和 calibration 两个半区都参与开发，后续结果都属于探索性开发结果。论文继续暂停。

## Task 1 — 取得并验证 STACOM 2011

**Files:**
- Create: `echo_routing/geometry/stacom.py`、`tests/test_geometry_stacom.py`
- Create: `scripts/stacom_verify.py`、`scripts/summarize_stacom_verify.py`
- Create: `docs/results/2026-10-03_stacom_verification/README.md`

- [x] 先下载一例（v1）。确认 DICOM 与 VTK 逐体素完全一致，帧时间取自 DICOM，帧数与官方 readme 一致。
- [x] 拟合探头扇区：顶点、开角、最大深度、中心线倾斜。
- [x] 坐标检查：用专家左室地标做对比和平移检验，并检查 EchoSlicer 由图像选出的 A4C、A2C 平面是否经过地标环中心。
- [x] 视野和时间：左室各层深度、地标到扇区边缘的距离、底部以外的空间、A4C 平面倾斜后仍可见的比例、每秒体积数和每周期帧数。
- [x] 其余 14 位志愿者下载完成，帧数均与 readme 一致。
- [x] 15 例全部验证完成，结果在 `docs/results/2026-10-03_stacom_verification/`（只含数字）。
  - 统一坐标约定为方位轴 −y、相对 EchoSlicer 约定镜像，13/15 例单独评估时也以它为最佳。v7、v8、v15 的 A4C 单独搜索偏向另一轴，因此两轴都跑过一遍。
  - 在该约定下，A5C 搜索方向的中位分数为 0.79，绕声束转 180° 后为 0.14，13/15 例前者更高，由此确定 A5C 一侧。
  - 15 例全部纳入。A5C 分数 ≥0.5 的有 10 例。
  - 专家地标只能宽松地支持坐标：分类器选出的 A4C、A2C 平面都在 10 mm 内经过三个环中心的只有 7 例。

## Task 2 — 切片管线与坐标检查

**Files:**
- Create: `echo_routing/geometry/{planes,volumes,pairs}.py`、`tests/test_geometry_pipeline.py`
- Create: `scripts/geometry_pipeline_check.py`

- [x] 平面参数化：ψ 绕声束旋转，α 绕平面自身横轴扇形倾斜（A4C 到 A5C 的动作），探头在图像顶端。在 ψ=0 时，α 等于 EchoSlicer 的 θ−90；ψ=90 正是 EchoSlicer 的 A2C 起始平面。
- [x] 两种体积：EchoSlicer 球面网格和 STACOM 笛卡尔网格。只有当一个点的全部三线性插值支撑都可见时，它才算"在体积内"，这样边缘的 0 值不会混进切片。在合成场景中，两条路径对同一平面的渲染误差小于 2/200 灰度。
- [x] 切片对采样（三类）、固定掩膜、掩膜内标准化、对比度特征。
- [x] 显示方向：ψ 与 ψ+180 是同一平面的左右镜像。真实数据检查发现，最初的成像列方向会使所有切片左右镜像（EchoSlicer 自己的 A4C 平面在其分类器上只得 0.01–0.05），已改为与标准显示一致，改后得分约 0.99。ψ 的采样范围定为 [−150°, 30°]：A4C 在 0 附近，A2C、A3C 在负 ψ（B_0：−60° 处 A2C 0.90，−90° 到 −150° 处 A3C 0.87–0.98）。几何相关单元测试共 30 项，全部通过。
- [x] 真实数据检查（启动条件 3）：
  - EchoSlicer 的 A4C 平面经管线渲染后，3 例全部通过（0.997/0.990/0.994，EchoSlicer 自身为 0.991/0.915/0.985）。
  - STACOM 两条路径在 15 例中全部一致（r ≥ 0.998，平移为 0）。
  - 在 29 个 EchoSlicer 体积和 15 例 STACOM 中，随机倾角下掩膜都不越界。
  - 管线渲染的 A4C 分数有 14/15 例通过。v14 为 0.975，EchoSlicer 自身为 0.847，比它高出 0.128，按预先写的双侧 0.1 标准判为不通过。分数偏高不是渲染错误的表现，是否改为单侧标准留给启动审查决定。
  - 按覆盖率规则排除 EchoSlicer 的 D_0、D_1、D_3，剩 26 个体积。

  原始检查项：
  - EchoSlicer 的 A4C 平面经本管线渲染后，分类器分数与 EchoSlicer 自身相差不超过 0.1；
  - 同一 STACOM 体积的笛卡尔路径与细网格球面重采样渲染同一平面时，共同像素上 Pearson r ≥ 0.95，且最佳整数平移为 0 像素。额外一次重采样会带来散斑层面的差异，这是预期的；位姿错误则会表现为图像平移。
  - 在随机抽取的倾角下，固定掩膜都不越出体积。

## Task 3 — 切片对库与对比度基线

**Files:**
- Create: `scripts/render_pair_bank.py`、`scripts/contrast_baseline.py`、`tests/test_pair_bank.py`

- [ ] 按种子渲染切片对库，存在服务器上，记录摘要（digest）。G1 和 G2 读取同一个库、同样的顺序。按人划分，记录每个人、每个体积贡献的切片对数量。
- [ ] 标准化用的目标均值和标准差，取 EV9V 训练帧扇区内的统计量，只计算一次并写入配置。
- [ ] 对比度基线：岭回归，输入为两张切片的原始特征、标准化后的特征及二者之差，目标 d_alpha。在训练人上拟合，在保留人上报告 MAE 和 R²。

## Task 4 — 训练脚本与冒烟测试（不正式训练）

**Files:**
- Create: `scripts/geometry_pretrain.py`、`echo_routing/geometry/losses.py`、`tests/test_geometry_losses.py`
- Reuse: `echo_routing/temporal/joint_visual.py`（`load_vision_transformer`、`VisualEncoder`）

- [ ] 编码器沿用第六轮 V1：`resblocks.11`、`ln_post`、`proj` 可训练，前缀不求梯度。InfoNCE 投影头 512-512-128，温度 0.1。几何头的输入为 [z1, z2, z1−z2, z1·z2]，经 256 维隐层输出 d_alpha/15°，损失为 smooth L1。
- [ ] AdamW（编码器 1e-5，头 1e-4，权重衰减 1e-4），预热 300 步后余弦衰减，梯度裁剪 1.0，共 3000 步，每步 32 对，种子 0、1、2，取最终权重。G1 与 G2 唯一不同的是 lambda_geo 为 0 或 1。
- [ ] 冒烟测试只跑几十步，检查损失下降、G1 与 G2 的批次逐位相同、TF32 关闭、显存不超过 20 GB；不保存为正式结果。

## Task 5 — 主环境特征提取与评估路径空跑

**Files:**
- Create: `scripts/extract_encoder_features.py`、`tests/test_extract_encoder_features.py`
- Reuse: `scripts/train_feature_probe.py`、`scripts/model_development_queue.py`、`scripts/report_model_development.py`

- [ ] 现有提取器只接受完整的 ViFi-CLIP checkpoint，需要另写主环境提取器：`load_vision_transformer` 加 `VisualEncoder`，TF32 关闭。它要写出同样的 `raw/<variant>/<vid>.npz` 和 `extract_meta.json` 字段，帧索引取自 v3 缓存，并拒绝 test 行。
- [ ] G0 经新提取器得到的嵌入，与现有 E1 原始嵌入的最大绝对差不超过 2e-2。
- [ ] 新编码器各自重训 probe（E1 配方），缓存键随 checkpoint 摘要自动变化。native-F1 下限对所有组固定为 0.973/0.980。轮次脚本里硬编码的 "E1" 规格键，在新配置里用组名映射。
- [ ] 在 G0 上空跑 probe、B3、ASRF，复现已记录的 E1 数字。

## Task 6 — 启动审查

- [ ] 逐条核对配置中的 `start_conditions`，写一页核对表，列出每条的证据与结论。全部满足后交项目负责人批准；未经批准不启动正式训练。

## Task 7 — 评估与报告

**Files:**
- Create: `scripts/geometry_round_report.py`、`docs/results/<date>_geometry_pretraining/README.md`

- [ ] 阶段 A（分类）在两个半区上报告：
  - A4C/A5C 区分（原生 A4C 与 A5C 片段上，片段平均 probe 差值的 AUC）；
  - A4C 召回和 A5C 召回；
  - native macro-F1；
  - 录像内部错误变化（逐帧 probe 的 native_internal_cuts_per_clip）。
- [ ] 阶段 B（分段）：ASRF（3 个种子）和 B3 tuned。报告 segment F1@50、0.5 s 内的边界召回、native macro-F1 和录像内切分，并与最强旧配置（0.9344/0.9721）对照。
- [ ] 对比 G2−G1、G2−G0、G1−G0，按种子做配对 bootstrap；只有三个种子的区间都同向不含 0 才称"一致"，否则写"未检出一致差异"。按配置中的 `interpretation` 解读，并单列机制证据（保留人上的 d_alpha 误差及其与对比度基线的比较、同平面不同时间的表示变化）。
