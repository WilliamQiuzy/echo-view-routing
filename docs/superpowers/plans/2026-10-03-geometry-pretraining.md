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

- **数据。** CETUS 申请已由项目负责人发出，正在等待回复；本计划不等它。现有动态三维数据是 EchoSlicer 的 29 个体积（4 人）和 STACOM 2011 的 15 位志愿者（各一个体积），共 19 人。CETUS 获批后只增加数据量，判定规则不变。（2026-10-04 更正：组织方回复只有 ED、ES 两个三维体积及标注，CETUS 会同时改变时间采样和数据分布，"只增加数据量"的说法撤回。）
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
  - 管线渲染的 A4C 分数有 14/15 例通过。v14 为 0.975，EchoSlicer 自身为 0.847，比它高出 0.128，按预先写的双侧 0.1 标准判为不通过。这条原始记录保留。
  - 2026-10-04 按项目负责人的意见，没有改成"分数升高就通过"，而是先对齐比较条件：同一时间点、同一组物理采样点、同一显示方向和灰度处理，比较原始图像（`scripts/geometry_aligned_check.py`）。18 个体积的采样点都在管线平面上（最大偏离 6e-14 mm），像素 Pearson r 为 0.994–1.000，最佳平移都是 0；对齐后分类器分数差 17 例 ≤0.01，v14 为 0.037。原标准把显示差异混进了比较（固定网格上 18 例全部是管线更高），因此在训练前对所有体积统一修订一次：正确性以对齐条件下的坐标和像素一致性判断，分类器分数只作辅助检查。修订写在配置的 `amendments` 里。
  - 覆盖率规则按最终掩膜（1° 倾角步长、2 像素边距）在采样实际使用的 181 个整数 ψ 上计算：排除 EchoSlicer 的 D_0（0%）和 D_3（10%）；D_1 为 21%，保留（此前用 5 个 ψ 估计为 18%）。剩 27 个体积，共 42 个体积、19 人。

  原始检查项：
  - EchoSlicer 的 A4C 平面经本管线渲染后，分类器分数与 EchoSlicer 自身相差不超过 0.1；
  - 同一 STACOM 体积的笛卡尔路径与细网格球面重采样渲染同一平面时，共同像素上 Pearson r ≥ 0.95，且最佳整数平移为 0 像素。额外一次重采样会带来散斑层面的差异，这是预期的；位姿错误则会表现为图像平移。
  - 在随机抽取的倾角下，固定掩膜都不越出体积。

## Task 3 — 切片对库与对比度基线

**Files:**
- Create: `echo_routing/geometry/bank.py`、`scripts/geometry_bank.py`、`scripts/ev9v_sector_stats.py`、`scripts/geometry_contrast_baseline.py`、`tests/test_geometry_bank.py`

- [x] 按种子渲染切片对库，存在服务器上，记录摘要（digest）。G1 和 G2 读取同一个库、同样的顺序。先按人均匀抽，再在人内按体积均匀抽（EchoSlicer A 的 15 个体积算一个人），记录每个人贡献的切片对数量。保留人单独一个库（4,000 对，种子 100），训练库每个种子 96,000 对（3000 步 × 32 对）。第一轮不做翻转、旋转和强度增强。
- [x] 标准化用的目标均值和标准差取 EV9V 训练帧扇区内的统计量（55.16、65.64），只计算一次（`data/geometry/ev9v_sector_stats.json`）。扇区由逐像素时间标准差、最大连通域和凸包得到，已目视抽查。最初用 10 次闭运算的版本会从边框向内侵蚀，每段录像都得到同一个矩形，已弃用。
- [x] 对比度基线：岭回归（λ=1，事先固定），输入为两张切片的均值、标准差、锐度、相对锐度及二者之差，目标 d_alpha（另做 |d_alpha|）。在训练人（种子 0 的库）上拟合，在保留人上报告 MAE 和 R²，分别在原始切片和模型最终输入（标准化、截断、缩放到 256、中心裁剪 224 之后）上计算，并列出"总预测 0"的参照。

## Task 4 — 训练脚本与冒烟测试（不正式训练）

**Files:**
- Create: `scripts/geometry_pretrain.py`、`echo_routing/geometry/losses.py`、`tests/test_geometry_losses.py`
- Reuse: `echo_routing/temporal/joint_visual.py`（`load_vision_transformer`、`VisualEncoder`）

- [x] 编码器沿用第六轮 V1：`resblocks.11`、`ln_post`、`proj` 可训练，前缀不求梯度。InfoNCE 投影头 512-512-128，温度 0.1。掩膜显式写出：锚点是 same_plane 对的两张切片，正样本是它的搭档；分母含批内其余每张切片，搭档只出现一次，锚点自身不出现。几何头的输入为 [z1, z2, z1−z2, z1·z2]，经 256 维隐层输出 d_alpha/15°，损失为 smooth L1。
- [x] AdamW（编码器 1e-5，头 1e-4，权重衰减 1e-4），预热 300 步后余弦衰减，梯度裁剪 1.0，共 3000 步，每步 32 对，种子 0、1、2，取最终权重。G1 与 G2 唯一不同的是 lambda_geo 为 0 或 1；G1 的几何项只在 no_grad 下记录，不进计算图，几何头既不训练也不受权重衰减。
- [x] 冒烟测试（18 步，不作正式结果）：G1 与 G2 的批次摘要逐位相同；TF32 关闭；训练峰值显存 1.5 GB。梯度检查：单独的几何项和单独的 InfoNCE 都给 15 个可训练张量有限、非零的梯度（最小范数 1.0e-3 和 2.8e-2），冻结张量没有梯度。

## Task 5 — 主环境特征提取与评估路径空跑

**Files:**
- Create: `scripts/extract_encoder_features.py`、`scripts/encoder_tensor_parity.py`、`scripts/geometry_g0_reproduce.py`（提取器没有单独的单元测试，正确性由同一张量对照和 G0 复现检查）
- Reuse: `scripts/train_feature_probe.py`、`scripts/model_development_queue.py`、`scripts/report_model_development.py`

- [x] 现有提取器只接受完整的 ViFi-CLIP checkpoint，需要另写主环境提取器：`load_vision_transformer` 加 `VisualEncoder`，TF32 关闭。它要写出同样的 `raw/<variant>/<vid>.npz` 和 `extract_meta.json` 字段，帧索引取自 v3 缓存，并拒绝 test 行。
- [x] 容差按项目负责人意见从 2e-2 改为 1e-4（同一输入张量、同一权重、FP32）。结果：同一输入张量上与原编码器最大差 9.5e-6。完整管线对 float32 原始缓存最大差 1.48e-4（orig）和 1.14e-4（gamma090），各 238,572 帧；已定位原因：两个环境裁剪结果完全相同（Pillow 都是 12.3.0），原模型自己在 24 帧的批次里重算也与缓存差 1.26e-4，但放回原提取时的 256 帧批次里差为 0，说明差异来自 GPU 求和顺序随批次形状变化（`scripts/encoder_tensor_parity.py`）。
- [x] 新编码器各自重训 probe（E1 配方），缓存键随 checkpoint 摘要自动变化。native-F1 下限对所有组固定为 0.973/0.980，边界召回 ≥0.95 写回配置。
- [x] 在 G0 上不重新调参，复用已有 probe 和解码设置（`scripts/geometry_g0_reproduce.py`）：probe 的 477,144 个逐样本预测无一改变（logit 最大差 2.6e-4）；三个种子的 ASRF checkpoint 在两个半区解码出的每一帧都与存档相同；B0 和 B3 的 111 张表数值完全相同，工作点相同，推导出的阈值只移动 9e-8。

## Task 6 — 启动审查

- [x] 逐条核对配置中的 `start_conditions`，写一页核对表，列出每条的证据与结论（`docs/results/2026-10-04_geometry_pretraining/start_review.md`）。项目负责人 2026-10-04 的决定：五处修正通过检查后直接启动。8 条全部满足（3a 按修订后的规则），2026-10-04 05:40 UTC 启动。
  - 对比度基线不是启动条件，但结果需要说明：在模型最终输入上，它对带符号倾角差的解释（R² 0.21，MAE 5.85°）强于原始切片（R² 0.07，MAE 6.51°），"总预测 0"为 6.59°。主要来自切片均值：标准化在整块掩膜上做，之后的 0 截断和中心裁剪使均值重新随中心区域的内容变化（与切片自身倾角 r = −0.25）。按配置，G2 几何头的误差若不低于 5.32°（即基线不超过它的 1.1 倍），G2 结果要带"捷径"警示。
  - 启动前又统一修订一次：报告自检发现 G0 的片段平均 AUC 已到天花板（selection 1.000、calibration 0.998；两个半区只有 11 和 18 个原生 A5C 片段），规则 2 无法检出改善。规则改用逐样本 AUC（同一差值、不先按片段平均；G0 为 0.915 和 0.882），片段平均 AUC 照常报告。此时还没有任何 G1/G2 结果。

## Task 7 — 评估与报告

**Files:**
- Create: `scripts/geometry_round_run.sh`、`scripts/geometry_round_report.py`、`configs/experiments/geometry_downstream_v1.yaml`、`docs/results/2026-10-04_geometry_pretraining/`

- [x] 阶段 A（分类）在两个半区上报告：
  - A4C/A5C 区分：原生 A4C 与 A5C 片段的逐样本 probe 差值 AUC（规则使用），片段平均差值的 AUC 一并报告；
  - A4C 召回和 A5C 召回；
  - native macro-F1；
  - 录像内部错误变化（逐帧 probe 的 native_internal_cuts_per_clip）。
- [x] 阶段 B（分段）：ASRF（3 个种子）和 B3 tuned。报告 segment F1@50、0.5 s 内的边界召回、native macro-F1 和录像内切分。每个系统都对照 native 下限（0.973/0.980）和边界召回 ≥0.95，并与 B3、ASRF 和最强旧配置（0.9344/0.9721）比较。最终目标沿用前几轮：三种子平均 segment F1@50 在每个半区都比最强旧配置高至少 1 个百分点，且每个种子、每个半区都满足阈值。
- [x] 对比 G2−G1、G2−G0、G1−G0，按种子做配对 bootstrap；只有三个种子的区间都同向不含 0 才称"一致"，否则写"未检出一致差异"。按配置中的 `interpretation` 解读，并单列机制证据（保留人上的 d_alpha 误差及其与对比度基线的比较、同平面不同时间的表示变化）。

**结果（2026-10-04，`docs/results/2026-10-04_geometry_pretraining/README.md`）。** 按预先固定的规则：G2 没有通过筛查；G1、G2 都不优于 G0；没有任何系统达到最终目标。按配置，这个方向在当前数据规模下停止。项目负责人 2026-10-04 决定关闭这条路线：保留结果、权重（服务器 `models_frozen/geometry_pretraining/`，只读）和停止决定，不追加训练。CETUS 获批不是重跑 G1/G2 的理由：它只有 ED/ES，会同时改变时间采样和数据分布。在提出新的、区别于本轮的可检验机制之前，不启动下一轮模型训练。
- G2−G1 在两个半区的任何分类和分段指标上都没有一致差异，segment F1@50 每个种子相差 −0.4 到 +0.3 个百分点。
- "未检出一致差异"不等于没有损失：六个 G1/G2 的 probe 都把 A4C/A5C 判定点移向 A5C（排序不变），calibration 上同样两个 A4C 片段被判成 A5C，native macro-F1 从 0.985 降到 0.969–0.979；G1/G2 特征上的 B3 在每个种子都低于两个 native 下限；G1 特征上 B3 在 calibration 的录像内切分一致增加。
- 机制证据：G2 的倾角头在保留人上误差 4.65°–4.71°（"总预测 0" 6.59°，对比度基线 5.85°，无捷径警示）。G1 的倾角头没有训练，所以两者的误差比较不能证明 G2 的编码器新增了几何信息，那需要在冻结的 G0/G1/G2 编码器上做同等训练的读出，本轮没有做，也不为关闭本轮补做。G2 让"同平面不同时间"和"同时间不同倾角"的特征距离都变大，两者比值仍约 1.55，没有显示出更能区分心脏运动与平面变化。
