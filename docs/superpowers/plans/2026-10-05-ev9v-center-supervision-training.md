# EV9V 视频识别与目标时刻监督：配对训练任务书

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实际训练同一个视频模型的三组监督对照，检验目标时刻监督能否在学习原生录像分类的同时改善时间分段，并与 B3、ASRF 及既定最强配置比较。

**Architecture:** E1 的 CLIP ViT-B/16 编码每帧，512 维帧特征进入两层带时间位置的 Transformer，输出一个五类窗口预测和辅助逐帧预测。三组网络完全相同，均微调视觉编码器最后四个 block；主要对照只改变混合窗口的监督目标。窗口概率经过同一套 B3 得到时间分段，不增加边界头、gate、蒸馏或其他解码器。

**Tech Stack:** 现有 PyTorch 主环境、E1 导出权重、cache_v2 帧索引、冻结 EV9V protocol、既有窗口规划和 B3 评估工具。

**状态与授权：** 用户已结束文书任务并明确要求恢复模型训练、由 Claude 执行。本文件是新一轮有界训练指令，不是已完成实验。完成输入、实现及训练正确性检查后，直接执行三组各三个种子，共九次训练；不再以旧 B7 的 R1 数值作为开训条件。Codex 本次只写任务书，不启动另一份重复队列。README 写给项目组，用英文。

---

## 1. 决策依据、改变了什么、不能声称什么

- 最新依据：`docs/results/2026-10-05_b7_window_target/README.md`（47a4f01 / faa2a43）。B7 对帧排列不敏感；12/16 中心类帧的配对检查中，异类上下文导致准确率下降 2.3 点，新增错误涉及 25 段目标录像、18 个 session。这支持检查目标时刻监督，未证明新模型会改善。
- 旧 R1 将固定 B7 判错的混合窗口换成参考标签 one-hot，再用固定 B3 解码。selection 仅达到 0.907，native macro-F1 保持 0.957。这支持停止“只修 B7 混合窗口”的局部方案。R1 是一种标签知情干预，不是所有可训练模型的数学上界，也不是所有概率输出和解码设置的严格最优值。
- 本轮同时训练原生窗口的五类识别和混合窗口预测，视觉权重会改变。它不假设原生错误已被解决；能否修好 A5C/SC4C 必须直接报告。只改善混合窗口、原生分类仍不达标，仍算最终目标失败。
- 9 月 25 日 V2b/V2c/V4c 的数字因读帧问题作废。本轮是借用既有的分解式视频 Transformer 思路，使用 E1 初始化、正确输入和明确对照重新检验。不是原实验的逐参数复现，不称为新架构发明。
- Round 2 是 E1 与 M5/ASRF 的长序列联合微调；Round 6 是 ASRF 固定区间上的二类平均特征分类。本轮是固定视频窗口上的五类目标时刻监督，推理不读 ASRF 区间。不要把前两轮的负结果删除或改写。
- 最后四个 block 是本轮事先固定的共同优化范围，不另做深度搜索。它与以前只训练最后一个 block 不同，因此本轮不能单独证明“多解冻三层”的作用。
- 不重开等权融合、条件蒸馏、M5、固定区间读出或几何路线。不使用 Pediatric 及其适配权重，不下载新数据，不改标签，不剔除困难录像。
- 两个验证半区都多次用于开发，E1 的上游 checkpoint 也在历史验证集上选择过。本轮依然是探索性研发；当前不读取 test 像素、预测或指标，不称旧 test 从未被使用。

## 2. 固定问题与训练组

| 组 | 输入 | 窗口主监督 | 用途 |
|---|---|---|---|
| N：原生训练对照 | 每个 batch 八个单一来源窗口 | 来源录像的五类标签 | 判断新的共同训练配方本身表现怎样 |
| S：混合内容对照 | 四个原生窗口 + 四个跨切面窗口 | 每类在窗口内的帧占比；原生窗口为 one-hot | 控制模型看过混合输入、预算和容量 |
| C：目标时刻候选 | 与 S 逐窗口逐帧相同 | 指定时刻所属来源录像的五类标签 | 唯一主要候选；检验目标时刻监督 |

**主因果对照是 C−S。** 两者的网络、初始化、图像、帧顺序、增强、批次、优化器和更新次数相同，仅窗口主损失的 target 不同。N 与 S/C 的差异同时包含上下文输入和主监督变化，不能把 S−N 全部归因于“数据增强”。三组都有相同权重的辅助逐帧损失。

不采用“混合窗口多数类”作为唯一控制：只有一个切换点的等间距窗口里，多数类经常就是中心类，会让两组目标几乎相同。S 使用类别占比，**不加类别权重或 label smoothing**，避免旧的加权 soft-target 目标再次改变类别比例。

## 3. 输入与窗口合同

**只读已有输入：**
- `configs/experiments/model_development_v2.yaml`，以及引用的原 protocol。
- `runs/20261002-051140-model_development_protocol-fa45e7f4-s0/protocol.json`，sha256 `a897483b776e8cff1ba8b92774c6a82537758bcba6abd362f2d8603f37997593`。
- `models_frozen/e1_echoviewclip_probe/v1/visual_encoder_fp32.pt`、对应 JSON、`probe.pt`、`feature_manifest.json`。
- `models_frozen/echoviewclip_stage1/v1/best.pth` 只用于来源核对，不从 Pediatric 导入权重。
- 数字排序后的原始帧、cache_v2 的实际 `frame_idx`、原训练库 1,500 条流，以及原生训练录像。

**类别顺序固定：** `[PLAX, PSAX, A4C, A5C, SC4C]`；训练五类子集的全部 3,364 段录像，不加入原来被映射排除的类别。

**时间约定：** 10 Hz，窗口最多覆盖连续 16 个样本，模型读取 16 张图；窗口推理 stride 为 5 个样本，原生和构造流共用相同路径。复用 `window_centres`、`window_samples`、B7 的等距取 16 帧规则和 `assign_to_samples`。不能在参考切换点重置窗口。

**指定时刻不一定是输入第 8 张图。** 片段开头、结尾和短片段会移动或重复采样。每个窗口必须记录真实的目标样本索引 `c`、取出的样本索引 `s[0:16]`，主标签必须来自 `label[c]`。各输入 token 的相对位置为 `s[i] - c`。这个位置来自采样时间轴，不来自标签或参考边界。推理接口只接收图像和相对索引，不接收类别、边界、录像 ID 或窗口类型。

**训练采样（生成后固定）：**
1. 每个 seed、epoch、update 的前四个窗口为共同的原生窗口：均匀抽类别，再均匀抽该类别录像，再均匀抽该录像的目标样本。所有训练录像均可被抽到；不按验证错误挑样本。
2. 后四个窗口：从原 1,500 条训练流枚举实际混合切面窗口。均匀抽目标类别，再均匀抽目标来源录像，再均匀抽该录像可用的混合窗口。以“至少两种标签”定义混合，跨录像但同类不算。记录每类可用录像及窗口数量；训练库必须五类均有可用混合窗口，否则属于实现/数据合同问题，报告而非改用验证数据。
3. S 与 C 使用完全相同的后四个窗口。N 将它们换成目标来源录像自身的原生窗口，并保留同一目标帧和目标帧的对比度版本。N 的前四个窗口与 S/C 完全相同。三组各读取 8×16 帧，每次各更新一次。
4. 原生窗口对比度以概率 0.5 取 `orig`，以概率 0.5 取已有 `gamma090`。混合窗口保留冻结训练流的各片段变体。不得新增按照类别设置的增强。N 的替代窗口沿用被替代目标帧的 gamma。
5. 小于 16 样本的原生录像沿用 B7 等距重复取样规则，不丢弃、不时间倒序，也不把重复帧当新病例。类别占比及辅助损失按实际输入的 16 个位置计算；重复采样的数量单独记录。
6. 按来源录像保存曝光次数。数据生成的 RNG 与模型 dropout RNG 分开，S/C 的完整 manifest 必须逐字节一致；每个 epoch 保存输入 digest，不能只核对初始权重。

所有帧标签都继承公开的录像级标签；它们是代理任务监督，不是专家逐帧确认的真实切面状态。

## 4. 固定模型和损失

**编码器：** 复用 `echo_routing/temporal/joint_visual.py` 的 `load_vision_transformer` 与 `VisualEncoder`。解冻 `transformer.resblocks.8..11`、`ln_post` 和 `proj`；其余视觉权重冻结。不更换 backbone，不增加局部 patch 分支。

**时间头：** 借用 `echo_routing/vit/models.py::FactorizedVideoViT` 的结构，但使用 E1 的 512 维嵌入。固定 2 层 TransformerEncoder，8 个 attention heads，FFN 2048，GELU，pre-norm，dropout 0.1。输入为一个可学习 CLS token 加 16 个帧 token。帧 token 加相对目标时刻的位置 embedding（整数偏移 −15..15）；CLS 单独有位置参数。CLS 经最终 LayerNorm 和 `Linear(512,5)` 输出窗口 logits。位置表必须检查索引范围，不能静默裁切。

辅助 `Linear(512,5)` 读取时间头之前的帧特征，输出 16 个逐帧 logits。窗口分类头和辅助分类头都从同一个冻结 E1 probe 的 weight/bias 初始化，之后独立训练。Transformer、CLS、位置参数按 seed 初始化；三组从同一份完整初始 state_dict 出发。

Token 流：`B×16×3×224×224 → B×16×512 → B×17×512 → B×5`；辅助支路为 `B×16×512 → B×16×5`。这是带位置的时间 self-attention；本轮不使用 cross-attention，也不把 self-attention 自身称为贡献。

```python
# logits: [B,5], frame_logits: [B,16,5], labels: [B,16]
# target_labels: [B] from the actual target sample c, not a hard-coded slot
soft = one_hot(labels, num_classes=5).float().mean(dim=1)
target = soft if arm == "S" else one_hot(target_labels, 5).float()
loss_main = -(target * log_softmax(logits, dim=-1)).sum(dim=-1).mean()
loss_frame = cross_entropy(frame_logits.reshape(-1, 5), labels.reshape(-1))
loss = loss_main + 0.5 * loss_frame
```

N 的原生标签恒定，因此其 `target_labels` 也是整窗标签。三组不加额外正则、pair loss、蒸馏、对比损失、置信阈值或输出池化。单一切面窗口上，S 与 C 的主监督必须完全一致。

## 5. 训练预算、模型选择和资源

- seeds 固定 `[0,1,2]`；N/S/C 每组 10 epochs，每 epoch 375 updates，有效 batch 8，共 3,750 updates/模型、30,000 windows/模型。九个最终权重全部报告，不因 seed 0 结果好坏决定是否继续其余种子。
- AdamW；视觉后四层 lr `1e-5`，新时间头和两个分类头 lr `1e-4`，weight decay `1e-4`，grad clip 1.0；前 375 updates 线性 warm-up，之后 cosine 到 0。
- 固定 fp32、关闭 TF32。继承主环境与 `ECHO_GPU_MEM_FRACTION` 限制。逐模型顺序运行，不抢占其他项目进程；显存不足时只允许使用 activation checkpointing 或等价梯度累积，不能偷偷缩短窗口、减少 batch 或冻结不同层。变更须在九次正式运行之前固定并对三组一致执行。
- 所有组使用最终 epoch，不使用 early stopping，不按任一验证半区挑 checkpoint。正式训练期间只记录训练损失、梯度、时间和资源，九次训练完成后才做最终下游选择与对比。
- 初始化 hash、可训练/冻结参数 hash、逐 epoch 帧 digest、批次顺序、优化器状态和随机状态写入 run manifest。相同 seed/arm/config 的短 GPU 重跑应可复现；浮点容差在看任何验证结果前固定。
- 不以 N/S/C 的训练准确率当泛化证据；不要求新模型开训前已能超过旧模型。启动条件只检查输入合法、实现正确、训练能运行、配对成立及资源可用。

## 6. 下游评估与预先固定的选择规则

**原输入保留：** 两个验证半区的 native、pair、multi banks 不变，公开标签不变，主表包含全部录像。新模型和 N/S 对照共用与 B7 相同的窗口、赋值时间轴和后处理。

**推理两行：** 每个模型先报告窗口概率直接 argmax 的结果，再报告 B3。主系统预先固定为 `C + B3`，不得看到结果后改选 N/S 或换解码器宣称该候选成功。新模型编码每帧独立，可按“checkpoint + frame + gamma”缓存特征；不能跨模型复用训练后特征。

**B3 选择：** 每个模型、每个 seed 使用既有同一 40 格网格：half_window `[5,7,9,11,15]` × quantile `[.80,.85,.90,.95,.97,.99,.995,.999]`，nms radius 5。quantile 只由该模型的 calibration native 概率计算。在 calibration 满足 native macro-F1 ≥ .980、multi boundary recall ≥ .95 的设置中取 segment F1@50 最大者；同分按窗口、quantile 升序取先出现者。冻结后只在 selection 应用一次。

**无可行设置时也要交付结果：** 标记 B3 系统不可行，不用 validation 选另一种回退。固定诊断行使用 half_window=9、quantile=.95、nms=5，阈值仍只从 calibration native 计算。这行在两半区都报告，但不能算通过任何成功条件。直接 argmax 行始终报告。不得因为无可行 B3 就隐藏该 seed。

**对照：** 复用并验证 E0+B3、E1+B3、B7+B3、E1+ASRF 三种子的现有冻结输出；保留原生监督/架构不完全等预算的说明。这些是性能参照，C−S 才是本轮监督机制的配对比较。ASRF 本轮不重训、不换特征、不用新视频模型概率当输入。若未来需要同新表示上的 ASRF，那是另一项明确比较。

**每个 seed、每个半区完整报告：** native 五类 macro-F1、每类 precision/recall、A4C↔A5C 与 SC4C→PLAX 的修好/新错及独立来源数；multi segment F1@50、boundary precision/recall/F1（0.5 s）、额外切分/录像；pair 指标只作次要结果。区分新模型窗口识别、直接赋值以及 B3 后结果。

复用窗口类型普查与上下文探针作诊断，分别报告原生、跨来源同类、混合切面窗口，不再从探针挑新的训练超参数。上下文配对比较必须按目标来源/session 统计；不把重复窗口当独立人群证据。

**不确定性：** native 使用现有 session 配对 bootstrap。构造流的现有 bootstrap 仅作为近似探索性区间，明确复用来源录像的依赖，不能断言区间一定保守或一定偏窄。报告来源去重的修好/新错和集中程度作为补充，不把它们称为解决了依赖。三个训练种子共享验证集，不是三个独立样本群。

**最终目标不变：** 同一个 `C+B3` 在两个半区的三种子均值 segment F1@50 均至少比旧参照高 .01；selection 参照 `0.9344097`，calibration `0.9721422`（以既有归档的全精度值核对）。每个种子、每个半区的 boundary recall ≥ .95，native macro-F1 ≥ .973 / .980。另对 calibration 的后续 `0.9739` 系统做敏感性参照，不改主门槛。

**机制结果与最终目标分开判定：** 预先报告 C−S 的逐种子差值、均值和配对区间。只有某半区三个区间均在零以上，才称该半区“检出一致提升”；区间含零不称相等。即使 C−S 有局部收益，未达到全部最终条件就明确写目标未完成。C 未优于 S 不能证明所有目标时刻模型无用，但本轮配方停止，不自动扩层、改 loss 或追加 sweep。

## 7. 实施任务与文件

### Task 1：冻结配置和训练窗口 manifest

**新增：** `configs/experiments/ev9v_center_supervision_v1.yaml`、`echo_routing/vit/target_windows.py`、`scripts/prepare_center_supervision.py`。

- [ ] 新配置完整列出本文件的参数；不要通过继承旧 ViT 配置意外带入 raw9、class weights、旧抽帧、验证早停或测试集。
- [ ] 核对输入及 checkpoint digest，将解析后的配置、protocol、类顺序和抽样 RNG 约定固定；训练计划和结果写入新目录，不覆盖旧配置或缓存。
- [ ] 将训练窗口生成与验证窗口规划分开。训练数据集只允许 protocol 的 train 来源；原生、混合、native 替代窗口保存实际 SampleRef 和时间索引。验证标签只能在评分器中读取。
- [ ] 添加 `tests/test_center_supervision.py` 的输入合同测试：错误帧序或缓存版本被拒绝；train 混入 validation/test 来源被拒绝；N/S/C 帧数与目标对齐；同类窗口 S/C target 相等；混合窗口 S target 和为 1；边缘和短序列的目标索引正确。
- [ ] 提交配置和管线，再生成各 seed 的 manifest；只提交自己的文件，不推送、不触碰暂存理论稿及其他改动。

### Task 2：模型、训练器及必要冒烟检查

**新增：** `echo_routing/vit/target_video.py`、`scripts/train_center_supervision.py`、`scripts/queue_center_supervision.py`。只复用旧公共组件，不修改旧实验行为。

- [ ] 在新模块实现 E1 + 时间头；测试 512 维特征、相对位置、主/辅助输出形状。时间头必须允许改变目标时刻影响输出；仅改变同一输入集合的 token 顺序并同步相对索引，不应伪造新的目标。
- [ ] 实现主损失并用明确的小 tensor 验证公式。全部同类时 S/C loss 及梯度相同；混合标签时两者只有 target 不同。前向一次，主损失与辅助损失共享同一批视觉特征。
- [ ] 复用导出参考帧验证初始 E1 特征：与 fp32 参考最大差 ≤ 1e-3，与已有 fp16 缓存 ≤ 2e-2。核对解冻的仅为最后四层、ln_post、proj；时间头与分类头均有梯度，冻结层不变。
- [ ] 用训练集窗口完成短 GPU smoke 和恢复运行检查。检查非有限值、梯度、损失更新、同 seed 复现、S/C 输入配对。不能用验证集分数决定是否开始正式九次训练。技术错误修复后重跑受影响检查，记录修复；不是追加研究假设。
- [ ] 先记录实测显存与每步耗时，估算九次训练预算；保持既有共享 GPU 配额。队列的跳过条件必须匹配完整 run identity 和完成 manifest，不能只看目录或 checkpoint 存在。
- [ ] 独立实现审查后提交；不得以“代码已完成”代替后续正式训练。

### Task 3：执行九次正式训练

在服务器 `/home/william/echo-view-routing` 的现有 `.venv/bin/python` 下提供以下 CLI；实现后逐条执行：

```bash
.venv/bin/python -m pytest tests/test_center_supervision.py tests/test_joint_visual_temporal.py -q
.venv/bin/python scripts/prepare_center_supervision.py --config configs/experiments/ev9v_center_supervision_v1.yaml --out runs/center_supervision_v1/manifests
.venv/bin/python scripts/train_center_supervision.py --config configs/experiments/ev9v_center_supervision_v1.yaml --arm C --seed 0 --smoke --manifest-root runs/center_supervision_v1/manifests
.venv/bin/python scripts/queue_center_supervision.py --config configs/experiments/ev9v_center_supervision_v1.yaml --manifest-root runs/center_supervision_v1/manifests --status runs/center_supervision_v1/status.json
```

- [ ] queue 按 seed 0/1/2，各 N/S/C 顺序执行九次。每次输出明确的 arm、seed、run_id、起止时间、已完成 updates 和最终 checkpoint hash。
- [ ] 使用现有 detached runner 启动单队列；只有读到进程、日志和已完成 update 才报告“训练已开始”。只配置队列不算开始，checkpoint 存在不算全部完成。
- [ ] 可恢复中断运行，必须恢复优化器、step、sampler/RNG 和 run identity。不能重复计算一次 update，不能复用 smoke 权重。最终损失与训练曲线保留。

### Task 4：完整评估、冻结和英文报告

**新增：** `scripts/report_center_supervision.py`、`docs/results/2026-10-05_center_supervision_training/README.md`、`report_tables.md` 和机器可读结果；权重留服务器。

```bash
.venv/bin/python scripts/report_center_supervision.py --config configs/experiments/ev9v_center_supervision_v1.yaml --status runs/center_supervision_v1/status.json --out docs/results/2026-10-05_center_supervision_training
```

- [ ] 报告脚本必须先检查九次训练完整、三组输入配对和最终权重身份。calibration 选参函数只得到 calibration 对象；selection 仅应用冻结设置。
- [ ] 每个训练后 encoder 重新提取自己的特征并生成窗口输出；不得读取旧 E1 特征冒充训练后表示。只使用主窗口 logits 作为该模型的部署输出，不事后择优改用辅助头。
- [ ] 主表三组 × 三种子 × 两半区完整展示，附原强基线、native 每类结果及来源去重的修好/新错。记录端到端推理时间，不能只计时间头宣称架构加速。
- [ ] 使用 `scripts/freeze_model.py` 的现有格式冻结九个模型、配置、输入与权重摘要。结果不佳也保留，主报告不得只展示胜出的 seed。
- [ ] 只提交本轮代码、测试、配置与汇总报告，不推送。论文继续暂停。本轮训练后停止，不自动继续新模块。

**最终必须回答四句话：** 实际训练了哪九个模型；C−S 是否有收益；原生录像分类是否一起改善或退步；同一个完整系统是否超过既定最强配置并满足全部约束。不能用“又发现一个机制”替代模型结果。

## 计划审查记录

2026-10-05，独立计划审查通过，未发现会导致实现受阻或主比较失效的严重缺口。审查覆盖 C−S 配对、短片段/边缘窗口的目标对齐、四层解冻兼容性、B3 选择和失败交付。本记录只证明计划经过审查，不代表代码已实现、九次训练已运行或方法有效。
