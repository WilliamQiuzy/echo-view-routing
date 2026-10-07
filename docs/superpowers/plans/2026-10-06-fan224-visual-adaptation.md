# Fan224 visual adaptation Implementation Plan

> **For agentic workers:** Use the executing-plans workflow to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. The intended executor is Claude; the author of this plan has not started model training.

**Goal:** 在完整扇形的 224×224 输入上，检验常规视觉适配是否改善 A4C/A5C 判定，并把收益传给 B3；用配对对照分清输入变化和训练变化。

**Architecture:** E1 ViT-B/16 → 原有 512 维全局特征 → 五类线性分类头 → B3。比较两种输入与两种训练范围的 2×2 设计。暂不加入时间 Transformer、额外 attention、残差、蒸馏或新损失。

**Tech Stack:** PyTorch、PIL、NumPy、现有 EV9V protocol、版本化特征缓存、B3 和 session 配对 bootstrap。

---

## 0. 决定、依据和授权范围

本轮执行的是一次有边界的常规适配比较，不是新架构贡献。准备、输入检查和独立实现审查通过后，按此任务书启动 12 次训练，不需要另等一次泛泛的“可以开训吗”。检查失败时修正实现；若必须改训练设计或无法保证裁剪保留图像区域，则报告具体阻断项，不静默换任务。

研究目标仍是心超切面识别与时间分段。论文暂停。保留全部公开标签、来源录像和现有开发划分。不读取 EV9V test，不使用 Pediatric 数据或它的派生产物；可以参考现有适配代码，但新脚本不能 import 会读取 Pediatric 的入口。

**已有证据：**

- 第四轮中，B3 与至少两个 ASRF 种子共同失败的 54 个参考段里，38 个是 A5C→A4C，另有 13 个 A4C 段；它们来自 29 段录像。只修类别的诊断收益比只修分段更大。它支持优先检查类别证据，不证明标签错误、视觉网络的上限或任意新模型的上限。
- C 的时间读出减少逐时刻错误，却把少数整段录像改错。R 保住部分整段判定，但相对 C 的原生录像内部切换从 0.08–0.12 增至 0.18–0.31 次/段。C/Z/R 均未超过强基线。这条读出搜索继续关闭。
- `docs/results/2026-10-06_fan224_pilot/README.md` 记录本次 30 段训练录像、480 帧预处理试做。旧中心裁剪已去掉大部分外围文字；完整扇形输入还改变了视野和比例。这不是“去字必然提高分类”的证据。
- 第六轮训练的是 ASRF 给定区间上的二分类读出；本轮是新输入契约下原生录像的五类视觉适配。两者不同，但不得据此称本轮有方法新意。

**本轮明确不回答：** 真实连续扫查的性能、像素级解剖分割、哪些帧真正显示流出道、跨患者泛化，以及裁剪收益是否专门来自文字消除。

## 1. 冻结比较与训练设置

| 组别 | 输入 | 可训练部分 |
|---|---|---|
| L0 | 原始图像经当前 E1 的 256/224 中心裁剪 | 五类线性头 |
| L1 | 与 L0 完全相同 | 同一个头 + ViT block 11、ln_post、proj |
| F0 | 经训练图像检查后冻结的扇形输入，等比例补边至 224 | 五类线性头 |
| F1 | 与 F0 完全相同 | 同一个头 + ViT block 11、ln_post、proj |

每组种子 0、1、2，共 12 次；F1 是事先指定的适配候选，不在看 selection 后换成另一个候选。所有行完整报告，F0 若更好也如实记录。

每个种子四组从相同的冻结 E1 图像编码器及 E1 probe 起步。使用 `ev9v_center_supervision_v1.yaml` 的 visual_export/probe/protocol 路径和 sha256，核验后复制进新配置。不得从 C、R 或失败微调权重继续训练。

**固定训练配方：**

- 仅协议中的 3,364 段训练录像，五类均保留。每步 16 段 × 每段 4 帧，共 64 帧。
- 每次先均匀抽五类，再在该类均匀抽来源录像。每段在现有 10 Hz 采样索引中分散取 4 个样本，短于 4 时有放回；复用 `spaced_indices` 的确定性规则。
- 每段以各 0.5 概率取 orig 或 gamma090，四组共享同一 gamma 和原始帧索引。gamma LUT 在任何遮罩/空间变换前应用。无随机裁剪、旋转、翻转或新增增强。
- 均值逐帧五类交叉熵。录像标签传播到帧是弱监督，不能描述为帧级专家标签。不新增 clip pooling、对比损失或时序损失。
- AdamW：分类头 lr=1e-3，L1/F1 的视觉可训练层 lr=1e-5；weight decay=1e-4；梯度范数上限 1；前 100 步线性 warm-up，之后 cosine 到 0。
- 固定 1,500 次更新，只取最终权重；无 early stopping、无验证选 epoch。fp32、TF32 关闭。
- 每种子独立数据 RNG，四组共用保存的批次清单。每步记录原始样本身份 digest；L/F 像素不同，不能要求二者像素 digest 相同。L0/L1 及 F0/F1 的输入像素须逐步相同。
- head-only 必须显式冻结整个编码器，包括 ln_post 和 proj。当前 `VisualEncoder(trainable_blocks=0)` 仍会把这两处设为可训练，不能直接拿它当全冻结模式。

## Task 1: 把探索性裁剪变成可验证的输入契约

**Files:**
- Read: `scripts/pilot_fan224.py`, `echo_routing/temporal/joint_visual.py`, `echo_routing/ingest/frames.py`。
- Create: `echo_routing/ingest/fan224.py`, `configs/experiments/fan224_visual_adaptation_v1.yaml`。
- Create: `scripts/check_fan224_input.py`, `tests/test_fan224_input.py`。

- [ ] 固定训练图像 QC 清单：包含试做的 30 段，再用 seed=20261007 每类从其余训练录像均匀取 14 段，共 100 段。另扫描全部训练文件的尺寸，所有非 320×240 或内容为空的记录列为异常，不删除它们。
- [ ] 每段检查首、中、末三帧的原图、遮罩轮廓和输出。这里只检查图像是否明显被截断、外围界面是否残留，不重标切面、不把 AI 判断当解剖真值。单独记录残留刻度与明显截断。当前 pilot 模板在扇形边缘仍有刻度，不能直接宣称验收通过。
- [ ] 若统一模板不适合某些布局，仅在训练图像上改成明确的布局模板/检测规则并重复 QC；记录修改。规则不得读取标签、来源 ID、split 或预测分数。不得逐帧按心脏内容缩紧框，不用 inpainting 猜测被遮住的组织。无法避免明显截断则停止，不用只保留“成功裁剪”的子集训练。
- [ ] 冻结遮罩/框、颜色顺序、插值、padding、gamma 次序及代码摘要。以 `fan224_pilot_v1` 等明确版本命名；团队正式交付的版本若不同，必须另记版本，不覆盖同一缓存。
- [ ] `fan224` 路径断言输出 224×224、RGB uint8；最终模型入口只做既有 mean/std 归一化。保留原 `crop_uint8` 不改，让旧实验仍可重现。输入类型由 manifest 显式指定，不能仅看尺寸判断是否已经处理。
- [ ] 输出 PNG 与 manifest，保留源帧号、时间戳、frame-order version、源文件/变换摘要。无患者标识或来源录像编号进入公开 README；映射留服务器。原始 JPEG 不覆盖。原始图像已经压缩，PNG 不会恢复丢失细节。
- [ ] 写并运行必要测试：224 输入不再次 resize/crop；原路径与 `crop_uint8` 像素一致；数字顺序 0/1/2/10；L/F 原始帧与时间戳一一对应；固定录像模板不随帧变化；不支持的尺寸或缺失帧报错；不得从训练入口读 validation/test。

Run: `.venv/bin/python -m pytest tests/test_fan224_input.py -q`，应全部通过。

## Task 2: 固定清单并实现 12 次配对训练

**Files:**
- Read: `scripts/pediatric_adapt.py`（只参考训练实现，不读取数据）、`scripts/train_feature_probe.py`、`scripts/extract_encoder_features.py`。
- Create: `scripts/train_fan224_visual.py`, `scripts/queue_fan224_visual.py`, `tests/test_fan224_visual.py`。
- Update: `configs/experiments/fan224_visual_adaptation_v1.yaml`。

- [ ] 配置写全四组、三个种子、上述超参数、变换和参考权重摘要；无隐式默认值。清单生成只看训练 manifest 和协议 train IDs。训练脚本只接受这四组，不引入其他搜索参数。
- [ ] 每种子存一次初始状态、批次清单及其 digest。再次生成须一致。缓存身份包含图像变换、编码器、分类头和帧顺序摘要；缺少身份信息拒绝使用。
- [ ] 用每种子前 6 步做可重现/恢复检查，不读验证集。L0/F0 的全部编码器参数和状态保持逐字节不变；L1/F1 只有指定末层变化，梯度有限且非零。初始 L0 前向复现现有 E1 fp32 至最大绝对误差 1e-3 内。
- [ ] 全冻结编码器可以预提特征以节约运行时间，但必须对齐相同 1,500 步和 batch 清单，不复用旧 probe 的验证选 epoch 逻辑。验证 cached 与 online 前向在 fp32 容差内一致。
- [ ] 写一次独立实现审查，解决影响结果的缺陷；在训练及任何新验证指标产生前提交配置和代码，只 stage 本轮文件。估算单次耗时和总预算，确认 GPU 空闲，然后按 seed 0→1→2、每个 seed L0/L1/F0/F1 排队。
- [ ] 训练期间只记录训练损失、更新数、耗时、显存及身份信息，不看验证表现改变计划。每个运行目录绑定 arm/seed/config/transform/batch/weight 摘要，只跳过身份完全匹配的已完成作业。

Run: `.venv/bin/python -m pytest tests/test_fan224_input.py tests/test_fan224_visual.py -q`。

新 CLI 约定（实现这组入口，不把示例当已存在的命令）：

```bash
.venv/bin/python scripts/train_fan224_visual.py --config configs/experiments/fan224_visual_adaptation_v1.yaml --arm F1 --seed 0 --smoke
.venv/bin/python scripts/queue_fan224_visual.py --config configs/experiments/fan224_visual_adaptation_v1.yaml
```

## Task 3: 相同分段器下评估，区分输入和训练的作用

**Files:**
- Read/reuse: `scripts/calibrate_simple_decoders.py`, `echo_routing/evaluate/dev_metrics.py`。
- Create: `scripts/report_fan224_visual.py`, `tests/test_fan224_report.py`。
- Create: `docs/results/2026-10-06_fan224_visual_adaptation/{README.md,report_tables.md,summary.json}`。

- [ ] 全部训练完成后，提取原协议的 train 和两个 validation half 的 10 Hz 特征/概率，含 orig、gamma090；两种输入共享原始帧索引和构造流 recipe。推理使用逐样本视觉头，没有时间窗口和 sparse centre fill。
- [ ] 固定 E1 不适配的 L/F 前向也报告，但只能叫输入敏感性对照：F 上零样本退步不等于重新训练后不能改善。可保留 mask-only+legacy 的少量训练图像预览，不能把它当正式第三训练组或因果分解。
- [ ] 每组每种子接同一 B3 算法与同一搜索预算。half_window=[5,7,9,11,15]，quantile=[.80,.85,.90,.95,.97,.99,.995,.999]，nms_radius=5。阈值由各模型 calibration 原生分数的相同分位数规则计算。
- [ ] 仅在 calibration 上选参数：native macro-F1≥.980、multi boundary recall≥.95 的可行设置中取 segment F1@50 最大者；同分按 (half_window, quantile) 升序。然后在 selection 应用一次。无可行项则报告失败；9/.95 的诊断行不得通过任何成功门槛。
- [ ] 同时报告无 B3 的逐帧/原生结果。不同模型的 B3 数值阈值可能不同，因此完整系统的全部变化不能归因于编码器。不得按 selection 改解码或从候选中挑赢家。
- [ ] 四组都列：五类 native macro-F1、各类召回、A4↔A5 混淆、逐样本错误、录像内标签切换、segment F1@50、边界 precision/recall。主问题是识别改进能否传到分段，不能只给 AUC。
- [ ] 对比 F0−L0、F1−L1（输入作用）；F1−F0、L1−L0（末层微调作用）；另给两者的交互差值，避免把一般微调收益算作扇形特有收益。不把“未检出差异”写成相等。
- [ ] 所有修好/新错按来源录像、session 去重。报告两个方向的 A4C/A5C 变化；说明多少 segment 变化由同一录像重复造成。置信区间沿用 2,000 次 paired bootstrap：native 按 session、流按流；流间复用素材使后者近似、探索性。种子共享验证样本，不能当独立病例增加样本量。
- [ ] 汇报现有 E0+B3、E1+B3、B7+B3、E1+ASRF 冻结参考。新扇形输入不能复用旧 ASRF 特征缓存；本轮不重训 ASRF。因此只能声称相对历史系统/本轮匹配 B3 对照的表现，不能声称超过“同等适配后的全部基线”。

## Task 4: 固定解读、停止和交付

**本轮的主问题：** F1 是否同时优于同输入的简单适配 F0 和同预算的旧输入 L1？如果只有 F1−F0 好、F1−L1 不好，只能说“支持扇形输入下的微调收益，尚未证明 F1 优于 L1”。是否存在两种输入共有的微调收益，还要看 L1−L0；是否有扇形特有收益，还要看交互差值。如果只有 F0−L0 好、F1−F0 不好，应优先保留简单适配，不宣称改了表示才有效。

“一致改善”仍需对应半区三个种子的配对 95% 区间都在零以上；未达到时报告点估计和不确定性，不断言零作用。两半区已多轮参与开发，本轮不能变成新的独立确认试验。

**最终开发目标不变：** 同一个预先指定系统 F1+B3 的三种子平均 segment F1@50 在 selection≥0.944409687184662、calibration≥0.9821421709894332；每个种子在每个半区 boundary recall≥.95，native macro-F1≥.973/.980。另报 calibration 参考改为 .9738935784747763 时的 +.01 敏感性。只在一个半区好不算完成，诊断回退行不算完成。

- [ ] 没有收益或识别/分段互换错误：本轮停止，保留必要输入接口，不能自动再扫 lr、扩大 ViT、加时序模块，不能说所有裁剪方法无效。
- [ ] 有部分改善但目标未过：明确“视觉/输入适配的局部正结果，目标未达”；指出改善来自多少独立录像。是否值得用匹配特征重训 ASRF 留给下一次决策，不能马上追加。
- [ ] F1+B3 达到历史开发目标：仍需下一份独立计划为相同输入重训 ASRF 等强对照，并解决独立确认数据问题，才能把它写成研究优势。当前不解除 test 的访问限制。
- [ ] README 用英文，开头直接回答做了什么、改善来自哪里、哪些指标仍未达标。训练曲线和全部失败行保留。私有逐录像表及图片留服务器；仓库只放汇总、代码和协议。冻结 12 组权重及清单，独立检查表格、保存本轮限定范围提交，不推送、不触碰理论稿和旧论文。

## 当前状态

已完成的只有训练图像预处理试做、源代码检查及本任务书。团队版本尚未交付，prototype 尚未完成 Task 1 的扩展 QC。没有训练本轮模型，没有新的分类或分段性能结果。
