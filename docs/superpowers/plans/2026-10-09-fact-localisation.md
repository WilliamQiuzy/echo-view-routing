# FACT Short-Segment Localisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Claude is the executor; this document authorises the bounded diagnostic below, not another training round.

**Goal:** 用冻结 FACT 权重和训练序列，定位短参考段在何处丢失，并检验最后一个 TDU 块的合段位置是否限制恢复。

**Architecture:** 普通推理逐块记录 E1、FACT 帧分支、token 与最终输出。在同一冻结网络的最后一个 `U` 块，比较原分组、参考边界细分和等数量均匀细分。后两种是使用参考信息的诊断干预，不能作为模型成绩。

**Tech Stack:** PyTorch、官方 FACT 固定提交 `7bd81bda2b84618a3e23876a2637a82f24881122`、现有 E1 缓存、pytest、英文 Markdown/JSON 报告。

---

## 负责人决定与依据

上一轮按原判定表结束。FACT 在 selection 上相对 ASRF 三个点估计均为正，但两个半区未达到一致提升，最终目标也未达到。保留这些事实，不把结果改写成“FACT 无效”或“已经超过强基线”。

下一步只做本任务的一次事后诊断。不开 Breakfast，不续训，不扫学习率或损失，不改 token 数，不改 E1 或 224×224 输入，不读取开发半区、test 或 Pediatric。

已经核对的依据：

- `scripts/train_fact_asrf.py::monitor` 经过 `head.decode` 用 eval 模式。因此 16 条训练监测序列的 F1=0.90–0.96 不是在开启 dropout 的模式下测出来的；它也不足以证明整个训练库欠拟合。
- 官方 `models/blocks.py::UpdateBlockTDU.temporal_downsample` 用上一块的逐帧 argmax 合成区间，平均帧特征，再用 GRU 和 cross-attention 更新。
- 最后一个 U 块的 a2f 注意力按这些区间展开回帧；帧分支仍有保留原帧特征的路径。因此“先合段可能影响短段恢复”是可检验假设，不能说“合掉后绝不可能恢复”。
- 报告中没有 IoU≥0.5 预测段的漏检，既可能是漏段，也可能是过长、过短、碎片化或偏移，不能直接等同于“类别判对但被合并”。

代码依据：[官方 TDU 实现](https://github.com/ZijiaLewisLu/CVPR2024-FACT/blob/7bd81bda2b84618a3e23876a2637a82f24881122/models/blocks.py)、[合段与注意力展开](https://github.com/ZijiaLewisLu/CVPR2024-FACT/blob/7bd81bda2b84618a3e23876a2637a82f24881122/models/basic.py)。

## 范围、预算与文件

总预算上限 **1 GPU 小时**，CPU 分析和开发耗时另记。只推理，不创建优化器，不调用 backward。先在固定 16 条训练序列上验证，再跑规定集合。检查通过后直接运行，不再申请开跑确认；若投影或实际耗时超过上限，保存进度并报告，不挑样本或另开搜索。

**新增文件：**
- `configs/experiments/fact_localisation_v1.yaml`：固定输入、集合、种子、诊断定义和预算。
- `scripts/diagnose_fact_localisation.py`：三个步骤 `--prepare`、`--check`、`--run`；使用同一配置和新目录，不覆盖旧产物。
- `tests/test_fact_localisation.py`：分区、干预隔离和逐段统计测试。
- `docs/results/2026-10-09_fact_localisation/README.md`、`summary.json`、`report_tables.md`、`preparation.json`：英文，明确标注 post hoc / training-set diagnostic。

**复用但不改语义：** `scripts/train_fact_asrf.py::monitor_set/load_plan/make_head`、`echo_routing/temporal/fact_head.py`、现有匹配与来源去重工具。不要直接实例化原 `Ctx`：它调用的 `TrainOnly` 会先抽取整张索引的标签。本轮在新脚本建立兼容上述函数字段的诊断上下文，先用训练允许名单筛行，再建立 labels 和训练库。官方代码、正式训练配置、旧结果与判定均不改。

## Task 1: 固定数据和记录单位

- [ ] 配置引用上一轮三个 FACT 最终权重及 SHA，配对 ASRF、E1 缓存、训练 protocol、训练清单和代码提交。源配置哈希不匹配即停止。
- [ ] 对每个种子使用该种子 **最后一个 epoch（索引 49）的全部 375 个实际批次**：1,500 条既有构造流 + 1,500 个实际抽到的原生窗口。逐批与上一轮保存的摘要核对。不根据预测、短段错误或训练分数筛选序列。
- [ ] 另复现原来固定 16 条监测序列的最终逐序列/总体 F1 和 transcript 计数。它们只核对监测口径，不替代整组训练序列结果。
- [ ] 构造流和原生窗口分开报告；原生窗口不是整段原生录像。不同种子的原生窗口可能不同；同一种子内 FACT、ASRF、E1 用完全相同的窗口。报告录像/session 去重数量，三个种子不当成新病例。
- [ ] 输入和预测逐序列记录只留服务器的 `runs/fact_localisation_v1/`。仓库只收汇总与摘要。读取索引元数据后，按 `protocol['train']['clips']` 筛选 `video_id`，保持原行序；仅从筛选后的行抽取 `label_index` 构造标签字典和训练库，不加载其他 split 的特征。报告如实说明索引本身含有其他 split 的行，不声称从未接触这些元数据。检查训练录像无缺失、重复或越界，且生成的流与批次摘要完全复现。
- [ ] 先提交本轮配置、脚本、测试和任务书，再运行任何诊断或报表。不碰其他暂存文件，不推送。

## Task 2: 同一前向的逐块定位

- [ ] 冻结三个 FACT、配对 ASRF 和 E1；全程 no_grad。FACT 用上一轮完全确定性设置，ASRF 用它原有设置。输入 tensor 和权重在诊断前后不得改变。
- [ ] 对普通 FACT 前向记录：E1 缓存概率 argmax；`i`、`u`、`U` 各块帧分支 argmax；最终官方输出；各块 token 类别/null；进入 U 的分区；最终 a2f 分配。i 块没有与后续块相同的 a2f 输出，不伪造该指标。
- [ ] 同一序列逐阶段报告 frame error、class-aware segment F1@50、匹配参考段数。说明全库汇总方式，并给逐序列分布；不要把 16 条监测序列的平均 F1 与不同聚合口径直接比较。ASRF 和 E1 仅作为同输入参照，不能把训练内分数叫泛化结果。
- [ ] 覆盖所有参考段；重点分组为 ≤16 样本的段、全部 X→Y→X 中段、A4C/A5C X→Y→X 中段。报告数量和来源去重，不只选“前级对、后级错”的病例。
- [ ] 对每个参考段在每个阶段记录：是否一对一匹配、忽略类别的最大 IoU、同类最大 IoU、该区间内部 argmax 类别计数、两端是否在原 0.5 秒容差内找到边界。交叉报告“前级匹配→后级失配”和相反方向。
- [ ] 若无匹配，保留以上指标，不把它们强行压成单一因果标签。特别区分 IoU 不够、标签错误与一对一竞争；同一段可能同时存在多种问题。
- [ ] 数一数进入 U 的每个预测分区覆盖多少个参考段，以及不同参考类别的占比。这里只说明分组混合，不证明这是失败的原因。

## Task 3: 最后一个 U 块的冻结权重干预

全部使用相同冻结权重、相同 E1 特征、相同进入 U 的帧/token 特征；只替换 `temporal_downsample` 生成的分区。后续均调用原有平均、GRU、类别投影、cross-attention、上采样、帧分支和官方解码。不调用训练损失或按参考类别挑 token。

三个条件：

- **A：原实现。** 完全不改分区。
- **B：参考边界细分。** 保留全部 A 的分界；只把落在 A 区间内部的参考类别变化点加入分界。不删除已有分界，不按标签改输出，不替换 token 类别。原生窗口只有一个参考标签，所以 B 应与 A 完全相同。
- **C：等数量均匀细分。** 对每个 A 区间，新增与 B 完全相同数量的分界，但位置在这个 A 区间内均匀放置。若区间为半开 `[s,e)`、长度 L、B 新增 k 个分界，则 C 新增 `s + floor(j*L/(k+1))`，j=1…k。位置必须唯一且严格位于区间内部。C 同样是诊断：新增数量取自参考边界计数，不是可部署对照。

- [ ] 全部构造流都运行 A/B/C，含同切面拼接和无需新增边界的序列。不要只运行 B 可能修好的错误子集。原生窗口验证 B=C=A；可复用 A 的输出避免重复推理，但必须在单元测试和固定 16 条检查中真实执行相等性核对。这种原生输出不变由诊断构造保证，不是未来可部署模型能够保住原生识别的证据。
- [ ] 在项目自己的诊断适配层临时替换分区构造；不要编辑官方目录或正式 adapter。对分区对象使用位置和长度，不能从参考类别填充网络输入。项目诊断使用半开 `[s,e)`，但官方 `utils.Segment` 是闭区间且 `.len=end-start+1`，构造时明确转换为 `Segment(action=原 u 分支在 s 的预测类别, start=s, end=e-1)`；action 不取参考标签。人工构造相邻同类但不同分区时，不得被 `parse_label` 再次合并。
- [ ] A 的 U 输入和输出须与普通官方 forward 逐位相同。B/C 进入 U 前的全部张量也须与 A 相同；禁止通过训练标签改变前两块。
- [ ] 在同一序列按 A→B→C→A 执行，最后 A 必须复现第一次 A，排除 hook 或状态污染。不加新切点时，B/C 必须逐位复现 A。
- [ ] 逐种子报告 B−A、C−A、B−C 的全序列 segment F1@50、边界 recall/precision、修好/新错参考段。对参考标签定义的短段和 A4C/A5C 子集，只报告匹配数/总数、recall、修好/新错参考段，不定义没有明确假阳性归属的子集 F1 或 precision。按参考段、来源录像和 session 分别计数。
- [ ] 结果明确说明：B−C 检查的是“在相同局部新增分区数量下，参考边界位置的价值”。它不能证明能从图像预测这些边界。A/B/C 比较同时改变了 GRU 处理的段序列，不能把效果单独归因于平均池化或注意力。

## Task 4: 检查与运行

- [ ] 单元测试覆盖：训练允许名单筛行后才抽取标签和加载特征；半开区间首尾；参考切点已在原边界上时不重复；长度为 1；k=L−1；相邻同类的子区间不被合并；B/C 局部区间数相同；单标签序列无改动；交错运行不污染模型状态；来源去重。
- [ ] 复现原 16 条监测结果和 A 的普通推理。差异出现先排实现/确定性/输入，不能据差异宣布新机制。
- [ ] 独立代码审查检查是否只有分区变化，标签是否进入类别输入，以及 report 的分母和修好/新错方向。检查通过后提交记录并直接运行。
- [ ] 先计时固定 16 条，预测预算；全过程保存进度、运行身份和逐序列产物，避免上一轮只有权重而没有存预测的问题。

计划中的入口（实现后运行，不代表当前已有）：

```bash
.venv/bin/python -m pytest tests/test_fact_localisation.py -q
.venv/bin/python scripts/diagnose_fact_localisation.py --config configs/experiments/fact_localisation_v1.yaml --prepare
.venv/bin/python scripts/diagnose_fact_localisation.py --config configs/experiments/fact_localisation_v1.yaml --check
.venv/bin/python scripts/diagnose_fact_localisation.py --config configs/experiments/fact_localisation_v1.yaml --run
```

## Task 5: 解释与停止

本轮只交付下列三个答案。不建立新的训练成功门槛，不因病例数或区间含零自动判定机制不存在。

1. **16 条是否代表全库？** 报告 1,500 条构造流和实际原生窗口的拟合，指出主要错误类别、长度和独立来源。不能从训练损失平台或小样本 fit 推断“只差训练更久”。本轮不拆损失、不做优化器搜索。
2. **短段最早在哪个阶段失配，又有多少被后续修回？** 如果进入 U 前就没有可识别短段，不能说 U 是唯一原因；如果 U 前匹配、之后失配，也先称定位现象。
3. **参考位置是否带来不同于一般细分的恢复？** B 相对 A 和 C 的差值、修好与新错都报告。若 B/C 相近，只支持检查分辨率或一般细分，不能宣称边界位置机制。若 B/C 都没改善，这个干预没有支持；不能据此排除重新训练后可能改变的行为。

即使 B 明显优于 A/C，也只说明冻结网络在给定参考分区时可恢复一些训练段。不能称为可实现的提升、全模型上界或达到 0.944/0.982 的证据。只有拿到这个结果后，负责人才能决定是否提出不读取参考边界的训练方案，并配固定细分等简单对照。

- [ ] 英文 README 首句注明本轮没有训练、没有开发集结果、B/C 使用了参考分区信息。三种子方向不一致时直接写明，不选最好种子。
- [ ] 独立复算计数、普通推理复现、干预范围和来源去重。保留上一轮结论、配置和历史指标。
- [ ] 提交本轮文件和复算命令，不推送；原数据、权重、暂存理论稿和许可证草稿不动。到此结束，不自动追加训练。

项目仍是心超切面识别与时间分段。论文继续暂停；最终目标、部署约束、封存 test 和 Pediatric 限制保持原样。
