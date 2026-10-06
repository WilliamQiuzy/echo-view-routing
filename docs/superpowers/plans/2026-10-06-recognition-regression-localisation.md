# Recognition Regression Localisation Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this bounded diagnostic task. Steps use checkbox syntax. This document authorises inference with existing models, not another training round.

**Goal:** Locate the native-recognition regression in the completed N/S/C experiment before selecting another model change.

**Architecture:** Read the frozen E1 predictor and the nine completed models. Compare dense frame predictions, the same predictions sampled on the existing window-centre grid, and the trained temporal-window predictions. Keep the images, timing, labels and aggregation identical within each contrast.

**Tech Stack:** Existing PyTorch inference, NumPy, project feature stores and evaluation helpers, YAML protocol, English Markdown report.

---

## 依据、问题与边界

本任务面向 Claude 实施。结果 README 用英文，面向项目组。

本轮训练已关闭，结果以 `2e6b1f2` 的 `docs/results/2026-10-05_center_supervision_training/README.md` 为准。C 在混合窗口上的目标时刻错误较少，但完整系统没有达标。N 只用原生窗口训练，也出现原生分类下降，所以现有结果不能把下降单独归因于中心监督或混合窗口。

代码中的两个分类路径不同：

```text
原 E1：图像 -> E1 embedding -> 已训练的 probe -> 五类 logits
新模型：图像 -> 微调后的 embedding -> 新时间 Transformer 的 CLS -> LayerNorm -> window head
                                      \-> 辅助 frame head -> 五类 logits
```

`TargetTimeHead.init_from_probe()` 只复制线性层权重。新的 window head 接收不同的表示，所以这不保证完整模型在训练前复现 E1。当前报告没有评价训练后的辅助 frame head，尚不能确定退步发生在哪个可观测输出阶段。

**唯一问题：退步主要表现为中心网格采样损失、时间模块前的逐帧判定退步，还是时间窗口读出额外改错？也允许几种同时存在。**

本任务不训练，不调阈值，不修公开标签，不剔除困难录像，不读取 test，不使用 Pediatric。所有既有成功标准不变。诊断路径不追认为上一轮的新候选系统。没有新的 oracle 达标门槛，也不把本次定位等同于找到了有效修复方法。

## 文件与复用

- Create: `configs/experiments/center_recognition_diagnostic_v1.yaml`：固定输入摘要、路径定义、聚合与统计口径。
- Create: `scripts/diagnose_center_recognition.py`：身份检查、推理、配对误差归因、生成报告。
- Create: `tests/test_center_recognition_diagnostic.py`：中心索引、网格映射、配对计数及路径复现。
- Create: `docs/results/2026-10-06_center_recognition_diagnostic/README.md` 和 `report_tables.md`：英文汇总。
- Reuse: `scripts/report_center_supervision.py` 中的 `validation`, `extract`, `windows_of`, `verify`, `evaluate`。
- Reuse: `scripts/train_center_supervision.py::build_model`，`echo_routing/evaluate/window_target.py::nearest_centre`，`echo_routing/evaluate/dev_metrics.py::native_table`。
- 旧配置、训练代码、冻结权重、原结果与用户其他文件不修改。逐录像编号、帧级输出与特征只存服务器的新结果目录，不提交仓库。

### Task 1: 固定五条输出路径

- [ ] 读取原配置与 report，记录协议、九个最终 checkpoint、E1 probe/cache、代码版本的摘要。保留原模型身份，不生成新模型版本。
- [ ] 在任何新指标生成前提交诊断配置和代码。下面五条路径全部报告，不看结果后选择路径：

| 路径 | 概率来源 | 时间采样与映射 |
|---|---|---|
| F-dense | 原始冻结 E1 的逐帧五类概率 | 原有完整 10 Hz 样本序列 |
| F-grid | 同一份 F-dense 概率 | 只取旧窗口的实际目标样本 c，再用旧 nearest-centre 规则回填 |
| J-dense | 对应 N/S/C 训练后 encoder + 同一 checkpoint 的辅助 frame head | 与 F-dense 完全相同的全部样本 |
| J-grid | 同一份 J-dense 概率 | 与 F-grid 相同的 c 和回填规则 |
| T-window | 同一 checkpoint 的时间 Transformer + window head | 原有 16 帧窗口、相对位置、stride=5 与回填规则 |

F 两条只需生成一次。J/T 对 N、S、C × 三种子全部生成；C 为主要定位对象，N/S 为现有训练对照。辅助头不重新训练，不换 probe，不调整温度或归一化。**不把旧 probe 接到新 encoder 的交叉组合用于证明表示丢失**，以免把头与表示的不匹配误称为信息损失。

- [ ] 精确用序列的 `refs[c]` 读取目标帧；不能假定目标是输入窗口的第 8 帧。边缘窗口和重复采样的短录像也使用相同规则。
- [ ] 原始 F 概率优先直接读身份匹配的冻结 E1 cache，以复现原评估；记录 dtype。J/T 共用同次提取的 fp32 特征。既有输入 parity 要求保持不变，不能以旧 E1 特征代替训练后特征。

实现时按以下关系构造，`centres`、`nearest_centre` 和 `window_frames` 全部复用旧函数：

```python
# Each probability array below has shape (sequence_samples, 5).
f_dense = frozen_e1_prob_for_exact_sequence_refs
j_dense = softmax(model.head.frame_head(trained_features_for_sequence_refs))
centres = window_centres(len(f_dense), 5)
assignment = nearest_centre(len(f_dense), centres)
f_grid = f_dense[centres][assignment]
j_grid = j_dense[centres][assignment]
t_window = windows_of(model, all_features, ref_to_index, sequence, 5, device)["prob"]
```

E1+B3 的固定解码参数取自 `runs/20261002-144641-md_simple_decoders-38bbcf7c-s0/selected.json` 的 `selected.b3.main`，与旧报告 `references()` 的读取一致。保存该文件摘要和完整参数，不从本轮 C/S 各自选出的配置中挑一个。

### Task 2: 只做必要的一致性检查

- [ ] 用合成数组测试：F-grid/J-grid 确实只读 c；回填长度、最近中心及并列时的规则与旧脚本一致；小于 16 样本与窗口靠边时不偏移。
- [ ] 测试原生预测的聚合仍为逐样本 argmax 后多数投票，平票规则不变；不改成平均概率或只在窗口中心投票。
- [ ] 测试修好/新错方向和来源去重。一个来源录像在多条流重复出现时，单位数与来源数必须分开。
- [ ] 九个 T-window 的逐序列预测/汇总与上一轮对应 argmax 行核对。有旧逐序列存档就逐序列核对；没有则明确核对范围，不假称已有逐序列存档。不一致先停下定位，不能重写旧结果。
- [ ] 用原存档 E1+B3 的**实际数值参数**解码 F-dense，复现既有 E1+B3 的指标。检查结果和容差写入配置；浮点差异必须说明，不得静默接受导致标签改变的差异。

Run on server:

```bash
.venv/bin/python -m pytest tests/test_center_recognition_diagnostic.py tests/test_center_supervision.py -q
```

Expected: pass indexing, aggregation, pairing and identity checks. 通过测试不等于研究问题已有答案。

### Task 3: 一次性诊断推理和报告

- [ ] 复用已有特征（若有完整且身份匹配的存档）；否则每个冻结 encoder 仅重新提取一次，再同时读辅助头和时间头。单 GPU 顺序运行。无需生成训练库新预测或重跑训练。
- [ ] **主比较不用 B3：** 对五条路径的 argmax 输出报告两半区 native macro-F1、五类召回及分母、错录像数、内部切分，以及按来源录像等权的目标样本错误率。对相同窗口中心，另报 single-recording / mixed-view 两类的错误率。
- [ ] 每个种子报告以下三个配对差值及修好/新错表：
  1. F-grid − F-dense，以及 J-grid − J-dense：中心网格采样/回填造成多少变化。
  2. J-grid − F-grid（并列附 J-dense − F-dense）：训练后的逐帧预测路径变了多少。这里 encoder 和 frame head 都变了，不能单独归因于 ViT。
  3. T-window − J-grid：同一个 encoder 下，时间读出相对辅助逐帧读出额外修好或改错多少。同样不能单独归因于 attention，window head 和 LayerNorm 也参与变化。
- [ ] 对 C 的原生录像按 F-grid / J-grid / T-window 的三个对错布尔值列出互斥且穷尽的八格表，校验八格总数等于录像总数。由此导出 F-grid 正确而 J-grid 错、J-grid 正确而 T-window 错、两级均错、以及相反方向的修好，避免重复归因。与 F-dense 的采样差异单列，不能遗漏。
- [ ] 原生置信区间按 recording session 配对 bootstrap，沿用 2,000 次和旧随机种子 20261002。给出原始分子/分母，不把 selection 的 11 段 A5C 当成大样本证据。窗口反复出现时按来源录像等权并报告 session 数。
- [ ] **补充系统表**只使用一套原存档 E1+B3 的完整实际参数（含数值 JS 阈值），对五条路径统一解码，不重新做任何分位数映射或选参。另保留原 T-window 的旧调参结果作为历史行。统一参数表是诊断，不是公平最优性能排名。
- [ ] 补充表报告 segment F1@50、边界召回、native macro-F1、内部切分。构造流置信区间仍标为近似与探索性；不把重复素材算成独立病例。

Proposed new CLI contract:

```bash
.venv/bin/python scripts/diagnose_center_recognition.py \
  --config configs/experiments/center_recognition_diagnostic_v1.yaml \
  --out docs/results/2026-10-06_center_recognition_diagnostic
```

### Task 4: 给出定位结论，停止

- [ ] README 开头用三句话回答：网格采样影响多大；时间模块前是否已经退步；时间读出额外修好和改错多少独立录像。两半区、全部种子都报，允许结论混合或不明确。
- [ ] 根据结果写下一步的**建议**，不自动训练：
  - 若 J 已经退步，且 T 相对 J 没有相应的额外损失：优先考虑同一个 C 目标和时间头、只冻结 E1 的配对训练，以隔离视觉联合微调的作用。这只是候选，不称为已证明的遗忘。
  - 若 J 保留的正确判断主要在 T 丢失：优先考虑让目标帧的已有 logits 直接进入最终输出、时间模块只学修正量；必须配当前 CLS 读出的控制，不能只展示新模型。残差连接不是新发明，也不保证训练后不会退步。
  - 若主要由网格造成：先针对同一权重比较更密集的推理，另立固定协议；不增加训练层。
  - 若多处同时退步或证据不清：明确各部分计数，不据此同时堆叠几种修复。
- [ ] 不用“理想修复达不到最终 +1”来排除一个有限机制，但也不能用一个阶段的改善声称整体模型达标。下一轮是否训练，要根据这份定位表明确决定。
- [ ] 只提交本轮配置、脚本、必要测试和英文汇总，不推送。用户其他文件保持原样。给出实际推理耗时，结束本任务；论文继续暂停。

## 交付状态

这是一份待执行的任务书。它没有产生新的预测、诊断数字或训练结果。

2026-10-06，独立计划审查通过，未发现实质性实施缺口。按审查建议补充了互斥、穷尽的八格配对计数。审查只覆盖任务书，不代表诊断已执行。
