# Frozen Trained Time-Token Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Claude is the intended executor. Within the fixed scope, complete source checks, implementation review, calibration, one selection evaluation, independent verification, documentation and round-only commits without waiting for the owner. No training is authorised in this task.

**Goal:** 检验A02已经学会切面标签的时间token，直接命名目标时刻时能否改善完整时间分段，而不是只将其作为训练辅助项。

**Architecture:** 保留A02九个冻结模型和完全相同的Sparse输入窗口。每窗口按输入位置确定一个训练锚点，读取已保存的`time_logits[:,j,:]`；与同一模型原CLS输出、另两组相同token读出及冻结Ordered强参照比较。无新参数、训练、RGB变化、请求密度变化或输出融合。

**Tech Stack:** A02的18份NPZ预测、metadata-only窗口重建、NumPy、既有B3/配对bootstrap、pytest、英文README。

---

## 负责人决定与问题

A02按no gain / end结束；接受`13f286c`的索引更新。冻结Ordered+简单分段仍是强参照；A02 aligned+CLS另保留为满足全部native/recall约束的结果，但未满足完整F1目标、未证明超过强参照。不能因calibration只差0.0004而改标准。

报告口径要分开：A02 selection的aligned−CLS continuation **segment F1**为+0.011/−0.003/+0.034；三个点估计都负的是**mixed窗口CLS准确率**。不要把两者写成一个结论。

**前序记录：**
- V02替换patch读出并用目标CE训练，没有完整系统收益；不能重开其读出/预算搜索。
- A02新增了8个位置的标签监督，时间logits在全部mixed锚点的正确率约0.85–0.88，两个对照约0.65–0.72；正式推理只用CLS。锚点总体与CLS目标总体不同，不能直接把这两个正确率相减。
- V03加密窗口请求只改善位置偏移，没有一致系统收益。本轮保留Sparse请求，改变的是读哪个已保存的预测。
- A01在冻结512维全窗口表示后增加时序头，没有超过强参照。本轮不再训练时序头。

**有限问题：** 在同样的请求时刻和输入窗口上，直接使用被监督过的时间logits，是否比同一冻结模型的CLS更准，收益是否传到类别正确的分段？这不证明CLS无法传递信息，也不证明token只看其锚点帧，更不预先宣称新架构或论文创新。

## 固定来源与规则

- A02代码/配置、九个最终冻结权重和18份预测全部只读。源配置`configs/experiments/video_dense_supervision_v1.yaml` sha `49f8782ad0d7702f65244dd6b637f2fc20dc5dd0aab40c2de4389d69f0eb1087`；源结果`docs/results/2026-10-13_video_dense_supervision/summary.json` sha `bea3ae472bfb6c45960e11ddbeec843ef587b9b4cead274cf899f8a44cae3915`，README sha `a5738431a20e0481142771cfe6cc75b4222bc99d3993d740592a9238887b9701`。结果commit `05a5228`，索引commit `13f286c`。
- 保存预测在服务器`runs/video_dense_supervision_v1/evaluation/{calibration,selection}/{center_only,aligned_dense,shuffled_dense}_s{0,1,2}.npz`，含`identity/sequence_ids/seq/centre/marker/logits/time_logits/sample_offsets/sample_probs`。准备阶段在计算任何新指标前固定18个文件sha；必须与A02 summary.runs的训练identity、final权重sha和既有窗口metadata核对，不能只检查数组shape。
- 五类、源protocol、orig/gamma、10 Hz、16槽、stride5、`window_centres/window_frames`、marker和nearest-centre映射与A02相同。不新增Dense请求；不改预处理，模型输入仍224×224。
- 先从原protocol和已允许的两开发half元数据重建每窗口`s[16]`、请求`c`、marker槽m；与保存的seq/centre/marker和完整来源digest逐项对应。不调用会解码整库原始帧的A02完整evaluate入口，也不原样调用`report_center_supervision.validation()`/`Inputs`：它们会eager加载训练标签/特征及两开发half。新增允许名单过滤的metadata adapter，calibrate阶段只评价calibration；不载入训练特征或非允许split标签。新选择函数只接收`s,c,m`。
- 固定训练锚点`a=[0,2,4,6,8,10,12,14]`。按以下排序选**唯一**j：`(|s[a[j]]-c|, |a[j]-m|, j)`。优先最接近请求样本的锚点，再接近原marker槽，最后取较小j。精确匹配、重复槽和奇数marker都用同一规则。
- 不用标签、置信度、attention、敏感度峰值、native/mixed标记或任何模型分数选j。不将锚点改为2j+1；A02已经按2j的标签训练，较高输入敏感度不是重标的依据。
- 每窗口取`time_logits[i,j[i],:]`，五类softmax与原helper相同；概率再沿用原nearest-centre映射到10 Hz。保留预测所属的请求时刻c，不把centre平移到`s[a[j]]`。报告二者的sample偏移及标签是否不同；偏移非0的窗口也必须留在主评估。

```python
# 纯位置函数；模型分数和参考类别均不进入选择。
ANCHORS = (0, 2, 4, 6, 8, 10, 12, 14)
def selected_anchor(s, c, marker):
    if len(s) != 16 or not 0 <= marker < 16 or s[marker] != c:
        raise ValueError("window/marker mismatch")
    return min(range(8), key=lambda j: (abs(int(s[ANCHORS[j]])-int(c)), abs(ANCHORS[j]-int(marker)), j))
selected_logits = time_logits[np.arange(len(time_logits)), chosen_j]
```

不插值、不平均多个token/多个窗口，不融合CLS，不加温度/阈值校准器。不尝试其他锚点规则或挑最好的arm。输出来自全局视频交互后的token，不称为未经上下文影响的单帧识别。

## 系统、比较和固定成功标准

| 系统 | 本轮做什么 | 地位 |
|---|---|---|
| Aligned token readout | aligned_dense冻结模型，固定j读时间logits | 唯一主候选 |
| Aligned CLS | 同模型A02原输出和原B3设置 | 同权重的读出对照 |
| CLS-only token readout / Shuffled token readout | 对另两组冻结模型应用相同j规则 | 未对齐监督/打乱监督的读出对照 |
| CLS-only CLS / Shuffled CLS | 另两组原CLS输出、原B3设置 | 复现与交互对照 |
| Frozen Ordered + change-point | V01存档系统及原设置 | 当前强参照 |

主比较为Aligned token−Aligned CLS的multi segment F1@50，逐seed逐half。另比较Aligned token−Frozen Ordered，以及Aligned token−两个token对照。原CLS行先按保存设置精确复现；它们不重新选参。

对三个新token系统，各seed只在calibration一次使用原40项B3：h `[5,7,9,11,15]`，q `[.80,.85,.90,.95,.97,.99,.995,.999]`，NMS5；native macro-F1≥.980、multi recall≥.95时取segment F1@50最高，精确并列按原网格顺序。参数冻结并提交后selection只应用一次；不计算selection上其他设置。没有可行点即失败，9/.95诊断行不算通过。新token系统的选点与A02原CLS均使用同样的40项开发预算，但系统差异也含实际选中的解码设置，不能全部归给读出。

- 额外报直接argmax，以及token概率使用同seed同arm的A02 CLS **数值B3参数**的固定工作点敏感性；不以敏感性挑系统。
- 系统paired bootstrap沿用2000 draws/seed20261002：native按session，构造流按stream；每half三个CI严格同向排除0才是consistent。三个seed不是三个新病例；开发数据反复使用，区间近似，不是独立验证。
- 同窗口同请求的分类比较：各token对其同权重CLS，分native、all-mixed、8/16、9–11/16、A4C/A5C对，以及精确锚点/非精确锚点。all-mixed按目标session主bootstrap，按stream作敏感性；若敏感性有consistent相反方向，该half的分类读法为不确定。只有all-mixed参与分类规则，其余预定子集只描述，不从中选结论。
- 原目标不变：三seed均值selection≥0.944409687184662、calibration≥0.9821421709894332；敏感性calibration≥0.9838935784747763；每seed每half recall≥.95、native≥.973/.980且B3可行。不舍入。
- **Complete model progress：** Aligned token对同权重CLS、冻结Ordered以及两个token对照的主F1，在两个half都consistent positive，且达到原目标及全部约束，才说本配方超过这些系统。只胜无监督/打乱对照不算进步。
- **Useful token readout：** Aligned token−Aligned CLS的all-mixed分类准确率在两个halfconsistent positive，且主F1也在两个halfconsistent positive，并满足全部约束，才说直接token读出有可用收益。旧数值目标/强参照是否通过另报，不能混为一个判定。
- **Labels learned, no usable gain：** 主候选既不满足Complete model progress，也不满足Useful token readout，就按事实写局部/不确定/无收益，本配置结束。CI含0不是相等，也不能由此说所有token方法无效。分类提高但分段无收益、原生分类/召回损失、或只一half成立均单独说明。
- 若只有分类/片段读出信号，不追加训练或换损失；若三个系统都差，不换锚点、不融合CLS、不扫概率规则。下一步须另有实质问题与新任务。

## Task 1: 固定配置与来源准备

**Create:** `configs/experiments/trained_token_readout_v1.yaml`、`scripts/trained_token_readout.py`（分阶段CLI）、`tests/test_trained_token_readout.py`。
**Reuse read-only:** A02 NPZ/配置/结果/冻结权重与既有metadata/B3评估helpers。

- [ ] 先核对索引A03 pending，再提交任务书和完整配置。固定18份预测sha、九个final身份、源protocol、窗口metadata与依赖版本；不计算新的token分类或分段指标。
- [ ] 写失败测试：位置选择不读标签；偶数、奇数、端部clamp、全部槽重复、多个精确锚点、并列选择、marker不符；人工`(N,8,5)`选logit；错identity/digest/class order/centre/seq拒绝；禁止读取test/Pediatric和非允许split标签。
- [ ] Run `.venv/bin/python -m pytest tests/test_trained_token_readout.py -q`，确认新模块缺失时失败，再实现metadata-only准备及锚点选择。源预测缺失/错身份立即停止，不默默重推整库或改读出。
- [ ] `--prepare`写新源清单/代码身份；`--check`精确复现18份CLS窗口/逐样本prob和A02原CLS的argmax、B3全部系统行。只用已有元数据和预测，不加载图像。
- [ ] 独立实现审查检查来源身份、标签隔离、重复槽与时刻映射、原CLS复现、B3选点只用calibration、失败状态/预算记录及数据输出范围。通过后提交检查记录，直接继续，不等负责人回来。

## Task 2: 校准冻结与一次selection评估

**Create:** `scripts/trained_token_readout_tables.py`、`tests/test_trained_token_readout_report.py`。

- [ ] 在新token指标产生前写报告/决策测试并提交。覆盖原CLS与token参数预算、无可行设置、只胜坏对照、分类提高而F1不升、单half、native/recall失败、敏感性冲突，以及少数正确例不夸大结论。
- [ ] `--calibrate`只处理calibration的新token系统与源CLS对照，保存每个设置的结果和唯一choice（或失败），参数数值和身份先提交；不读取/计算selection的新token指标。
- [ ] `--selection`验证参数冻结身份后只执行已选设置一次，保存逐窗口/逐样本/逐序列输出在服务器。重复运行只能复现同一身份，不能重选参数或覆盖不匹配旧结果。
- [ ] 汇总全部native五类recall/F1、A4C→A5C与反向修好/新错、label switches、multi segment F1、boundary P/R/F1、≤16样本段、X-Y-X中段、同类拼接误切。病例按录像/session去重，不把锚点/多次流复用算独立病例。
- [ ] 在same-target分类上直接比较token与CLS，不用A02全部锚点0.85–0.88和CLS target准确率跨总体比较。报告锚点到请求时刻的偏移分布、非零偏移窗口数、标签不同数及其分类/分段影响，所有窗口留在主结果。
- [ ] 复现并保留A02各CLS与Frozen Ordered参照；E0/E1/B7+B3、A01时序头等旧系统列原存档行，不重训、不将旧模型接新表示。缺失参照照实说明，不能用未完成阶段分数代替。
- [ ] 固定读法只执行一次；任何后加汇总标为post hoc，不改变候选、参数、样本和主结论。

## Task 3: 独立核查与关闭记录

**Create:** 英文`docs/results/2026-10-14_trained_token_readout/README.md`、summary、完整表、位置映射示意图（不用原始帧）。

- [ ] 独立核查用保存的原time logits和metadata重新实现位置排序/概率、重算全部系统、CI及判定，核对源文件/权重sha与calibration-before-selection顺序。源A02已GPU复推核查，本轮主要验证新增索引/解码，不声称又训练或又验证泛化。
- [ ] 仅在需要验证缓存时，可对每seed按原窗口顺序选8个（native4/multi4，不按错误选）的固定窗口作GPU复推，最多24个窗口；原帧/曲线只留服务器。不做新的全量模型推理。不因复推发现的实际实现错误提高阈值/改科学规则。
- [ ] **资源限额：GPU总0.25小时，CPU阶段累计墙钟2小时。** 通常只需CPU；GPU spot-check/失败尝试也计入。CPU、GPU分开计时，不能把源A02已花的推理时间当本轮新成本。超限保存进度并报告，不删除失败或减少候选。
- [ ] 写明收益来自“已训练模型的读出检查”，不是新训练、新架构或真实连续扫查验证。token总体能学到标签不等于同一目标的完整系统更好。
- [ ] 只提交本轮文件与索引更新、不push。A02/V02/V03/A01原结果、配置、权重及停止记录全部保留；theory8和无关dirty文件不动；不读test/Pediatric，不改公开标签。更新A03实际结果、源身份、提交和结束范围。

## 权限与结束

用户已授权在固定任务内直接执行，不需等待其回来；实现错误可修复并复验，不能借此改科学设计。准备、代码审查、校准、selection与核查各报真实状态。本文没有授权任何新训练、修改模型、改目标、继续旧解码搜索或开启下一轮自动实验。

## 计划审查记录

2026-10-14（America/New_York），独立只读计划审查为Approved，无阻断项。审查指出旧`validation()/Inputs`会eager载入本轮不需要的标签/特征，已明确要求允许名单过滤的metadata adapter。审查没有计算新指标或执行训练。计划审查通过不等于实现或读出比较已经完成。
