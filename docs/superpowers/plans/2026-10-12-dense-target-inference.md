# Dense Target-Time Inference and Remaining-Error Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Claude is the intended executor. This task authorises bounded inference and analysis only; it does not authorise another training round.

**Goal:** 在冻结的最佳视频编码器上，检验0.5秒一次的目标预测是否造成可避免的分段损失，并明确剩余错误发生在分类、采样、解码或区间匹配哪一步。

**Architecture:** 保留三个 raw_video_context_v1 Ordered 模型及原B3数值参数。比较旧稀疏预测、由同一稀疏概率作线性插值、每个10 Hz样本实际调用模型的密集预测。没有新权重、读出或优化器。

**Tech Stack:** 现有 TargetMViT、raw-video冻结输出、window_centres/window_frames/nearest_centre、项目分段和误差归因工具、PyTorch、pytest、英文 README。

---

## 负责人决定与证据

结束 target_patch_readout_v1 的读出与训练预算配置，保留正式 no gain 判定。它没有超过同预算CLS或冻结Ordered；不追加attention、gate、辅助loss或训练轮次。上一轮多帧识别与系统正结果保留。

当前冻结Ordered仍为下一步的固定来源，三种子均值selection .9543886531510294、calibration .9817001920287197；目标未达到，且seed2 selection native .9700744362565338低于.973。更改patch读出和额外3750更新未带来系统收益，不能继续以少数混合窗口改善作为加模块的理由。

另一个尚未单独检验的实现问题是：模型只在 `window_centres(n,5)` 请求预测，其余10 Hz样本用nearest centre赋值；匹配边界平均偏移约1.5样本。**这不是证明主要损失来自采样。** 本轮用冻结权重和冻结解码器直接检验，不因为距离门槛0.0004而重选阈值。

这是独立的新负责人任务，不改前轮“本轮无dense stride”的范围或任何旧结果。允许在 **4 GPU小时** 内完成实现、检查、推理、报告和核查；检查和完整推理预算预测通过后直接执行，无需再等负责人。超过上限保存状态并报告，不删种子、不调batch来改变数值、不自动加预算。

## 固定来源

- 五类、两个开发半区、协议、源帧/gamma、224×224几何、0–255归一化、16样本窗口、marker、fp32和环境，继承 `configs/experiments/raw_video_context_v1.yaml`。模型均eval，所有参数不更新；test/Pediatric不读。
- 只用三个冻结Ordered：`models_frozen/raw_video_context_v1/ordered_s{0,1,2}/final.pt`。sha分别为 `80f13a42e628fe2647634b79e59d5b8486fee5c3e0ae45f70bcc667c41258505`、`74276323ababf78fcb578be28d3172f3c220b3b7e66c2ff96a2072f39210c26f`、`923eb3afb0d846b1361897dbb63bb5aecf22401ce4b305b44fe990d48edc7f2c`。不选择某个最好seed，不用continued/patch权重。
- 源报告为 `docs/results/2026-10-10_raw_video_context/{README.md,summary.json}`；源逐窗口输出为 `runs/raw_video_context_v1/evaluation/{selection,calibration}/ordered_s{0,1,2}.npz`。准备时记录实际文件名和digest，并核对来源运行身份、窗口、marker、帧顺序和源权重。旧缓存只读，不写回旧round目录。
- 旧B3设置三种子都是half_window9、NMS5，阈值分别为 `.29269428317686846`、`.2845063773784935`、`.30845250187026374`。**直接继承这些数值**，不重算quantile、不搜索新设置。

## 三个推理条件

| 报告名 | 每个10 Hz样本的概率 | 模型调用 |
|---|---|---|
| Sparse | 保存的旧stride5窗口概率，按原nearest-centre赋值 | 复用旧窗口 |
| Interpolated | 对同一旧窗口概率按centre位置逐类线性插值 | 不新增调用 |
| Dense | 每个样本c都请求16帧窗口，输出该c的概率 | 新增旧窗口未覆盖的c |

- `window_centres(n,5)`通常从2开始，再7、12…，并按现有规则处理最后样本；**不能擅自改成0、5、10…**。Dense centres为 `np.arange(n)`，所有旧centres必须是它的子集。
- Sparse必须从旧保存的logits构造概率，精确复现旧逐样本预测及全部旧系统行。
- Interpolated对每条序列每个类别执行 `np.interp(np.arange(n), old_centres, old_prob[:,k])`，区间外保持首/末值，结果统一cast为float32，不再次softmax或重新校准。检查非负、行和在1±1e-6、旧centre的概率不变。仅一个centre时全程常数。插值可能产生新的argmax类别，不预设它安全或更好。
- Dense每个c都调用原 `window_frames(c,n,16,16)`、原target marker和Ordered图像顺序。对旧centres，**直接复用旧logits**；只推理新centres后按c拼齐，确保三条件共享旧观测。
- 新推理只改变目标请求频率，既不缩短窗口，也不使用类别、拼接位置或不确定度决定哪里加算。对于构造流，先按原recipe组装逐样本帧引用，再取16帧窗口；不得先独立编码各来源录像后拼接，否则输入条件改变。
- 每个模型两半区、全部native/pair/multi序列均纳入，不能只加算错误附近。不能在看指标后改为stride2/3/4、局部触发或其他插值。

## Task 1: 固定协议与纯函数检查

**Create:** `configs/experiments/dense_target_inference_v1.yaml`、`scripts/dense_target_inference.py`、`tests/test_dense_target_inference.py`。

- [ ] 先提交任务书、配置和新的实现入口；只包含本轮，不改旧训练/评估函数或输出。新root为 `runs/dense_target_inference_v1/`、结果为 `docs/results/2026-10-12_dense_target_inference/`。
- [ ] 先写测试再实现：n=1、短于16、整除与非整除5的序列；旧centres包含在dense内；Sparse使用原tie规则；Interpolated端点/旧centre/行和；Dense旧centre复用且每个c恰有一次输出。
- [ ] Run `.venv/bin/python -m pytest tests/test_dense_target_inference.py -q`，检查索引与插值，不写只镜像实现的测试。加入源文件身份不匹配即拒绝读取的测试。
- [ ] 运行身份覆盖三条件、全部源checkpoint和缓存digest、协议、源帧与窗口/marker实现、预处理、数值B3设置、环境、推理与报表代码。旧输入只读，逐序列/编号/媒体留服务器。
- [ ] 模型forward只接RGB/marker。预测序列的labels、reference边界、is_native不进入模型或加算决策。eval读取开发标签只用于报告，test/Pediatric不加载。

## Task 2: 启动检查与密集推理

- [ ] 仅在16个按序列顺序选择的旧窗口做source parity，不读取目标预测正确与否选例。重新推理原窗口，logit最大差≤1e-4且argmax相同；旧缓存身份/像素必须正确。Sparse全部行与存档精确复现。若差异超限先查实现，不变模型或升容差。
- [ ] 检查每条序列的Dense目标帧与原10 Hz引用一致，marker仅由s和c产生；核对短序列/端部重复和拼接处gamma。共享窗口与新增窗口使用相同预处理。
- [ ] 原batch32、只读推理：按固定顺序预热1batch再计时10batch新增窗口，估算三seed/两半区的完整新增调用、解码和报告成本。预算包含这些调用和2小时评估/核查预留，超过4小时则停止并报告，不能只处理子集。
- [ ] 独立实现审查通过后直接完成三seed/两半区推理。模型hash前后相同；没有optimizer或训练步骤。沿用GPU显存限制、单任务执行与其他项目保护。
- [ ] 保存每个c的logits、来源与每种映射后的概率，来源只在服务器。允许身份完全相同的缓存恢复，不允许通过挑batch或重跑选更好预测。

## Task 3: 冻结解码比较与误差归因

**Create:** `scripts/dense_target_inference_tables.py`、`tests/test_dense_target_inference_report.py`、英文README、完整表/summary与计算成本。

- [ ] 先全部推理完，再统一算结果。三条件各报直接argmax与**同一旧数值B3**；没有新的calibration选择步骤。旧Sparse source行精确复现才能报告Dense差异。
- [ ] 主比较：Dense − Sparse 的multi segment F1@50（旧B3）；另报Dense − Interpolated、Interpolated − Sparse。三seed逐half沿用2000paired bootstrap、seed20261002；native按session，构造流按stream近似，源录像复用与共享案例局限保留。
- [ ] 报边界P/R、匹配边界平均/中位/90分位绝对偏移（samples与seconds）、≤16样本短段、A4C/A5C X-Y-X中段、同类拼接误切、原生标签切换及每类native recall/macro-F1。仍使用0.5秒主容差，不在看结果后改容差或IoU。Sparse和Dense必须在同一完整10 Hz轴比较，不直接比较两组不同窗口数量的准确率。
- [ ] 逐原生录像比较三个阶段的公共标签结果：argmax逐样本多数票、整段mean概率标签、B3后的多数票。单列raw正确但decode错误、raw错误但decode正确、两者错误；每类修好/新错按来源录像/session去重，尤其A4C→A5C。分数波动不归因心动相位。
- [ ] 对全部参考段用现有 `unit_outcomes(hit=5, near=10)` 分类：label、missed_cut、extra_cut、boundary_shift，并统计未匹配预测段；配置固定这两个值。near10是1秒的诊断匹配范围，必须与主边界recall的hit5（0.5秒）分开，不能把near内匹配都记作主指标命中。沿用固定label优先顺序，但同时报告是否两端切换在主容差内检出、内部额外切换、参考区间内argmax多数票/mean概率是否支持参考类，避免把同时发生的错误归给唯一机制。
- [ ] 按源录像/session去重报告各原因。参考区间仅用于上述诊断，不作为推理/修复信息；不能把这些归因计数当作oracle可达成绩，不能把某种平均分数不支持label推成视觉特征无信息。
- [ ] 对native下限未达的seed与calibration剩余损失，给逐来源的分数曲线/输出对比并报独立来源数；个例图在服务器，仓库只存不含编号的汇总。公开标签不修改，AI判读不当作专家复核。
- [ ] 检查原目标保持不变：均值selection≥.944409687184662、calibration≥.9821421709894332，每seed/halfrecall≥.95且native≥.973/.980；另报calibration敏感性.9838935784747763。旧设置在Dense上的约束重新检查，不默认仍可行。
- [ ] Run `.venv/bin/python -m pytest tests/test_dense_target_inference.py tests/test_dense_target_inference_report.py -q`。独立核查复算主差值、阈值、共同窗口与误差去重，并用冻结模型核对少量新增窗口。提交本轮文件，不推送，其他未提交与暂存文件不动。

## Task 4: 固定解释和结束条件

- [ ] Dense系统一致改善且约束满足：支持更密目标请求的实现收益；仍是同一个已训练模型，不称新架构或顺序机制。只有目标全部满足才称旧开发协议达标；test不自动开启。
- [ ] Dense只减少偏移，但F1/分类/约束无一致改善：说明定位更细，没有证明剩余主要损失来自网格。结束本配置，不扩步长/插值/解码搜索。
- [ ] Interpolated与Dense收益接近或差值区间含0：不能证明两者等价，也不能把全部收益归于新观测。按成本与实际计数报告，不事后重选主条件。
- [ ] Dense增加误切或类别错误：报真实收益与代价，保留原Sparse为固定参照；不重调阈值弥补。
- [ ] 若原生损失主要发生在raw正确、decode错误阶段，仅提出相应解码研究依据；若raw与decode共同错，仅提出相应识别研究依据。两种都只是定位，不能自动触发新训练或模块。报告须明确可由采样改善的部分与仍未解释的部分。
- [ ] 本轮到报告即结束，无新训练、dense temporal head、重训ASRF/FACT、蒸馏、expert伪标或外部数据。论文保持暂停，patch读出配置与FACT细分配置保持结束。

## 运行入口约定

`python scripts/dense_target_inference.py --config configs/experiments/dense_target_inference_v1.yaml --check` 完成身份/共享窗口/预算检查。

`python scripts/dense_target_inference.py --config configs/experiments/dense_target_inference_v1.yaml --run` 完成全量新增目标推理与报告；必须要求已通过同身份check，不创建训练进程。
