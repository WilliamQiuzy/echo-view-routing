# A62：固定收益与候选的动作级风险及锚点上下文对照 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` or `superpowers:subagent-driven-development` task by task **only after explicit execution approval**. One server executor: `training_execution`; reviewers are read-only. **DRAFT_FOR_USER_REVIEW — 本卡未获执行授权。** 当前只写卡及只读审卡；不得创建A62训练/工程优化任务、GPU队列、源缓存、执行worktree或监控。计划审查的Approved不等于用户批准启动。

**Goal:** 在A61固定S-L收益头、原候选和原N门控下，检验动作级unsafe事件证据R1及完整锚点上下文R2，能否拒绝参考帧害、保住A08并保留真实JEPA短段增量。

**Architecture:** 冻结A08/J、B3、候选、utility及N。R0-legacy保留旧阈值，R0-match只改变校准来源；R1/R2学习相同动作级unsafe目标，模型容量、初始化、FIT候选及draw完全匹配，唯一差别为新增上下文是否提供新信息。先完成四个新头的固定6000步，再在一个来源互斥的CAL组按同一规则校准，在AUDIT组只作诊断；全模型/阈值/端点实际Git封存后才读取outer head结果。

**Tech Stack:** 现有Python/NumPy/PyTorch CPU FP32、只读A61后验/score缓存、原五类IoU50 matcher及A61单执行者资源/真实OS协议；实际新增跨join teacher缓存使用现有L40S。没有backbone训练、utility再训练或新标注。

**Status:** `execution_authorized=false; training_started=false; implemented=false; calibration_partition_materialized=false`。本文件给出完整规则，未产生分组名单、recipe、阈值或训练结果。精确已有对象及预算基准在同目录[机器可读basis](2026-11-04-a62-anchor-context-risk.basis.json)。本卡所有命令均为未来待实现接口，当前不得执行。

---

## 1. 前序证据、改变条件和排除范围

真源：原主工作区`docs/EXPERIMENT_INDEX.md`的A59/A60/A61结题条目；`docs/EXPERIMENT_LEDGER.md`仅作截至2026-10-01的旧史，不用其旧表推断当前状态。

- **A59**：raw-supported动作域过窄，新构造另一半可恢复union0/78；完成4头/24000更新而未建立稳定净增益。
- **A60**：JEPA class-run扩大候选容量，但收益/风险误判可新增原生参考错位置，原count-only oracle不保证帧安全。
- **A61**：60000更新、314行、240×5000、60oracle+16复现、56native全部完成。主S-L-C保住native逐位置输出，排序隔离对比20个区间含0；新构造保留部分恢复，但旧A60selection seed461仍H40/R37，新另一半两seed仍各H9。
- **事后案例**：旧例为old-raw前缀40个PLAX→PSAX，动作H40/R0，真实v−1.8917却预测+2.4708；新例为同一JEPA动作H9/R2，丢A4C长11和A5C长9的匹配，真实v−3.2165仍为正分。两例均无安全非空候选，正确选择no-op。证据见`docs/results/2026-11-04_a61_residual_case_analysis/README.md`及匿名JSON。

本卡检验两个假设：**H1**同一校准人群下，动作级事件风险包R1比旧位置证据包R0-match改善拒绝/恢复；**H2**在相同事件模型下，R2的锚点/有序上下文提供增量。R1−R0-match包括监督与证据聚合变化，不能称纯损失因果效应；R2−R1用于信息增量对照。仅使用两seed，不能推断模型族或所有随机初始化。

**明确不做：** 不重训/选S-L头，不加listwise系数，不扩/裁/拆候选，不改G、maxONE、no-op、N或五类matcher，不加fallback，不硬编码PLAX/A5C/9/40特例，不开新的网络/阈值/seed救援臂。候选细分属于后续待议方向，不属于A62本卡；不自动A63。

## 2. 不可变对象及动作合同

原A61源封存`0aeb8527423d491920c0f9e9e578a8496592cf85`，原主工作区集成`31b04d61d816a4762c428b670bdffa6f4e725bc3`。运行时逐一用actual Git blob和文件SHA核验basis，不信JSON内passed标志。四个固定CPU头如下；路径前缀均`runs/a61_action_safety_ranking_v1/models/`：

| 原seed | 固定utility step6000 | 固定risk step6000 |
|---|---|---|
| 2026110461 | `seed2026110461/S-L/step6000.pt`，SHA `039bbec11f6be5b1379e50b9334934c09250ab7ef035bcf630014bb38f0c1a80` | `seed2026110461/risk/step6000.pt`，SHA `313c7995c9189a4b527f8324dd8b37e75719524a0d021d9864014d8665af2311` |
| 2026110462 | `seed2026110462/S-L/step6000.pt`，SHA `b3eaf1bdf3a33a0d98928ba834d05d5b6f945d6065db9d25e241468af51b8af2` | `seed2026110462/risk/step6000.pt`，SHA `f82c28f3f091644233d8f2c6cd83e68e9daf06bff2c9ab8d934cc3f746797432` |

A08为原stride5+B3，权重SHA `9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc`；J为A57 core-J-high6000，SHA `2fa26f2b3ffbffd4d8481c6ac1ff2190572f18648e13f13dbf37333a17bdd985`。B3原参数`half_window=7, threshold=.19660949291632382, nms_radius=5, threshold_quantile=.999`不变。

候选逐字复用`proposal_actions.py`：RA≠B的最大分歧分量+RJ最大常类run，trim只去两端不变位置，dedup及来源flag不变；完整轴majority G，maxONE+no-op。旧轴候选键/顺序/替换逐位置必须与A61相同；新增轴也只运行同一已pin的纯proposal函数。R2不能修改proposal或对候选做预筛。所有源输入class order固定`PLAX,PSAX,A4C,A5C,SC4C`。

utility原109特征/normalizer/RMS不变，RMS=1.229706395330679。对于每条轴、每个seed：

```python
top = first_sorted_action_of_highest_positive_frozen_SL_score_or_None
N = top is not None and old_predH[top] <= old_predR[top] and old_pFP[top] < .5
E_prediction = apply(Base, top) if top is not None else Base
N_prediction = apply(Base, top) if N else Base
C_L0 = N and old_r[top] < original_A61_SL_tau[seed]
C_M0 = N and old_r[top] < tau_M0_CAL[seed]
C_R1 = N and q_R1[top] < tau_R1_CAL[seed]
C_R2 = N and q_R2[top] < tau_R2_CAL[seed]
```

拒绝即Base，永不转选第二候选。新q不取代N中的原pFP，不和old_r作OR/max门控，否则无法检验保住恢复覆盖。R0-legacy原τ分别`.007963286712765694/.006616456434130669`。E/N和所有C共享完全相同utility/top。旧缓存可复用时0新增旧头forward；新轴必要的固定utility/risk推理必须如实计数，不能说A62无NN。

## 3. 数据、校准划分及来源隔离：固定方案

### 3.1 拟合数据不变

只用A61既有FIT：2736原生/402session+固定1500构造，共4236轴；G内4183候选（native265、constructed3918），非空query137/1431，16688 changed-position训练多重集。使用A61 prepared dataset SHA `b7ba5812e4ade5eb87339cef51603d3a710774d8a93b36072a8c6a168bec76ff`。新标签仅为每候选`y = int(H>0 or dFP16>0)`；H/R/dFP沿用完整轴参考标签和原matcher。不会从outer或CAL/AUDIT拟合任何参数/normalizer。

### 3.2 将原inner的100个session确定性分成CAL/AUDIT

原inner628 native属于100个session，训练FIT/inner/selection/calibration原source/session overlap为0。未来metadata阶段：

```python
sessions = sorted(unique(session_of(native.id) for native in original_inner_628))
assert len(sessions) == 100
order = sorted(sessions, key=lambda s: (sha256(b'A62-inner-v1\0'+s.encode()).digest(), s))
CAL_sessions, AUDIT_sessions = set(order[:50]), set(order[50:])
CAL_natives = all_original_inner_natives_whose_session_in(CAL_sessions)
AUDIT_natives = all_original_inner_natives_whose_session_in(AUDIT_sessions)
```

分别保证50个session、native数合计628、每原生轴完整未裁剪、两个组五类均非空，且与FIT和两个outer source组互斥。若资格不满足，metadata阶段失败，不换seed重分、不按风险/效果重抽；修订卡须先重新审阅。具体native数/类计数只能由这条规则产生，当前未运行，不虚填314/314。

**不把原300条跨session inner构造硬塞入两组。** 它们保持原样只作历史参考，完全不参与新τ、新AUDIT或拟合。避免一条流混入CAL/AUDIT两边session。

从CAL natives新建150条CAL mixed，从AUDIT natives新建150条AUDIT mixed。两组实际`N_CAL_native+150`及`N_AUDIT_native+150`轴，共928轴，但不是原928的重新命名。每条构造只能使用本组来源，同一条不重复source。新recipe来源集合和原生source footprints逐项验证。

### 3.3 新确认与固定生成规则

新增A62 confirmation selection/calibration各120条，只用原selection284 native/36session、原calibration237 native/36session各自来源。四个新mixed角色共540轴：CAL150、AUDIT150、new-selection120、new-calibration120。

统一调用原`Pool.from_index`/`make_multi_recipes`，参数`n_fragments=(4,8), min_len=4, max_len=80, edit_variant='gamma090', edit_prob=.5, reuse=True, p_same_next=.35`；每条source不重复。使用独立RNG：`np.random.default_rng([2026110463,62,d])`，d=0/1/2/3依次上述四角色。不重试recipe、不按指标重采样。相邻同类按原RLE合并规则，source join不冒充语义边界。

在任何新event头输出/更新之前，冻结100session分组、四份recipe、引用帧/变换、原native映射、classmap、来源交集核验、各角色T/window/chunk上界及全部端点的SHA/actualGit。分组名单/原ID留server，公开仅计数/校验hash。

新的540条跨join源后验必须真实运行固定A08/J，共1080个source payload；不能拼接native logits。原native及旧外层5表使用已资格化完整context缓存，只读复用，不重forward。构造阶段clip-family元数据和瞬时label parse仅用于固定recipe/参考标签；不得用它们生成特征、挑source或看任何A62 head效果。

来源互斥不抹掉历史暴露：A08见过inner，CAL/AUDIT也属于被反复使用的OPEN开发来源；新outer只换recipe，不是新患者或独立临床连续扫描。Case A/B已用于假设设计，均为开发回归项，不能再作为未见成功证据。test/sealedtest/Pediatric/Unity训练及21隔离患者继续禁用。

## 4. R1/R2的精确可实现匹配

### 4.1 信息及参数容量同时控制

原候选特征记x109。为避免R2只因多输入参数占优，两个新模型均输入205维：

- **R1：** `[normalize_A61(x109), normalize_A61(x109)[:96]]`。后96维是已有输入的确定性重复，不增加信息，不用全0 dummy列。
- **R2：** `[normalize_A61(x109), normalize_FIT(c96)]`。前109完全相同；c96为下述新观测上下文。

这使R1在信息上仍为“原109候选输入”，但物理输入/参数容量与R2相同。两个模型均`Linear(205,32)→GELU→Linear(32,1)`，6625参数、末层权重/bias初始0、无dropout。相同seed的全部初始tensor逐字相同，两个臂共用同一candidate draw。R2−R1解释为新增上下文信息及其输入分布效应，不声称特定单个feature的独立因果效果。

### 4.2 c96固定布局：80个有序概率槽+16个几何/锚点槽

先取Base连续同类runs中所有与action `[s,e)`相交的完整runs，再包含紧邻的一个左run和一个右run（存在时）。其连续包络`[a,b)`为anchor envelope；所有操作仅依赖Base、原A/J概率和action。

80维：把`[a,b)`按`floor(k*(b-a)/8)`切成8个按时间排列的等长bin。依次A08、J，每个bin按固定5类取均值，共2×8×5；空bin填5个0，非空bin概率和为1，故空bin可由该块辨识。不是把全部时序无损保存。

随后16维严格按表排序，log均为log1p：

| 槽 | 定义 |
|---|---|
| 0–3 | log(T)、log(changed K)、log(action L)、log(envelope length) |
| 4–5 | `(s-a)/(b-a)`、`(e-a)/(b-a)` |
| 6 | log(Base边界中严格位于s/e之间的数量) |
| 7–8 | log(apply action后被删除的Base边界数)、log新增边界数；包括端点，按全轴差集 |
| 9–10 | log(与action相交的最短完整Base run长)、log最长完整run长 |
| 11 | 与action相交各run中`overlap_length/full_run_length`的最小值 |
| 12 | K/L |
| 13–14 | 相交完整runs的A08段均值对其Base类与最大其他类的margin：最小值、按完整run长度加权均值 |
| 15 | 原B3 half_window7产生的JSD boundary score在s/e两端的最大值；轴外端点记0 |

R2的96维均值/std仅从同4183个FIT候选计算，std下限1e-6。原109 normalizer不重拟合。新增特征不含GT、ID、fragment join、真实边界、H/R、短段匹配或未来帧标签。特征构造也不改变Base/B3输出。保存逐槽名称、单位、空值和轴截断规则及FIT normalizer hash。

### 4.3 监督、暴露和训练

四个新头=R1/R2×seeds2026110461/62，各6000，总24000正式更新。仅候选级unweighted BCEWithLogits，`y=(H>0 or dFP16>0)`；不加focal/类别权重/困难样本挖掘、位置CE或H/R回归辅助。新q不输出预计H/R，不能把其值冒称gross count。短真段丢失不加入当前训练目标，作为独立严格验收端点。

复用A61每seed的risk candidate draw：每步32 native候选+32 constructed候选，6000×64=384000物理candidate曝光/头。复用同一FIT candidate索引及顺序；不是64query、更不是全候选数。两个臂同seed的每一步曝光完全相同，不按失败案例改变训练分布。

AdamW lr1e-3/weight_decay.01/betas(.9,.999)/eps1e-8；warm100+cosine至6000，FP32 CPU、clip norm1、无early stop。τ只在final6000后产生。2000/4000/6000各完整state+latest100更新，4头合计12 milestone+4latest=16记录；均含model/Adam/RNG/cursor/norm/draw/input/code/source身份。中间checkpoint只记录FIT训练及可复现状态，不读CAL/AUDIT决定checkpoint。old utility/risk权重、normalizer和RMS必须始终逐SHA不变。

## 5. CAL、AUDIT和匹配阈值

在四个final6000封存后，在CAL上计算原S-L top及原N。各模型/控制的阈值人群是**完全相同的N-admitted top集合**，而不是全部候选、全部位置或各自重新选top。标签仅用于校准这一阶段：unsafe仍为真实H>0或dFP16>0。

对每seed和M0/R1/R2分值s分别执行同一公式：

```python
admitted = CAL_records_with_original_N_true
unsafe = [r for r in admitted if r.H > 0 or r.dFP16 > 0]
tau = min(r.s for r in unsafe) if unsafe else (1.0 if admitted else 0.0)
accept = original_N and s < tau  # strict; ties reject; no fallback
```

M0分值为old_r，R1/R2为各自sigmoid q；产生6个新τ。原L0两τ原封保留，共8条阈值引用。CAL上M0/R1/R2 zero-observed-unsafe是构造性质，不是独立保证。有限0/1饱和按原float值处理，不加epsilon；nan/inf或缺行即实现失败。

公开各seed的CAL总轴/N-top/unsafeN-top数、τ决定点数/并列数、接受/unsafe/全no-op、native/mixed来源、分数分布。预设校准信息量提示线为每seed至少20个N-top、5个unsafeN-top；不满足则标`calibration-limited`，不得重抽/补采/放宽阈值或宣称已识别H1。边界规则仍照上式执行，保持完整负结果矩阵，不以少样本选择性停掉某臂。

同一CAL阈值在AUDIT不变应用，只作来源互斥的诊断；AUDIT不训练、不选择臂/seed/阈值/步骤。AUDIT出现H或回归案例没修好是科学负结果，不触发重训，也不删后续完整外层矩阵；只有实现/数据/资源门失败才停止执行。原928 inner及其原τ输出继续作为历史证据，不改写。

R0-match−R0-legacy识别校准来源/响应变化；R1−R0-match评估事件风险包；R2−R1评估上下文。没有“改CAL又把收益全归模型”的对比。

## 6. 固定全矩阵、指标及先后顺序

### 6.1 端点计数

外层6表：main/A58stress/A59confirm/A60confirm/A61confirm/**新A62confirm**，各selection/calibration两半。每个table-half恰13系统：Base A08-s0+B3一行；每seed各E、N、C-L0、C-M0、C-R1、C-R2六行。因此**156 outer metric rows=6×2×(1+2×6)**。

CAL/AUDIT各同样13系统，**26 inner diagnostic rows**。Base及E/N对照不是新增拟合；native/pair跨6表严格复用main预测，只有mixed换bank。所有计数/score来源/复用是否实际forward必须显式列出。

每table×half×seed固定10个contrast，**240×5000 paired bootstrap**：

1 R2−Base；2 R1−Base；3 R2−R1；4 R1−M0；5 M0−L0；
6 R2−M0；7 R2−N；8 R1−N；9 M0−N；10 N−E。

沿用原whole-session(native)/whole-stream(multi)/whole-pair配对单位及pooled-M/P/T重算；原缺失比率/undefined/null记录保留。全部95%条件区间；另对新A62的H1/H2×2seed×2half共8个主要对比，用同5000抽样给99.375%双侧条件区间（quantiles .003125/.996875，Bonferroni口径），不新增抽样、不宣称超越原shared-source bootstrap假设。

main的6policy×2seed×2half=**24个完整5×5×5 native联合归因cube**，另列2个Base native参考。开发回归Case A/B按4个C策略×2seed×2axis=**16个记录**；同一个B不能计为两条独立病例。公开只匿名动作/门控摘要。

oracle只诊断固定expanded/G/maxONE/no-op域：旧5表×2half×3安全域的30条A61结果精确复用；新A62两半各unrestricted/net-safe/gross-safe，**只新算6条**，共36条引用/结果。不重跑A61的旧域库或原60/16复现。先冻结全部外层模型读出再做新oracle。gross-safe仍H0且dFP≤0，不事后把定义改为无短段丢失；短段丢失另报。safe-only oracle不是unsafe模型的普适上界，union不是joint attainability。

**另补固定top可达支持诊断，不能拿上段全候选oracle替代。** 6表×2half×2seed=24个cell；每cell分别用两种真值过滤，独立表共48条support记录：

- `event-safe`：只允许`{no-op, 原N-admitted S-L top}`，该top另需真实H0且dFP16≤0。
- `acceptance-safe`：同一固定域，另要求short16 gross lost=0。不得改成另选candidate；原N不成立、无positive top或过滤失败时只有no-op。

每条记录报告原N-top总数、过滤后可用top数、H/R/u、正u top数、gross真段及JEPA-exclusive short16可恢复/丢失、union覆盖、**精确pooled multi-F1最优accept/no-op选择**的M/P/T及实际joint恢复。pooled目标仅multi；native/pair的固定top支持/H/R另作单独计数，不混入F1 oracle。原生/对子跨表重复只标复用，不充当新证据。

求解平分沿用`anchor_oracle.fractional_oracle`的精确整数/分数规则：同一分式目标相等时优先no-op，其次按原canonical action key字典序；固定top域只有no-op/top。实际joint恢复按最终确定的choices计算，不在同F1最优解中事后挑短段恢复最多者。旧30条引用不改其原平分结果。

48条记录均在全部outer读出封存后，仅读原top/N、缓存标签及actions求解，不新增NN。它区分“固定utility/N已无安全恢复支持”与“q/阈值未取得已有支持”。该oracle可按标签逐轴选accept/no-op，而q是同阈值策略，故可达支持不等于某个q阈值必能实现；unsafe/有短段丢失的实际策略也不受strict-safe上界约束。不据此重训、改N/q/目标、减掉负结果矩阵或降低验收标准。

### 6.2 必报内容和科学验收

完整五类native macro/混淆、逐位置混淆/accuracy、H/R/net及按真类流向；mixed/pair完整M/P/T和segmentF1@50；边界P/R、same-class joins、cuts；short≤10/16 FN/FP、gross恢复/丢失、JEPA-exclusive恢复；候选/接受/veto/no-op/长度/K/比例/极值；原N及各C覆盖、事件BCE/Brier/固定5箱q的unsafe率、r/q估计对象差别；Case A/B gate和RLE；训练/资源/物理曝光。q仍未获临床概率或有限样本保证。

**主系统事前固定R2，R1是对照，不在R2失败后改挑R1作为主胜出。** 分开判三件事：

- **执行有效性：** 四新头24000真实Adam/trace、16完整state、6新τ、全部182行/240CI/24cubes/6新全候选oracle+30旧引用/24个可达支持cell×2过滤=48记录/16回归/图审计和真实OS账完整。不能拿软件检查或partial矩阵代替。
- **联合经验目标：** R2在6表×2half×2seed的每个固定cell均H0、短FP不高于Base、short16 gross丢失0；main native逐位置H0，main segmentF1点估计不低于Base并同时给CI（不由含0宣称非劣）。原A61 confirmation的JEPA-exclusive gross short16恢复，逐cell不低于原primary的selection6/5、另一半4/3，且lost从2变0。新A62确认每half每seed至少1个JEPA-exclusive真短段恢复且不丢短段；拒绝全no-op胜利。
- **归因证据：** H1/H2只依据其事前匹配contrast；在新A62四个half×seed上方向一致、对应主要条件区间下界>0，才称该F1增量得到支持，同时满足上述安全目标。安全改善而F1区间含0，可报告拒绝改善但收益证据不足；R0-match独自改善不能算R1/R2模型增益。240比较其余部分描述性，不挑显著cell。

全部数值仍相对传播参考标签/共用OPEN构造；没有专家逐帧/新患者/自然临床流证据。部分达标、H>0、旧短段恢复缩水、no-op退化或无法证明归因，分别诚实报告；不自动换模型/seed/τ/候选/数据/主指标，不自动A63。

## 7. 预算：明确承接旧账，新增上限待用户批准

旧24GPU-jobh/72CPU-accountingh/40,000,000,000B累计额度**不清零**。A61结题root最终账为GPU18.4888181300h、CPU19.7818594369h、server tracked保守上界37,904,365,798B。另有本次事后诊断已测CPU5.611943秒：server5.278390+传输/匿名化.217111+公开context提取.101185+验证.015257；故已知列明CPU基准**19.78341831h**。原0.75h历史allowance及所有未知OS/重叠/未测行政尾保留；当前审卡/文件写出等未测开销另列，不能伪报0或把已包含child的CPU再加一次。

原deadline `1793820844.1596537`（2026-11-04 **14:34:04纽约/19:34:04UTC**）已结束。**本卡不能沿用为一个仍有效时间窗，也不自动重置它。** 提议的新窗口为用户以后明确批准A62执行的T0起12小时；T0与精确epoch必须写入未来授权记录。仅审阅本卡不产生T0，不授权延时或训练。

| 项目 | 提议A62新增硬上限 | 说明 |
|---|---:|---|
| GPU occupied job wall | **1800秒（0.5h）** | 仅540新跨join双producer的完整entry/load/IO/cleanup/失败attempt；新风险训练CPU-only |
| summed measured CPU accounting | **21600秒（6h）** | 全metadata/新特征/工程72/训练24000/校准/182行/240CI/6新全候选oracle+48固定top支持记录/图/审计/collector/根集成及失败/review；其中3600秒保留收尾 |
| tracked storage conservative upper增量 | **1,250,000,000B** | 全new namespace、540×2后验、R2特征、16state、score/trace、临时atomic副本、图和收据；不复制旧raw/weights/caches |
| absolute wall | **T0+12h（仅待批准提案）** | 不是本卡写入时间；所有准入/科学工作须能连同收尾完成 |

预计边界：GPU18.488818+.5+保留2h closing reserve=20.988818<24；CPU19.783418+6+额外.25h预测不确定性buffer=26.033418<72；storage37.904366GB+1.25GB+.5GB预留=39.654366GB<40。buffer/reserve/forecast不是实耗。local公开副本与Git行政开销按实际scope分列，不伪称server全文件系统峰值。

A61 240新流完整双producer实耗271.649473 GPU秒；540流按相同平均体积线性参考约611.211秒。这只是排程锚点，**不是admission**。冻结真实recipes后计算实际T/windows/chunks、最大T/C/K/包络成本，并结合已有真实teacher及新205维CPU图的测时保守放大，覆盖全矩阵、1080 payload、所有attempt/temporary、独立审计及收尾。新GPU/CPU阶段限额不能相加超整轮增量上限。

CPU阶段排程上界：metadata/新cache CPU3600秒、特征+72工程+24000正式5400秒、校准/读出/CI7200秒、oracle/native/图/终审/collector/root同步3600秒、剩余1800秒机动=21600秒；每笔实际rusage扣账，不能重复计nested。只允许已批准预算内的cost-only实测及窄审修正工程估计，不改目标、步数、matrix、阈值规则或硬上限。若全剩余admission不合，记录失败并诚实闭环，不先开训练赌后面能塞下。

## 8. 未来执行图与数据/模型barrier

目前本节均未执行。用户以后若明确批准实施/训练及新wall窗，才创建独立A62 worktree（拟`/Users/williamqiu/.config/superpowers/worktrees/echo-view-routing/a62-anchor-context-risk`）、私有`runs/a62_anchor_context_risk_v1`和公开`docs/results/2026-11-04_anchor_context_risk`。不得在A61目录续训或覆写seal。

1. **资格/metadata freeze：** 先查collaboration，仅同一`training_execution` server executor；核live boot/PID/start/active/pending/实际ledger/flock/GPU owners。协调已有claim，不另queue、不kill/bill foreign、不购买。旧A57–61只读。冻结new code/config、100session分组、四recipe、端点计划和source identities，不读取新q结果。
2. **全剩余预算+teacher阶段准入：** source未开前覆盖整轮全部端点，明确540跨join源成本和2GPUh reserve。允许在已冻结recipes之后先做独立计费的CPU-only cost-only synthetic205维/长包络边界测时（0SGD、0真实outer head/标签结果），以便把新图成本纳入source前完整forecast；不是正式头或效果pilot。若该forecast不合即不启动GPU。通过后唯一source阶段真实forward固定teacher；metadata/构造瞬时label parse例外公开。source阶段不实例化新q、不跑utility/risk head，不算指标。
3. **新FIT图：** 完整核4183候选/目标/原109字节、R2 c96、两臂205输入及drawplan；只做新图72工程更新，rusage/OS/Adam/梯度/重复/磁盘resume/I/O/worst负载。不得重跑A61的130工程或源库资格化。
4. **actual formal admission → 四头24000：** 重新扣除全部实耗并覆盖所有剩余matrix+closing。固定顺序seed461 R1→R2、seed462 R1→R2；latest/12milestones实际Adam/RNG完整。CPU import前隐藏CUDA；异常context立即failclosed，整个context occupied wall保守计GPU一次。
5. **final audit → CAL → AUDIT：** 独立验证4final/16states/24000physicaltrace及4原fixed heads未变；CAL只生成6新τ，AUDIT只诊断，保留负结果。实际Git冻结全8头引用、norm/draw、6新+2旧τ、182行/240CI/24cubes/6新全候选oracle/24支持cell×2过滤/16回归端点后，才允许任何新outer utility/risk/q输出。
6. **只读源adoption → 外层156行：** teacher precache逐1080payload/journal核source/input/axis/class/规范化/完整context/SHAs，保留生产时new_event_updates0和绑定时24000两时间身份；不改旧journal、不重NN。新head cache每seed储存冻结C6旧utility/risk引用+新(C,2) q、actions/identity，不能冒充旧A61 C6 reader。缺precache/adoption失败须拒绝，不能fallback另一路NN。
7. **全部读出封存 → 240CI → 6新全候选oracle/48固定top支持记录/24native → 图与终审。** 新图统一6类：R0校准效应、R1/R2配对、grossH/R与短段、风险/覆盖、上下文及匿名回归、训练/实际资源；PNG/PDF/CSV完整并逐图视觉QA。旧30oracle逐实际Git/原source pins引用，不重新求解旧库；新增支持记录也纳入CPU和临时空间准入，不改变6h上限。
8. **实际资源/OS/保护集成闭环：** collector自身持queue锁，不套同锁supervisor；parent真实wait，SELF OS未知保持null，snapshot后的caller/CPUtail另列且不重复。实验完整不等于paper-goal实现。新exact公开manifest、caller状态及root只选新A62路径同步，Index当前prefix追加留unstaged；原用户所有修改/staged完整保护，不push/网页/外联。完成或真实hard-limit/infrastructure终止全部阶段后才诚实结题；本卡不自动创建监控，也不自动A63。

## 9. 实现文件图及任务清单（仅以后批准后执行）

所有既有`proposal_*`/`safety_*`/B3代码与A61config/报告只读，不为A62改旧对象。新增：

| 文件 | 单一职责 |
|---|---|
| `configs/experiments/anchor_context_risk_v1.json` | 本卡常量/身份/预算/端点，拒绝unused字段 |
| `echo_routing/temporal/event_risk_data.py` | 分组/recipe、FIT事件标签、共享drawplan/旧cache映射 |
| `event_risk_features.py`（同目录） | 精确c96/205、FIT norm、无标签观测合同 |
| `event_risk_model.py` | 6625参数MLP、BCE、Adam/RNG/fullstate及物理trace |
| `event_risk_policy.py` | 固定top/N、L0/M0/R1/R2、CAL strict τ |
| `event_risk_cache.py` | headless producer身份、1080payload只读adoption、q-cache |
| `event_risk_metrics.py` | 182/240/24/6/16计划、原matcher复用、报告 |
| `event_risk_budget.py` / `event_risk_audit.py` | actualcaps/ledger/OS/tail及冻结核验；不改旧账 |
| `scripts/{prepare,cache,check,admit,train,calibrate,readout,aggregate,package,close}_event_risk.py` | 对应单阶段CLI、默认只读/dryrun，显式执行flag及approved-plan身份 |
| `tests/test_event_risk_{data,features,model,policy,cache,scope,budget,closure}.py` | 新合同测试；不全量重跑旧研究测试 |

`event_risk_*.py`简称均位于`echo_routing/temporal/`。旧CLI含A61全局RUN/PUB常量时不得直接import成A62写入口；只能复用经pin的纯库函数或新namespace wrapper。

**实现任务与实际运行分开：** Task1–3先完成接口/必要软件测试和整图审查；不在Task1先开GPU、再等Task2实现测时模型。真实source、完整FIT工程72、正式训练及读出统一在Task4按§8依赖顺序各执行一次；已有阶段凭真实收据复用，不重跑。72指完整FIT repeat/resume检查，最小软件fixture若产生真实优化器更新须另列工程计数及实际费用，不藏成0，也不计入24000正式更新。

### Task 1 — 输入/来源与预算合同
- [ ] 保存未来新main工作区before/index/用户修改保护快照，不强制回退到本卡记录的历史HEAD；记录新的合法A62 parent。
- [ ] 写新data/budget合同RED，覆盖100session精确50/50、混合流仅本组source、role互斥、4固定RNGdomain、旧300mixed排除及禁止重抽。
- [ ] 实现metadata-only接口，GREEN后冻结counts、inputmanifest、partition/recipe/source code实际Git；独立SPEC→QUALITY，不运行任何q训练。
- [ ] 实现全剩余预算/source准入及1080payload物理计数接口；真实cost-only/source执行留到Task4，须先有Task2/3已审的新模型/特征/阶段代码。

### Task 2 — 事件特征/模型新图
- [ ] RED：FIT标签H>0或dFP>0、H/R取消不能使标签安全；原109和4183候选/排序完全一致，R1后96为确切duplicate，不是新信息。
- [ ] RED：8bin顺序/空bin、完整Base run包络、全轴边界差集、端点/短轴/多run、96槽顺序、normalizer FIT-only；交换GT而观测不变时特征必须不变。
- [ ] 实现features/model，固定205/6625、同seed初态与同draw、BCE均值及完整状态；GREEN/窄SPEC→QUALITY。
- [ ] 实现并准备新FIT工程检查4头各repeat6+repeat6+resume3+3=72的阶段接口；真实运行留Task4，证明finite、历史梯度/Adam/drift、精确重复/resume、全部state字段和durable I/O。工程头不得warmstart正式头。

### Task 3 — 门控、校准与端点
- [ ] RED：E/N exact reuse、四C同top、拒绝无fallback，q不能进入原N，q与r意义不混，CAL strictties/0/1/noN/noUnsafe边界、AUDIT不可改变τ。
- [ ] RED：182rows/240contrasts/24cubes/6新全候选oracle+30旧引用/24固定top支持cell×2过滤/16regression唯一键；支持只能取原N-top或no-op、event-safe与strict-safe嵌套及精确pooled M/P/T，native/pair复用、NULL分母及95/99.375分位数；旧Case B去重说明。
- [ ] 新cache/adoption/预算/OS pending/durable-intent失败路径RED→GREEN。fake child退出须真实wait，原失败与非0退出保留；不把manifest或PID消失当OS0。
- [ ] 新完整graph SPEC→QUALITY，冻结新代码/配置/端点，具备实际full formal admission入口；不是toy passed即启动，真实准入留Task4。

### Task 4 — 正式与实际读出
- [ ] 按§8先复核唯一metadata/recipe冻结收据，运行新图synthetic cost-only测时及全剩余source准入，再唯一540双producer cache；随后一次完整FIT工程72和实际full-formal-and-closure准入。任何之前已完成真实阶段只核原收据，不重复执行。
- [ ] 严格图序4新头×6000；实查真实PID/start/Adam推进，四固定头与旧代码SHA不变。
- [ ] 独立final/fullstate/曝光审计；CAL/AUDIT26记录、6新τ和2旧τ引用实际Git封存，outer head仍不得提前。
- [ ] 源只读adoption及完整outer admission；执行156行及全240CI，随后6新全候选oracle、48固定top支持记录、24native、16回归；不选择性丢失败臂，也不因标签显示支持不足而改固定N或阈值。

### Task 5 — 图、科学/资源审计与主目录保护
- [ ] 六种PNG/PDF/CSV真实render/逐图QA，自包含报告全部方法/半部/seed、原始负结果和H1/H2/联合目标分别判定。
- [ ] 独立科学/资源history/actualGit/真实OS审计；父wait/caller tail/私有模型保存与total ledger完整，不计foreign。
- [ ] exact新A62 allowlist选择性集成。A61专用Git helper的allowlist不包含A62，**不能直接调用或改旧helper蒙混通过**；若普通Git仍受exclude I/O影响，另建独立A62窄helper并经SPEC→QUALITY/隔离repo测试，fresh检查hooks/split/sparse/并发HEAD/index，不绕新增用户gate。
- [ ] main当前Index只追加，完整保留所有当前staged/unstaged；commit不夹用户8项或任何后来修改；核actualGit、copy SHA和保护范围，不声称不可读旧文件已重算hash。只在真实闭环后宣告实验结题；无自动A63。

## 10. 拟实现CLI协议、核验与停止规则

**以下不是现有可运行命令，不在审卡阶段运行。** 实现需让每个CLI默认不执行，变更/训练必须同时要求`--execute`和已批准card SHA/授权记录。未来单worker顺序示例：

```bash
.venv/bin/python scripts/prepare_event_risk.py --config configs/experiments/anchor_context_risk_v1.json --metadata-only --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/admit_event_risk.py --stage sources --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/cache_event_risk.py --stage headless-sources --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/check_event_risk.py --stage full-FIT-new-graph --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/admit_event_risk.py --stage full-formal-and-closure --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/train_event_risk.py --all-four-final6000 --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/calibrate_event_risk.py --CAL-only-thresholds --AUDIT-diagnostic --seal-before-outer --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/readout_event_risk.py --all156 --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/aggregate_event_risk.py --all240 --bootstrap5000 --new-oracle6 --reachable-support24 --support-filters2 --native24 --regression16 --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/package_event_risk.py --all-plots --execute --approved-plan-sha <APPROVED_CARD_SHA>
.venv/bin/python scripts/close_event_risk.py --execute --approved-plan-sha <APPROVED_CARD_SHA>
```

`<APPROVED_CARD_SHA>`只可由未来真实用户授权后最终卡版本生成，是运行身份参数，不是当前缺失研究条件。代码审查后需补exact命令、source freeze和本地/远端环境路径到新`EXECUTION_SEQUENCE.md`，未完成之前不能启动GPU或优化器。新合同测试命令为`.venv/bin/python -m pytest tests/test_event_risk_data.py tests/test_event_risk_features.py tests/test_event_risk_model.py tests/test_event_risk_policy.py tests/test_event_risk_cache.py tests/test_event_risk_scope.py tests/test_event_risk_budget.py tests/test_event_risk_closure.py`，只检新图，期望有真实RED及修正后的GREEN/费用收据。

**执行停止：** 旧对象漂移、数据/来源泄漏、不可恢复pending/checkpoint ambiguity、缺实际Git seal、未经授权的CUDA、任一累计/增量资源硬限额或absolute deadline不合，立即failclosed并完整记费/退出/报告；不删失败或并发状态、不kill foreign。非预算性的科学负结果继续既定矩阵并报告，不改变设计。不得因某臂先失败、某seed好看或图已出而提前称整体完成。

**当前交付边界：** 只完成此卡、basis与只读文档审查。没有物化新分组/recipes、没有建立A62运行目录/队列/监控、没有新源/模型前向或训练。等待用户审阅；未来是否授权实施/训练及新12h窗口由用户明确决定。
