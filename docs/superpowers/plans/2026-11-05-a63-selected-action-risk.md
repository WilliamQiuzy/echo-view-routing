# A63：来源隔离的危险 N-top 供样与选择动作增强采样 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` or `superpowers:subagent-driven-development` task by task **only after explicit execution approval**. One server executor: existing `training_execution`; reviewers are read-only. **DRAFT_FOR_USER_REVIEW：当前只写卡/审卡，不启动任何训练，包括辅助 cross-fitting；不生成 recipe/分折名单、不做供样、不连接服务器、不创建执行 worktree/队列/监控。** 文档审查 Approved 不等于用户授权执行。

**Goal:** 先验证能否在不接触 CAL/AUDIT/outer 拟合标签的条件下，获得来源分离、去重且覆盖足够的危险选中动作；准入通过后，只检验风险训练采样变化能否改善冻结部署 S-L/N 的低尾风险与安全收益覆盖。

**Architecture:** 部署 A08/J、B3、候选/G、最终 S-L、原 N、maxONE/noop 全部固定。阶段 S 在 FIT 的两个互斥来源半区内构造训练流，使用另半区训练的辅助 S-L/N 选择器供样，同时记录冻结部署选择器真正选中的动作。覆盖不合格即结题，不能训练 C0/C1。覆盖合格后，C0 全候选采样对照 C1“选中动作增强＋全候选保留”；两臂同一数据、R2 表示、网络、目标、初始化、步数和优化器，仅 draw 分布不同。

**Tech Stack:** 现有 Python/NumPy/PyTorch CPU FP32 小头、原五类 IoU50 matcher、只读 A61/A62 posterior/C6 缓存；仅新跨 join 流使用现有 L40S 上固定 A08/J 真前向。新增文件独立命名，不改旧实验的实现/配置/结果。

**Status:** `execution_authorized=false; partition_materialized=false; recipes_materialized=false; auxiliary_training_started=false; coverage_admitted=false; C0_C1_training_started=false`。本卡是待审设计，未实施；文中接口都是未来接口，不能现在执行。精确前序 pin、预算快照及规则镜像见同目录 [basis.json](2026-11-05-a63-selected-action-risk.basis.json)。

---

## 1. 前序证据与本轮唯一研究问题

真源为当前 main 的 `docs/EXPERIMENT_INDEX.md` A61/A62/POST-A62-SCORES，以及 [A62 报告](../../results/2026-11-04_anchor_context_risk/README.md)、[冻结分数诊断](../../results/2026-11-05_a62_frozen_score_diagnostic/README.md)。旧 `docs/EXPERIMENT_LEDGER.md` 的 dated table 截止2026-10-01，不用其旧 pending 状态替代新证据。

- A61 保住主要原生表现并保留部分 JEPA 短段增量，但旧构造单动作 H40/R0（同单元别处 R37，净增3）与另一个 H9/R2/丢两个短段案例仍有害。
- A62 的四头/24000正式更新与全矩阵已完成；R2 22/24 cells 全 noop，24/24 无 JEPA short16 恢复，H1/H2不支持。不能因两旧例被拒而宣称风险问题解决。
- POST-A62：FIT4183唯一候选、unsafe1806，但原 N-top 危险仅13/980、9/995，原生危险 N-top均0；这些危险动作只占384000 draw的633/405次。R2 N-top AUC在FIT为.9656/.9528，在CAL为.6343/.6667。CAL首个危险q8.94e−8/2.57e−7导致大量安全动作被拒；新确认四cells仍有12/13/15/15个安全JEPA收益 N-top动作。

**H63：** 在共同的新供样池上，C1 的预声明选择动作增强采样，相比 C0 全候选采样，能否在不牺牲原生/gross帧安全的情况下，改善固定部署 N-top 的风险排序和可接受真实收益？

本轮不是再次检验 R2−R1 上下文增量；不改变风险事件目标，不更换网络，不调阈值规则。辅助供样、共同训练数据改变是两臂共享背景；C1−C0仅识别整套采样干预（含选中动作/事件类别/去重权重）的效果，不独立识别其中每个子因素。C0−旧A62 R2含数据变化，不称纯采样或 OOF 因果效应。

## 2. 冻结部署合同与身份

基准 main HEAD `db0d04c6543813e2bc3d3456d3a6ecdfd09f327e`；A62 source HEAD `dda29a526bd1ac1d9611b7edabd81974a683811d`；A62模型manifest SHA `2421febaf3acbf47e0651fa09058813b7db30f215ba2f709d1aabb58f638e4a4`。执行前用 actual Git 对象与实际文件SHA核basis，不能信嵌入的passed。

| 对象 | 冻结身份 |
|---|---|
| A08 原stride5 | weights SHA `9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc` |
| J=A57 core-J-high6000 | SHA `2fa26f2b3ffbffd4d8481c6ac1ff2190572f18648e13f13dbf37333a17bdd985` |
| B3/Base | half_window7、threshold.19660949291632382、nms_radius5、threshold_quantile.999；全轴原顺序 |
| S-L seed2026110461/62 | step6000 SHA `039bbec11f6be5b1379e50b9334934c09250ab7ef035bcf630014bb38f0c1a80` / `b3eaf1bdf3a33a0d98928ba834d05d5b6f945d6065db9d25e241468af51b8af2` |
| 原 N 的risk seed461/462 | step6000 SHA `313c7995c9189a4b527f8324dd8b37e75719524a0d021d9864014d8665af2311` / `f82c28f3f091644233d8f2c6cd83e68e9daf06bff2c9ab8d934cc3f746797432` |
| 候选/语义 | 原RA≠B分量＋RJ最大常类run、trim ends、dedup及old/J flag；完整轴majority G；最多1动作/noop；五类顺序PLAX,PSAX,A4C,A5C,SC4C |
| 部署输入/目标尺度 | 原109/47 normalizers、utility RMS1.229706395330679固定；不得用辅助fold normalizer替换 |

对每个部署轴/seed，先选**同一个**正 S-L 最大动作，数值并列按原action.key字典序：

```python
D_top = min((i for i in G_candidates if fixed_SL[i] > 0),
            key=lambda i: (-fixed_SL[i], action_key[i]), default=None)
D_N = D_top is not None and old_predH[D_top] <= old_predR[D_top] and old_pFP[D_top] < .5
C0_accept = D_N and q_C0[D_top] < tau_C0
C1_accept = D_N and q_C1[D_top] < tau_C1
```

拒绝即 Base，无 second-best/fallback，无新q替换原 N 的 pFP，无按class/长度硬规则救援，不裁/拆/增候选。辅助模型绝不进入此部署解码器。所有动作效果仍为完整轴 H/R、ΔFP16、原u/v和真实短段matcher；标签是clip-family传播参考，不是专家逐帧可见性标签。

## 3. 阶段 S：源会话隔离供样（含真实辅助训练）

### 3.1 分折与固定构造：当前不物化

只使用原 FIT2736 native/402 source sessions；原1500跨来源FIT混合流保留只读历史，不直接过滤后冒充fold内数据。`session_of(id)=id.rsplit('_',1)[0]`，不是患者身份。

```python
order = sorted(FIT_sessions, key=lambda s:
    (sha256(b'A63-supply-v1\0' + s.encode()).digest(), s))
F0, F1 = set(order[:201]), set(order[201:])
assert len(F0) == len(F1) == 201 and not F0 & F1
```

按完整native来源归属两个组；不按label/模型错误重排，原生数不预填1368/1368。每组原生五类必须非空，否则结构停止，无新hash重分。

每组新建**1500条**mixed，总3000条；每条4–8个不同source clip、片段长度4–80、`gamma090`概率.5、`reuse=True,p_same_next=.35`，调用原A62 recipe构造器。RNG `default_rng([2026110563,63,d])`，d=0/1分别F0/F1。同轴不重复source，所有refs必须落本组；不为危险数调长度/类别/recipe、不追加“更难”的流。先冻全部recipe/source refs/SHA/actualGit，再有任何辅助头输出。变换不改变五类或真边界规则。

新A63确认流两半各120条，使用原selection284 native/36session与calibration237 native/36session各自来源；RNG同上d=2/3。它们的recipe也在首个辅助结果前冻结，但**仅coverage通过后**才允许真实teacher缓存；不得在FIT供样阶段读取其参考效果。这里outer半名selection/calibration不等于大写CAL阈值组。

### 3.2 辅助交叉选择器：另一组训练，本组供样

每个seed2026110461/62、每个held fold f，独立新初始化一组辅助 `SL_cf[f,seed]` 和 `Risk_cf[f,seed]`：

- 只训练于 `F(1-f)` 的完整native＋其1500条新mixed；held fold的任何source不得出现在训练native或任一训练流refs中。
- 用原S-L109→32→1、position47→32→3及candidate109→32→1架构，原S-L MSE＋全unequal-pair logistic/fixednoop0，原risk位置CE＋ΔFP>0 BCE；原6000步/AdamW1e−3/.01/warm100/cos6000/FP32/clip1。
- 辅助S-L的F_fit、v目标/RMS及109 normalizer、辅助risk的109/47 normalizer全部**仅在该辅助训练组**拟合；不能复用全FIT normalizer/targets。计算其训练native＋mixed的全轴M/P/T确定fold F_fit。模型从相同base seed的fresh init开始，不从全FIT最终S-L/risk warm-start；fold/seed之间不共享参数、Adam、RNG或latest。
- Utility每步64非空queries，32native＋32mixed，使用每query全候选list；risk每步64candidates，32native＋32mixed及其全部changed positions。任何池空/非有限则结构失败，不能改draw或悄悄fallback。
- 先完成6000步、实际审计/Git封模型及normalizer/draw/训练source footprint，然后才给held fold的native＋mixed打分。其N规则仍为正utility top、predH≤predR、pFP<.5，无fallback。

共 **8个辅助逻辑模型×6000=48000辅助优化更新**；Risk模型各有两个物理MLP。辅助训练不是“0训练诊断”，必须计入CPU、状态、trace和真实Adam审计。只有最终6000用于供样，不以held效果挑step/fold/seed。2000/4000/6000＋latest共32个辅助完整states；latest每100更新含全部optimizer/RNG/draw cursor。

此处“OOF”只指辅助S-L/N拟合来源互斥。A08/J及既有部署模型有历史FIT暴露；不声称整个视觉系统未见过这些来源，更不能称患者OOF。

### 3.3 原部署选中集合 D 与辅助选中集合 X 必须分开

对新3000条FIT mixed的所有G内候选，分别保存：

- **D：** 原冻结部署S-L/N承认的top；它最接近将来实际决策，但仍有历史FIT来源暴露。
- **X：** 对应held fold的辅助OOF S-L/N承认的top。
- canonical key=`(immutable_axis_key, action.key)`去重，标记D-only/X-only/both。定义`X_only=X−D`；训练可能包含同轴两个不同候选，**部署仍仅D_top一个**。

先保存纯特征/分数/选中标记及selector seal，才用FIT参考标签计算效果。这里u/v与安全收益membership使用冻结原A61 F_fit=.8917446116307443；辅助训练自己的fold F_fit仅供该辅助选择器目标/RMS，不能混入部署或membership。危险目标不变：`y = int(H>0 or dFP16>0)`。安全收益池定义`y==0 and short16_lost==0 and (R>0 or u>0 or Jexclusive_short16_recovered>0)`，u仍为surrogate。仅short-loss而H/FP均不增的动作仍是原BCE的y=0，留在全候选池；不能暗改目标来修验收。

选中供样只增强**mixed**部分，因为前序缺口全在mixed；native32曝光两臂完全匹配。必须报告原生D/X的危险数（允许0），不能由此声称学到了原生危险识别；原生保护靠后述硬验收。

### 3.4 真正的去重/来源覆盖，不能用整条流的footprint膨胀样本数

每个危险动作记录：

1. H位置的`(source_clip,original_frame,Base_class,edited_class)`；若ΔFP>0，还记录新出现的未匹配≤16预测段及其有序原始refs/预测类。完全忽略构造轴ID、绝对拼接offset和gamma；H tuples按(source_clip,original_frame,Base_class,edited_class)排序；新增FP段内保留refs顺序、段间按规范内容排序，使用UTF-8 canonical JSON（sort_keys=True,separators=(",",":")）的SHA为**event signature**。
2. **责任source sessions**只取H位置或上述新增短FP段所引用的source，而不是整条流所有4–8个片段。A4C↔A5C/PLAX→PSAX指至少一个H位置的参考类→edited类，不以整个动作多数类代替。H/FP-only/both、H≥8、替换多数类、A4C↔A5C、PLAX→PSAX、完整run覆盖1/部分覆盖<.5均单列。
3. 安全样本的draw signature取所有changed位置有序原始refs/Base/edited类；同源帧的不同拼接/gamma版本不当作新的独立事件。若同signature有不同标签/效果，保留冲突、按membership分别统计，不能删除不利版本。

这些ID/signatures/逐case数组只留server，公开计数、来源概率、source-sharing情况和pins。唯一动作、signature、source会话均不等于独立患者或IID试验。

**准入全部按去重signature或责任session计；动作数仅作描述。** 对每seed的危险D/X_only分别构造去重集合S_D、S_X；signature的责任session集合必须在全部成员间一致，否则审计失败。C1危险抽样各4个槽，故对任一session s：`p_D(s)=|{e∈S_D:s∈responsible(e)}|/|S_D|`，`p_X(s)`同理，实际危险增强槽的来源参与概率`p_mix(s)=.5*p_D(s)+.5*p_X(s)`。三者的最大值各须≤.20；不能用candidate轴数、gamma复本数或整个流的session数作分母。跨D/X的同一signature在union只计1，两个membership的概率分别如实计算。

H/FP/类转换由signature事件内容定类型；几何覆盖某bucket仅当该signature在相应membership的全部成员都满足时计入，跨bucket不一致者单列不替代配额。安全J收益signature也须全部该membership成员满足，冲突照录。复制、改轴ID或追加同signature/同membership的gamma复本不增加任何准入数量，也不降低上述来源概率；这条复制不变性是必须通过的结构fixture。recipe首轮已全冻，禁止追加版本来改变membership救准入。

## 4. Coverage Gate：两个seed都全部通过，才准入C0/C1

以下是**事前工程覆盖门槛，不是统计功效证明或安全保证**。按source/action/event signature独立列分母；两seed不相加。门槛在任何新结果之前固定，不足不能加recipe、换fold或降低floor。

| 每seed、仅新mixed供样 | 必须满足 |
|---|---:|
| D危险event signatures | ≥30；责任sessions≥10（唯一动作数另报，不作floor） |
| X_only危险event signatures | ≥30；责任sessions≥10 |
| D∪X危险 | ≥60不同event signatures、≥20责任sessions；跨membership同事件不重复 |
| 来源集中度 | max p_D(s)、max p_X(s)、max p_mix(s) 各≤20%，按上述实际signature抽样概率 |
| H>0 / H≥8 | ≥40 / ≥15不同event signatures |
| pure-FP危险（H=0,ΔFP>0） | ≥8不同event signatures，责任sessions≥5 |
| A4C↔A5C参考害 | ≥15不同event signatures，责任sessions≥6 |
| PLAX→PSAX参考害 | ≥8不同event signatures，责任sessions≥4 |
| 完整run覆盖1 / 部分覆盖<.5的危险 | 各≥15不同event signatures，按上述全成员一致性归桶 |
| D安全收益 / X_only安全收益 | 各≥100不同draw signatures；各含≥30个安全J-exclusive short16收益signatures |
| 类型与隔离 | 两fold原生五类非空；所有refs/候选/normalizer/targets/optimizer source隔离通过；无非法池fallback |

所有5替换类的危险/安全计数（包括0）都公布，但不制造稀有类样本以“填满表”。没有覆盖的类/原生危险类型不获得学习充分性声明。

**Fail = A63供样/覆盖负结果。** 保存48000辅助更新及其费用、完整覆盖表/全部失败项，停止，不实例化或优化C0/C1、不跑outer策略矩阵。不能把供样失败写成采样假设已被训练证伪。审计错误与真正覆盖不足分开；工程修复不能改分组/配方/门槛。仅在原合同和剩余预算内修复/续接，不重复已完成teacher或训练。

**Pass不是开训充分条件：** 还须用实际spend、空间、全部后续24000步/校准/238行/140paired/56supports/34native/图QA/独立审计/closure预测重新完成全caps准入，并冻结coverage证据、draw、数据与actualGit。

## 5. 通过后：C0/C1精确匹配采样对照

共同训练池是2736原native＋3000个新fold内mixed的全部G内候选；native候选应与既有265个逐key一致，mixed数量必须实际数，不填4183。两臂都能访问同一完整池；不使用CAL/AUDIT/outer标签拟合。

- 两臂固定A62 **R2表示**205维（原109＋context96）及A62 normalizers，不重拟合；同`205→32→GELU→1`、6625参数、末层0初始化、相同seed逐tensor init。
- 目标均为原unsafe事件，普通逐candidate BCEWithLogits mean；无新增ranking/focal/严重度权重，sigmoid仅推理，不clip0/1。推理固定为每轴全G候选的FP32 batchshape，空轴跳过NN；不在CAL/outer间切换分批/量化以改变极小q或ties。
- 每臂6000更新，batch64，原AdamW/学习率/FP32/clip1；两seed共4模型/24000正式风险更新。不得早停/挑seed/挑step。16完整states（12milestone＋4latest），latest每100含全部state。

预生成全部draw：

| batch64部分 | C0 All | C1 Selected+Replay |
|---|---|---|
| native32 | 全native候选均匀 | **完全相同32个索引** |
| mixed前16 | 全mixed候选均匀 | **完全相同前16个索引** |
| mixed后16 | 全mixed候选均匀 | D危险4、X_only危险4、D安全收益4、X_only安全收益4 |

C1选中部分：先在对应membership内均匀抽signature，再在signature内均匀抽canonical动作，可有放回；同一signature全部成员的责任session集合须一致，故§4的p_D/p_X/.5混合概率正是实际来源参与概率。每候选保留全池非零曝光概率；75% batch属于全候选replay。C0的mixed后16也有放回。C1的label平衡/去重层级是采样包的一部分，不能声称“只改变了N-top标志、其他梯度分布完全一致”。

RNG固定域：共同native `[seed,63,0]`；C0全mixed `[seed,63,1]`（前16共享）；四selected slots的signature/member分别`[seed,63,2,slot,0/1]`。每步每slot4个；统一预生成6000行和源索引，actualGit封SHA后开训。不能根据训练loss、CAL或outer采样。

OOF辅助模型8个与正式风险模型4个合计**12逻辑模型/72000真实优化更新**；辅助48000与formal24000分账，工程toy/speed checks另列，不能宣传“只训练24000步”。旧A62 R1/R2仅冻结参考，不重训。

## 6. 校准划分与屏障

**不重洗已暴露的100个inner source sessions。** 逐pin复用A62 `partition_mapping.private.json`（SHA`2e42c56d376d75d8e064df9e7d00cfd0b8622a2810337b6f3c09a1c1f61332d3`）：

- CAL：原50 sessions、319 native＋原A62的150组内mixed＝469轴。
- AUDIT：另50 sessions、309 native＋原A62的150组内mixed＝459轴。
- CAL/AUDIT/FIT以及原selection/calibration外来源两两session互斥；旧300条inner跨来源mixed继续不进入新CAL/AUDIT。
- 它们均已在A62和本诊断暴露，故标记为固定开发校准/审计集，不声称新独立holdout。禁止按A63输出重分、追加或移走低分危险点。

两新头均final6000后，只在同一CAL原部署D_N集合分别校准：

```python
bad = [r for r in CAL if r.D_N and (r.H > 0 or r.dFP16 > 0)]
tau = min(r.q for r in bad) if bad else (1.0 if any(r.D_N for r in CAL) else 0.0)
accept = D_N and q < tau
```

总4个新τ；CAL两seed D_N应仍为80/92、unsafe10/14，与已封原C6/参考逐项核对，否则先修IO不能新划分。仅short-loss不新增到校准unsafe目标；它可以导致验收失败，不能事后改τ来消掉。

AUDIT只读出，绝不选模型/τ。先独立核4模型/16states/24000trace、4τ/8新inner记录/全部端点和实际Git对象，再冻models/calibration/readout-plan三者actualGit；此后才允许任何A63 outer风险头输出或GT指标。辅助模型不能代替此屏障。

两新确认半的recipe先冻、teacher可以coverage后geometry-only precache；此时不加载A63事件头、不读GT效果，不算u/H/F1；构造器瞬时label parse例外明确记录。独立producer identity/journal不含finalheadseal；最终屏障后逐payload只读adoption绑定24000，保留两个独立训练计数轴：FIT producer为aux=0/formal=0；coverage后CONFIRM producer为aux=48000/formal=0；final adoption为aux=48000/formal=24000。formal=0绝不代表全轮0训练；保留时点和费用，不覆写journal或重NN。未adopt/partial cache一律拒绝，禁止fallback forward。

## 7. 固定全矩阵、验收与停止结论

### 7.1 完整矩阵（coverage pass后不可缩）

7表＝main、A58stress、A59confirm、A60confirm、A61confirm、A62confirm、**新A63confirm**；每表两半、两个seed。8个policy＝E、N、冻结L0/M0/A62-R1/A62-R2、C0、C1。

| 项目 | 精确数量/规则 |
|---|---|
| outer rows | 7×2×(1 Base＋2seed×8policy)=**238**（56新C0/C1＋168冻结policy参考＋14Base） |
| CAL/AUDIT rows | 2×(1Base＋2seed×8policy)=**34**（8新记录＋26参考） |
| paired | 每14表半×2seed×5contrast＝**140**；各5000个同unit paired draws |
| contrasts | C1−C0、C0−Base、C1−Base、C0−冻结A62-R2、C1−冻结A62-R2 |
| primary family | 新A63四cells的C1−C0及C1−Base，共**8**；同5000样本同时报95%和99.375%CI（quantiles.003125/.996875） |
| 固定D_N容量 | **56 support记录**：A62旧48条精确只读复用，新确认2半×2seed×event-safe/acceptance-safe＝8个新solve；见下文 |
| native cubes | main两半×2seed×8policies＝**32**＋2Base；其他表复用main native，不增加独立样本数 |
| 风险诊断 | FIT供样D/X/all、CAL/AUDIT与14outer source roles，all-G与原D_N分开；unsafe类型、q0/1/BCEinf、whole-tie低尾、coverage/rejection/regret/真实曝光及成本 |
| 回归 | CaseA/B×2seed×8policy＝32潜在/实际效果记录；其中新C0/C1共8，旧L0/M0/R1/R2原16条只读复用，E/N另8条从冻结C6与原case证据只读派生、0新参考NN；不是新确认 |

新确认240条必须完整跨join A08/J前向，不拼native logits。**旧12个source roles**及其C6/C2严格只读，新确认增2个，合计14；所有raw score/action/input/model/semantics/journal/payload逐pin核。新增C0/C1小头在这些角色的前向必须如实计数；固定参考头只在缺新角色缓存时前向，不能把共用gate重复计NN。

不重跑已完成的A61/A62全域oracle或collector。本轮用逐候选effects与冻结D_N的可达容量诊断，旧oracle只读引用；不把safe-only oracle gap当纯ranking gap。旧48条support来自A62 `N_top_support.csv`（SHA `c4ccbe57409510ff80b4c7c0fd024b5b7aee432f2fb00e262ea2ff8d8ac5969d`），以原axis/input/D_N/action/effect pins核定后精确复用，不重solve。新确认8条必须在全部readout实际Git封存之后，求解仅{noop,原D_N}的精确pooled multi F1选择；event-safe为H=0/ΔFP≤0，acceptance-safe另加short16_lost=0，严格noop-first并列。native/pair可用数及source-role效果分开报告。它们是标签辅助可达容量，不保证任何单q阈值实现、不冒充模型输出，不新增全候选域joint oracle。所有5类混淆、native accuracy/macroF1、frame H/R/ΔE、cuts、short10/16 FN/FP/gross recovered/lost、A4C↔A5C、边界、same-class joins、长度/覆盖分层与资源均保留。

Bootstrap单位沿用完整source sessions（native）、完整固定流/固定pair，配对draw每unit只生成一次并供全部指标/CI复用；共享来源、重复构造及只有两seed的限制照录，不冒充新增患者/完全独立事件。undefined ratios和finite-replicate条件化原约定保留，不通过丢掉失败replicate制造显著性。

### 7.2 三个判定轴，不混成一个胜负

**硬保护（C1必须在全部28个表半/seed cells通过）：** native最终clip标签、accuracy/macroF1及5类confusion相对Base不降；native/multi/pair各自gross H=0；每kind全任务ΔFP16≤0、每kind真short16 gross loss=0。逐class FP/FN/害修复全列，任何类新增gross害直接失败；不以别处修复抵消H。native参考错帧的安全R修复允许，仍完整列changed positions，不要求机械全帧相同。

**安全有效性目标（C1）：**

1. A61confirm selection两seed各保留至少6/5个原JEPA真short16恢复；另半至少4/3。按原source-role、原matcher/gross units计，不把动作数代单位。
2. 新A63confirm四cells每个至少1个安全J-exclusive真short16恢复；且C1−Base pooled multi F1@50的family-adjusted99.375%CI下界均>0。
3. 全noop、仅少切、仅净错帧下降、仅总AUC上升均不能通过。

**H63采样增量：** 上述硬保护/有效性全部通过，并且新四cells C1−C0 pooled multi F1@50的99.375%CI下界均>0，才称本有界设置支持采样包增量。C0也改善但C1不稳定领先时，保留“共同供样/数据条件下可改善、采样增量未证实”的分离结论；不得用C0救援C1的primary失败。

不以AUDIT、旧外层表或CaseA/B选择checkpoint/seed/τ。不事后删除CAL极低分危险点；q=0导致τ=0而全拒绝时照录负结果。只说明本配置，不否定全部风险模型族，不称conformal/临床安全保证。

### 7.3 硬停止

- 无执行授权：停止在本卡；所有trained/produced counts=0。
- 来源/身份/标签绑定失败、stage全caps不合、原时间窗/资源上限耗尽、意外CUDA或混入禁用数据：failclosed，保留partial/actual费用/OS，不降floor/改科学配置救援。
- 任一seed coverage Gate失败：停止于阶段S，C0/C1正式更新必须0。
- 正式阶段通过准入后按既定6000完成，不根据早期CAL/outer中途选优；真实预算/硬件故障可中止并诚实incomplete，不缩矩阵称完成。
- 完整验收失败：结题保留负结果；无自动A64、额外seed/模型/阈值救援。

## 8. 待批预算、累计不重置与执行窗口

这是一份**待批上限**，不是已获新训练额度。T0=null；拟在用户明确批准开始时记录实际UTC/epoch，固定12小时总窗口，deadline=T0+43200s。现在不计时、不排队。它不擅自延后A62原2026-11-05 08:32:41UTC deadline，也不借A62剩余额度提前跑A63。

最新结题快照来自POST-A62 `PUBLICATION_RECEIPT.json`（包含partial测量/明确unknown尾）：

| 资源 | 已累计快照 | A63拟增量硬上限 | 在上限处的累计 |
|---|---:|---:|---:|
| 实际GPU job occupied wall | 18.657838855h | **1.5h=5400s** | 20.157838855h；另留继承2GPUh closing reserve后22.157838855h<24h |
| CPU accounting | 20.231205854833334h | **8h=28800core-s** | 28.231205854833334h<72h |
| conservative tracked storage upper | 38,650,409,858B | **1,000,000,000B**，含最终所有server/local/Git/临时副本 | 39,650,409,858B<40,000,000,000B |
| elapsed window | 新T0待用户批准 | **12h**，其中2h留收尾 | 不重写前轮时间账 |

上表已累计不是新live事实；审批后先锁内刷新所有后续审卡/实施/RED/review/部署/attempt/OS尾实际费用，保留unknown项，重算remainingcaps。预测/reserve不是实耗，不做“预算清零”。foreign任务不计A63、不kill、不购买新机。

拟GPU分配：FIT3000新mixed source stage≤4500occupied s；coverage后新confirm240 source≤600s；可审计cost-only GPU探针≤60s，余240s用于attempt/entry/tail风险；总5400s不可突破。若真实预估装不下，不实施source stage。已有A62单机测时仅是估计输入，不当作当前速度保证。

拟CPU工程上界：准备/新cost probe1200s、source全CPU6000s、辅助训练与supply9000s、正式风险6000s、全readout/CI/图QA/独立审计/closure6600s，总28800s；其中closure至少1CPUh。storage增量中至少128,000,000B留最终打包/副本/回执。阶段份额是准入forecast，不记实耗；同科学合同内可依据新实际测时审计调整份额，但总cap/完整矩阵/closure reserve不可削弱。

供样阶段最坏约3000×2=6000个新source payload；通过后confirm240×2=480，最大**3240真实新流/6480双teacher payload**。原native/CAL/AUDIT/旧outer不重forward。所有partial/failed/resumed attempt逐实际NN单位与占用wall计费，不能用最终提交payload数代全部实耗。

每次长阶段先完整forecast到收尾，包含48000辅助＋24000formal、所有small-head inference、238/34/140×5000/56supports/34native/图与独立审计/真实parentwait/保护main集成。只可通过新的cost-only测量＋窄工程审查/Git修正错误估计，保留失败原件；不能无证据降floor、删端点或改阈值。

CPU阶段在import torch前隐藏CUDA、线程数固定1并实查context；意外context则失败，整个occupied wall保守计一次GPU。GPU source阶段计整个claim从entry/load/IO到cleanup的wall，不只kernel。累计CPU明确区分SELF+waitedchildren、父尾补集、本地SSH/admin，nestedfixtures不双计；模型推理/工具/未测setup不伪称worker CPU。

## 9. 实施文件与逐任务执行图（未来接口，现在不得运行）

授权后新worktree拟为 `/Users/williamqiu/.config/superpowers/worktrees/echo-view-routing/a63-selected-action-risk`，server新namespace `runs/a63_selected_action_risk_v1`，公共目录 `docs/results/2026-11-05_selected_action_risk`。T0/parent/birth/actualcurrentGit须当时记录，不能把本卡日期当birth。唯一server执行者继续`training_execution`；先查collaboration/live boot/PID/start/active/pending/flock/ledger/GPUowners，不另派executor/重复queue。

| 任务 | 新建文件及责任 | 检验与产物 |
|---|---|---|
| T1 合同/身份 | `configs/experiments/selected_action_risk_v1.json`; `scripts/a63_common.py` | actualGit/input/producer函数/旧weightSHA；default dry-run；owner authorization＋T0 receipt |
| T2 来源/recipes | `echo_routing/temporal/selection_supply.py`; `scripts/prepare_selected_action_risk.py`; `tests/test_a63_source_isolation.py` | 固定2×201/2×1500/2×120；所有refs完整组内、禁用数据0、real acrossjoin身份 |
| T3 独立producer | `scripts/precache_selected_action_risk.py`; `tests/test_a63_lifecycle.py` | phase-specific caps/partial不fallback、单attempt journal、source units与parentOS |
| T4 辅助CF | `scripts/crossfit_action_selectors.py`; `tests/test_a63_crossfit.py` | complement-only targets/norm/Adam、8模型/32states/48000trace、actualGit先于held-score |
| T5 供样/门槛 | `echo_routing/temporal/selection_coverage.py`; `tests/test_a63_supply_gate.py` | D/X/both分离、event/责任来源dedup、双seed完整gate、coverage.json＋negative closure |
| T6 matched draw/风险 | `echo_routing/temporal/selection_sampling.py`; `scripts/train_selected_action_risk.py`; `tests/test_a63_sampling_pairing.py` | sharednative32/sharedmixed16，4selectedslot、6000draw冻结、4模型16states/24000trace |
| T7 seal/readout | `scripts/audit_selected_action_risk.py`; `scripts/evaluate_selected_action_risk.py` | 4τ/34inner/actualGit/adoption屏障；238outer、140paired、56supports（仅8新solve）、34native及风险/32回归 |
| T8 图审计/终账 | `scripts/report_selected_action_risk.py`; `scripts/close_selected_action_risk.py` | PNG/PDF/CSV逐图视觉QA、独立科学/资源核验、真实callerOS、exactpublicmanifest、root保护main集成 |

每任务：

- [ ] 先写明确手算/结构fixture并观察对应RED，示例：held source夹入某个mixed必须拒绝；给其他source的同事件轴/gamma增加重复副本不能稀释D/X来源概率并使gate从fail变pass；即使只重复一个责任来源也不能靠整条流多source使gate通过；D_top被拒不能换CF_top；同一风险事件不同gamma不增加signature数。
- [ ] 只实现该单元最小接口，跑对应GREEN及实际必要检查；不重跑旧A61/A62整套工程checks。
- [ ] 新scope独立SPEC→QUALITY审查，修复保留原RED/非0OS与费用；源代码/配置/数据/normalizer/draw按actualGit和SHA封存，不能靠review Approved代运行准入。
- [ ] 合同内常规修复/检查/准入/全矩阵/结题按批准后授权自动推进，不每阶段再问用户。科学设计/预算/数据权限若需越过本卡则停止提修订，不能自行换方案。

未来入口均默认dry-run，必须存在执行授权pin且显式`--execute`才可mutate。例如：

```bash
.venv/bin/python scripts/prepare_selected_action_risk.py --stage metadata --execute
.venv/bin/python scripts/precache_selected_action_risk.py --role FIT_supply --execute
.venv/bin/python scripts/crossfit_action_selectors.py --execute
.venv/bin/python scripts/audit_selected_action_risk.py --stage supply --execute
.venv/bin/python scripts/precache_selected_action_risk.py --role CONFIRM --require-coverage-pass --execute
.venv/bin/python scripts/train_selected_action_risk.py --require-coverage-pass --execute
.venv/bin/python scripts/audit_selected_action_risk.py --stage model-calibration-seal --execute
.venv/bin/python scripts/evaluate_selected_action_risk.py --require-actual-Git-seal --execute
.venv/bin/python scripts/report_selected_action_risk.py --execute
.venv/bin/python scripts/close_selected_action_risk.py --execute
```

这是待实现CLI合同，不表示命令当前存在或已测试。producer确认分支只能coverage后进入；collector自身取queue锁，不套相同flock supervisor。独立审计先核真实对象/完整矩阵/费用/actualOS，再由root集成；server snapshot complete不等于overall complete。

## 10. 保护、交付与结题

- A57/58/59/60/61/62及POST-A62命名空间/配置/公共manifest全部只读。新dataset、states、q/case、IDs、source-session映射只留新server RUN；公开aggregate/匿名画像/pins/成本，原禁止test/sealedtest/Pediatric/Unity训练/21隔离患者继续禁用。不造人工标签，不改网页/push/外联/购买。
- Main原用户8个staged理论变更和其他修改继续保护；当前rawindexSHA`19dd2b444410211e2922fcf3497af4b7f10c71a2a13036686d297045e21a35cf`，HEAD如§2，Index文本在POST-A62追加后SHA`271c166792b24087664c255d1d78155052f6fd638d14d0d31a99caf7e5b6ff26`，均需执行时fresh核并保留并发用户变化。
- shared `.git/info/exclude`历史读取ETIMEDOUT，不能删/改exclude/config/user index/泛删锁。旧A62专用publication helper不覆盖A63 allowlist，不能擅扩旧helper来写新scope；若需新helper须独立toy/SPEC→QUALITY，prepared ref/locks/durableintent/backup、失败partial不强恢复。apply前重查hooks与完整index/split/sparse资格，不绕用户新增gate。
- 仅exact新A63路径选择性集成，不夹8个用户stage；Experiment Index仅append当前完整prefix、留unstaged，并核复制SHA/actualGit与用户index保护。读不了普通status不成为破坏ignore的理由。
- 每阶段真实OS/directwait、物理更新/forwardunits、全费用与失败历史齐全。Coverage fail、真实hardlimit中止、完整矩阵后goal fail分别结题，不把任一叫作成功风险模型。
- 只有完整已接纳矩阵＋实际独立审计＋真实OS总账＋模型保存＋exactpublicmanifest＋main保护集成完成，或完整诚实incomplete/negative闭环，才overall closed；若获准创建A63监控，仅删除该轮监控，不影响其他监控。常规未变/健康运行不通知，实质结果/失败/需用户操作/结题才通知。**无自动A64。**

## 11. 本次交付状态

本次只交付此卡及basis/只读审查记录。所有数据生成、辅助训练48000步、风险训练24000步、source前向均为**计划数，当前实际训练/源前向为0**；本地写卡、审查行政CPU另列，不称总资源消耗为0。预算门槛与12小时窗口尚待用户审阅；未获得执行授权。
