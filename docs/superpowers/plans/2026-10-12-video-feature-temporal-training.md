# Frozen Video Features with Matched Temporal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Claude is the intended executor. Within the fixed scope and budget, complete preparation, checks, training, one evaluation, independent verification and round-only commits without waiting for the owner to return.

**Goal:** 检验成熟的学习型时序模型，能否在已有效的视频表征上找回简单变化点解码丢掉的切面段，并同时达到原生识别与分段要求。

**Architecture:** 冻结三个 raw_video_context_v1 Ordered 编码器，保留其16帧/224×224输入及512维投影。每条完整序列先按原stride5窗口提特征，再nearest-centre映射到10 Hz轴；三个训练头读同一 `(T,512)`：线性分类头、官方MS-TCN、官方ASRF。视频权重不更新，无自创gate/attention/辅助loss。

**Tech Stack:** TargetMViT、序列级上下文特征缓存、既有training-bank和epoch-batch规则、`dev_heads.MSTCNHead/ASRFHead`、PyTorch、pytest、英文README。

---

## 负责人决定与问题

密集请求配置结束，保留 offset_only / interpolation_close / native_localisation / end 结论；不继续步长、插值或旧输出上的解码搜索。patch读出配置与FACT细分配置保持结束。冻结Ordered的多帧与系统正结果保留。

现有依据指向两处不同的损失：selection中Dense仍漏掉的参考段，有13/11/7段的Dense多数票支持参考类别，但解码后失败；calibration剩余段多数的原始多数票已错。原生剩余错误主要是少数A4C→A5C。这些都是定位，不能证明一个机制同时解释两半区，也不能把参考区间多数票正确当成可实现的oracle成绩。

本轮提出新、有限的问题：**在固定视频表征上，学习一个完整的时序分类/分段系统，是否优于只重训类别读出，以及视频分类器加简单解码？** 以前在E0/E1/L1上的时序头结果不等于这个视频特征组合的结果；本轮也不是将那些已关闭配方重新搜索。

ASRF、MS-TCN均为已有方法。本任务先检验强表征与成熟时序模型的组合，不声称原创架构；系统差异不能单独归因于边界loss、某个卷积层或心动信息。

## 固定来源与共同输入

- 五类、protocol、原始帧顺序、gamma、短边256/中心裁剪224、0–255归一化、10 Hz采样、16样本窗口与marker，继承 `raw_video_context_v1.yaml`。只用Ordered真实顺序。
- 三个完整冻结来源为 `models_frozen/raw_video_context_v1/ordered_s{0,1,2}/final.pt`；sha分别 `80f13a42e628fe2647634b79e59d5b8486fee5c3e0ae45f70bcc667c41258505`、`74276323ababf78fcb578be28d3172f3c220b3b7e66c2ff96a2072f39210c26f`、`923eb3afb0d846b1361897dbb63bb5aecf22401ce4b305b44fe990d48edc7f2c`。encoder、marker、投影及源classifier全部保持冻结/eval；新线性头是源classifier的独立副本。
- **保留原Sparse stride5**，不选Dense作为新输入。`window_centres(n,5)`、`window_frames(c,n,16,16)`和nearest-centre规则均原样，含2/7/12…的centre偏移和端部处理。边界并非每0.1秒独立视频预测。
- 每个窗口取 `model.embed(rgb,marker)` 的512维输出，不L2/标准差归一化、不读patch token。先获得 `(K,512)`，再按原nearest-centre得到 `(T,512)`，float32存储。原源classifier作用在这些特征上应复现原窗口logits，最大差≤1e-4且argmax相同。
- **上下文敏感缓存必须按完整输入建立。** 对构造流先组装每个10 Hz样本的 `(video,frame_idx,gamma)`，再在完整构造流上取窗口编码。严禁将独立录像的MViT特征直接拼接；窗口会跨拼接处。模型/完整窗口引用/marker均进入缓存身份，不能只按目标帧缓存。
- Native先对完整orig录像编码；时序头训练时的≤256样本native crop再从该完整native的特征中切取，保留其原完整录像的视觉上下文。该约定训练和完整native评估一致，不将crop当作另一段重新编码的输入。
- 不混合三个编码器的512维坐标。seed k的三种头只读Ordered_s{k}特征。三种头同seed的每个输入浮点数组、有效长度、labels、joins与批次都完全相同。

## 三个训练系统与参照

| 英文报告名 | 训练什么 | 固定推理 |
|---|---|---|
| Linear + change-point | `Linear(512,5)`，从该源classifier复制W/b | 概率接一次calibration选点的原B3网格 |
| MS-TCN | 官方4-stage、10-layer、64 channels，in_dim=512、classes=5 | 官方最后stage逐时刻argmax |
| ASRF | 官方512→64，10层、ASB/BRB各4-stage、5类 | 官方refinement_with_boundary，boundary_th=.5 |

**主候选固定为ASRF。** 主比较ASRF − (Linear + change-point)，另比较ASRF−MS-TCN及ASRF−冻结Ordered+B3。MS-TCN作为另一个强学习型对照；如果它领先，照实报告，但不事后更改主候选。

- MS-TCN官方commit `33ed91c0c7576650a2367efc602553af4c5295b1`，沿用 `MSTCNHead` 的4-stage CE + .15 TMSE。这里直接输入512维是任务适配，不零填充为2048，记录参数数目；不加载以前不同输入维度的权重。
- ASRF官方commit `9623f1e8d9a1171333a4eeb65d190997b6c44a95`，沿用 `ASRFHead` 的官方weighted CE + GS-TMSE（实际权重**1**）+ .1 boundary loss，boundary_th=.5。不要继承旧文档误写的GS-TMSE .15；不修改官方库。
- ASRF的class weights与boundary pos_weight按该seed epoch0训练序列的官方规则计算并固定。Linear用**同一组class weights**作masked CE，以控制简单类别重校准。MS-TCN保留官方unweighted CE；比较完整系统，不能将ASRF−MS-TCN作为单一loss消融。
- 新线性头保留源W/b。时序头seed_everything(seed)后建立，各自确定性初始化，不在不同结构间声称起始权重相同。若旧 `runs/fact_asrf_paired_v1/inits/ASRF_s{k}.pt`可用，应与重建的同结构seed初始state核对并复用；缺失则记录重新建立的digest，不据结果换初始化。

## 训练库、批次与预算

- 只使用现有protocol训练3364录像与固定1500条multi-bank。重建bank必须保持原训练行相对顺序和原recipe digest；不另造开发集决定的新短段/病例。
- 训练准备只使用protocol.train和训练允许名单。索引先读取id/split/n_samples筛选，再只取训练行label_index；raw_label和其他split标签不读。E1缓存仅供训练帧索引/时间与bank元数据，不将其特征或预测替代MViT特征。不能直接调用会将全部索引labels读入的旧 `TrainOnly`。
- 三seed `[0,1,2]`；50epoch×375更新=18750更新/头，batch为4条完整bank stream+4个native window；native_per_epoch1500、native_window256。沿用 `TrainData.epoch_batches` 的RNG/抽样逻辑和protocol clip顺序，**在新适配器中保留float32**，不走旧native_seq/stream_seq的fp16转换。
- 每epoch bank streams用seed/epoch打乱；native录像与crop start使用原RNG `[seed,epoch]`。用 `train_l1_asrf.batch_digest` 核对ids/lengths/labels/joins；如旧配对ASRF plan存在，逐更新比对这些结构digest。三种新头必须完全配对。
- 全部头用AdamW lr3e-4、weight_decay1e-4、betas(.9,.999)、eps1e-8、clip1；750更新warm-up后cosine到0。feature augmentation/额外归一化均none；官方网络内dropout保留。正式18,750更新取**最后权重**，无早停或开发epoch选择。
- fp32、TF32 off、cuDNN deterministic/benchmark off。记录官方CE算子的确定性限制；恢复需还原参数/优化器/调度器和Python/NumPy/CPU/CUDA RNG。源视频不反向、不加入优化器，hash前后相同。
- 每seed queue为Linear、MS-TCN、ASRF，合计9次训练。主候选不在训练曲线后改成某个head/seed。
- **本轮总GPU上限8小时**，包括训练特征、检查、9次训练、开发特征、评估和核查。预留2小时给开发提取/报告/核查，其余按实际训练-only计时与样本总数估算。预算与实现检查通过后直接执行，不等负责人回来；超预算保存状态并报告，不减少某组预算、不放弃对照、不自动扩预算。
- 正式训练全部结束并冻结后才提取/加载本轮开发特征与新指标。可提前读取旧汇总与来源digest；它们不能选择新epoch/数据/设置。test/Pediatric不读；公开标签不变。

## Task 1: 协议与上下文特征接入

**Create:** `configs/experiments/video_feature_temporal_v1.yaml`、`echo_routing/temporal/video_sequence_features.py`、`scripts/prepare_video_feature_temporal.py`、`tests/test_video_sequence_features.py`。

- [ ] 先提交任务书与固定配置，保留前轮停止记录。引用source config/权重/metadata/协议/epoch规则和环境digest。旧输出/模型不修改。
- [ ] 先写测试再实现：由组装后frame refs取窗口、same-view joins不产生真边界、跨类joins不作为模型输入；目标/marker/nearest-centre对齐；改变固定refs以外的labels/joins/native flag不改变feature；短序列/端部/gamma均覆盖。
- [ ] Run `.venv/bin/python -m pytest tests/test_video_sequence_features.py -q`。新缓存为 `runs/video_feature_temporal_v1/features/{train,development}/...`，不写回旧feature cache。
- [ ] 在训练源固定16序列上提窗口features，source classifier(feature)与同输入source forward logits差≤1e-4、argmax相同；同上下文窗口才允许去重；frame_idx来自原numeric缓存，不新抽图。
- [ ] 保存训练-only序列清单、每个原生/完整bank流的上下文特征、labels/joins独立文件、按seed的50epoch批次清单、初始状态和class weights。模型只收到features/length mask，prob/reference/bank cell/id不进入前向。
- [ ] source encoder及classifier被冻结，完整state前后hash一致。缓存float32；报告embedding norm分布/样本数仅作检查，不据数值添加归一化。
- [ ] 先在按清单顺序选的训练序列计时窗口提取，按**全部训练windows数×三个source模型**预测成本；预算不足就报告，不只编码bank或只保留少数native。

## Task 2: 头训练入口与开跑检查

**Create:** `scripts/train_video_feature_temporal.py`、`scripts/queue_video_feature_temporal.py`、`tests/test_video_feature_temporal_training.py`。
**Reuse without editing:** `dev_heads.MSTCNHead/ASRFHead`、官方固定仓库、batch_digest、lr_factor与冻结工具。不要修改旧round的config或run identity。

- [ ] Linear用源classifier副本，输入 `(B,512,T)` 转 `(B,T,512)` 逐位置线性，masked weighted CE。MS-TCN/ASRF调用已有adapter。进入loss前按valid mask将padding labels规范为ignore值、padding features置0；不更改官方有效样本上的loss。不要要求不同padding长度时官方TMSE数值不变。
- [ ] 检查三组同seed批次结构和features逐位相同；loss中仅有效labels作监督，gt boundaries只进入训练loss。推理替换所有同长度labels/joins/id字段，输出不变；padding labels改值后规范化loss不变。
- [ ] 按训练清单顺序选择相同16序列，每个头做100更新的数值/梯度记录：有限loss、梯度与权重真实更新即可，不设满分门槛、不据拟合值调整配方。
- [ ] 同初始state的6更新重跑及3+3恢复，沿用已有temporal smoke容差1e-6；若官方CUDA算子限制导致超限，先核对状态/实现，记录具体差异并报告，不自动升容差或换精度。源视觉状态不变。
- [ ] 每种头实际batch8预热5更新、测50完整更新，计入monitor/保存估算。budget ledger覆盖失败检查、特征、训练与后续预留；只允许运行身份相同的完成任务跳过。
- [ ] 独立实现审查重点看“先组装后编码”、fp32缓存、特征坐标不混用、labels隔离、padding、权重/批次来源与预算。修复实际问题并复验通过后提交记录，直接启动9次队列。
- [ ] Run `.venv/bin/python scripts/queue_video_feature_temporal.py --config configs/experiments/video_feature_temporal_v1.yaml`。更新计数18750与最终权重确认后，用freeze_model新增 `models_frozen/video_feature_temporal_v1/{linear,mstcn,asrf}_s{seed}/`。

## Task 3: 一次性评估

**Create:** `scripts/report_video_feature_temporal.py`、`scripts/video_feature_temporal_tables.py`、`tests/test_video_feature_temporal_report.py`、英文 `docs/results/2026-10-12_video_feature_temporal/README.md`、完整表/summary/训练曲线/token流程图。

- [ ] 九次完成并冻结后才建立开发序列feature，按完整native/pair/multi流各自组装窗口，不拼接各native特征。开发也直接建立float32 `Seq`，不能调用 `EvalSet.sequences()`（它内部的stream_seq会转fp16）。特征上的source classifier与旧窗口logits最大差≤1e-4且argmax相同；冻结Source参照优先用原存档概率精确复现旧Sparse/B3系统行，不能用浮点重算差异冒充新收益。超差/对齐错误先定位实现，不更换source。
- [ ] Linear的B3仅在calibration一次选原40项：h `[5,7,9,11,15]`，q `[.80,.85,.90,.95,.97,.99,.995,.999]`，NMS5；native≥.980、multi recall≥.95下segment F1最高，同分原网格顺序，无可行点即失败。参数先冻结再selection一次应用。它是新训练分类器的标准评估，不重开旧冻结输出的解码搜索。
- [ ] MS-TCN只有官方最后stage argmax；ASRF只有官方refinement boundary_th=.5作为主输出，另报其frame branch作为诊断，不追加B3/阈值/epoch选择。
- [ ] 主比较ASRF−Linear+B3，另ASRF−MS-TCN、MS-TCN−Linear+B3、两时序头−冻结Ordered+B3。逐seed逐half报paired bootstrap2000、seed20261002，native按session/流按stream近似。全程同protocol、10 Hz轴、.5秒容差与class-aware IoU .5。
- [ ] 复现/列出旧E0/E1/B7+B3、历史/配对E1+ASRF、Original FACT与G1+ASRF；旧系统使用其自己的输入与存档设置，绝不把旧E1 ASRF直接接到MViT特征上。若训练计划/初始化与旧配对ASRF完全一致，报告该完整输入替换比较，注明精度/预训练/上下文都不同，不能单独归因动态信息。
- [ ] 报native宏F1、各类recall、A4C→A5C/反向修好新错、原生label switches、multi segment F1、边界P/R、≤16样本段、X-Y-X中段、同类拼接误切；所有病例按来源录像/session去重，seed不是额外病例。
- [ ] 对原Sparse B3已定位的“原分类多数票正确但段未匹配”参考段作**冻结诊断清单**，新系统修好/新错分别报告；不可用清单作训练采样/选参，不因只修selection就宣称整体机制成功。
- [ ] 对ASRF拆分候选边界是否提出、官方refinement是否合并、类别是否判错；区分source visual正确但ASRF class branch改错，与class branch正确但refinement改错。参考区间只用于诊断，GT不进入推理；大多数票/mean分数错不等于特征无信息。
- [ ] Linears与时序头都比较源classifier识别：若Linear已修好同样病例，不能把这部分收益全归时序。ASRF−MS-TCN不作为单一边界loss消融。
- [ ] 源已训练模型+简单解码仍为强基线。主候选必须同时达到原目标/约束，并有对Linear+B3及冻结Ordered+B3的收益证据，才能说推进了新完整模型；只超过历史较弱输入不够。
- [ ] 源/新头模型、输入与参数digest、成本、实际GPUledger、命令、原生与流的限制、偏离齐全。独立核查重算主指标/判定/病例去重并少量重推理，报告不含编号/媒体。
- [ ] Run `.venv/bin/python -m pytest tests/test_video_sequence_features.py tests/test_video_feature_temporal_training.py tests/test_video_feature_temporal_report.py -q`。只提交本轮，不推送；其他未提交/暂存、理论稿、论文、许可、演示不动。

## Task 4: 固定解释与停止规则

一致差异仍需某half三seed的95%区间严格同向排除0；区间含0不是等价，不以重用病例的seed数扩大有效样本。

1. ASRF对Linear+B3和冻结Ordered+B3在两half的完整segment F1一致更高，且每seed/half约束满足：支持此完整时序整合的系统收益；不能单独归因边界loss或称新架构。
2. 只修selection解码丢失，calibration分类/原生约束未改善：报告局部进展，整体目标未完成，本配置结束，不加视觉微调或loss补救。
3. Linear与时序头修好同类病例，时序头无额外系统收益：类别重校准可解释部分结果，时序增益未检出；不能把它叫新时序机制。
4. 时序头找回段但改错原生录像/多切抵消：报告权衡，本配置结束。不得再把source logits残差/gate接入或重调官方阈值。
5. 无一致收益：结束这个视频特征/时序配方，不否定所有时序模型。MS-TCN若优于主候选，照实报告，不事后改主候选或开新选参。
6. 旧目标保持：三seed均值selection≥.944409687184662、calibration≥.9821421709894332；每seed/half边界recall≥.95，native≥.973/.980；另报calibration敏感性.9838935784747763。达标只代表重复开发协议，test不自动打开。

本轮到报告即结束。无FACT重训、读出再搜索、encoder微调、外部数据、伪专家标签或新模块。论文暂停；所有历史负结果与正结果原样保留。
