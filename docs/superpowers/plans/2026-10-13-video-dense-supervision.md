# Video Token Supervision with Preserved CLS Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Claude is the intended executor. Within this fixed scope and budget, complete implementation, checks, independent implementation review, training, one evaluation, independent verification and round-only commits without waiting for the owner to return. Preparation is not training; report the actual verified state.

**Goal:** 在现有强视频模型上，检验给视频交互后的时间 token 增加位置对应的切面监督，能否提高类别正确的时间分段，同时满足原生识别要求。

**Architecture:** 三个原 Frozen Ordered MViTv2-S 各自作为三个配对系统的共同起点。所有系统保留原 CLS、512维投影、五类头、marker、16帧输入和Sparse推理；训练时另读取最后norm后的8个时间位置，空间平均后共用原投影和分类头。三组只改变辅助监督：不使用、按真实输入位置对应、或在窗口内打乱对应标签。推理仍只用原CLS输出和原B3规则。

**Tech Stack:** TargetMViT、既有raw-video/target-patch数据清单与token-grid工具、PyTorch、pytest、原B3评估、冻结工具、英文README。

---

## 负责人决定与前序证据

- **A01已结束。** 接受`bce15e1`的结果索引更新，包括增加三个系统行与将标题改为已完成任务。保留trade-off / no gain / end；不继续该冻结特征时序头配方。
- **V01的正结果保留。** 同切面内，多帧比复制目标帧的识别更好；Ordered+B3在两个半区一致优于E1+B3。Ordered−Shuffled没有证明真实顺序额外有效。冻结Ordered+B3仍是强参照，尚未满足全部目标。
- **V02已检验读出变化和继续训练。** 替换为target/global patch平均没有达成主比较与系统收益。其监督仍是一个窗口目标类别，不是对视频交互后各时间token逐一监督。
- **R11已有逐帧辅助损失。** E1图像编码器的辅助帧头有0.5权重；不能说我们从未用过逐帧监督。但那个损失作用于进入时间Transformer前的独立图像输出，不等于本轮对MViT视频交互后token的监督。那轮失败不能被忽略，也不能当作本轮已检验。
- **V03只改变请求密度。** 边界偏移减少，没有一致系统收益。A01在冻结全窗口512维输出之后再学习时序，也未超过强参照。它们不证明压缩丢了信息，更不能证明新辅助损失会奏效。

**有限假设：** 在同一视频窗口内部保留可读的时间位置类别，可能比只要求CLS答对一个目标，更有利于跨切面的目标判断。这是训练目标的检验，不是新架构或创新声明。真实顺序、生理相位、探头移动、标签错误和真实扫查上的效果均不由本轮自动证明。

来源：`docs/EXPERIMENT_INDEX.md`中的R11、V01、V02、V03、A01；对应的修正README与配置。MViTv2是已有骨干：[原论文](https://arxiv.org/abs/2112.01526)。本轮辅助目标是本项目的实验适配，不是声称复现该论文的训练方案。

## 三个系统：前向与参数相同，只改变监督

| 内部arm / 英文报告名 | 主损失 | 辅助时间token损失 | 正式推理 |
|---|---|---|---|
| `center_only` / CLS continuation | 原CLS在请求时刻的CE | 权重0，仍执行辅助前向以配对随机数消耗 | 原CLS + B3 |
| `aligned_dense` / Aligned token supervision | 同一CLS CE | 正确时间锚点类别，固定权重0.5 | 同一CLS + B3 |
| `shuffled_dense` / Shuffled token supervision | 同一CLS CE | 将同窗口8个锚点标签固定打乱，固定权重0.5 | 同一CLS + B3 |

主候选固定为`aligned_dense`。`shuffled_dense`是故意破坏时间对应的监督对照，不是实用baseline。它与aligned使用同样的8个标签、多重集合、参数、loss权重和预算；在单标签native窗口两者完全相同。只击败shuffled不能算成功。

所有arm都计算辅助logits，不能仅在候选组执行Dropout/投影而改变Torch RNG消耗。主CLS前向先执行，再通过同次forward捕获的norm输出计算辅助logits。辅助标签绝不进入模型forward；在loss函数外换标签，eval的CLS logits必须不变。主CLS输出也不能被hook替换、叠加或融合。

```python
# 所有组同一前向；以下规定张量语义，不允许从aux标签构建模型输入。
# X: 最后encoder.norm输出 (B, 1+8*7*7, 768)，保留autograd。
# z_cls: 完整原TargetMViT.forward(rgb, marker)输出 (B,5)。
P = X[:, 1:].reshape(B, 8, 7, 7, 768).mean(dim=(2, 3))
z_time = model.classifier(model.encoder.head(P))  # (B,8,5)，共用原dropout、768->512投影及五类头
L_center = cross_entropy(z_cls, target_label, reduction="mean")
L_time = cross_entropy(z_time.reshape(-1, 5), aux_labels.reshape(-1), reduction="none").reshape(B, 8).mean(1).mean()
lam = 0.0 if arm == "center_only" else 0.5
loss = (L_center + lam * L_time) / (1.0 + lam)
```

normalisation固定在计算结果前。aligned与shuffled的主损失比例相同；相对center_only的变化也含主/辅助梯度权重分配，不能把aligned−center_only单独解释为时间标签位置的因果效果。aligned−shuffled才控制这个差别，但仍只是在构造监督下的比较。

不新增参数、aux head、时序头、boundary loss、gate、residual、teacher或解码器。推理时关闭aux计算，CLS路径与原模型相同。时间token已与其他时间/CLS交互，不能称为未受上下文污染的局部信息。

## 固定数据、来源和位置映射

- 使用`configs/experiments/raw_video_context_v1.yaml`和`target_patch_readout_v1.yaml`所固定的五类、protocol、Ordered权重、RV清单、SC清单、环境与预处理。仍输出224×224。输入帧/顺序/gamma/marker不变，不接尚未交付的团队扇形输入，不修改fan224结论。
- **起点是V01 Ordered最终权重，不是V02 patch权重，也不是A01头。** 三个source sha沿用V01：s0 `80f13a42e628fe2647634b79e59d5b8486fee5c3e0ae45f70bcc667c41258505`；s1 `74276323ababf78fcb578be28d3172f3c220b3b7e66c2ff96a2072f39210c26f`；s2 `923eb3afb0d846b1361897dbb63bb5aecf22401ce4b305b44fe990d48edc7f2c`。每seed的三组从同一个完整state开始，不重置原分类头。
- 重用V01冻结的Ordered RV训练窗口：3750更新，每次4个native和4个mixed窗口；8×16帧、目标、pixel digests和批次完全相同。
- RV不保存逐槽标签。**只从已固定的训练SC清单补取其`labels`**，先逐项核对该seed SC与RV的`vid/frame/gamma/s/c/target/kind/seq/repeats/videos`及marker，确保是相同窗口。保留补充标签文件与独立digest，不能改原RV或SC文件。所有录像必须在原protocol.train允许名单内。
- norm后网格必须由模型实际几何计算为(8,7,7)，每个block的query temporal stride必须为1。stem的时间kernel/stride/padding/dilation必须为(3,2,1,1)。否则停止，不能临时换映射。
- 时间锚点固定为输入槽`[0,2,4,6,8,10,12,14]`，即stem query中心`j*stride - padding + (kernel-1)*dilation/2`。取这些槽的原SC类别作为8个aux目标。它们是输入时间锚点，不是最后token只依赖该帧的receptive field。
- 槽重复/端部clamp照原清单保留，不伪造新时间，不据预测筛选。重复槽仍在8个位置的平均loss内；报告重复比例，不能将其算成8个独立病例。
- 对shuffled，每窗口用独立NumPy `default_rng([20261013, seed, update_index, window_index]).permutation(8)`生成一次排列。标签多重集合必须相同，允许identity/同类不变；不强制重抽，不动像素，不动target label，不消耗全局Torch RNG。报告排列及标签真正变化的比例。
- 这些是公开clip标签与人工拼接得到的训练标签，不是专家逐帧的解剖真值。不能用已有开发错误清单挑窗口、重标或删录像。
- 准备与训练不读开发frames/features/labels，也不读test/Pediatric。训练索引按允许名单先筛选，再读取标签；禁止为方便载入其他split的标签。

## 训练预算与复现约定

- seeds `[0,1,2]`，每seed三个arm。每次3750更新，最后权重；无开发epoch选择。
- 沿用V02/V01的AdamW、encoder lr1e-5、新参数lr1e-4、wd1e-4、clip1、375更新冻结预热和原warmup/cosine。预热冻结encoder所有参数，但保留通过encoder到marker的autograd；之后全部encoder可训练。head/projection共享，不另建optimizer参数组。
- fp32、TF32 off、原stochastic depth与dropout。全部arm辅助计算顺序相同。新center_only也是本轮新训练的配对对照；因新增辅助前向的RNG消耗，不能要求它逐位复现历史Continued CLS，也不能借此换随机数规则。
- 正式队列为每seed center_only、aligned_dense、shuffled_dense，共9次。参数、起点、帧、数据顺序、Torch RNG起点和更新数均配对。取最终权重；指标出来后不追加lr/权重/epoch/种子搜索。
- **总GPU上限8小时**，包括检查失败、准备、九次训练、评估和核查。预留2小时给开发推理/报告/核查。按实际完整更新计时预测剩余成本；预算不够保存产物并报告，不删对照、不缩某组预算、不自动扩大上限。
- 只用echo-l40s本项目环境；一条训练队列，`ECHO_GPU_MEM_FRACTION=0.6`。不暂停或修改其他项目的GPU作业；负载只报告已测证据。
- 沿用V01已记录的MViT max-pooling CUDA不确定性说明。6更新/3+3短检查及跨冻结边界恢复用已固定的状态逐位还原和既有漂移规则，不能据本轮结果提高容差。状态可以还原不等于训练权重或性能逐位复现。

## Task 1: 固定协议和辅助标签接入

**Create:** `configs/experiments/video_dense_supervision_v1.yaml`、`scripts/prepare_video_dense_supervision.py`、`tests/test_video_dense_supervision_data.py`。
**Reuse read-only:** raw-video/target-patch配置、RV/SC清单、protocol与源权重。

- [ ] 先核对`docs/EXPERIMENT_INDEX.md`已有A02 pending（无则添加），再提交此任务书和完整固定配置，写明R11/V01/V02/V03/A01前序记录与本轮变化；旧报告与停止规则不改。
- [ ] 先写测试：时间锚点0..14，SC/RV任一像素引用/marker/目标不对应立即拒绝；源清单digest错误拒绝；非train录像拒绝；同类native排列不变；mixed排列label多重集合相同；不受全局Torch RNG影响。
- [ ] Run `.venv/bin/python -m pytest tests/test_video_dense_supervision_data.py -q`，首次应因新模块缺失而失败；只实现最小接入后重跑通过。
- [ ] 新aux标签/排列文件只写`runs/video_dense_supervision_v1/manifests/`。训练-only先生成并冻结三seed的清单摘要。仓库仅写计数和digest，编号/逐窗口标签文件留服务器。
- [ ] 检查每个输入窗口的source pixel digest不变，目标槽标签和原target一致。冻结源权重/清单的身份进入新run identity。

## Task 2: 同一CLS前向、辅助读出和loss

**Create:** `echo_routing/vit/video_dense_supervision.py`、`tests/test_video_dense_supervision_model.py`。
**Reuse read-only:** `target_mvit.py`、`target_patch_readout.token_grid`、原encoder构建与projection/classifier。

- [ ] 先写测试：token grid和人工tokens的8个时间读出；三arm state key/参数数目相同；同state eval的CLS logits与原TargetMViT最大差0；aux标签改变不改变前向；hook不替换token0且异常后删除；lambda0主loss和参数梯度与原CLS CE相同。
- [ ] Run `.venv/bin/python -m pytest tests/test_video_dense_supervision_model.py -q`，确认失败后实现capture hook及上面的loss公式。
- [ ] 保留autograd的norm capture，捕获次数必须恰好1，不把tokens.detach()。只在本次forward注册hook，在finally移除；禁止在检查代码再挂冲突的同一hook。
- [ ] 所有组返回同样的`(z_cls,z_time)`训练输出；推理只走CLS。只在loss外指定本轮aux标签/权重。没有标签/边界/id/native标记进入模型前向。
- [ ] 训练模式同state同RNG的三组在loss前logits和结束后的Torch RNG状态必须相同；native窗口aligned/shuffled loss和梯度相同。aligned aux梯度可传到shared head/projection、marker，以及解冻后骨干。
- [ ] Run两份新测试并提交。规范之外的实现错误可修并复验；不按指标更改设计。

## Task 3: 训练入口、独立审查和直接开跑

**Create:** `scripts/train_video_dense_supervision.py`、`scripts/check_video_dense_supervision.py`、`scripts/queue_video_dense_supervision.py`、`tests/test_video_dense_supervision_training.py`。
**Reuse read-only:** source loaders/digests、lr_factor、freeze_model、既有恢复规则。

- [ ] 先写失败测试覆盖run identity（全部相关代码/来源/标签文件/监督/预算）、完成跳过规则、optimizer真正绑定载入权重、三组批次与RNG配对、预算含失败任务、无开发路径访问。
- [ ] 实现并通过测试；先提交代码再做正式检查。CLI统一用`--config configs/experiments/video_dense_supervision_v1.yaml`，单次训练另用`--arm ... --seed ...`。
- [ ] --check：训练清单顺序的固定16窗口，原CLS eval parity；人工时间锚点对齐；三组100更新数值/梯度检查；六更新重跑/3+3恢复；缩短的冻结边界检查。只要求有限值、正确更新及冻结状态，不设由拟合分数决定的开训门槛。
- [ ] 计时：解冻后每组5次预热+20次完整更新，包括aux、loader和保存估算。记录显存与实际用时，预测9次总成本并保留2小时评估。
- [ ] 独立实现审查必须检查CLS路径未变、共同随机数、SC/RV配对、stem时间锚点、真实/打乱标签、共享projection、loss scaling、标签隔离、状态恢复和预算。修复后重新核对。
- [ ] 检查/审查/预算通过后**直接启动九次训练**，不等负责人回来。用实际进程、更新数和日志确认开跑，再报告“已开始”；不把prepare或队列脚本存在当成训练开始。
- [ ] 全部最终权重冻结到`models_frozen/video_dense_supervision_v1/{center_only,aligned_dense,shuffled_dense}_s{seed}/`；原source冻结模型hash前后不变。

## Task 4: 一次性评估、固定读法与记录

**Create:** `scripts/report_video_dense_supervision.py`、`scripts/video_dense_supervision_tables.py`、`tests/test_video_dense_supervision_report.py`、英文`docs/results/2026-10-13_video_dense_supervision/README.md`、完整表、summary、训练曲线及token示意图。

- [ ] 在产生任何开发输出之前写报告测试和决策函数；提交。测试必须覆盖native/recall失败、不可行B3、只胜shuffled不胜center、机制有信号但系统无收益等情况。
- [ ] 九次全完成并冻结后，才加载本轮开发帧。保持原Sparse stride5、16槽、marker、nearest-centre与原五类/10Hz/0.5s容差。只用CLS概率作为系统输出；aux预测仅作预定诊断，不拼接为新的系统。
- [ ] 每个系统每seed只在calibration选一次原40项B3：h `[5,7,9,11,15]`、q `[.80,.85,.90,.95,.97,.99,.995,.999]`、NMS5。native macro-F1≥.980且multi recall≥.95下取segment F1@50最高，精确并列按原网格顺序。无可行点判失败，诊断设置9/.95不得参与达标。参数冻结/提交后selection仅应用一次。
- [ ] **主比较** aligned−center_only 的multi segment F1@50；必须同时列aligned−Frozen Ordered+B3和aligned−shuffled。另报centre-only、shuffled、冻结Ordered、历史Continued CLS，以及E0/E1/B7+B3、历史/配对ASRF、Original FACT、A01 ASRF/MS-TCN/Linear的原存档行；先复现，不将旧权重接新特征。
- [ ] paired bootstrap沿用2000 draws/seed20261002；native按session、constructed streams按stream，三个seed共享病例。每half“consistent”仍需三个CI严格同向排除0。CI含0不叫相等。stream reuse使区间近似且开发集已反复使用。
- [ ] 报native宏F1/五类recall、A4C→A5C及反向修好/新错来源数、label switches、multi segment F1、boundary P/R/F1、≤16样本段、X-Y-X中段、同类拼接误切。
- [ ] 预定机制诊断：相同mixed窗口的目标准确率（≤7/8/9–11/12–15目标槽及A4C/A5C对）、aux锚点准确率及class-wise recall；参考类别变化处的两侧锚点分类是否都对。计数按源录像/session去重，低于10个单位注明；不以少数稀有病例触发追加实验。锚点指标不等于可部署的逐帧分段结果。
- [ ] 同一源Ordered numeric B3设置上的三arm结果为解码敏感性，单独报告；它不挑候选、不替代主系统。不能把不同calibration工作点带来的系统差异全归于token监督。
- [ ] **原目标不变：** 三seed均值selection≥0.944409687184662、calibration≥0.9821421709894332；calibration敏感性0.9838935784747763；每seed每half boundary recall≥.95、native≥.973/.980，B3可行。不舍入判断。
- [ ] **实际模型进步**另判：aligned−center_only及aligned−Frozen Ordered+B3在两个half的主F1均consistent positive，且候选满足全部约束和目标，才说超过当前强系统。达到旧数值目标本身不足。
- [ ] **位置监督机制支持**另判：在全部mixed窗口（不是只挑8/16子集）的目标准确率上，aligned−center_only和aligned−shuffled在两个half均consistent positive；并且aligned−center_only的multi segment F1@50在两个half均consistent positive，候选在每seed每half满足B3可行、native和recall约束。四项同时成立，才说本配方支持时间对应监督；这个规则不要求该读法本身满足旧F1均值目标，也不等于超过冻结强参照。mixed准确率主bootstrap按目标源session，另按stream作敏感性；若敏感性在任一half有consistent相反方向，该half的机制读法改为不确定。若只系统改善则保留系统结果、机制未定；只胜shuffled不胜center无实用收益。
- [ ] **取舍/无收益**：候选有局部信号但native/recall失败或无主系统收益，按事实报告；本配置结束。不加teacher、boundary loss、gate、new head，不扫lambda、不增加epoch/seed，不据开发病例改采样。三组都无收益也不能推断所有dense supervision无效。
- [ ] 独立核查从保存的窗口输出重算系统/CI/判定/来源去重，并核对冻结权重、pixel计划、标签隔离和时间顺序。保存逐序列输出在服务器，不声称未保存的文件存在。
- [ ] 只提交本轮允许文件、不push；保护theory8文件和所有无关dirty变更。更新`docs/EXPERIMENT_INDEX.md`的A02为实际结果，保留全部前序结论；公开标签/test/Pediatric状态不变。

## 交付与权限

报告写给项目负责人/组员，用英文。完成“实现、准备、开跑、训练结束、已评估、已核查”分阶段报真实状态。训练与检查出错可在既定科学设计内修复、补回归测试并复验，无须等负责人。若改变科学设计、无法满足固定预算或源身份不符，则保存证据并停止依赖步骤；不得默默回退到别的模型。

本文授权的是以上一个新、配对的辅助监督实验。前序闭合配置保持闭合。没有授权发布论文、推送仓库、修改权限/协议、导出患者级资料或开启下一轮自动搜索。

## 计划审查记录

2026-10-13（America/New_York），独立只读计划审查初稿后指出机制判定中的“主系统不退步”缺少可执行定义。已在Task 4明确mixed窗口总体、比较对象、两half的逐seed区间规则、主segment F1要求、每seed约束及stream-bootstrap冲突规则；Task 1也明确先核对A02 pending。复审为Approved。此记录仅证明任务书已审查，不证明代码完成、检查通过或训练开始。
