# FACT on Frozen Video Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Claude is the intended executor. The owner authorizes preparation, routine repairs, checks, the three formal runs, freezing, one staged evaluation, independent verification and round-only commits within this task. Do not wait for the owner to return once the checks and budget projection pass.

**Goal:** 检验官方片段-token分段系统在已有效的视频表征上，能否超过相同输入的ASRF、线性分类加简单分段，以及冻结Ordered强参照。

**Architecture:** 只训练官方GTEA FACT，读取A01缓存中的冻结Ordered 512维视频特征。每个seed使用自己的坐标、既有训练序列和批次计划；正式训练三个head，不重训视频编码器或既有对照。用官方输出做主评估，不加边界细分、残差、辅助loss或B3。

**Tech Stack:** PyTorch; pinned official FACT and existing FACTHead adapter; A01 float32 sequence caches; half-bounded development reader; pytest; English results README.

---

## 1. 负责人决定、依据和新问题

A05关闭。其原任务书下的局部转录改善和native损失已通过带日期的说明补记；原操作性flags和结果不改。不延长A05、不扫描loss权重、不把A04的重叠窗口现象直接变成一致性损失。

本轮编号 **A06**，是**新表征与已有分段模型的组合比较**，不是原创模型声明。

- **V01**：多帧视频输入相对重复目标帧有一致收益；真实顺序超出同帧打乱的额外价值未证实。冻结Ordered+简单解码仍是强参照。
- **F02**：FACT在E1逐帧图像特征上没有一致超过ASRF。**F03/F04**发现和检验的是这套输入下的定位/细分，细分路线已关闭。
- **A01**：强视频特征上已试线性头、MS-TCN、ASRF，没有试FACT。ASRF有召回损失，不能将这个结果外推到全部片段-token模型。
- **A02–A04**：窗口内token分类改善未转成分段收益；不是整条流上的FACT比较，也没有证明跨窗口一致性训练有效。
- **A05**：只改冻结特征上线性头的训练目标，未显示合格系统增益；没有检验本轮模型。

**新问题：** 在完全相同的视频特征与训练流上，官方整流FACT是否比A01的ASRF及线性系统更好，并找回强参照漏掉的段？这是未检验的问题，不预先承诺有效。A06与F02的差别是输入从逐帧图像特征换为16帧视频上下文特征；与A01的差别是完整下游模型。FACT与ASRF的结构、优化器和增强不同，结果只能归于整套配方，不能单独归于cross-attention或某项loss。

方法来源：[FACT论文，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html)及[官方仓库](https://github.com/ZijiaLewisLu/CVPR2024-FACT)。不追踪上游最新版，沿用已接入提交。

## 2. 固定来源和数据边界

以下为本地已核对的完整SHA256；配置在任何新计算之前照抄并核验。报告文案可另增带日期说明，不用README的措辞变更替换运行来源。

| 来源 | SHA256 |
|---|---|
| `configs/experiments/video_feature_temporal_v1.yaml` | `3b065e7fe908ca7432ba0a407fd6b82432882c1d72dabd1d22b899b79d790690` |
| `docs/results/2026-10-12_video_feature_temporal/preparation.json` | `5f7d48db3a3b2f54f0788d9dc66a53dae2ff3dbf5105d9c1033b78a2f417d021` |
| `docs/results/2026-10-12_video_feature_temporal/summary.json` | `3a1601512bc423a6ad97b5c88ed29e8ebf949a6df802bc0a76fb6fc13c503acf` |
| `configs/experiments/fact_asrf_paired_v1.yaml` | `5caffd566fd560593c5d60e829446fe0ecc97debd59cbd5c627ce13aec421ab2` |
| `docs/results/2026-10-08_fact_asrf_training/summary.json` | `32c2665f485981d8b38d9c63c283027b1fd02568d04c36816d162fe88066bafa` |
| `configs/experiments/segment_objective_v1.yaml` (A05 source pins only) | `3f6980942cdf1ff6aec550608605888cd6dac1a3ea454d2442e927a5762826bd` |
| `docs/results/2026-10-14_segment_objective/preparation.json` (full cache hashes) | `b02b362da058a93f2c29ce20915e0e7cc0519eb83e6396dcb638716693dbb735` |
| `configs/experiments/trained_token_readout_v1.yaml` (metadata adapter sources) | `c2e90d605316a2c7d9679728464dbf3a534a7eafc6ea0760a966025be72fc8b4` |

- 只读A01已有 `runs/video_feature_temporal_v1/features/train/ordered_s{k}.npz`、`train/sequences.npz`、`plans/plan_s{k}.json`。配置继承A01的protocol、source weights、bank digest、类别顺序、gamma和224×224输入来源，不生成新像素或新表征。
- 特征是完整构造流先拼接再编码得到的 `(T,512)` float32；stride5请求后nearest-centre映射至10Hz，保留重复行。不换Dense、不按参考边界重置窗口、不归一化特征、不混用三个编码器坐标。
- 按A05的source/preparation记录核验训练文件完整SHA、identity及序列文件SHA；development缓存训练前只可整文件哈希，禁止打开数组或标签。把新核实的完整SHA写入本轮preparation和identity，不冒充旧preparation已经包含未记录字段。
- 用A01准备记录和summary解析计划、初始态及冻结ASRF/Linear/MS-TCN的路径和SHA；冻结Ordered的权重/存储输出/原B3设定继承A01配置。缺失或不匹配须定位后停止，不重新挑来源。
- 训练划分3364原生录像及1500构造流；每个epoch375批，每批4条流+4个原生裁剪，原生最多256样本；50epoch，总18750更新。逐批结构digest与A01计划一致；逐元素抽查float32特征。每个seed只用该seed的输入。
- metadata只读允许split的标签；其他split仅允许编号、split、样本数用于隔离。模型forward不接收参考边界、类别、transcript、原生标志或编号；官方接口要求的推理标签固定全零。

## 3. 模型、初始化和固定训练配方

**恰好三个正式训练：FACT_s0、FACT_s1、FACT_s2。** ASRF/Linear/MS-TCN/Ordered复用冻结模型和存档设置，明确标作历史冻结对照；不重训它们，不宣称与新模型同时训练。

- 使用 `echo_routing/temporal/fact_head.py::FACTHead`，官方提交 `7bd81bda2b84618a3e23876a2637a82f24881122`，`configs/default.py + gtea.yaml`。官方源文件保持不变。prepare除查HEAD外，还要确认官方tracked文件无修改并保存实际所用源码digest；有差异即定位停止，不reset别人的修改。
- 512维输入、5类、无背景类、`trans=false`、`match=o2o`。从训练标签重算最长transcript=8、平均=2.654333333333333、ratio=1.5、token=12；相邻同类合并，X-Y-X保留三个段。null权重仍用官方规则。任何不符都不是改token数的理由。
- 初始化复用F02的 `runs/fact_asrf_paired_v1/inits/FACT_s{k}.pt`，与其summary中`runs/FACT_s{k}/init_sha256`核验，再严格加载。只复用未经训练的初始态，**不加载F02最终或过拟合权重**。配置展开、tensor名和shape必须完全吻合。
- F02的官方dropout、channel masking、temporal masking、matching和每项loss权重不变。Adam lr=1e-4，betas=(.9,.999)，eps=1e-8，weight_decay=0，grad_clip=10。完成11719更新后lr乘.1；无warmup，最终18750更新权重，不选epoch。
- fp32、TF32关闭、cuDNN deterministic、`torch.use_deterministic_algorithms(True)`；沿用F02确定性设置。优化器绑定实际训练对象；恢复包括优化器、scheduler、RNG和completed更新数；错误身份拒绝续训。
- 每epoch固定监测A01的前8个原生裁剪和前8条构造流，记录各项loss、frame accuracy、segment F1、transcript及梯度/LR。只作训练拟合，不选点、不延长。监测集与F02原监测集不同，报告说明，不声称逐位重放F02全过程。
- 官方最终block解码为主输出，不接B3、不搜索mwt/切换代价/threshold。不加载真实推理标签。帧分支直接argmax只作同权重读出诊断，不称去token消融。

## 4. 文件与任务

新增文件集中在本轮，不改旧模型、缓存、choices或运行代码来迁就新结果：

- `configs/experiments/fact_video_feature_v1.yaml`：来源、三次训练、预算、完全列明的判读条件。
- `scripts/train_fact_video_feature.py`：prepare/check/queue/单run/freeze；组合A01训练adapter及FACT官方wrapper，禁止调用旧prepare的特征提取入口。
- `scripts/report_fact_video_feature.py`：只打开允许half、参考复现、官方预测存储、统计和判读。
- `tests/test_fact_video_feature.py`、`tests/test_report_fact_video_feature.py`：来源/配对/隔离/恢复/边界条件及所有判读分支的表驱动测试。
- 结果 `docs/results/2026-10-15_fact_video_feature/`；运行 `runs/fact_video_feature_v1/`；冻结 `models_frozen/fact_video_feature_v1/FACT_s{k}/`。

### Task 1 — 固定配置、红测试、实现

- [ ] 配置先提交，写入任务书SHA和上表pins，加入A06 pending索引；补齐来源SHA由固定preparation/summary解析，不能按成绩选。
- [ ] 先提交针对本轮接口的失败测试，运行 `python -m pytest tests/test_fact_video_feature.py tests/test_report_fact_video_feature.py -q` 确认因缺实现失败，再实现。禁止为了“红”破坏现有文件。
- [ ] 复用 `scripts/train_video_feature_temporal.py::TrainSet`、`echo_routing/temporal/video_sequence_features.py::Batches` 的训练规则，逐批核对A01plan；不要调用会建模或重提特征的准备路径。
- [ ] FACT loss前按length去padding，保持 `(T,512)`，与从源缓存直接裁剪的序列逐元素一致。测试改变padding标签/特征不改变有效loss及预测。
- [ ] queue只跳过identity完全相符且全部更新完成的run；同名不同身份拒绝。identity覆盖本轮配置、相关执行源码、官方commit/展开配置、缓存hash、批次和初始权重。
- [ ] 相关测试通过后提交实现，再做独立实现审查；阻断项/应修项修复并复核，不能把“审查已排队”当通过。

### Task 2 — prepare/check与直接开跑

- [ ] `python scripts/train_fact_video_feature.py --config configs/experiments/fact_video_feature_v1.yaml --prepare`：核验来源、保存每seed计划/初始态、训练数据结构及所有source hashes；development只整文件哈希。
- [ ] `... --check`：验证训练抽样/全部批次结构、12token规则、初始化严格一致、有效padding隔离；在初始和过拟合状态都用五种假标签验证推理输出不变。
- [ ] 6update重跑及3+3恢复差≤1e-6；CPU或GPU比较策略、具体对象和误差须事前固定，遇不确定算子不放宽科学标准来开跑。
- [ ] 用A01固定16训练序列做独立smoke，最多2000更新。要求每条transcript正确、segment F1@50=1且每个匹配token正确；官方loss不要求趋零。未过则保存实际fit曲线并停止本轮，不追加updates/换优化器。smoke权重不用作正式初始化。
- [ ] 从同样初始态计时5warmup+50正式batch更新，估算训练、每epoch监测/存盘、推理和核查成本。用seed0实测/各seed长度核算；预计全程含1.25GPU小时预留≤12小时才允许启动。
- [ ] check/审查记录提交后，直接 `... --queue`，无需负责人回来。启动后读实际进程、日志与完成更新数才报“正式训练开始”。
- [ ] 三个正式run结束后核验18750更新、final SHA及完整manifest，使用项目冻结工具；全部冻结后才能打开development数组。

### Task 3 — 一次分阶段评估

**来源读取：** 沿用A05 `scripts/report_segment_objective.py::DevAccess, dev_half` 的bounded seek reader及A03 metadata helpers（`half_index, meta_stream, half_sets`）；它们可被只读导入。不调用 `validation()`、`dev_run.Inputs`、A01的eager development loader或encoder fallback。

- [ ] 测试I/O spy：calibration阶段只读calibration特征/标签；selection被拒绝，直到committed evaluation manifest的git-object proof校验通过。combined NPZ的目录/编号和offset可读；另一half的特征/标签行不可读。
- [ ] `python scripts/report_fact_video_feature.py --config configs/experiments/fact_video_feature_v1.yaml --calibration`：全体新权重冻结后，只评估calibration，重现A01 ASRF/Linear/MS-TCN及Frozen Ordered存档行；不选FACT设置、epoch或候选。reference完整校验失败即停。
- [ ] 所有参照保持原解码参数。A01 Linear与A05 F相同，不重复作为独立对照。再列E0/E1/B7+原B3及F02 FACT；F02对照标为完整输入来源替换，不能归于单一视频运动因素。无法加载的参考不默默略去。
- [ ] 校准阶段dry-run报告和表驱动判读测试通过后，将全体模型/source sha、固定官方解码、参照设置及判读定义写入 `evaluation_manifest.json` 并提交。这里没有候选选参；不为了形式造choices搜索。
- [ ] `... --selection --manifest-sha256 <sha> --manifest-commit <commit>` 在server验证该提交确含manifest的git-object proof，然后selection仅运行一次；proof可从本地复制到server，**不push**。中断时只可按相同identity复用已存输出，不重新选参数。
- [ ] 每half列native/pair/multi。主比较 **A06 FACT − A01 ASRF** 的multi segment F1@50，按seed配对。必列 **FACT − A01 Linear+B3、FACT − Frozen Ordered+B3、FACT − A01 MS-TCN**。F02与弱参照只作次要报告。
- [ ] 配对bootstrap 2000次、seed20261002；native按session、constructed按stream，95%区间。每half三seed严格同侧排除0才叫consistent；否则保留方向和区间，不称等价。六项固定指标：multi segment F1@50、boundary F1、boundary recall、native macro-F1、native内switch/clip、same-view-join cut rate。FP/FN计数及其配对区间另列。
- [ ] 报告救回/新丢匹配段及FP，按来源录像/session去重；≤16样本短段及A4C/A5C X-Y-X中段单列原始分母，少于10session标小样本。native两向混淆分别数。不可把三seed病例当三倍独立样本。
- [ ] 训练16监测曲线之外，正式结束后在全部1500训练流作一次官方解码，报告训练拟合、漏段长度/类别；只定位欠拟合，不因曲线自动加预算或跑Breakfast。无需重新跑F03分块/参考边界诊断。

### Task 4 — 独立核查和关闭

- [ ] 独立核查从保存的预测重算系统行、主/必列区间、targets、全部判读、恢复/新丢及来源数；少量GPU复推核验冻结权重。用新写的统计核对，不只调用原报告函数。
- [ ] 英文README写明：实际训练的是三个FACT head，视频编码器未更新；其他head是历史冻结对照；token示意、固定配方、训练拟合、成本、偏离、约束、效果和不能归因的机制。
- [ ] 修改只限本轮产物和索引；结束时将A06 pending换为实际结果。原始媒体、编号、逐序列输出只留server；repo只汇总数与hash。保留所有历史配置、结果、choices和停止记录。

## 5. 完整判读规则（非互斥，计算前原样落配置并测试）

以下所有“consistent positive/negative”都是**同一half三个seed的95%区间严格同侧**。按每个half分别报告，不因其他half相近而忽略失败。

1. **complete_system_progress**：FACT对ASRF、Linear+B3和Frozen Ordered+B3，在两个half的主F1均consistent positive，且满足全部约束和原目标。至多称该开发协议上的系统进步，不称原创、临床验证或cross-attention因果贡献。
2. **learned_system_gain_without_strong_reference_win**：对ASRF和Linear均在两个half consistent positive，但未同时满足对Frozen Ordered的比较、约束或目标。明确哪一关未过，不称总体胜出。
3. **one_half_or_local_effect**：任一half任一必列F1比较consistent positive，或只在短段/恢复数的点估计改善；逐项列指标、分母和区间。局部点估计与consistent统计证据分开，不把次要机制指标当主成功。
4. **trade_off**：任一seed/half相对同输入Linear或Frozen Ordered存在救回段，同时有新增未匹配预测、native下降或任一约束失守；列对应seed/half计数和区间。此flag描述共存，不要求跨seed显著，也不证明因果；没有该模式便不应用。
5. **uncertain_or_no_qualifying_gain**：完整进步条件不成立即记录，并报告主比较在各half的方向；不把不显著称相等，不将一个模型配置否定整个家族。
6. **end**：一次报告后该官方GTEA/视频特征/预算组合关闭。无额外种子、Breakfast、增token/层数、encoder微调、loss/优化器/epoch/decoder搜索、参考边界、boundary细分、teacher、gate或融合自动续做。

所有目标照旧：三seed均值multi segment F1@50 selection≥**0.944409687184662**、calibration≥**0.9821421709894332**；calibration sensitivity **0.9838935784747763**。每seed/half boundary recall≥**.95**，native macro-F1≥**.973/.980**。FACT无B3可行选点条件，因为其主输出没有B3；不得因此放宽recall/native下限。超过旧目标也不能代替对强参照的比较。

## 6. 预算、权限与保持事项

**总上限12 GPU小时 + 4 CPU执行小时**，包括失败检查、smoke、三次训练、训练流/开发推理、独立核查，账本追加，不漏计、不双计重叠阶段。正式队列前预留至少1.25GPU小时给评估核查；每run启动前重估全部剩余任务，超额即停存进度。只共享空闲资源，遵守项目GPU配额，不干预其他项目进程。

运行中新文件在独立目录，source只读；科学范围或预算变化不授权。常规实现修复、恢复和同身份中断续做已授权。无需在检查通过后再等负责人。test继续封存，Pediatric不读，公开标签不改，论文暂停；8个暂存理论文件及其他未提交改动原样保留。不要push。

限制必须写入README：开发集反复使用，session不是patient，构造流不是连续扫查，少数录像反复出现，native标签非专家逐帧标签。这个任务是检验尚未试过的组合，绝不是要求制造一个正结果。

## Plan review record

独立计划审查于2026-10-15完成，结论 **Approved，无阻断问题**。审查核对8个来源pins、F02初始态与512维/12token配置的兼容性、A01批次、半区隔离和manifest屏障、预算及判读。建议额外检查官方checkout的tracked修改，已纳入第3节。正式实现审查、server端严格加载和检查尚未执行；本记录不代表训练启动或模型结果。
