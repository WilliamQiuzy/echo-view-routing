# Target-Anchored Video Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Claude is the intended executor. Keep the three matched readouts and the fixed stopping rules below.

**Goal:** 在已取得多帧收益的 MViT 上，检验把最终读出绑定到目标时间位置，能否减少跨切面上下文造成的目标误判，并保留原生录像识别与完整分段收益。

**Architecture:** 同一个已训练的 TargetMViT，保留全部16帧、目标标记、全局注意力、512维投影和五类头。只改变最终归一化token的读出：CLS、全部patch平均、目标时间位置的patch平均。三组参数数目相同，无新注意力、gate、边界头或辅助损失。

**Tech Stack:** 现有 torchvision MViTv2-S / TargetMViT、raw_video_context_v1 的训练窗口与像素摘要、PyTorch、项目评估与冻结工具、pytest、英文 README。

---

## 决定与证据范围

上一轮保留为正结果：多帧识别收益成立，Ordered + B3 对 E1 + B3 的系统收益在两个半区一致。真实顺序的额外收益未检出；完整目标未达到。不能把校准半区0.0004的差距当作可忽略误差，也不能放宽种子2原生分类下限。

本轮依据是另一个明确现象：混合窗口中目标切面占8/16时，多帧组目标准确率为0.53–0.70，而目标帧重复为0.95–0.99；目标占12–15帧时，多帧组较好。这些旧计数没有区间，来自构造流，只支持检验上下文覆盖目标的假设，不证明某个token或层已被定位为原因。

本轮问题：**使用同一套视频patch特征时，目标位置读出是否优于不区分时间位置的读出？** 不缩短输入、不丢上下文、不更改训练窗口，不搜索阈值来追逐门槛。边界平均偏移约1.5个样本的采样网格问题暂不处理，推理stride仍为5。

此任务是一次已有视频模型的读出适配与机制比较，不预先称为论文创新。即使目标读出成功，也不能证明跨界混合仅发生在最后读出：patch已经经过全局注意力和时间卷积，仍可能包含其他切面。

## 三组与配对初始化

| 报告名 | 最终768维读出 | 控制的因素 |
|---|---|---|
| Continued CLS | 最终norm输出的CLS token | 同预算继续微调的现有读出 |
| Global patches | 所有时间、空间patch的均值，不含CLS | 换成patch特征本身的影响 |
| Target patches | 目标位置对应时间槽内的空间patch均值，再按固定目标权重合并 | 在相同patch特征上绑定目标位置 |

**主候选固定为 Target patches。** 主机制比较为 Target patches − Global patches；实用比较同时包括 Continued CLS 与上一轮冻结的 Ordered。不能在结果后把赢的对照改为主候选。

每个种子三组都从该种子上一轮 Ordered 的同一完整权重严格加载，包含编码器、512投影、已学习的marker和分类头。**不重新初始化头或marker。** 新优化器与新3750更新计数均从0开始，这属于配对继续微调，不是恢复旧优化器状态。

| seed | 来源 `models_frozen/raw_video_context_v1/ordered_s{seed}/final.pt` SHA256 |
|---|---|
| 0 | `80f13a42e628fe2647634b79e59d5b8486fee5c3e0ae45f70bcc667c41258505` |
| 1 | `74276323ababf78fcb578be28d3172f3c220b3b7e66c2ff96a2072f39210c26f` |
| 2 | `923eb3afb0d846b1361897dbb63bb5aecf22401ce4b305b44fe990d48edc7f2c` |

## 目标patch读出：明确实现

现有MViT执行 `conv_proj -> pos_encoding -> blocks -> norm -> x[:,0] -> head`。本轮在项目wrapper中对 `encoder.norm` 注册临时hook，只替换送给原head的第0个token；保留其他token、不修改官方代码或已完成轮次的脚本。hook必须在finally移除。Continued CLS不替换，输出需与原模型逐位一致（eval / 同样输入）。

1. norm输出为 `X: (B, 1+T'*H'*W', 768)`。丢掉第0个后reshape为 `(B,T',H',W',768)`。从实际源conv、pos_encoding和各block query stride计算形状并断言一致。当前固定MViTv2-S应为 `T'=8,H'=7,W'=7`；所有block query时间stride均为1。若源实现不符，停止并报告，不偷偷硬编码reshape。
2. 先作空间均值：`P = patches.mean((2,3))`，得到 `(B,T',768)`。Global patches为 `P.mean(1)`。
3. 从输入marker得到 `(B,16)` 的单个1位置。用**全1的固定1D卷积核**按RGB stem的时间kernel/stride/padding/dilation下采样marker，不引入可训练参数。当前几何为kernel3、stride2、padding1、dilation1。得到非负 `w: (B,T')`，按时间归一化到和1。
4. Target patches为 `sum_t w[t]*P[t]`。偶数位置如slot8对应stem时间槽4；奇数位置如slot9对应槽4/5，各0.5。slot0/15、短序列重复和非中央目标也必须按该规则计算，不能用类标签或固定slot8定位。
5. 将选出的768维向量写入 `X.clone()[:,0]`，调用原dropout、512投影、五类头。严禁detach。marker的学习分支保持原样；固定读出权重不能替代输入marker或读取参考切分。

位置权重仅绑定网格位置，不是patch真实感受野或解剖范围。投影原先是在CLS上训练的，patch读出会改变其输入分布；Global patches对照用于区分该变化与目标绑定，不能只凭Target−CLS声称位置机制成功。

## 固定数据、训练与预算

- family5、224×224、16帧、10Hz、目标中心标签、像素预处理完全继承 `configs/experiments/raw_video_context_v1.yaml`。只使用 Ordered 的图像顺序。
- 复用 `runs/raw_video_context_v1/manifests/index.json` 与各 `RV_s{seed}.npz`。训练允许名单、源清单、每窗口Ordered像素摘要均核对。三组同种子使用同一批次、帧、gamma、marker和目标。每更新4原生+4混合窗口；3750更新、batch8、最终权重。
- 种子0/1/2，三组共九次。唯一损失仍为目标五类CE。类别/录像采样不变，不添加loss、蒸馏、class weights或标签平滑。
- 优化器、学习率、dropout、stochastic depth、fp32/TF32/确定性设置继承上一轮：encoder含投影lr1e-5；marker+分类头lr1e-4；AdamW、wd1e-4、clip1；前375更新冻结encoder但保持autograd，此后全解冻；375 warm-up与3750步cosine。
- 各组开始时都重置为同一个源state并设模型RNG为seed；数据RNG独立。新模型只能复制冻结版本，不能在models_frozen中覆盖。
- 开训后不读开发帧或计算开发指标，九次全部结束并冻结后统一评估。旧汇总用于本任务依据与来源核对，不挑新epoch或初始化。
- **本轮GPU上限12小时**，包括检查、九次训练、冻结和评估，不沿用上一轮未花完的预算。完整更新计时和评估预留2小时预测在内，且实现审查/检查通过后，Claude可按这份负责人任务书直接启动；如超预算保存状态并报告，不删对照、不缩减某组更新数、不自动扩预算。
- queue顺序固定：seed0 Continued CLS / Global patches / Target patches，然后seed1、seed2。沿用单GPU任务与项目显存限制，不干扰其他项目。

## Task 1: 记录协议、实现纯读出与wrapper

**Create:** `echo_routing/vit/target_patch_readout.py`、`configs/experiments/target_patch_readout_v1.yaml`、`tests/test_target_patch_readout.py`。

- [ ] 先提交任务书与固定配置。记录源权重、源训练清单、环境、源报告与所有引用文件digest，不改上一轮结果、规则或任务书。
- [ ] 先写pure-function测试：用每时间槽取常数的token检查Global均值与Target权重；slot0、8、9、15的权重；重复目标只使用现有单平面marker；CLS排除与总权重1；不同目标marker选择不同位置；标签/来源名无法作为函数参数。
- [ ] Run `.venv/bin/python -m pytest tests/test_target_patch_readout.py -q`，先确认新功能未实现时失败，再实现上述固定读出并确认通过。
- [ ] wrapper严格加载完整旧state，检查参数名称、数量、起始权重三组一致；不新增任何参数。保存真实token形状与输入token图。
- [ ] 用训练窗口比较 Continued CLS 与源模型的eval输出，最大差0；Global/Target手工重算768维与512输出，fp32最大差≤1e-5。hook在正常/异常路径均清除。
- [ ] 记录所有源query时间stride为1。读取真正的网络结构，不根据参考类别、真实边界或窗口“纯/混”标志选择readout。

## Task 2: 新运行入口与启动检查

**Create:** `scripts/prepare_target_patch_readout.py`、`scripts/train_target_patch_readout.py`、`scripts/check_target_patch_readout.py`、`scripts/queue_target_patch_readout.py`。
**Reuse without editing:** raw-video帧/清单读取、现有窗口与冻结工具。不要直接套用旧Ctx而把产物写回旧round目录。

- [ ] 独立输出为 `runs/target_patch_readout_v1/`、`checkpoints/target_patch_readout_v1/`、`models_frozen/target_patch_readout_v1/{cls,global_patches,target_patches}_s{seed}/`。run identity必须覆盖readout、源权重、完整相关代码、模型实现、环境、清单、像素摘要、loss、优化器和3750更新。
- [ ] 所有train rows先允许名单筛选再读取标签/帧；test/Pediatric不读。检查新三组输入摘要逐窗口相同，已学marker正确载入。
- [ ] 用训练清单顺序固定16个窗口覆盖短序列、端部和native/mixed，运行三组各100更新的数值/梯度冒烟（源encoder冻结，恒定marker/head lr1e-4）。报告loss、accuracy但不设置拟合满分门槛；不据结果改readout或配方。
- [ ] 检查projection/head/marker梯度路径；冻结期间encoder不变，解冻后实际更新；不能对encoder使用no_grad。state恢复需覆盖所有参数、optimizer、step和CPU/CUDA RNG。
- [ ] 复用上一轮已记录的非确定性算子检查规则与固定操作性容差，区分逐位恢复与后续训练漂移。不能再用观察到的差值提高容差，不能把漂移检查当性能稳定证据。九次均记录是否中断。
- [ ] 在三组实际解冻batch8上各预热5次、计时20次完整更新，含解码与hook；预测九次训练和完整评估成本。预算通过后提交检查记录。
- [ ] 独立实现审查：重点检查token展平顺序、stem目标对齐、Global/Target只差时间权重、完整state配对与无开发信息输入。修正实质问题后再开跑。
- [ ] 完成以上检查后执行：`.venv/bin/python scripts/queue_target_patch_readout.py --config configs/experiments/target_patch_readout_v1.yaml`。最终权重和运行记录确认真实完成再冻结。

## Task 3: 一次性评估与报告

**Create:** `scripts/report_target_patch_readout.py`、`tests/test_target_patch_readout_report.py`、`docs/results/2026-10-11_target_patch_readout/README.md`、完整表/summary与曲线。

- [ ] 九次完成后才载入开发帧，使用旧stride5和同一centres/16帧/marker。报告前原设置复现上一轮三个Ordered系统与旧强基线；旧逐窗口输出只作固定参照，不选择新参数。
- [ ] **主机制指标：** 参考目标类恰占8/16的混合窗口上，五类argmax目标准确率的 Target patches − Global patches。另报Target−CLS、所有mixed、≤7/9–11/12–15各组、A4C/A5C类别对与端部。不能在报告后改主比例组。
- [ ] 给上述混合窗口补配对bootstrap：按目标来源session抽样，2000次、seed20261002；同时按整条构造流抽样作为敏感性。context/录像复用仍未完整建模，区间近似。每类来源录像/session、每段贡献与修好/新错计数都报告；没有足够类别/抽样支撑则标不可估，不换组。
- [ ] 原生A4C/A5C balanced accuracy、每类recall、AUC、全原生目标accuracy、native宏F1与误切都报，按session配对；尤其单列A4C→A5C与A5C→A4C，不能只报A5C找回。公共录像标签不是专家逐帧真值。
- [ ] 映射仍nearest-centre；先报argmax。各新系统按原40项B3网格在calibration各选一次：half_window `[5,7,9,11,15]`、quantile `[.80,.85,.90,.95,.97,.99,.995,.999]`、NMS5；native≥.980且multi边界recall≥.95下segment F1最高，同分旧网格顺序，无可行点则失败，不改回退。
- [ ] 新参数冻结后selection只应用一次。另用各源Ordered的原numeric B3设置，对同种子三组作固定解码敏感性：h9，threshold s0=.29269428317686846、s1=.2845063773784935、s2=.30845250187026374，NMS5。新系统概率标度可能不同，此项不能代替正式选点或达标判断。
- [ ] 主完整系统为Target patches+B3。比较Continued CLS、Global patches、冻结Ordered、E0/E1/B7+B3、历史/配对ASRF、Original FACT和G1+ASRF。当前冻结Ordered也是强基线，不以只超过早期弱系统宣称超过全部基线。
- [ ] 多段segment F1@50、边界P/R、≤16样本段、X→Y→X中段、同类拼接误切和边界偏移沿用原paired bootstrap，逐seed逐half报告。三个seed仍是同一批病例，不是三批患者。
- [ ] GPU/推理成本、来源digest、训练曲线、偏离、全部模型输出保存位置与命令齐全；逐录像输出/媒体/编号只在服务器，仓库只有汇总。
- [ ] Run `.venv/bin/python -m pytest tests/test_target_patch_readout.py tests/test_target_patch_readout_report.py -q`。独立核查从保存输出复算主比较/约束/修好新错、重跑少量推理。仅提交本轮文件，不推送，不碰理论稿/论文/许可/演示/其他未提交文件。

## Task 4: 固定解释与停止表

沿用一致差异口径：在某半区三个seed的95%区间严格同向排除0才称一致；区间含0不是等价。按目标session的主区间决定，stream敏感性若方向明显冲突则降为不确定，不能任选有利抽样法。

1. Target−Global patches在两个半区的主混合指标一致为正，且Target−CLS也一致为正：支持本配方下目标位置读出改善跨切面目标识别。仅一个半区成立则限该半区，不扩大为总体机制。
2. Target仅优于CLS而不优于Global patches：不能把收益归给目标位置；可能来自patch读出或优化。结束本比较，不自动加attention/gate。
3. Target改善mixed但原生退步或系统未改善：报告上下文鲁棒性与原生分类的权衡。结束本比较，不加辅助loss补救。
4. 没有一致机制或系统收益：结束这一readout/训练预算配置，不否定所有目标条件模型。
5. 达到旧目标与超过新的强基线分开判断。目标仍为三seed均值selection≥.944409687184662、calibration≥.9821421709894332；每seed每halfrecall≥.95，native≥.973/.980；另报calibration敏感性目标.9838935784747763。不能因为已接近就降低或按舍入达标。
6. 声称新方法胜过已有模型，另需Target对Continued CLS及冻结Ordered完整系统的配对收益证据；只达到旧数值门槛不自动证明这一点。所有开发结论仍探索性；test不自动打开。

报告保留上一轮多帧与系统正结果，不因本轮失败改写它。论文暂停；FACT细分路线继续关闭。无dense stride、更改loss、阈值扩网格或追加seed/epoch/新模块。

## 实现依据

- `echo_routing/vit/target_mvit.py`：上一轮marker接口与512投影/五类头。
- 当前固定环境 `torchvision/models/video/mvit.py` 的 `MViT.forward`：最终 `norm` 后读 `x[:,0]`；MViTv2-S的query时间stride均为1。须记录服务器实际实现digest，不能把本机版本摘要当服务器摘要。
- `docs/results/2026-10-10_raw_video_context/{README.md,summary.json,report_tables.md}`：多帧正结果、mixed目标覆盖、旧约束与来源模型。
