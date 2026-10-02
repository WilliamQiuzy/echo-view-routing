# 第二轮：视觉表示与联合训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 交给 Claude 执行；本轮已决定继续训练，不需要再次询问是否启动 Task 5a。

**Goal:** 训练出超过当前强基线的模型；本轮检验更强视觉表示能否改善识别，以及可学习分段模块是否仍有独立增量。

**Architecture:** 保留已有 ASRF、M3p 和 M5，通过更换同一视觉输入做受控比较。先提取 EchoViewCLIP stage 1 的逐帧视觉特征并训练时序头，再按下文条件进行一次配对的视觉末块联合微调；不再扩充时序模块或继续堆解码网格。

**Tech Stack:** 现有 PyTorch 主环境、EchoViewCLIP 独立环境、版本化缓存、family5_v1、10 Hz、第一轮 protocol。

## 决定与证据边界

本计划承接 commit 36cd881 的第一轮结果，取代旧任务书“必须先超过 ASRF 0.5 个百分点才能尝试更强编码器”的启动限制。它授权一次新的编码器对照实验，不改变第一轮失败结论，也不降低最终胜出标准。论文定稿继续暂停。

- M4 未超过强基线；M5 与 ASRF 未检出跨种子一致差异，不等于已证明等效。M5 的均值仍较低，selection 上 B3 也是强对照。
- 共同错判少数 A5C 片段支持检查视觉表示，但不能排除整段标签与局部证据不一致。约 0.95 不是已证实的模型上限。v3 encoder 最佳 epoch 为 0，也是需要保留的训练现象，不足以单独证明实现错误。
- M3p 已有边界监督和边界控制的输出池化；它只去掉 attention 中的路径项。M5 若无法超过 M3p，就不把路径项当贡献。
- 第一轮 20 倍是不同 batching/解码实现下的时序头计时，不能作为同条件架构加速或完整系统加速结论。
- 本轮不改标签、不删除难例、不反复重新拆验证集、不读取 test。两半 validation 都参与过开发，后续结果均为探索性开发结果。

## Task 1 — 冻结增补协议，同时准备特征抽取

**Files:**
- Create: configs/experiments/model_development_v2.yaml
- Modify: scripts/model_development_queue.py（新增 --config、实验标识和恢复校验，旧默认行为兼容）
- Create: tests/test_model_development_round2.py
- Reuse: echo_routing/evaluate/dev_metrics.py、第一轮 protocol.json

- [ ] 保留第一轮 protocol、两半会话、各自 banks、1,500 条训练流、标签映射、帧索引和 gamma 变体；新配置记录父协议摘要与本轮新增字段，不能就地改旧协议。
- [ ] 沿用 checkpoint-selection 与 calibration 的职责、native-F1 下限（各编码器逐帧投票基线减 0.005）、召回下限 0.95 和原有解码网格。本轮启动前固定第一轮报告的回退顺序：满足两项、仅满足召回、仅满足分类、均不满足。所有未达标行继续明确标记；回退不能算通过最终目标。
- [ ] 所有方法同时报告默认解码和同一 B3 网格解码，两者各自成行，不跨行挑最佳指标。ASRF 另保留官方边界 refinement 默认行；不新增解码家族或事后扩大网格。
- [ ] checkpoint 只在 selection 上选，decoder 只在 calibration 上选。所有方法同一选择规则。候选架构的后续选择按下文固定规则，不能把 calibration 说成未看过的确认集。
- [ ] 队列必须支持新配置，运行身份包括 config/protocol/features/head/seed 的摘要；已完成作业只有全部匹配才能跳过。独立日志与输出目录不得按秒数单独命名，也不得覆盖第一轮 status。
- [ ] 用短 CPU 测试确认：更换 feature/config 摘要不能误复用完成状态；缺少 train/validation 样本或读入 test 时拒绝执行；维度从 manifest 读取。

## Task 2 — 有限的错误归因；不等待它完成才抽取特征

**Files:**
- Create: scripts/model_development_oracle_diagnostic.py
- Create: tests/test_model_development_oracle_diagnostic.py
- Read: 第一轮冻结预测、echo_routing/temporal/dev_heads.py、echo_routing/temporal/models/boundary_context.py

- [ ] 对 B0、ASRF、M3p、M5，三个 seed、两个半区分别报告原生片段分类与构造流分类，不能用构造流逐帧优势推出原生片段分类最好。
- [ ] 在构造流上提供一个明确标为 ORACLE_DIAGNOSTIC 的结果：用代理参考的真实类别变化位置分段，段内平均该模型类别概率后预测，再合并相邻同类输出。同类来源拼接不能当真边界。普通预测完全保持原状。
- [ ] 此诊断只问“给定参考分段后，这个分类器还错多少”，不是可部署结果，也不是数学上的性能上界。不把原生片段标签假装成专家逐帧真值，不把真边界传入正常推理路径。
- [ ] 同时列出每类混淆、已知边界附近错误、片段内部错误、整段错误。给每个错误来源录像一次计数，另报它在拼接流中出现次数，避免把同一病例重复出现当成多个独立问题。
- [ ] M5 的输出已经池化，oracle 作用于该输出时仍含原模型混合错误；明确这一限制。若需要池化前概率，另存一个只用于诊断的输出字段，不能替换默认输出或借此重选 checkpoint。
- [ ] 不为这一步重训或遍历新网格；交付一张诊断表即可。它用于解释后面的真实训练结果，不作为暂停全部训练的门槛。

## Task 3 — EchoViewCLIP 视觉缓存与分类探针

**Files:**
- Create: scripts/extract_echoviewclip_features.py
- Create: scripts/train_feature_probe.py
- Create: tests/test_echoviewclip_feature_contract.py
- Reuse: scripts/run_windowed_echoviewclip.py、echo_routing/features/cache.py、echo_routing/temporal/dev_run.py
- Read: docs/models_frozen/echoviewclip_stage1/v1/MANIFEST.json

- [ ] 首选已有冻结 stage 1 checkpoint，摘要以服务器实际校验为准。该权重是本项目 EV9V 9 类训练、按历史 validation 选出的版本，不是一个从未见过本项目开发反馈的外部编码器；在报告中明确这一点。
- [ ] 通过 pinned 官方模型工厂加载图像编码器，取逐帧投影特征，记录精确层名、维度、dtype 和预处理。检查 missing/unexpected keys，视觉权重缺失时立即失败，不接受静默随机初始化；仅允许有记录的文本缓存键排除。保留原始完整视频 TML 模型作为基线，不能把逐帧特征加线性头称为官方 EchoViewCLIP。
- [ ] pinned 代码的首选取值点为 model.image_encoder(image)：512 维，已经经过 image_encoder.ln_post 和 image_encoder.proj，位于整段时间聚合及外层 model.projector 之前。用实际模型与短样本再次核验；不把同一帧的 512 维向量与外层视频投影结果混用。
- [ ] 先核验所取层真正是单图像函数；如果包含跨帧计算，则不得从原生整段预计算后再拼接。首选绕过时间聚合的逐帧视觉层；无法提供时记录阻塞并保留原模型训练结果，不暗中泄漏源片段上下文。
- [ ] 10 Hz 帧索引、帧顺序和 gamma 与第一轮严格一致。EchoViewCLIP 沿用其既定预处理；编码器间差异包含预处理，不把它全部归为权重变化。train/validation 逐样本对齐，生成独立 manifest，加入 visual checkpoint、预处理、层名和探针 checkpoint 的摘要。
- [ ] 在独立 EchoViewCLIP 环境抽特征，主环境训练时序头，避免修改已冻结模型依赖。用短视频检查在线与缓存一致；同一张图嵌入不得因外部视频帧改变而变化。
- [ ] 冻结视觉权重，训练五类线性分类探针：AdamW lr=1e-3、weight decay=1e-4、batch=256，最多 20 epochs、patience=5，seed=0；每 epoch 每训练片段抽 16 个可用特征（短片段有放回），采用与 v3 相同类别采样策略。以 selection 原生 macro-F1 选探针，calibration 不选权重。三个时序 seed 共用这一探针和缓存。
- [ ] 特征抽取先产出 raw embedding 与索引，供探针训练；探针冻结后，再为 orig/gamma 变体写齐正式缓存要求的 frame_idx、t_sec、feat、prob、logit。frame_idx/t_sec 与旧协议逐样本一致；logit 是同一冻结探针的五类原始输出，prob=softmax(logit)。不能用真标签或零数组占位后直接交给 B3/时序评估。最终 manifest 同时绑定 visual 与 probe 摘要，并用现有 load_video_features 实际读取一份缓存作接口检查。
- [ ] E0 使用原 v3 编码器及原分类头；E1 使用新视觉层及线性探针。它们是两个完整视觉输入配置，不声称只改变一组预训练权重。分别报告逐帧、原生片段及 oracle 诊断指标，不能直接把两个历史测试表的数字相减。
- [ ] 在相同两半 banks 上补官方 EchoViewCLIP stage 1 滑窗输出，使用流中实际可见帧，作为完整系统对照。已有预测仅在 bank、采样和 checkpoint 摘要一致时复用。

## Task 4 — 实际训练矩阵，先得到可归因的结果

**Files:**
- Modify: scripts/model_development_queue.py、echo_routing/temporal/dev_run.py（仅新增 manifest 兼容检查）
- Reuse: scripts/run_asrf_official.py、scripts/train_boundary_context.py、scripts/calibrate_simple_decoders.py、scripts/posterior_decoders.py
- Create: scripts/report_model_development_round2.py

| 视觉配置 | ASRF / M1 | M3p：无 attention 路径项，有边界输出池化 | M5：完整池化模型 | B3 |
|---|---|---|---|---|
| E0：v3 ResNet-18 | 复用种子 0/1/2 | 复用种子 0/1/2 | 复用种子 0/1/2 | 复用同轮校准 |
| E1：EchoViewCLIP 逐帧视觉层 | 训练种子 0/1/2 | 训练种子 0/1/2 | 训练种子 0/1/2 | 新探针概率上校准 |

- [ ] E1 共九个时序训练，先 seed 0 的 ASRF/M3p/M5，再完成 seed 1/2；不需要等 M5 单次胜出才确认种子。输入维度不同只改各头入口维度，其余结构/损失/optimizer/update 上限保持第一轮配置，不为本模型单独增加训练预算。
- [ ] 既有 E0 结果满足协议和配置摘要时直接复用；新队列不能误跳过 E1。旧冻结权重不覆盖，缓存不混用。
- [ ] 训练入口示例（先实现 queue 的 --config）：

```bash
python scripts/model_development_queue.py --config configs/experiments/model_development_v2.yaml --jobs M1,M3p,M5,simple --feature-manifest ACTUAL_E1_MANIFEST --protocol ACTUAL_ROUND1_PROTOCOL --seed 0 --status runs/md_round2/e1_s0_status.json
```

种子 1/2 使用各自 status 和日志；simple 只需运行一次。示例参数需从真实 manifest 取得，命令不是已完成的执行记录。

- [ ] 主表同时列 native macro-F1、各类召回与分子/分母、multi segment F1@50、边界 recall/F1、内部切分、同类误切、逐样本错误率；报告每个 seed，不能只给均值。
- [ ] 分别计算 E1−E0（各头的视觉输入增量）、M5−M3p（路径项增量）、候选−ASRF/B3（新模型增量）。若所有方法同步提高但新模型仍不领先，收益归给视觉输入，不能宣称新机制成立。
- [ ] native 按会话重采样；构造流逐流区间保留“复用录像导致依赖未完整建模”的限定，不把多个 seed 当多个独立测试集。补逐来源录像错误计数，而非再扩大统计审计范围。

## Task 5b — 有条件执行一次配对联合微调

这一步不再以“M5 已超过 ASRF”为条件。E1 的冻结结果完成后，按以下固定规则决定；不新增模型家族。

- [ ] 联合微调的启动条件固定为 (A 或 B) 且 C。A：E1 seed-0 线性探针相对 E0 原冻结分类头的 native macro-F1，在 selection、calibration 两半都严格提高（差值 >1e-6）。B：仅用这两个逐帧分类器的概率，在各自半区固定的 multi bank 上按 Task 2 参考分段做平均概率分类；以“判错的参考类别段数 / 全部参考类别段数”为错误率（相邻同类来源先合并，每参考段计一次），E1 在两半都降低 >1e-6。不用时序头输出替代这一 B 指标，也不按新模型结果挑病例；这些片段复用的诊断不当作独立样本检验。C：预先列明的 ASRF 官方 refinement、M3p argmax、M5 argmax 中至少一种默认解码，在 calibration 的三个 seed 各自都达到 recall >=0.95（使用未四舍五入值）。不以事后 B3 解码或三个 seed 平均召回代替 C。
- [ ] A/B 是单个共享编码器/探针的方向性诊断，C 是三个时序 seed 的最低召回检查；它们只决定是否值得继续实验，不是统计胜出声明。若条件不满足，完成 Task 6 并报告“没有证据支持按本轮规则继续联合微调”，不自动堆另一个大编码器；不要把这次有界实验未通过等同于项目终止。
- [ ] 从 E1 的 M3p/M5 中选择一个候选：按 selection 半默认解码的三个 seed 平均 segment F1@50 选；差距小于 0.005 时固定选较简单的 M3p。该规则仅选进入微调的结构，不改变每个模型原来的 checkpoint。
- [ ] 对该候选和 ASRF 做 seed 0/1/2 的配对联合微调（六个运行）。两者从同一 E1 视觉 checkpoint 出发，接各自相应 seed 的冻结训练头，解冻相同的最后视觉 Transformer block、最终归一化和投影；其余视觉层冻结。记录实际可训练参数名称和摘要。
- [ ] backbone lr=1e-5、head lr=1e-4、AdamW weight decay=1e-4、最多 10 epochs、patience=3；保持相同训练 recipe、native/stream 比例和更新预算。允许微批与梯度累积，等效 batch=8，不因模型显存不同改变抽帧数量或训练流。
- [ ] 微批累积必须复现各方法原有整批损失归一化，不能把长短不同的微批 loss 简单平均。写一个关闭 dropout、fp32 的小例子，比较整批与累积后的各参数梯度；两种模型分别通过，且沿用整批的 padding/mask 口径，再进入正式微调。
- [ ] 每条训练流按 recipe 渲染实际图像，只把图像与 mask 输入模型；不能读整段来源视频补充上下文。为控制显存可按帧微批通过编码器，保持到解冻层的计算图；不把旧特征缓存当作可回传梯度。若用 frozen-prefix 缓存，需要逐样本保留最后可训练块的完整输入 token，并验证在线一致性。
- [ ] 在一个更新上确认：解冻视觉权重得到有限且非零梯度并发生变化，冻结层不变；两种系统的输入帧/recipe 一致。显存不足先缩微批并累积，不擅自缩小上下文窗口。
- [ ] checkpoint 仍只由 selection 选择；calibration 只在训练结束后用于固定网格解码。微调失败或退步保留原冻结版本与日志，不把尝试静默删掉。
- [ ] 这些 seed 改变时序训练和微调随机性，但共享历史预训练视觉权重；不能称为三个独立完整编码器预训练重复。

## Task 6 — 交付与下一轮边界

**Files:**
- Create: docs/results/2026-10-02_model_development_round2/README.md
- Create: scripts/train_joint_visual_temporal.py（仅 Task 5b 启动时）
- Create: tests/test_joint_visual_temporal.py（仅 Task 5b 启动时）
- Freeze: 新版本目录、manifest、checkpoint/预测/协议摘要；不覆盖第一轮

- [ ] 交付实际 E1 特征与探针、九个时序训练的权重/曲线/完整表，以及 Task 5b 的启动条件和完成或未执行原因。报告解释视觉输入、分段输出、路径项各自贡献多少。
- [ ] 纠正“打平”和“分类能力最好”的宽泛措辞；逐项说明未检出差异、原生/构造流和默认/额外解码的范围。若新模型只在逐样本错误率上更好，就只声明这一项。
- [ ] 对最终候选、ASRF、原始 EchoViewCLIP 做同硬件、batch=1、相同长度分组、预热与 CUDA 同步的计时；报告 head 和包含解码/预处理/视觉前向的完整系统时间。批量吞吐另列，不继续沿用混合 batching 的 20 倍主张。
- [ ] 第一轮失败结论不改。最终目标仍是相对最强基线有至少 1 个百分点的实用 segment F1 提升、native 分类超过最强完整系统、变化召回 >=0.95 且相对竞争者下降不超过 1 个百分点，并降低切分。未满足的目标逐项保留，不按本轮结果重新定义“超过”。
- [ ] 若候选只与 ASRF 接近但更小/更快，记录精度与成本权衡；这不能代替用户要求的准确性进步。若候选在 E1 上胜出，再补同编码器 ASFormer/MS-TCN 及已有 STFM/EchoPrime 等完整基线表，才判断是否超过全部基线。暂不据这三头矩阵宣布全胜。
- [ ] 只提交本轮相关文件，保留当前暂存理论草稿、审阅 Word 和用户未提交更改。论文正文和投稿准备不推进。

参考：[EchoViewCLIP 官方实现](https://github.com/xmed-lab/EchoViewCLIP)明确区分 stage 1 分类、stage 2 OOD 与 stage 3 质量控制；本轮仍是 stage 1 对应的闭集任务。[ASRF 官方实现](https://github.com/yiskw713/asrf)继续作为类别/边界联合学习的强直接对照。
