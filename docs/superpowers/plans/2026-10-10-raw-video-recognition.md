# Target-Conditioned Raw-Video Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax. Claude is the intended executor; this is the lead's next decision after the predicted-partition FACT report.

**Goal:** 检验正确帧顺序下，直接从图像块联合编码空间与时间的模型，能否改善目标时刻的切面识别，并把收益传到时间分段。

**Architecture:** EchoPrime 初始化的 MViTv2-S 视频编码器，保留其 512 维投影，加五类分类头。用一个零初始化的目标时刻标记分支告诉全局读出应回答哪个帧位置。三组结构完全相同，只改变 RGB 帧输入：目标帧重复、目标帧保持而其他帧打乱、真实顺序。标签仅用于训练监督和评估。

**Tech Stack:** torchvision MViTv2-S、已有 EchoPrime 权重、项目窗口清单/帧读取/评估工具、PyTorch、pytest、英文 README。

---

## 负责人决定与已知范围

**结束 FACT 最后 U 块细分路线。** 固定细分对原 FACT 的六个点估计都为负；预测细分也没有稳定超过固定细分或原 FACT。这个资源决定不把“未检出一致差异”改成等价，也不否定全部 token 模型。

**不把“最早在 E1 未匹配”解释成视觉特征不足。** 这个指标会把逐帧标签跳变计为段失配，不能单独定位视觉识别错误；本轮不以它作为开训条件。

本轮提出一个尚未有效完成的比较：**同样图像集合下，视频时间顺序是否带来额外的目标识别信息？** 旧 V4/V4c 的结果来自乱序帧；2026-10-05 的正确输入实验是独立帧 E1 编码器加时间 Transformer，不是 MViT 图像块联合时空编码器的有效重跑。其他基线各有自己的输入方式，本任务不声称它们全部缺少动态信息。

这是补做已有视频模型的一次有明确输入对照的任务，不预先称为论文创新或已证实的解法。即使顺序有用，也只支持“顺序相关信息”，不能据此宣称识别了流出道运动、心动相位或探头变化。

**允许完成接入、检查与九次正式训练。** 本轮总 GPU 上限 **24 小时**，包含检查、九次训练和评估。实际完整更新的计时预测在预算内，且实现/恢复/目标定位检查通过后，直接开跑，不再等负责人确认。超预算则保存准备或训练进度并报告，不削减某组更新数、不换模型、不自动加预算。

## 固定模型与初始化

- 官方 source：EchoPrime，固定 commit `03874a5f203695f38968068c21584656d475b6b1`。
- 已取得 checkpoint：`third_party/EchoPrime/model_data/weights/echo_prime_encoder.pt`；SHA-256 `7ca32e8bfde248bd6d8c7e46fdb7440385169af4dc2f416b5de840bdc2e64f3b`。
- 按官方 `load_for_finetuning.py` 建立 torchvision `mvit_v2_s`，把其投影头改成输出 512，再严格加载这份权重。**保留此投影**，在外面加 `Linear(512,5)`；不直接用旧 `MViTClip`，它会把预训练投影替换掉。
- 记录并固定实际 torch/torchvision 版本与源代码摘要，不升级环境。不得把旧版本 V4/V4c checkpoint 当成初始化。
- 输入为 RGB `(B,3,16,224,224)`。用现有 `load_frame_u8` 的 gamma、short-side 256 / center-crop 224 几何，保留现有 224×224 路径，不重开 fan224。
- RGB 归一化采用 EchoPrime 的 0–255 口径：mean `(29.110628,28.076836,29.096405)`、std `(47.989223,46.456997,47.20083)`。不能再除以 255。这个裁剪几何是项目适配，不能称为官方完整预处理复现。

## 目标位置必须可识别

窗口在序列端部会被截取，目标不总在第 8 个位置。不能把一个不接收目标位置的全局窗口输出当成所有目标的标签。

- 给每个窗口建立 `(B,1,16,224,224)` 的 0/1 标记张量；一个时间槽整平面为 1，其余为 0。标记位置仅由调用者请求的样本 c 和窗口索引 s 确定，不读任何类标签。
- 若 s 中多次出现 c，用这些位置中的中间一个（排序后索引 `len//2`）作标记。所有出现 c 的位置都属于下述 shuffle 的保留位置。
- 新建 `Conv3d(1, stem_out_channels, ...)`，kernel/stride/padding/dilation 与原 `conv_proj` 相同，无 bias、权重全 0。将它的输出加到原 RGB `conv_proj` 输出，之后调用原 MViT 路径。不修改官方 repo；在项目 wrapper 中用临时 hook，或等价的隔离适配，并在 finally 恢复。
- 零标记分支初始状态下，视频编码器的 512 维输出必须与原三通道编码器相同。标记分支本身不产生类别输出。
- 新五类头用每个种子一次保存的共同初始化：weight 正态 std=0.02，bias=0；三组共用初始状态，标记分支初始全 0。

目标标记是任务接口的适配，不是单独待证明的创新。本轮三组都使用它，避免比较中把“知道目标在哪”与“有时间顺序”混在一起。

## 三个训练组

| 英文报告名 | 输入 RGB | 目标标记 |
|---|---|---|
| Target-only | 将该窗口真正目标帧及其 gamma 复制到全部 16 槽 | 原窗口的标记位置 |
| Shuffled context | 与 Ordered 完全相同的 16 张图像，保留所有 s==c 的槽，只打乱其余槽 | 原位置不变 |
| Ordered context | 原窗口的真实时间顺序 | 原位置不变 |

Shuffle 不移动原始时间戳/位置标记来让模型重建顺序；模型只接 RGB 与目标标记，不接帧编号、源录像名、labels、is_native 或切面比例。除保留目标槽外，非目标槽跨前后位置打乱；同时移图像及其 gamma，不能只乱 metadata。

训练 permutation 在清单生成时用独立数据 RNG 保存：`default_rng([20261010, seed, update_index, window_index])`，对非目标槽做 permutation；不根据预测或标签选择排列，不强制 resample 直到“足够乱”。报告重复帧/未产生变化的比例即可。

推理 permutation 每个模型种子和窗口固定；用 `SHA256(seed + '|' + private_sequence_id + '|' + centre_index)` 的前 8 bytes（big endian）初始化独立 RNG，仍只打乱非目标槽。这个 ID 只用来固定顺序，不作为模型输入。

## 数据、监督与预算

- family5：PLAX、PSAX、A4C、A5C、SC4C。沿用当前训练/两开发半区 protocol；test/Pediatric 不读。
- 复用 `runs/center_supervision_v1/manifests/index.json` 的同种子 **SC 窗口清单**（S/C 当时共享的一份），10 epoch × 375 更新，每次 4 个原生窗口和 4 个混合切面窗口。复制并冻结来源摘要，不使用旧 C 模型权重。
- 窗口按现有规则在 10 Hz 时间轴取 16 个样本，端部截取/重复规则不变；推理 centres 间隔 5 样本。原生和构造流用同一规则。
- 同种子三组的请求目标、标签、批次和 Ordered/Shuffled 图像多重集合完全一致。Target-only 有意减少不同图像的观测量，不能声称它与多帧组信息量相同。
- native 与 mixed 的目标标签都为请求样本 c 的公共参考类别，不取窗口多数类。标签只供 CE 与报表，窗口内其他标签不进入前向。
- 三种子 `[0,1,2]`，各组 **3,750 次更新**，最后权重，不在训练中建立/评估开发集合，不早停、不选 epoch。
- 唯一损失为目标五类 CE，batch 平均；不加 frame auxiliary、边界、对比或蒸馏损失，不设 class weights/label smoothing。训练窗口清单原有类别/录像均匀采样已经固定。
- AdamW：encoder lr=1e-5（包括预训练投影），新五类头与 marker stem lr=1e-4；weight_decay=1e-4、clip=1。前 375 更新只训练新头/marker，encoder requires_grad=False 且 lr=0；随后全部 encoder 解冻。
- 冻结只关闭 encoder 参数更新，不能把整个 encoder 前向放进 `no_grad()`；marker 在输入 stem 注入，仍需经过冻结的 trunk 反向传播才能训练。冒烟检查必须确认 marker 权重实际变化。
- 新参数 lr 前 375 次按现有 `lr_factor` warm-up，此后 cosine 到 0。encoder 在前 375 次为 0，此后与新参数使用相同 cosine 因子乘 1e-5。不遗漏冻结后需解冻的参数组；优化器从开始就为实际模型保存全部参数组。
- fp32、TF32 off、cuDNN deterministic、benchmark off。按现有视频模型可运行设置处理缺少确定性 CUDA 内核的问题，记录容差与具体算子；不私自换成混合精度来压预算。
- 首次正式训练后不追加超参、窗口长度、marker 结构、预训练或解冻范围搜索。

## Task 1: 接入与清单

**新文件：** `echo_routing/vit/target_mvit.py`、`configs/experiments/raw_video_context_v1.yaml`、`scripts/prepare_raw_video_context.py`、`scripts/train_raw_video_context.py`、`scripts/queue_raw_video_context.py`、`scripts/report_raw_video_context.py`、`tests/test_raw_video_context.py`、`tests/test_raw_video_context_report.py`。

- [ ] 在源 FACT 细分任务书尾部加负责人日期决定：本轮报告判定不确定，结束此细分配置路线。保留旧规则与结果；不要把它改成“所有 FACT 无效”。
- [ ] 配置固定全部来源、窗口与控制定义、训练和报告规则；只提交本轮文件与上述决定，不动论文/理论稿/许可草稿/演示。
- [ ] 训练允许名单筛选后才提取类别标签与载入帧/缓存。读取源 SC 清单时核对其所有录像都在训练名单内；不要直接调用会读开发标签的旧 preparation 或 raw9 的训练入口。
- [ ] 保存 Ordered 的逐窗口源帧/gamma、Target-only 重复来源、Shuffled permutation、目标槽、三组像素摘要和共同初始化摘要。端部/短录像都检查；不得用 reference labels 找目标槽。
- [ ] 对 source RGB encoder 严格 load/check；在相同已处理张量上验证源三通道模型与新模型零 marker 分支的 512 维输出，fp32最大差≤1e-4。只做预处理/模型一致性，不声称当前任务性能。
- [ ] 运行身份含相关模型/输入函数、torchvision 模型实现、环境、checkpoint、marker 定义、清单/排列、初始化、optimizer、loss、总更新数与代码摘要。独立数据 RNG 与模型 RNG 分开。

## Task 2: 开跑检查

- [ ] 用训练源固定 16 个窗口检查短录像重复、目标 at 0/末尾、native/mixed 和多个目标重复槽。数据选择只按清单顺序/覆盖，不读开发集。
- [ ] 单元测试确认 Ordered/Shuffled 像素多重集合相同、所有目标槽不动；Target-only 每槽都与目标像素一致；标记恰有一个时间槽且位置正确；转 tensor 后各组监督相同。
- [ ] 模型前向只收 `(rgb, marker)`。替换参考标签、源名字、native 标志不改变输出；改变目标 marker 能进入对应分支，但不要求随机初始化时输出必然变化。
- [ ] 检查 warm-up 中只更新新头/marker，encoder 权重不变；解冻后 encoder 与投影有有限梯度并实际更新；marker hook不残留，连续多前向与异常退出后仍正确。
- [ ] 同初始状态重跑和 3+3 恢复对连续 6 更新，用 loss差≤1e-4、参数差≤1e-5；另用小模型或短状态检查跨第375次更新解冻边界的恢复和lr。失败先修实现，不换精度或调参。
- [ ] 在相同小训练窗口上每组 **300 次更新**的拟合记录，训练新头/marker（encoder仍冻结，恒定新参数lr=1e-4）。报告 loss/F1/grad，不设“满分才过”的性能门槛，不把冻住 encoder 的拟合记录当正式学习效果。数值、更新、目标和隔离检查是硬条件。
- [ ] 在实际解冻状态、正式 batch=8 上预热5次后测50次完整更新；含解码/增强/marker开销，测峰值显存并预测九次训练及报告成本。不能只测分类头或冻结前向来批准预算。
- [ ] 独立实现审查通过并提交检查记录后，预算内直接跑：seed0 Target-only/Shuffled/Ordered，再seed1、seed2。只允许身份相同的完整运行跳过；中断从完整状态恢复。

## Task 3: 一次性评估

- [ ] 九次完成并冻结后再加载开发帧。保存每窗口 logits/target marker/排列和映射后的逐样本输出，逐序列信息留服务器。
- [ ] 原生录像与构造流分开。主机制比较为 **Ordered − Shuffled** 在原生 A4C/A5C 窗口上的五类 argmax balanced accuracy（两类 recall 均值）；同时报告 margin `logit(A5C)-logit(A4C)` 的 AUC、全部原生目标准确率及 Ordered/Shuffled − Target-only。AUC增益单独不等于决策改善。
- [ ] 参照 E1 分类器对同一请求样本 c 的输出，核对目标像素来源一致。只在相同 centres 做识别比较，不能把稀疏窗口准确率与旧dense逐帧结果直接相减。
- [ ] 对 mixed windows 报告目标判定、目标类帧比例和修好/新错来源数。它们检验跨切面上下文，不能独自证明心脏运动信息有用。
- [ ] 每个10 Hz样本取最近centre输出，平局按现有 `nearest_centre`；原生clip仍按逐样本多数票统计。先报直接 argmax，再用相同标准简单变化点解码网格。
- [ ] 三组每种子仅在 calibration 上选一次现有网格：half_window `[5,7,9,11,15]`，quantile `[0.80,0.85,0.90,0.95,0.97,0.99,0.995,0.999]`，NMS5。规则沿用中心监督轮：native F1≥0.980、boundary recall≥0.95下 segment F1最高，同分按网格原顺序；无可行设置则标失败，诊断设置9/.95只报告不达标。
- [ ] 选定后冻结，selection一次应用。记录每组实际工作点；不能把每组不同解码器工作点带来的收益全部归到视频编码器。直接识别与固定解码比较分开。
- [ ] 复现强参照的原设置：E0/ E1 /官方 EchoViewCLIP +简单解码，历史和配对 ASRF、Original FACT，先前 C目标监督模型，以及G1+ASRF的较高calibration参考。不能据当前结果重选某个旧seed/旧encoder。
- [ ] 主系统是事前指定的 **Ordered context +简单变化点**；Target-only/Shuffled也分别检查目标。若一个对照反而赢，照实报告，不事后把主候选改名。
- [ ] 既有bootstrap2000、seed20261002；原生目标/clip按session成组，流按流近似；重复录像与窗口不是独立病例，种子共享案例。所有差值逐种子逐半区给出，沿用三种子区间严格同向排除0的“一致”口径。
- [ ] 报告≤16样本短段、A4C/A5C中段、label switches、同类拼接误切、native宏F1/每类recall、边界P/R与端部误差；修好/新错去重到来源录像/session。原生参考仅是公共录像级标签。

## Task 4: 结论与结束条件

系统目标不变：三种子平均 multi segment F1@50 selection≥0.944409687184662、calibration≥0.9821421709894332；每个种子每个半区 recall≥0.95，native F1≥0.973 / ≥0.980。另报告calibration敏感性目标0.9838935784747763。

- [ ] Ordered若一致优于Shuffled的原生A4C/A5C目标判定，可报告“时间顺序相关的识别收益”。仍不能把它直接说成流出道、心动周期或探头运动机制。
- [ ] Ordered/Shuffled若均优于Target-only，但Ordered未检出对Shuffled一致优势，支持多帧条件下的收益，未证明额外顺序价值；不是两组等价。重复帧与预训练输入分布不同也可能影响比较。
- [ ] 若只有AUC变好，准确率/分段不改善，报告排序改善而非可用系统增益；不补阈值搜索。
- [ ] 识别改善但系统仍退步，分清类判断、上下文吞段、多切和边界偏移；本轮结束，不自动另加时序头或解码搜索。
- [ ] 若无稳定机制或完整系统收益，结束这个模型/输入/预算的比较，不否定所有视频模型。旧FACT细分路线保持关闭。
- [ ] 达到目标也只称当前开发协议达成，未建立真实连续扫查或外部患者泛化；test不自动开启。
- [ ] 英文 `docs/results/2026-10-10_raw_video_context/README.md`、完整表/summary、曲线、模型输入 token 图、实际GPU/推理成本、冻结权重/摘要、运行命令、启动与偏离记录齐全。独立核查数据对齐、主比较、目标与去重，只提交本轮，不推送。

## 来源与实现范围

- [EchoPrime 官方模型加载](https://github.com/echonet/EchoPrime/blob/03874a5f203695f38968068c21584656d475b6b1/load_for_finetuning.py)：原视频encoder输出512维，16帧224×224输入。
- [torchvision MViTv2-S](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.mvit_v2_s.html)：标准视频编码器；实际环境版本必须记录。
- 旧 `echo_routing/vit/models.py` 与新模型可借结构，不借旧乱序结果；旧 `vit/datasets.py` / `train_vit.py` 的raw9、验证选点和标签读法不直接复用。

与E1、STFM、FACT等的整体比较同时改变了预训练/架构/归一化/训练方式，不能单独证明“局部patch优于全局向量”。三组内部对照才检验本任务指定的输入条件。它是一个待验证的模型比较，不是已建立的新论文贡献。
