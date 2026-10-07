# L1 features with matched ASRF Implementation Plan

> **For agentic workers:** Use the executing-plans workflow to implement this plan task-by-task. The intended executor is Claude. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 检验旧输入上普通视觉微调得到的 L1 特征，能否通过匹配训练的 ASRF 改善完整的切面识别与时间分段系统。

**Architecture:** 两组冻结的 512 维视觉特征，各接同一官方 ASRF。A 组使用原 E1，B 组使用 fan224 轮的 L1；每组对应三个种子，共六次 ASRF 训练。输入仍为旧 L 路径输出的 224×224，视觉编码器和视觉分类头均不再训练。

**Tech Stack:** PyTorch、已固定版本的 ASRF、现有 EV9V protocol、版本化特征缓存、现有分段评估和配对 bootstrap。

---

## 1. 决定和证据边界

fan224 轮正式结束，不把 L1 事后改写成上一轮的预设候选。本文件是读取上一轮结果后提出的新探索性任务，候选固定为 B，即 L1+ASRF。

依据：`docs/results/2026-10-06_fan224_visual_adaptation/README.md` 和 `report_tables.md`。

- L1+B3 的 segment F1@50 在两个半区、三个种子都高于 L0+B3，均值约 .909/.969，但尚未检出一致提升，也未超过强基线。
- L1+B3 的 selection 原生 A5C 为每种子 11/11、native macro-F1 约 .989。**这些是 B3 后的结果**，不能说每个原始帧头都判对了所有 A5C，更不能说明每个截出的短片段都正确。
- L1+B3 的 selection 流逐样本错误仍为约 6.0–6.6%；种子 1 边界召回仅 .937。缺口可能同时包含短片段类别错误、漏切、多切及区间偏移，当前不能只归因于边界检测。
- ASRF 是既有强基线。它读取视觉 embeddings，并训练自己的分类和边界分支；**不会直接继承 L1 线性头已经做出的类别判定**。所以需要完整匹配训练，不能把新特征直接塞给旧 ASRF 权重，也不能保证原生分类会保住。

本轮是现成组件的组合验证，不主张新架构贡献。C/Z/R、fan224、额外 pooling/残差/gate 搜索均不重启。本文授权 Claude 完成下列检查和审查后直接启动六次训练，不需再等一次笼统确认；当前 Codex 只写任务书，没有启动队列。

## 2. 固定设计

| 组别 | 冻结视觉特征 | ASRF |
|---|---|---|
| A，匹配对照 | 原始 E1，旧 L 输入 | 从保存的初始状态训练，seed 0/1/2 |
| B，预设候选 | L1_s0 / L1_s1 / L1_s2，旧 L 输入 | 对应种子与 A 相同初始状态、批次清单和预算 |

重新训练三次 A 是为了保证与 B 的预算和选 checkpoint 规则相同。历史 E1+ASRF 只作为额外冻结参考，不替代本轮匹配对照。L0 编码器未变，**不另训练 L0+ASRF**；其线性头变化不属于 ASRF 输入。

只用已固定 EV9V train 与两个开发半区，不读 test，不使用 Pediatric，不改标签，不剔除困难录像。来源组是 recording sessions，不是已验证的患者身份。两半区都已经反复用于开发，本轮不是独立验证。

## Task 1: 核验并接入完整特征缓存

**Files:**
- Read: `scripts/report_fan224_visual.py`、`echo_routing/features/cache.py`、`echo_routing/temporal/dev_run.py`。
- Create: `configs/experiments/l1_asrf_integration_v1.yaml`、`scripts/prepare_l1_asrf.py`。
- Test: `tests/test_l1_asrf_inputs.py`。

- [ ] 等待现有训练特征提取正常完成，以文件和内容核验为准，不以预计完成时间为准。读取 `cache_v2/fan224_visual_adaptation_v1/train/L1_s{seed}` 的 index/identity；检查每个种子 3,364 段、orig/gamma090 两份、帧数和原始 frame_idx 完整。仅 index.json 存在不算完成。
- [ ] 核验 L1 checkpoint、原训练配置和输入变换摘要；来源记录在 `models_frozen/fan224_visual_adaptation_v1/L1_s{seed}` 及相应 run manifest。原 E1 来自 fan224 配置固定的 visual_export 和 probe。记录真实路径/摘要，不凭目录名推断模型身份。
- [ ] 将已有 L1 特征接入标准 `VideoFeatures`/manifest，保存源 frame_idx、从原缓存复制并核验的 t_sec、512 维 feat、对应视觉 prob 和真实 logits。旧 train 导出只有 feat/prob 时，不能把 log(prob) 当成原始 logits；若下游要求 logits，按相同权重重算或明确改变 schema，不伪造。
- [ ] 验证集可复用上一轮 eval_cache 中已有的特征，但必须覆盖所有协议原生样本和两种 variants；缺少的用冻结权重补提，不能拿稀疏 stream triples 假装完整缓存。先做身份和对齐，不计算新的探索性评分。
- [ ] A 组使用同一 E1 特征库。可复用已验证的 E1 原缓存；对固定训练帧与当前 L 路径在线前向作误差检查，fp32≤1e-3、历史 fp16≤2e-2，并统一喂给 ASRF 的存储/解码精度。不能把 L0 的新头分数冒充 E1 probe 的分数。
- [ ] 所有组复用原 protocol、原 1,500 条训练流 recipe、原帧索引/对比度、原 validation banks；recipe digest 必须一致。保留原 index 顺序，避免仅重新排序就改变训练流组成。新缓存路径独立，不覆盖旧缓存。
- [ ] 测试：漏一段、漏 variant、帧错序、嵌入维度不符、出现 test 行、摘要错误都应拒绝；相同 recipe 在 A/B 的来源、标签、时间戳和边界必须完全相同。只允许 embeddings 和各自分类分数不同。

Run: `.venv/bin/python -m pytest tests/test_l1_asrf_inputs.py -q`，应全部通过。

## Task 2: 六次同预算 ASRF 训练

**Files:**
- Reuse: `echo_routing/temporal/dev_heads.py` 的 `ASRFHead`、`asrf_weights`，`echo_routing/temporal/dev_data.py` 的 `TrainData`/`collate`。
- Create: `scripts/train_l1_asrf.py`、`scripts/queue_l1_asrf.py`、`tests/test_l1_asrf_training.py`。
- Update: 本轮配置，显式保存以下全部值，不隐式继承旧的选参逻辑。

**固定配方：**

- 官方 ASRF commit `9623f1e8d9a1171333a4eeb65d190997b6c44a95`；输入 512，五类，64 features、10 layers、每分支 4 stages。复用现有官方 loss，包括 Gaussian-similarity TMSE；lambda_b=.1，boundary_th=.5。不改 loss，不增加特征归一化/白化或 masking augmentation。
- 每步 4 条构造流 + 4 个原生窗口；每 epoch 遍历固定 1,500 条训练流，抽 1,500 个原生窗口，native_window=256，沿用 `TrainData.epoch_batches` 的具体抽样算法。
- 固定 50 epochs，每 epoch 375 updates，共 18,750 updates/模型。AdamW lr=3e-4，weight_decay=1e-4，grad_clip=1，前 2 epochs warm-up，再 cosine 到 0；fp32、TF32 关闭。不 early stop，不重试另一 lr。
- 每个 seed 的 ASRF 初始权重先保存一份，A/B 从同一份载入；数据清单先保存。复现实际更新中的批次内容，不只比较初始化哈希。类权重和 boundary pos_weight 按官方方法由共同 epoch-0 标签计算，两组完全一致。
- 视觉网络保持冻结，ASRF 没有访问视觉权重的可训练路径。三对比较同时包含 L1 视觉训练种子与 ASRF 种子，不能宣称独立估计了两种随机性的方差。

**Checkpoint 规则（新比较统一执行）：**

每个 epoch 结束只用 calibration 评估官方 ASRF 输出；selection 不进入训练/选 epoch。训练仍完整跑 50 epochs。在 calibration 满足 native macro-F1≥.980、multi boundary recall≥.95 的 checkpoint 中，取 segment F1@50 最大者；同分取原生内部切换更少者，再取较早 epoch。没有可行 checkpoint 时，只报告最终 epoch 为诊断，该模型不能通过最终目标。

这与历史 ASRF 用 selection 选 checkpoint/early stopping 的规则不同，必须在报告中说明，所以必须有重新训练的 A 组。calibration 的选择后分数偏乐观，不能作为独立确认。

- [ ] 写测试和 training-only 冒烟检查：初始化相同、样本身份/有效长度/标签/边界目标相同；改变 padding 的无效标签不改变有效 loss；loss 有限、梯度非零；同种子短跑和中断恢复可复现；selection loader 在训练入口不可调用。
- [ ] 不直接复用 `dev_run.run`/`train_head` 入口：它们会访问 selection、选 epoch 和 early stop。可复用组件，但实现本轮显式训练/选择规则并加防护测试。
- [ ] 冒烟测耗时和显存、估算六次预算；独立实现审查并修正实质问题。在首个训练及 calibration 指标产生前，提交本轮完整配置/代码。确认 GPU 不与特征提取争用后顺序启动，seed 0→1→2，每 seed A 后 B。
- [ ] run identity 包含协议、特征、代码、ASRF 初始化、seed、批次清单和本轮配置；不能仅靠目录名跳过。所有新运行用独立目录，最终保留选定和最终 checkpoint 及完整日志。

Run: `.venv/bin/python -m pytest tests/test_l1_asrf_inputs.py tests/test_l1_asrf_training.py -q`。

新 CLI 约定（待实现）：

```bash
.venv/bin/python scripts/train_l1_asrf.py --config configs/experiments/l1_asrf_integration_v1.yaml --arm B --seed 0 --smoke
.venv/bin/python scripts/queue_l1_asrf.py --config configs/experiments/l1_asrf_integration_v1.yaml
```

## Task 3: 完整系统评估和一次性错误定位

**Files:**
- Reuse: `echo_routing/evaluate/dev_metrics.py`、`scripts/report_fan224_visual.py` 的配对/来源去重逻辑、既有 ASRF 故障定位代码。
- Create: `scripts/report_l1_asrf.py`、`tests/test_l1_asrf_report.py`。
- Create: `docs/results/2026-10-07_l1_asrf_integration/{README.md,report_tables.md,summary.json}`。

- [ ] 六次训练结束、calibration 的 checkpoint 选择全部冻结后，统一报告两个半区。主输出只有官方 `refinement_with_boundary`，boundary_th=.5。不新扫阈值，不加 B3 到 ASRF 后面，不用冻结视觉头替换 ASRF 分类输出。
- [ ] 完整表包括 A、B 三种子，已完成的 L1+B3 三种子，以及 E0+B3、E1+B3、B7+B3、历史 E1+ASRF。L1+B3 复现原有预测和原参数，不重调。
- [ ] 主比较为 B−A，次比较 B−L1+B3；在两个半区逐种子报告 segment F1@50、边界 P/R、五类 native macro-F1/召回、A4↔A5 混淆、原生录像内部额外标签切换。将原生修好/新错、流上参考段修好/新错按来源录像和 session 去重。
- [ ] 做一个固定的“原生正确是否传到短段”表：对每个流中的参考同类区间，用相同来源帧、相同 gamma 的视觉概率，先按多数帧 argmax 判类；另列平均概率判类。全来源录像也用相同 gamma 和同一个聚合定义比较，避免把多数票与均值差异当成截短效应。对由多个来源组成的同类区间，分别报告组成 fragment，不强行称为一个来源录像。
- [ ] 表中区分：来源录像判对但实际 fragment 判错、两者都判对但系统未匹配、来源录像已判错。再从系统输出细分标签错误、漏边界、多切/匹配不足；参考区间只用于诊断，绝不供模型推理使用。
- [ ] ASRF 的漏切按已有方法分为未提出候选、后续类别合并、偏移；同时记下有候选却多切的情况。说明它是否把 L1 视觉判对的病例改错。不要看到高原生 F1 就直接叫“纯分段问题”。
- [ ] 上述诊断在结果脚本中事先固定，一次性随主表计算。它们不是额外的开训门槛，也不因某个固定分类器的 oracle 不达标而取消 ASRF 训练：ASRF 自己会重新分类，该 oracle 不是它的性能上限。
- [ ] 统计沿用 2,000 次配对 bootstrap，seed=20261002，native 按 session、构造流按流；明确流复用素材、区间近似、种子共享开发集。“一致”仍指一个半区三个种子的区间同向排除零，未达到不等于等效。不得用几十个重复 fragment 代替独立录像数。

## Task 4: 成功规则和结束

预设候选 B 的最终目标保持原数值：三种子平均 segment F1@50 在 selection≥.944409687184662，calibration≥.9821421709894332；每个 seed/half boundary recall≥.95，native macro-F1≥.973/.980，且有可行 checkpoint。另报告 calibration 使用 .9738935784747763+.01 的敏感性。不能把仅优于较弱 A 组叫作超过原强基线。

- [ ] B 优于 A，但不优于 L1+B3：视觉适配可能对 ASRF 有益，但没有完整系统增益，不加模块继续追。
- [ ] B 优于 L1+B3，但最终目标未过：报告系统局部进步及剩余病例，不降低门槛。
- [ ] B 退步或未检出稳定增益：这一组合停止，不自动微调编码器、改 ASRF loss、替换输出头或扩训练预算。
- [ ] 即使过目标，也只是在反复使用的开发集上达标，需要另立独立确认计划；不解除 test 访问限制，不把常规组件组合直接称为新方法。
- [ ] README 用英文。保留全部运行、失败状态、训练曲线、选 epoch 依据及来源去重统计。权重/特征和含来源编号的表留服务器，仓库只保留代码/配置/汇总。检查报告与预测一致、冻结六个版本、只提交本轮文件，不推送、不修改旧论文和理论稿。

## 当前状态

本任务书基于 fan224 的已完成结果制定。没有启动本轮 ASRF 训练。现有后台特征提取的完成状态由执行者在 Task 1 核验，本文件不把预计完成时间当成完成证据。
