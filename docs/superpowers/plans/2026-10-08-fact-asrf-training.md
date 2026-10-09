# FACT–ASRF Paired Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Claude is the intended executor. This document supplies the lead's four decisions; it does not report a training launch.

**Goal:** 在冻结 E1 特征上检验官方 FACT 是否优于配对重训的 ASRF，并同时对照简单解码器和既有最强配置。

**Architecture:** FACT 使用官方 GTEA 底座，12 个 segment token、无 transcript 输入、一对一匹配和官方输出规则。ASRF 保留现有官方适配。两组共用输入、批次清单和更新次数，使用各自事前固定的优化器；本轮比较的是完整训练系统，不能单独归因于 cross-attention 或匹配损失。

**Tech Stack:** PyTorch、冻结 E1 512 维缓存、官方 FACT / ASRF、现有 EV9V 开发协议、YAML、pytest；单张 L40S。

---

## 负责人决定

1. **底座选 GTEA，不跑 Breakfast。** 作者的配置生成器明确把 GTEA 推荐给小数据集。本项目有 502 个训练 session、五类和最多八个参考段；构造流并不增加独立录像。先使用约 1,020 万参数的官方配置，比直接使用约 1.19 亿参数的 Breakfast 更适合作为本轮的资源选择。这是事前选择，不是 GTEA 效果更好的证据。它同时改变块数、宽度和增强，不能称作纯参数量对照。
2. **FACT 用官方 Adam 设置，ASRF 用已用过的项目配方。** 不强制统一优化器。共同约束是监督来源、批次、更新次数和选点机会。FACT 的批量和学习率衰减时刻适配本轮预算，不能称为完整复现论文训练配方。
3. **两组都取最终权重。** 每次 50 个 epoch × 375 次更新。训练时不计算两个开发半区的指标，不选最佳 epoch，不根据训练曲线追加训练。
4. **批准六次正式训练。** FACT / ASRF 各三个种子。正式训练总预算上限为 13 GPU 小时；先对 GTEA 实测估时，检查通过且预计在预算内即可自动启动，不需要再等一次负责人确认。检查与评估耗时单独记录。发生实现错误或超预算时保存进度并报告，不更换底座或缩短某一组来凑预算。

官方依据（固定提交 `7bd81bda2b84618a3e23876a2637a82f24881122`）：

- [小数据集底座建议](https://github.com/ZijiaLewisLu/CVPR2024-FACT/blob/7bd81bda2b84618a3e23876a2637a82f24881122/utils/gen_config.py)
- [GTEA 配置](https://github.com/ZijiaLewisLu/CVPR2024-FACT/blob/7bd81bda2b84618a3e23876a2637a82f24881122/configs/gtea.yaml)
- [官方优化器与学习率衰减实现](https://github.com/ZijiaLewisLu/CVPR2024-FACT/blob/7bd81bda2b84618a3e23876a2637a82f24881122/train.py)

## 固定设置

| 项目 | FACT | ASRF |
|---|---|---|
| 输入 | 冻结 E1 512 维特征 | 完全相同 |
| 预处理 | 现有 short-side 256 / center-crop 224，数字帧序，10 Hz | 完全相同 |
| 训练数据 | 既有 1,500 条流；每次更新 4 条流 + 4 个原生窗口，窗口最多 256 个样本 | 同种子逐更新共用清单 |
| 种子 | 0、1、2 | 0、1、2 |
| 更新次数 | 18,750 | 18,750 |
| 优化器 | Adam，lr=1e-4，weight_decay=0，clip_grad_norm=10；betas=(0.9, 0.999)，eps=1e-8 | AdamW，lr=3e-4，weight_decay=1e-4，clip_grad_norm=1；其余沿用现有配方 |
| 学习率 | 前 11,719 次更新为 1e-4，之后为 1e-5，无 warm-up | 前 750 次更新 warm-up，之后按现有 `lr_factor` cosine 衰减 |
| 选点 | 第 18,750 次更新后 | 第 18,750 次更新后 |
| 解码 | 官方输出，mwt=0.1，无搜索 | 官方 boundary refinement，boundary_th=0.5，无搜索 |

FACT 衰减比例来自官方 GTEA 的 250/400：`ceil(18750 * 250 / 400) = 11719`。按已完成更新数实现：当 `completed_updates < 11719` 时，下一次更新用 1e-4；否则用 1e-5。这是本轮明确的预算适配，不是把官方 epoch 250 原封不动搬进 50 epoch。

FACT 合并 `configs/default.py + gtea.yaml`；保留官方通道遮挡、时间遮挡和 dropout。只继承接入阶段已经定义的数据集覆盖项：五类、无背景、`trans=false`、`match=o2o`、12 个 token、训练平均 transcript 2.654、官方 null 权重规则。不要额外复制 Breakfast 的宽度或增强。不同模型的内部增强是系统配置的一部分；配对保证增强前的数据相同，不声称随机增强完全相同。

ASRF 模型、损失、类别权重、pos_weight 和输入维度沿用配对草稿。GS-TMSE 的实际权重为 1，plain TMSE 关闭；旧 YAML 的 0.15 是文字记录错误。旧报告加日期说明即可，不覆盖旧配置或据此重跑历史模型。

## Task 1: 固定执行配置与最小代码

**文件：**
- 修改 `configs/experiments/fact_asrf_paired_v1.yaml`：把草稿转成上述执行配置，清空四个 open items，保留旧稿的 Git 记录。
- 新建 `scripts/train_fact_asrf.py`、`scripts/queue_fact_asrf.py`、`scripts/report_fact_asrf.py`。
- 新建 `tests/test_fact_asrf_training.py`、`tests/test_fact_asrf_report.py`。
- 复用 `echo_routing/temporal/fact_head.py`、`dev_heads.py`、`dev_data.py` 和现有评估代码；不修改官方仓库。
- 新建英文结果目录 `docs/results/2026-10-08_fact_asrf_training/`，先放 preparation 与配置解释，之后追加结果。

- [ ] 在 YAML 中把两组优化器、学习率、底座和最终选点明确分开；缺键直接失败，不静默继承共享 AdamW。
- [ ] 复用 `TrainData.epoch_batches` 和既有批次摘要逻辑，但不要复用 `scripts/train_l1_asrf.py` 的 `Data` 类：它会直接加载 calibration。训练入口必须只能建立训练集，两个开发半区由报告入口独立加载。
- [ ] 保留原 protocol、缓存版本和 manifest 哈希核对。索引若列有开发标签，只允许核对完整性；不能用于训练、平均段数、null 权重或清单选择。
- [ ] 每个 arm × seed 从新初始化开始。两种结构不能共享权重；只共享种子编号和数据清单。不加载 16 条序列过拟合权重。
- [ ] 为训练入口实现 `--prepare`、`--smoke --arm FACT|ASRF --seed 0`、`--arm FACT|ASRF --seed N`；队列依次跑 FACT_s0、ASRF_s0、FACT_s1、ASRF_s1、FACT_s2、ASRF_s2。
- [ ] 运行身份包含官方提交、展开后的 GTEA 配置、适配代码、输入摘要、优化器、更新数、批次和初始化摘要。仅同身份且完整的结束记录可以跳过；不同身份不能覆盖。
- [ ] 保存并恢复实际被训练模型的权重、优化器、学习率进度、Python / NumPy / CPU / CUDA RNG 和已完成更新数。沿用已修复的对象绑定方式。
- [ ] 在执行准备和冒烟前提交本轮代码、配置及此任务书。只提交本轮文件；不推送、不改其他暂存内容。

## Task 2: 所选 GTEA 配置的开跑检查

旧的全库数据对齐结果可以在所有输入与清单摘要相同时复用。Breakfast 的过拟合、梯度与计时结果不能作为 GTEA 的通过证明。

- [ ] 增加有实际意义的测试：两种优化器类型和参数；FACT 在衰减前后及恢复后的 lr；最终选点不能读取开发指标；实际优化器绑定的是加载权重的模型；两组每次更新的数据清单一致。填充检查沿用既有测试：保持张量形状与有效样本不变，只改 mask 外的标签，损失不变；FACT 另核对去填充后的输入。不要把它扩成 ASRF 对改变 padding 长度也必须完全不变的新要求。
- [ ] 本地可执行的测试先跑；依赖官方仓库的测试必须在服务器上真正通过，不能用本地 skipped 充当通过。
- [ ] `--prepare` 只在训练集生成三个种子的清单和两组初始化摘要。重新生成须相同。
- [ ] `--smoke --arm FACT --seed 0` 使用正式 GTEA 配置和 Adam，在接入阶段同一组固定 16 条训练序列上交替更新 1,000 次。此处学习率恒为 1e-4，官方增强开启。要求每条序列 transcript 正确、segment F1@50=1、参考段匹配到不同且类别正确的 token；不要求总损失趋于零。失败时先区分实现错误和小样本拟合不足，不自动换回 Breakfast、调参或下结论“FACT 无效”。
- [ ] 对 GTEA 初始和上述过拟合权重复用推理标签隔离检查，预测与 logits 必须不依赖传入标签。核查所有损失梯度；已有结构性零梯度需在 GTEA 上确认解释仍成立，不套用 Breakfast 的参数编号。
- [ ] 两组分别检查同初始状态重跑，以及 3+3 次更新恢复与连续 6 次更新；使用接入/项目既定数值容差并报告最大差。FACT 另用不跑完整模型的调度测试覆盖第 11,719 次更新附近的恢复。
- [ ] 两组以正式 batch=8、fp32、TF32 off 实测：5 次预热 + 100 次更新，记录中位数、较慢批次、峰值显存。估算六次总耗时，预留保存开销；不把参数比例当耗时比例。
- [ ] 独立实现审查只检查实现、隔离、恢复和报告判定，不读取开发指标。处理会改变实验有效性的问题，提交通过记录后直接开队列。

建议的新入口命令（由本轮实现提供；不声称现在已存在）：

```bash
.venv/bin/python -m pytest tests/test_fact_integration.py tests/test_fact_asrf_training.py tests/test_fact_asrf_report.py -q
.venv/bin/python scripts/train_fact_asrf.py --config configs/experiments/fact_asrf_paired_v1.yaml --prepare
.venv/bin/python scripts/train_fact_asrf.py --config configs/experiments/fact_asrf_paired_v1.yaml --smoke --arm FACT --seed 0
.venv/bin/python scripts/train_fact_asrf.py --config configs/experiments/fact_asrf_paired_v1.yaml --smoke --arm ASRF --seed 0
.venv/bin/python scripts/queue_fact_asrf.py --config configs/experiments/fact_asrf_paired_v1.yaml
```

## Task 3: 六次训练与一次固定权重评估

- [ ] 队列开始后，以进程、首批日志和已完成更新数确认“已开始”。失败或预算问题暂停并保存，不在结果出来后补救超参数。
- [ ] 记录训练损失、梯度、学习率和固定训练监测序列的预测。监测数据只来自训练集，仅作拟合诊断，不能选 checkpoint 或改变预算。
- [ ] 六次全部完成后，核对每次确实更新 18,750 次，冻结最终权重和哈希；然后才运行报告入口。
- [ ] 复现 E1+B3、E0+B3、B7+B3、历史 E1+ASRF，以及 geometry G1+ASRF（calibration 三种子均值 0.9738935784747763，来源 `docs/results/2026-10-04_geometry_pretraining/report.json` 的 `goals/G1 ASRF/calibration`）。全部使用各自已存配置，不重调参数。B7+B3 和较高的 calibration 参考行用于覆盖历史比较，不用来挑本轮模型。
- [ ] 两个开发半区各报告 native、pair、multi；主比较为配对 FACT−ASRF 的 multi segment F1@50。另报告 FACT−E1+B3、FACT−E0+B3、FACT−B7+B3。逐种子和三种子均值都保留。
- [ ] 原生录像按 session bootstrap，构造流按既有流级配对 bootstrap（2,000 次，seed=20261002）。明确复用来源录像的相关性未被完整建模；三个种子共享病例。不能把六格或种子当作六份独立人群证据。
- [ ] 保留既有规则：一个半区三个种子的 95% 区间全部严格同向排除零，才称“一致”；其余称“未检出一致差异”，同时报告方向和大小，不写成等价。

英文 README 必须回答以下问题，机制指标只报告、不增加启动或成功门槛：

1. 与 ASRF 和 B3 相比，完整系统的 segment F1@50、边界 precision/recall、native macro-F1 改变多少？
2. 是否保住更多短参考段（≤16 个样本）和 X→Y→X 中段？A4C/A5C 单独报告；区分匹配失败与标签错误，并报告独立来源录像/session 数。
3. 是否以更多原生录像内 label switches 或同切面拼接误切换来了召回？原生标签是录像级标签，不能把所有 switch 硬称专家确认的错误。
4. 是否再次损失 A5C 识别？列出 A5C→A4C 与 A4C→A5C 的修好/新错数，去重到录像/session。
5. FACT frame-branch argmax 与官方完整输出分别怎样？这只是已训练系统的读出诊断，不是“去掉 token 训练”的消融。
6. 非 null token 数与参考段数怎样对应？输出是否有同一个 token 覆盖不连续区间？12 个 token 不代表输出最多只能有 12 个连续段。

记录参数、训练成本和同硬件推理成本。训练图若仍在改善，照实报告预算限制；不事后改成最佳 epoch。

## Task 4: 判定与交付

本轮沿用最终目标：三种子均值 selection ≥ 0.944409687184662、calibration ≥ 0.9821421709894332。另报告更高 calibration 参考的敏感性目标 0.9838935784747763。每个种子、每个半区都要求边界召回 ≥0.95，native macro-F1 分别 ≥0.973 / ≥0.980。不放宽任何下限。

- [ ] FACT−ASRF 未在两个半区均检出一致正向差异：结束这一套 GTEA 训练配置的比较。分别报告正向、负向和不确定，不否定所有片段 token 方法，不自动跑 Breakfast。
- [ ] 一致超过 ASRF，但未同时一致超过 E1+B3：报告相对 ASRF 的收益，目标未完成。
- [ ] 比较成立但任一系统约束不满足：报告收益与代价，不能用另一个 epoch 或解码参数修补。
- [ ] 比较与约束成立但最终目标未达：报告局部进展；是否后续研究另行决定。
- [ ] 比较、约束及原最终目标均满足：只称开发协议目标达成，同时报告 calibration 更高参考的敏感性结果。test 仍不自动打开。
- [ ] 英文 README、preparation.json、summary.json、report_tables.md、training_curves.png 写到新结果目录。逐录像编号、预测和权重留服务器；每个表有复算命令和配置/模型摘要。
- [ ] 独立核查复算主比较、约束、来源去重和判定表。只修报告或实现错误；如修复影响正式训练，必须标明原运行失效，不静默覆盖。
- [ ] 按本轮范围提交，未提交的论文/理论稿/许可草稿保持原样，不推送。报告实际跑了什么、哪些结果支持或不支持假设、是否达到目标。

整个比较始终是 EV9V 构造流上的探索性研究。开发半区已经多次使用，session 不是患者；真实连续扫查和外部泛化尚未验证。FACT 是已有方法，即使获胜也不能据此宣称自创模型或单独证明 cross-attention 的作用。论文继续暂停；test 和 Pediatric 不读取。
