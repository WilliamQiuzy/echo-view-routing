# EX7：Helena 输入和 A08 在 EchoNet-Pediatric 上的稳健性

Date: 2026-11-07
Owner request: 2026-11-07. The user asked whether the model trained on Helena's input is more robust on other datasets' test sets. They chose a sampled EchoNet-Pediatric test, about 2.5 GPU h.
Scope: inference only, no training. GPU cap 3 h.
Deliverables: this plan, the code, public aggregate JSON, the report, an index row

## 问题

EX3 已经在 LVH、Dynamic（抽 400 段）和 CAMUS 上比较过 H 和 A08，没有一致的进步。这一轮补一份新人群的外部测试集：儿童，两种切面（A4C 和 PSAX）。PSAX 也是第一次在外部测。问题是：在这份数据上，H 是否比 A08 更稳健。

## 数据

- 清单是 `data/manifests/pediatric/pediatric-external-v1/videos.csv`，2026-10-01 冻结，只用 `role == test` 的行（官方第 2–9 折）。
- 抽样在任何模型输出之前写定：
  - 种子是 2026110701。
  - 先 A4C 后 PSAX。每种切面在它的测试病人（`group`）里按名称排序，不放回地抽 400 名。
  - 每名病人在这种切面里按录像名排序，再用同一个随机数生成器抽一段。
  - 同一名病人可能两种切面都被抽中。
- 先前使用：Pediatric 在 2026-10-01（B0 外部检验）和 2026-10-05（E1 的头部适配）用过。A08 和 H 都没有在它上面训练或选择过。所以它对这两个模型是外部数据，但不是从没看过的数据。

## 系统

- A08：seeds 0–2，`models_frozen/video_within_view_v1/all_s{k}/final.pt`，用 A08 自己的中心裁剪。
- H：seeds 0–2，EX3 的冻结权重，用 Helena 的处理。
- 输入路径和 EX3 的外部路径完全相同：
  - H 的定位用 Helena 的代码（提交 53a2246）、EX3 的配置和 FastSAM-s 权重，每段录像算一次。定位失败时，按 EX3 用整帧方形。
  - 两个系统都按 10 Hz 采样，用 A08 的窗口计划，取窗口概率的平均。概率最高的类别就是这段录像的判读。

## 输出之前的检查

1. **路径复现。**在 EX3 Dynamic 组的前 8 段上：
   - A08 seed 0 走这条路径，要复现 EX2 存储的窗口平均概率，最大差 ≤ 1e-3；
   - H seed 0 用 EX3 已存的定位，要复现 EX3 存储的 H 输出，最大差 ≤ 1e-3。
2. **定位复现。**在同样 8 段上重算 Helena 定位。裁剪框要和 EX3 存储的完全相同，第一段遮罩 IoU 的中位数 ≥ .99。
3. **抽样。**
   - 清单文件的哈希要等于协议里记下的值。
   - 800 段都能解码。帧数和清单不一致的段数要记下。
   - 病人数、折的分布写进公开文件。

任何一项不通过就停下，不产生模型输出。

## 指标

- **主指标：**两种切面召回的平均（macro recall）。比较 H 和同号 A08，按病人整组重抽，做配对 bootstrap，给出 95% 区间。
- **次要指标：**
  - 每种切面的召回；
  - 交叉误判，即 A4C 判成 PSAX、PSAX 判成 A4C；
  - 判成其他三类（PLAX、A5C、SC4C）的比例；
  - 两个 AUROC：A4C 分数区分 A4C 段和 PSAX 段的能力，PSAX 分数同理。给出配对差和区间；
  - 每种切面的精确 McNemar 检验；
  - 9 对跨 seed 比较；
  - 各系统在 seed 之间的极差。
- **AUROC 是必须看的。**EX3 在 CAMUS 上发现，召回可能因为更爱说某一类而上升，而区分能力并没有变好。

## 判读（事先写定）

| 判读 | 条件 |
|---|---|
| H 更稳健 | 三个 seed 的主指标区间都高于 0，而且两个 AUROC 都不出现两个以上 seed 的区间低于 0 |
| A08 更稳健 | 三个 seed 的主指标区间都低于 0 |
| 没有一致差异 | 其他情况 |

报告同时列出 EX3 已有的 LVH、Dynamic 和 CAMUS 结果（包括 CAMUS 2CH 的检查），不重算。

## 资源和隐私

- GPU 上限 3 小时，预计约 2.5 小时，几乎都花在定位上。每段约 9 秒，按 EX3 的 Dynamic 记录估算。
- 磁盘至少保留 3 GB，GPU 显存占比 0.4。
- 逐段输出和录像名只留在服务器的 `private/` 和 `forward/` 下。公开文件只有汇总，写入前检查有没有 EV9V 和这份样本的标识。
- 不读 EV9V 测试集，不写 EX3 的文件夹，不改冻结模型。
- 报告和索引只在本地分支 `helena-default-input` 提交，不推送。
