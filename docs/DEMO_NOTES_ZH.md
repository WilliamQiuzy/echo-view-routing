# 组会演示提纲（2026-09-13）

> 面向：导师与项目组成员。目的：展示基线阶梯已在公开数据上跑通，并说明每个数字的含义与边界。

## 1. 已经搭好的东西

- **代码仓库**：GitHub 私有仓库 `WilliamQiuzy/echo-view-routing`，六模块结构（ingest / audit / features / compose / temporal / evaluate），架构文档 `docs/ARCHITECTURE.md`，运行手册 `docs/RUNBOOK.md`。
- **计算资源**：Nebius L40S（与另一个项目共用一张卡，我们的进程通过显存上限隔离），项目目录 `/home/william/echo-view-routing`，数据只在服务器，Mac 只有 6 段示范视频。
- **数据**：EV9V 全量（5,138 段，9 个原始编码）已下载、解压、建清单。原始划分 3,683 / 567 / 888；五类粗粒度任务 3,364 / 521 / 800（排除 PMPALA），与提案一致。
- **标签映射**：`configs/labels.yaml` 冻结为 `family5_v1`，同时保留数据卡与 STFM README 两套互相冲突的英文展开（PMASA / PMVLSA / PMPALA），待临床或作者确认。

## 2. 基线阶梯（提案 §7.1）

| 编号 | 方法 | 状态 |
|---|---|---|
| B-file | 逐文件平均概率，整文件接受/暂缓 | 已跑 |
| B0 | ResNet-18 逐帧分类 + argmax | 已跑（编码器：AdamW 3e-4，早停，最佳为第 1 个 epoch，验证集 cine 级 macro-F1 0.977） |
| B1 | 概率滑动平均 + 滞回 | 已跑 |
| B2 | 粘性 HMM + Viterbi | 已跑 |
| B3 | 左右窗口 JS 散度 + 持续标签变化 | 已跑 |
| B4 | MS-TCN（冻结特征） | 已跑（精简复现，非作者代码） |
| B6 | STFM 官方代码（EV9V 作者） | 通过适配器以固定 commit 运行，结果视训练进度而定 |

**结果表**：见 `runs/<最新 baselines run>/reports/ladder.md`，或运行 `python demo/demo_cli.py`（含四格图与风险-覆盖率曲线）。

## 3. 每个指标怎么读

- **native_cine_macroF1 / balAcc**：未修改的测试集 cine，逐 cine 多数投票后的分类质量。这是现有标签能支持的最干净终点。
- **frag_per_min**：单一切面 cine 内的预测跳变次数/分钟。是**稳定性代理指标**，不是真值边界。
- **四格设计**：同类/异类拼接 × 有无 gamma 扰动。`falseSplit_same_*` 是同类拼接处被误切的比例（越低越好）；`missed_diff_*` 是真实类别变化被漏掉的比例（越低越好）。两者要一起看。
- **boundaryF1@0.5s**：语义边界在 ±0.5 s 容差内的一对一匹配 F1。
- **tau@5% / coverage@5% / risk@5%**：阈值只在验证集拼接流上选（目标污染率 5%），然后冻结用于测试集。表中同时给出实际达到的污染率；没达到目标就如实报告。

## 4. 必须说明的边界

1. 拼接流是**特征空间**的拼接（冻结的逐帧编码器下与像素拼接等价），不是真实探头扫查；不能推广到床旁连续录像。
2. 目前是**两段式**拼接（每流 2 段），4–8 段的长流是下一步。
3. 没有患者 ID 映射，不确定性只能按 cine 分组。
4. 编码器在第 1 个 epoch 后即过拟合（验证集分数随后下降），学习率敏感性实验待做。
5. STFM 若未训练完成，只展示已固定的 commit、适配器与训练日志，不填任何预期数字。

## 5. 想请导师确认的三个问题（同中文摘要）

1. 第一阶段以研究团队的数据整理为目标、接受五类粗粒度范围，是否可行？
2. 能否安排医生核对类别映射（尤其 PSAX 各层面与 PMPALA）？
3. 若两周内时间分段不能证明优于逐文件分类，是否同意调整贡献方向？

## 6. 现场演示命令

```bash
cd ~/Desktop/Harvard/Capstone/echo-view-routing
remote/status.sh              # 服务器与任务状态（可选，需要网络）
.venv/bin/python demo/demo_cli.py   # 离线：表格 + 四格图 + 风险-覆盖率曲线
open runs/<run>/reports/demo_figure.png
open demo/samples/           # 六段示范视频（每个原始编码一段）
```
