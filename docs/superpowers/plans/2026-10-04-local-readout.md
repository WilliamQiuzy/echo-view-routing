# 第七轮：冻结 E1 的全局与局部读取对照 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 预注册配置：`configs/experiments/local_readout_v1.yaml`，在任何训练之前写定并提交。

**Goal:** 检验一个尚未验证的假设：冻结的 E1 编码器的局部 patch 特征中，是否存在全局向量不容易读出的 A4C/A5C 区分信息。这不是已发现的缺陷——ViT 本身已经处理了局部区域，我们只是尚未比较不同的读取方式。主要比较是 P2−P1。

**决定来源：** 项目负责人 2026-10-04（CETUS 验收之后）。CETUS 不加入训练；test 继续封存；论文继续暂停。注意力多实例学习（Ilse 等，ICML 2018）是成熟方法，本身不算创新；本轮只判断是否找到了一个可改进的具体瓶颈。

**Architecture:**
- P0：全局向量按时间取平均，就是第六轮 V0，必须逐序列复现其验证预测。
- P1：每帧的全局 token，经残差帧模块，再做可学习的时间加权（门控注意力），接同一个区间分类头。
- P2：每帧的 197 个 token（全局 token 加 196 个 patch token），先用门控注意力读取局部区域，再用与 P1 相同的时间加权和分类头。全局 token 也在 P2 的候选里，所以 P2 可以退回到 P1 的输入；P2 有收益就说明 patch 增加了信息。
- 同层同维：都取最后一个 block 的输出，经 ln_post 和 proj 到 512 维。P1 的残差帧模块与 P2 的 token 注意力参数量相同（131,712 对 131,457），总参数 329,091 对 328,836。
- 数据、帧、标签、权重、种子、优化器、批次和预算全部沿用第六轮：第六轮冻结的 ASRF 候选区间，每区间最多 32 帧，8 个 epoch，取最终权重，不用验证数据做任何选择。

## Task 1 — 预注册与代码

- [x] 配置 `configs/experiments/local_readout_v1.yaml`：问题、三组、特征层、读取方式与容量、预算、评估、统计、停止规则和启动条件。
- [x] `echo_routing/temporal/local_readout.py`（全 token 路径、门控注意力、残差帧模块、P1/P2 构造）；`tests/test_local_readout.py`（7 项）。
- [x] `scripts/round7_local_readout.py`（每个种子一次运行，直接调用第六轮的函数）；`scripts/round7_report.py`（三种子报告）。

## Task 2 — 启动检查

- [ ] token 0 与现有全局嵌入一致（≤1e-5）；第六轮的编码器一致性检查通过。
- [ ] 一次 sanity 运行（256 个区间、1 个 epoch）走通全流程，不作结果。
- [ ] P0 在三个种子、两个半区逐序列复现第六轮 V0 的验证预测。

## 训练期间的说明（2026-10-04，读取任何结果之前）

- 独立代码审查确认 P1/P2 公平（同一次前向、同样批次、同样调度和裁剪、时间模块和分类头初始化相同），P0 走的就是第六轮 V0 的函数。
- 审查指出"最强旧配置"在报告里是写死的。它沿用第三轮起固定的参照（selection：E0 的 B3 tuned 0.9344；calibration：B7 加 B3 0.9721），第六轮和几何轮都用它。之后在 calibration 上有更高的三种子平均：第二轮联合微调的 ASRF 0.9726，几何轮 G1 的 ASRF 0.9739（后者在 selection 上未达下限）。主判定仍用固定参照；报告另外写明，若改用 0.9739，最终目标的结论是否改变。
- `--workers 0` 时 DataLoader 会报错（默认 6，不影响本轮，第六轮也有同样写法）；为保持三个种子代码一致，队列结束后再修。

## Task 3 — 运行与报告

- [ ] 种子 0、1、2 依次运行，GPU 上一次一个任务。
- [ ] 报告：A4C/A5C 双向错误、区分度（AUC）、segment F1@50、收益涉及的独立来源录像和会话数；P2−P1、P1−P0、P2−P0；各组对 ASRF、E1 的 B3 tuned 和最强旧配置；最终门槛不变。
- [ ] 按预注册规则解读：P2 相对 P1 没有分类收益，就结束这个假设；只有分类收益，记为局部收益；两者都有，才值得进一步验证，最终目标另行判定。
