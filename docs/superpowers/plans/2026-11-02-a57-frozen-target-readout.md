# A57 冻结后完整目标任务读出计划

> **For agentic workers:** Use `executing-plans` to implement this plan. Keep the single existing execution owner; use read-only review for concrete issues. No new training, hyperparameter search, source gate or user approval is introduced here.

**Goal:** 在预定32,000步矩阵结束后，独立核对全部模型，再一次性输出固定2000/6000表与inner-selected checkpoint包；所有选择来自声明的训练内层，外层只作复用开发集探索。

**Architecture:** 冻结producer是55745e1，实际GPU队列保持其源文件、数据、配置和原始card身份不变。未来读出代码及本文单独冻结；不覆盖服务器仍被producer pin引用的原始strategy文件。完整科学合同及前瞻增加条款见[研究方案](2026-11-02-target-task-research-redesign.md)，本文件是实际读出执行清单，不修改已训练模型。

- [ ] 单一formal队列完成L/F/J-low/J-high首轮与F/selectedJ/L续程，第二seed F/selectedJ各6000；正式逻辑/物理更新及任何失败/重放分别保留。
- [ ] CPU模型审计验证逐步损失/梯度有限、205/56/2个optimizer张量绑定（实际数量以模型状态为准）、全部moment形状/有限性/步数/学习率，J149个活跃视频张量变化、9个未用张量不变，以及所有里程碑与inner缓存SHA。
- [ ] 冻结11个固定step行和6个inner-selected package行；每行绑定模型、源、inner输入、原始概率/预测/指标身份。相同checkpoint SHA重复行共用同一外层概率缓存。
- [ ] 在inner缓存上独立重数raw指标，核对expected(N,5)/finite/归一化、输入checksum、逐位置argmax预测及全部原始inner指标；通过后才运行原V01固定40点B3网格。选择顺序为multi segmentF1、native macroF1、较少A4C/A5C错误、较小网格顺序。实际选择文件先提交/冻结，再读取外层数据。
- [ ] 仅用已验证metadata-only half adapter准备selection/calibration原生、pair、multi输入；完整64位置footprint只包含本half来源。固定10Hz/tubelet2/stride32/正三角概率拼接保持producer定义。
- [ ] 核对原A08 all_s0/s1/s2权重SHA、完整缓存SHA、model identity及sequence/window/centre/marker/order，保留原B3选择；不为参考模型新增前向。三个seed的历史指标须完整重现；s0为预定配对参考，另外两个均报告。
- [ ] 提交/冻结新reader代码、本文、模型manifest、inner-B3选择和外层输入身份，再启动唯一GPU reader。使用仅两列路径清单视图；输入作用域披露保留producer曾暂时解析未用字段的事实。
- [ ] 每条序列原子写概率并提交cursor；已提交前缀校验SHA/journal positions/expected原始长度/五类概率shape/finite/归一化，含pairs。pending歧义停止审计，不静默重跑。记录逐checkpoint/half原图解码/模型/拼接/缓存成本；任何恢复时明确成本覆盖范围。
- [ ] CPU从缓存独立复现raw指标，并运行冻结inner-B3及A08原B3。匹配统计保留两seed的6000步J−F raw主比较和B3补充，2000次配对stream/session bootstrap；共享来源构造流的区间只条件于固定bank，不叫独立患者不确定性。
- [ ] 同时报告legacy短段≤16位置和本轮补充≤10位置，原生五类混淆、A4C/A5C双向错误、真实边界P/R、真短段损失/恢复与短假段；少切不单独算成功。图表只含聚合值。
- [ ] 核对21个当前聚焦测试（最终执行时以实际测试数为准）、所有模型/输入/source/reader身份和完整受保护数据边界。关闭本队列进程，记录实际GPU-job-wall、实际已计量CPU-core、计量覆盖/未计量准备限制及新增峰值；历史费用不同口径原样链接，不拼成虚假的同口径实际总计。
- [ ] 发布本地报告/index/结果commit，保持main用户暂存和私有IDs/图像/权重/逐例缓存。研究阶段结果不自动等于发表或临床目标完成；未来结构若受开发观察启发须另写探索card。

**Bounds:** 共享原新研究阶段24GPU-job-wall小时/72实际CPU-core小时/40GB/48墙钟小时，完整收尾储备至少2GPU小时；本读出不是新增预算。旧source133安全门槛不进入本轮决策。官方test、Pediatric、Unity与隔离媒体不产生新前向；其标签不用于训练或模型/解码选择。

**读出调度记录（2026-11-03，在任何新外层模型前向前）：** 为先提供预定主比较，`readout_execution_order.json`固定先执行seed1/seed2各6000步F/J四行，其余冻结行保持原相对顺序，全部17行和相同checkpoint去重语义保留。全部inner-B3选择提交后，先用已pin的A08原缓存/原B3参数在完全相同外层时间轴上重现三个seed的历史指标并补足≤16/≤10短段计数，才发布候选比较。完整已完成行的raw和冻结inner-B3聚合可以先公开，明确最终候选缓存独立审计尚待完成；不把进行中的残缺序列计为完整行。此调度不改变模型、端点、选择规则或MD成本档位。
