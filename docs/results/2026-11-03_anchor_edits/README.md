# A59：保留A08锚点的局部修复

状态：方案fff5366/7a3e0bf已冻结并通过独立审查；六项候选/特征检查及十五项模型生命周期检查、独立两阶段复核已通过；资源、构造bank、持久化缓存和精确候选空间oracle已完成局部检查；原stride5与无标签推理wrapper新增4项检查通过，完整trainer/readout/aggregate接口继续集成。正式模型更新0，新GPU作业0。用户2026-11-03明确授权结合A57/A58重规划并在剩余预算继续，不需常规确认。

A58的JEPA增量在短段压力calibration成立，但整段联合解码仍损害A08识别；g=0对照也退化。A59让A08原stride5+B3决定默认输出和所有替换类别，JEPA只作为局部修复价值的证据。最多一个修复，使训练的单动作utility与实际动作一致；多数类保护对所有序列使用同一规则，不能用native/真边界/sourceID作推理开关。

A14–17已经训练过局部路由，A59不声称路由或utility学习是新方法。新条件与完整控制见[执行卡](../../superpowers/plans/2026-11-03-a59-anchor-edits.md)。机械保住原生macroF1不等于保住帧准确率、真短段或临床识别，所有损害仍完整报告。

继承GPU24h/CPU72h/40GB与原更早deadline，不重置费用。起始GPU剩6.684573h，其中2h收尾储备；CPU conservative receipts+allowances剩54.036262h；tracked上界剩6375507422B。用户重启后SSH已于2026-11-03 16:10:52纽约时间恢复；root真实只读核对GPU0MiB/无app、A58closed、A59尚无runtime。唯一执行worker继续真实源资格化和实测全成本准入，正式更新仍0。

历史连接失败保留在readiness：21:06:59UTC最后一次TCP超时，21:10:52UTC用户重启后SSH恢复。原预算/费用/更早deadline均不重置。监督器launch-intent与锁内账本两个问题已在0507f02修复并通过独立SPEC+QUALITY，1a09c8f补齐锁内本地测试费用，原IssuesFound/失败与CPU费用保留。真实metadata与400/214源严格载入均已完成，两个worker由父监督直接观测OS退出0。root核对31个Git对象和server SHA全部一致；FIT/inner为4236/928条，下一步实测FIT源前向与CPU候选训练图，再做覆盖完整矩阵的成本准入。
