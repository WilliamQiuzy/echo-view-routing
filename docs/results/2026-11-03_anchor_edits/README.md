# A59：保留A08锚点的局部修复

状态：方案fff5366/7a3e0bf已冻结并通过独立审查；六项候选/特征检查及十五项模型生命周期检查、独立两阶段复核已通过；资源、构造bank、持久化缓存和精确候选空间oracle已完成局部检查；原stride5与无标签推理wrapper新增4项检查通过，完整trainer/readout/aggregate接口继续集成。四个局部修复头已各完成6000步，共24000次正式更新；全部12份内层诊断记录已完成并通过独立审计，正在封存后统一外层。用户2026-11-03明确授权结合A57/A58重规划并在剩余预算继续，不需常规确认。

A58的JEPA增量在短段压力calibration成立，但整段联合解码仍损害A08识别；g=0对照也退化。A59让A08原stride5+B3决定默认输出和所有替换类别，JEPA只作为局部修复价值的证据。最多一个修复，使训练的单动作utility与实际动作一致；多数类保护对所有序列使用同一规则，不能用native/真边界/sourceID作推理开关。

A14–17已经训练过局部路由，A59不声称路由或utility学习是新方法。新条件与完整控制见[执行卡](../../superpowers/plans/2026-11-03-a59-anchor-edits.md)。机械保住原生macroF1不等于保住帧准确率、真短段或临床识别，所有损害仍完整报告。

继承GPU24h/CPU72h/40GB与原更早deadline，不重置费用。起始GPU剩6.684573h，其中2h收尾储备；CPU conservative receipts+allowances剩54.036262h；tracked上界剩6375507422B。用户重启后SSH已于2026-11-03 16:10:52纽约时间恢复；root真实只读核对GPU0MiB/无app、A58closed、A59尚无runtime。唯一执行worker继续真实源资格化和实测全成本准入，正式更新仍0。

历史连接失败保留在readiness：21:06:59UTC最后一次TCP超时，21:10:52UTC用户重启后SSH恢复。原预算/费用/更早deadline均不重置。监督器launch-intent与锁内账本两个问题已在0507f02修复并通过独立SPEC+QUALITY，1a09c8f补齐锁内本地测试费用，原IssuesFound/失败与CPU费用保留。真实metadata与400/214源严格载入均已完成，两个worker由父监督直接观测OS退出0。root核对31个Git对象和server SHA全部一致；FIT/inner为4236/928条，下一步实测FIT源前向与CPU候选训练图，再做覆盖完整矩阵的成本准入。

真实FIT源GPU图检查已完成：7242由父监督直接观测OS退出0，GPU占用33.593256秒，原stride5 A08原始logit误差1.90735e-6、JEPA概率误差0。随后CPU小panel检查在12次工程更新后的逐tensor非零梯度断言失败（8696 OS1/CPU4.870384秒）；原失败保留，worker检查特征方差并单独冻结非退化工程control，正式数据/损失不变，全FIT图核验仍待缓存完成。正式更新0，这些均不是模型性能结果。

小panel失败原因已由冻结输入证实：仅1个构造候选、0个原生候选，两arm均只有1行特征，98列标准化后全零，因此第一层weight梯度应为0。单独冻结的v2检查保留这一实测false，并以不进入正式数据的非退化control验证全部tensor梯度；11485由父监督观测OS0/CPU3.390718秒/GPU0。72次新工程更新与原失败12次分列，正式仍0；完整四模型/70行/96对照成本准入后直接继续缓存，全FIT实际候选图在正式准入前再核验。

完整缓存/矩阵成本准入已通过并在c016469/71d3f9e封存：预测累计GPU21.434302小时（包含未花费的2小时收尾储备）、CPU accounting23.011658小时、tracked上界34.484451GB；预测不是实耗。root实查唯一14991/14974及start_ticks306343一致，A08FIT已提交1309/4236条，39个实际Git对象与server SHA一致。缓存完成后继续全FIT真实候选图与正式准入，再执行四模型各6000步。

正式训练33027/33010已由父监督直接观测OS退出0：143.888347秒wall/70.453209 CPU秒，CUDA隐藏、无GPU context。root独立读取四份latest：各update6000、全部Adam计数6000、各6000条physical trace；46个formal/64个audit实际Git对象与server SHA均匹配。全FIT实际553候选（121native/432constructed），148工程更新与24000正式更新分列。内层不选checkpoint；尚无新模型外层A08比较结果。下一步全部权重/12inner/policy actualGit封存，再全70行/96比较/oracle/闭环，监控继续。

70行固定读出已齐（ebd0526），root独立从5×5混淆与M/P/T重算主表并匹配server SHA。J-G两seed主selection片段F1 .9498525/.9507874、cal .9808429/.9779904，均低A08 .9563492/.9826923；J-G相对A-G只在selection点估计有增量，cal反向。short16 selection FN6/6对7，但FP17/16对9；cal FN4/5对5、FP11/13对8。G宏F1恒等是机械属性，seed360cal帧准确率.993437低于Base .993926；不能称稳定性/短段净收益成功。96配对区间和12oracle继续，不由负点估计调参或提前停止。
