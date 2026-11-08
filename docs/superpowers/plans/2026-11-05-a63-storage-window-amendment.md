# A63 窄修订：300MB 存储追加与新执行窗口（待用户批准）

状态：preparation_only；未授权新窗口，未修改生效预算，未启动科学任务。此修订继续 A63，不启动 A64。

## 具体问题与方案

最新完整存储预估为1,207,032,484B：完整 local/future envelope480,466,994B +只读实测旧remote RUN/PUB136,059,133B形成current616,526,127B，原remaining全部非负项加resume bundle48MB后得到完整预估。详见 `docs/results/2026-11-05_a63_resume_preparation/complete_remaining_preview_v1.json`。原300MB管理floor仍为max(current_upper,300MB)，没有另加300MB；本修订不改该公式或降低估计。

建议仅将 **A63 整个生命周期的增量存储总上限从1,000,000,000B改为1,300,000,000B**。追加300,000,000B，不是再发一份1.3GB，不重置任何已用量。原累计tracked-path conservative upper硬限额40,000,000,000B保持；原前序基准38,650,409,858B +新A63总上限1,300,000,000B =39,950,409,858B，仍小于40GB。

原A63 GPU5400 occupied-s与CPU accounting28800 core-s不变，全部原费用、准备/审查/失败/连接与后续actual attempt继续扣减。128MB storage、3600CPU、7200wall closing预留及累计2GPUh预留继续保留。追加额度不是实耗或物理空间峰值声明。

当前CPU完整remaining预估20,444.337709s、GPU4,795.601100s、wall24,044.337709s均为工程预估，不是实际准入或物理保证。真实teacher geometry、完整6000 predraw、gzip最坏共存、最新du/费用与原完整矩阵仍需实际重算；任一实际硬限额不fit就停止，保留资源负结果，不缩矩阵、改模型或降低覆盖/错帧门。

## 新窗口与顺序

拟授权新的固定12小时执行窗口，**从启动正式执行流程时冻结T0并开始计时**（包含actual admission/源缓存/训练/核验/结题，不以首次SGD另开钟）。启动前固化当前费用/存储cutoff、用户授权raw SHA与稳定源码actual Git证明；deadline=T0+43200。原04:44–16:44UTC窗口、原资源停止及其所有字节永久保留；新窗口一旦开始不因断线自动暂停或顺延。

优先执行原S阶段：metadata只读adoption→真实完整准入→原固定FIT双teacher跨join缓存→可观测geometry/完整predraw→8aux×6000→独立48000/8models/32states核验→actual-Git辅助头seal→held-fold无标签选择分数及seal→两seed全部23项覆盖门。所有既有source/metadata不重抽，不以源logit拼接代替跨join forward。

后半薄IO实现与独立审查可并行收尾，但已执行的源码版本/合同不能覆写。**C0/C1必须等覆盖全部通过、后半软件完整审查通过、新确认源完成及fresh完整剩余allcaps通过后才能启动**。四formal各6000=24000、34inner、238outer、140×5000 paired、56support、34native、32regression、图QA/独立科学与资源审计/真实parent OS/保存模型/保护main结题全保留。

若覆盖失败：停止CONFIRM/formal/outer，完整保留供样覆盖负结果。若资源失败或外部阻塞早于覆盖：覆盖未评估，不当作供样假设失败。两seed、候选、部署S-L/N、损失、R2表示、校准划分、阈值规则与所有验收标准均不变；不自动A64。

## 审批生效条件

需要用户明确批准上述300MB追加和新12小时触发方式后，才创建新的窄预算/窗口授权sidecar与相应校验代码版本。旧config/card/basis/owner/已发布840项不改。新schema须显式绑定本修订raw SHA，严格限定只有存储增量总额1.3GB，CPU/GPU/累计40GB不能由任意配置值扩大。

未批准前继续按1GB拒绝开跑，不把准备、历史v3 true或当前预估称作新的actual admission。
