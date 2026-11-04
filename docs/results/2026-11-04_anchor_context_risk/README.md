# A62：固定S-L的动作级风险与锚点上下文

用户已明确批准实施、训练、全矩阵、独立核验和结题，不自动A63。已批准卡原字节保留，独立授权见[execution_authorization.json](execution_authorization.json)。T0=2026-11-04 15:32:41纽约，截止2026-11-05 03:32:41纽约（1793867561 UTC epoch），不可重置。

当前 metadata 与首轮 cost-only 已实际执行并直接观察到 OS0；正式更新0、teacher source NN0、实际FIT72更新0，未形成模型效果结论。metadata冻结540条新构造流/1080双producer payload计划、CAL319/AUDIT309原生轴及50/50 session划分，未混入旧300 mixed；见[input_manifest.json](input_manifest.json)及[metadata实际收据](process_metadata_1.json)。首轮cost-only为79.485973秒wall/74.331037 CPU-core秒，含10次synthetic MLP forward、0SGD/0teacher，见[cost_profile.json](cost_profile.json)和[实际OS/费用](process_cost_1.json)。

原公式的剩余495980.593774 CPU秒仅为[保留的成本预览](cost_preview_v1.json)，不是actual admission、硬件下界或模型失败。正在按[COST_REFINEMENT.md](COST_REFINEMENT.md)独立审查合法候选/RLE/完整上下文/长K成本分项；新实测及完整remaining allcaps通过前不启动teacher。原metadata72/cost77 contracts及原公式/profile保留。

旧exclude I/O使常规worktreecheckout失败，已保留真实OS128/费用；只对本轮新index和Git blobs精确物化，原main/index/config/exclude未改。所有已物化blob SHA1核验通过。当前服务器/GPU所有权须以每次实际启动前的新资格化为准，旧快照不构成准入。

[已批准执行卡](../../superpowers/plans/2026-11-04-a62-anchor-context-risk.md)及basis/review不改写，历史DRAFT flags由独立用户授权覆盖。新增1800GPU秒/21600CPU秒/1.25GB上限，累计24GPUh/72CPUh/40GB不重置；实际各阶段/失败/review/fixture费用继续exactonce计入新账。
