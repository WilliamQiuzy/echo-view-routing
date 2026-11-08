# A63：资源门槛停止，供样覆盖未评估

A63 没有进入 FIT 新教师前向、48000 步辅助训练或 C0/C1 正式训练。供样覆盖门和 C0/C1 采样假设均未检验；没有本轮外层效果或安全性结果。本地collector和父进程等待已实际OS0；主工作区保护集成尚待独立记录。

用户授权窗口固定为 **2026-11-05 04:44–16:44 UTC**。原卡的历史 DRAFT 字节保留；[owner_approval.json](owner_approval.json) 是执行授权。完整经过、失败和限制见 [FINDINGS.md](FINDINGS.md)；明确停止决定见 [execution_stop_decision.json](execution_stop_decision.json)。

- 已完成：402 source sessions 的 201/201 固定划分、两组各1500新 FIT mixed 及两组各120确认配方冻结；两次 CPU 工程测时和预算估计修订；软件分 scope SPEC→QUALITY 审查。
- 未执行：FIT/确认教师、8个辅助模型、coverage、4个 C0/C1 模型、τ、238 outer /34 inner /140×5000 CI /56 support /34 native /32 regression。计划数不冒充输出。
- 真实历史准入：07:35 UTC 原准入 OS1（storage/window 不fit）；09:48:57 UTC v3 准入实际 direct-wait OS0/true，完整父回执保留。但原结果 JSON 尚因 SSH timeout 未镜像，不能称完整远端复核。
- 当前停止：10:24 UTC 对同一已审公式重算，剩余 wall=23684.337709s（含7200s收尾），预测结束16:59:03 UTC，已超过原deadline约903.13s。它是工程估计与启动门，不是物理下界。准备/镜像次序问题和网络不可达均保留，不把过去的 true 当现在启动资格。

[local_known_worker_receipts.json](local_known_worker_receipts.json) 列5个实际远端 worker：4个OS0、1个预算拒绝OS1。最后v3行来自实际父进程 stdout 的逐字段恢复，**不声称已取得原远端JSON字节**。当前远端boot/PID/GPUowners/active/pending/最终du未建立；不声称服务器空闲。工程小头和toy SGD并非0计算，逐scope费用保留；正式A63 source/aux/formal计数均为0。

本地collector 83093已由caller 83032实际wait OS0，外层也实际wait caller OS0。snapshot/parentwait分别见 [resources_resource_stopped.json](resources_resource_stopped.json)、[local_resource_stop_OS_receipt.json](local_resource_stop_OS_receipt.json)。真实OS与费用证据另见 [admin_local_stop_parent_wait_total.json](admin_local_stop_parent_wait_total.json)；整项CPU包含子进程，不能在snapshot/补集之外再次全量收费。资源测量区分known CPU、显式unknown allowance、last-known remote sample和prospective storage；未知尾部不填0。

已知问题的未执行 analysis 两路径及未完成 geometry CLI 不进入本轮 main payload，原分支代码/测试/IssuesFound/费用保留：[root_analysis_withdrawn_and_provider_observation.json](root_analysis_withdrawn_and_provider_observation.json)、[withdrawn_aux_geometry_draft.json](withdrawn_aux_geometry_draft.json)。

下一步动因仍是验证来源隔离后是否有足够独特危险N-top供样。本轮没有产生可用于决定 C1 是否有效的新模型证据；不降低覆盖门、改τ或重置窗口，不自动启动A64。最终main集成、原8条staged保全及Index仅追加的实际验证由独立root private publication receipt记录。

费用最终public截点见 [postclosure_fee_addendum.json](postclosure_fee_addendum.json)：known measured CPU 274.394235s，未知行政allowance 127.0s单列；之后source清单提交/main发布费用由private实际回执另补，不重收外层总量或修改已封资源。独立终审仅批准报告准确且边界充分，见 [final accuracy review](review_9158754_independent_final_accuracy.json)，不表示研究目标成功、远端闭环完成或main已集成。
