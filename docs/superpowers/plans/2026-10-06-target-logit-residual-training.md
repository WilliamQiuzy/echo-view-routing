# Target-Logit Residual Readout Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax. The project lead has selected the next bounded training comparison; after implementation review and technical checks, run the six jobs without requesting another research-direction decision.

**Goal:** Test whether a direct target-frame classification path reduces the native-clip errors introduced by the temporal readout while preserving useful temporal corrections and improving the complete system.

**Architecture:** Keep the C model's four trainable ViT blocks, relative-position temporal Transformer, windows and supervision. Compare a zero-initialised CLS classifier (Z) with the same classifier added to the current target-frame logits (R). Reuse the original three frozen C models as a required reference.

**Tech Stack:** Existing PyTorch E1 encoder, target-window manifests, AdamW training and checkpoint conventions, B3 evaluation, three paired seeds, English result report.

---

## 1. 决定与证据

**启动这一轮训练，方向只选“目标帧 logits 直达输出”。新增 Z、R 各三个种子，共六个模型。** 不同时冻结 E1、不增加损失、不换数据或预处理、不搜索新超参数。

依据是 `docs/results/2026-10-06_center_recognition_diagnostic/README.md`（结果提交 `502190d`、`b99f823`）：C 的窗口头减少了逐时刻错误，但每个种子、每个半区都新增 2–3 段此前两个分类路径均正确的原生录像错误。其逐帧头的退步较小，而且大部分被时间头修好。这是定位依据，病例少，不能表述为确定的因果机制。

本轮保留时间模块，检验能否在它修正零散错误时，让目标帧已有的分类分数参与最终决定。残差相加是常见设计，本项目尚无证据证明其有效或有方法新意。

原 N/S/C 轮继续按原标准记为未达标。新结果不回填旧表、不改旧结论。两验证半区已多次用于开发，仍属探索性；test 不读取，Pediatric 不使用，公开标签不修改。

## 2. 三个比较对象及其解释

令 `f_t` 为模型当前 encoder 的逐帧 embedding，`a_t = frame_head(f_t)` 为五类 logits，`c` 为实际目标样本。令 `h` 为原两层时间 Transformer 的 CLS 经原 LayerNorm 后的表示。

| 对象 | 最终 logits | window 输出层初始化 | 本轮训练 |
|---|---|---|---|
| C：原始控制 | `W h + b` | 原 E1 probe 权重 | 不重训；使用已有三个冻结 C |
| Z：初始化控制 | `Δ = W h + b` | W、b 全零 | 三个种子 |
| R：残差候选 | `a_c + Δ` | 与 Z 完全相同，W、b 全零 | 三个种子 |

**R−Z 为主要受控比较。** Z 与 R 的全部参数、初始状态、参数量和训练输入一致；区别仅为是否把目标帧 logits 加入最终输出，包括这条连接产生的梯度路径。两组的辅助逐帧头都从 E1 probe 初始化并继续训练。

**R−C 为必须同时报告的实际改进比较。** 仅超过 Z 不够：Z 的初始输出是均匀分布，不能把击败这个控制当成超过当前系统。Z−C 则说明只改变 window 输出层初始化有什么影响。R−C 同时涉及输出连接和初始化，不把其全部收益归于连接本身。

这里的 `a_c` 来自**同一模型当前的、可训练的逐帧头与 encoder**，不是另跑一个冻结教师。R 只在初始化时复现目标帧 E1 的 logits；训练后不保证保持 E1，更不保证 native macro-F1 下限。窗口中心采样仍为 stride=5，所以初始化也不能宣称复现 F-dense 的整段评估。

输出先相加 logits，再做 softmax。没有概率平均、学习 gate、固定混合系数、stop-gradient、蒸馏项、额外 margin loss 或基于参考标签的推理分支。

## 3. 文件与复用

- Create: `configs/experiments/target_logit_residual_v1.yaml`：新实验身份、两个训练 arm、配对来源、评估规则与解释规则。
- Create: `echo_routing/vit/residual_target.py`：复用 `TargetTimeHead`，仅增加目标 logits 的索引与相加。
- Create: `scripts/train_target_logit_residual.py`：加载配对初值、技术检查、训练与断点恢复。
- Create: `scripts/queue_target_logit_residual.py`：六次顺序运行、独立状态与日志。
- Create: `scripts/report_target_logit_residual.py`：冻结结果、三路输出诊断、完整系统表。
- Create: `tests/test_target_logit_residual.py`：目标索引、初始化函数等价、梯度与配置身份检查。
- Create: `docs/results/2026-10-06_target_logit_residual/README.md`、`report_tables.md`、无录像编号的 `summary.json`。
- Reuse unchanged: `echo_routing/vit/target_video.py`、`target_windows.py`、`echo_routing/temporal/joint_visual.py`；旧训练脚本中纯函数和 checkpoint helper；旧报告中的输入、窗口和评估 helper。
- 不覆盖旧模型、旧 manifest、旧配置、旧 run 目录。逐录像编号、特征和逐样本输出只存服务器。

## Task 1: 写入固定配置与配对初值

- [ ] 读取 `configs/experiments/ev9v_center_supervision_v1.yaml` 与旧 `runs/center_supervision_v1/manifests/index.json`，验证每个种子 SC manifest、原始 seed 初值、E1 export/probe 的摘要。
- [ ] 两个新 arm 均复用同种子 **SC manifest**，即每步 4 个原生窗口加 4 个混合窗口；均使用 C 的目标时刻硬标签。顺序、帧、gamma 和重复采样完全不变。不重新抽样，也不按新诊断中的错误录像加权。
- [ ] 每个种子从原 E1 export 和原 seed 的**训练前** head 状态开始，复制后将 `head.head.weight` 和 `head.head.bias` 清零一次。其余参数逐项相等。为 Z/R 保存同一份新初值及其摘要；不能从训练后的 C checkpoint 继续训练。
- [ ] 配置中明确新 run identity 包含新实验名、arm、全部有效配置、源训练配置摘要、manifest 摘要、新初值摘要和代码摘要。引用旧 C 时按其**原配置**校验身份，不能再用新诊断/新训练配置去计算旧 run identity。
- [ ] 在任何正式训练前提交配置与实现。预注册中保留旧评价目标的精确值，不四舍五入后再判定。

固定训练参数与 C 相同：

| 项目 | 固定值 |
|---|---|
| 输入 | 同一份 EV9V 五类训练库、旧 SC 窗口与数值顺序的 cache_v2 |
| 图像 | 旧 E1 图像路径：gamma、短边 256、中心裁 224、旧归一化；不切换 preprocessing 团队的新导出 |
| 时间输入 | 16 个样本，10 Hz，相对实际目标样本的位置，推理 stride=5 |
| encoder 可训练部分 | ViT blocks 8–11、ln_post、proj，其他冻结 |
| 时间头 | 原两层、8 heads、FFN 2048、GELU、pre-norm、dropout 0.1 |
| 损失 | `CE(final_logits, label[c]) + 0.5 * CE(frame_logits, frame_labels)` |
| 优化 | AdamW：encoder 1e-5、其余 1e-4；weight decay 1e-4；clip norm 1 |
| 预算 | 10 epochs × 375 updates，batch=8；375 步 warm-up 后 cosine；fp32，TF32 off |
| 权重选择 | 最终 epoch；训练过程不读取验证指标 |
| 种子与顺序 | 0、1、2；每个种子 Z 然后 R；单张 GPU 顺序运行 |

## Task 2: 实现唯一结构差异

- [ ] 在新模块中复用旧 `TargetTimeHead.forward()` 得到 `delta` 和 `frame_logits`，按如下规则形成输出；原模块默认行为不改。

```python
def readout(delta, frame_logits, rel, mode):
    # delta: B x 5; frame_logits: B x 16 x 5; rel: B x 16
    at_target = rel.eq(0)
    if not bool(at_target.any(dim=1).all()):
        raise ValueError("target sample is absent from the window")
    # Short clips may repeat the same target frame. Count it as one target.
    weights = at_target.to(frame_logits.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True)
    target_logits = (frame_logits * weights.unsqueeze(-1)).sum(dim=1)
    if mode == "Z":
        return delta, frame_logits
    if mode == "R":
        return target_logits + delta, frame_logits
    raise ValueError(mode)
```

- [ ] 用真实样本索引检查所有 `rel==0` 位置都指向同一来源帧及相同 gamma。不能写死第 8 个 token；边缘窗口和短片段重复样本均要测试。
- [ ] 两组使用同一参数树和同一 forward 中的 dropout 调用。mode 仅为输出规则，不创建额外可训练参数。不 detach target logits，R 主损失沿直接路径和时间路径一起反传。
- [ ] 新 checkpoint/optimizer/resume/status 均写入 `target_logit_residual_v1` 命名空间；旧 helper 若硬编码旧目录或只接受 N/S/C，不直接调用其 main，而写新适配入口。

## Task 3: 必要测试、冒烟与实施审查

- [ ] 单元测试覆盖 rel==0 索引、重复目标与缺目标失败；Z/R 参数完全相同；delta 为零时 R 等于当前目标帧 logits、Z 为零；非零 delta 时严格相加，不能变成概率平均。
- [ ] 用同一 fp32 embedding/同一 E1 probe 在训练帧上检查 R 初始化函数等价，max abs logits 差 ≤1e-5。encoder 对参考与 cache 的 parity 沿用旧阈值 1e-3 / 2e-2；分别报告，不能混成一个数。
- [ ] 梯度检查处理**零初始化的预期行为**：首次反传中，输出层之前的时间路径梯度为零是正常的；输出层和辅助逐帧路径应有有限的非零梯度。完成首次非零学习率更新后，再检查时间路径有有限非零梯度。不得为了通过旧的“第一步所有参数非零”检查而改初始化。
- [ ] 冒烟沿用 6 次更新、同 seed 重跑、3+3 resume 对 6 步、冻结参数摘要不变。Z/R 输入 digest 和新初值 digest 必须一致；不同 seed 的时间参数不能全部相同。
- [ ] 在 seed 0 上完成 Z/R 冒烟与报告 dry-run，只用训练样本/合成数据，不生成验证指标。确认旧 C 加载路径复现旧 forward，不改其模型。
- [ ] 独立实施审查重点检查：相加发生在 logits；target 对齐；Z/R 配对；旧 C 身份使用原配置；零初始化梯度；选参不读 selection。修复技术问题后提交，启动全部六次训练，无需再申请方向批准。

```bash
.venv/bin/python -m pytest tests/test_target_logit_residual.py tests/test_center_supervision.py tests/test_center_recognition_diagnostic.py -q
.venv/bin/python scripts/train_target_logit_residual.py --config configs/experiments/target_logit_residual_v1.yaml --arm Z --seed 0 --smoke
.venv/bin/python scripts/train_target_logit_residual.py --config configs/experiments/target_logit_residual_v1.yaml --arm R --seed 0 --smoke
```

以上为新 CLI 的实现契约，不表示脚本已经存在或测试已通过。

## Task 4: 跑完六个模型

- [ ] 实施新队列，完成状态、run manifest、最终权重摘要三者一致才可跳过已完成任务。失败时仅修技术问题；不得按早期验证表现删 arm、加 seed 或延长训练。
- [ ] 预计 GPU 训练约 3 小时，以旧 C 每次约 28.5 分钟估算；实际耗时、显存和恢复情况如实记录。启动后用进程、日志及 update 数确认，不把排队称为已训练。

```bash
.venv/bin/python scripts/queue_target_logit_residual.py \
  --config configs/experiments/target_logit_residual_v1.yaml \
  --status runs/target_logit_residual_v1/status.json
```

## Task 5: 固定评估，三个问题分开回答

- [ ] 六个新模型全部完成后统一推理。R/Z 各自提取训练后特征，不能共用旧 E1 特征。旧 C 三种子与 E0+B3、E1+B3、B7+B3、E1+ASRF 的冻结结果先复现。
- [ ] **主要系统输出**始终为各组规定的最终 logits：R 用相加结果、Z 用 CLS 结果。每组展示 argmax 和 B3，不能事后改用 frame head 作为参赛输出。
- [ ] **B3 设置**沿用旧 40 组网格：half-window [5,7,9,11,15]，分位数 [.80,.85,.90,.95,.97,.99,.995,.999]，NMS=5。每模型仅在 calibration 选满足 native F1≥.980、boundary recall≥.95 的最高 segment F1，平局按原网格顺序。冻结后只在 selection 应用一次。没有可行设置则报告失败，9/.95/5 固定回退仅作诊断。
- [ ] **识别是否保住：** 原生五类混淆、macro-F1、每类召回与分子/分母、源录像修好/新错。主要比较 R−Z，并列 R−C 和 Z−C；两半区、全部种子完整报告。
- [ ] **时间收益是否仍在：** 同样的原生/单录像窗口/混合窗口目标时刻错误率及来源录像等权结果；原 474 目标的配对上下文 probe；原生内部切分。不要只报 clip accuracy 而隐藏碎片或目标时刻错误增加。
- [ ] **完整系统是否赢：** segment F1@50、边界 precision/recall/F1、内部切分、同类拼接误切和 native macro-F1。R−Z、R−C、强基线均报告；不同 B3 选参下的差异属于完整系统效果，不能全部归于输出连接。
- [ ] 每个 R 模型同时做一次 **delta=0 的推理诊断**，其结果必须等于本模型 J-grid（只读当前 target frame logits）。对 J-grid 与最终输出列出修好/新错及片段碎片变化，量化时间修正保留了什么、破坏了什么。该路径不选参、不作为新的候选；native 及 argmax 系统结果即可。
- [ ] 保留无录像编号的汇总与服务器逐序列概率档案；此次保存完整逐序列预测，避免今后只能复现 summary 行。

统计沿用每种子配对 bootstrap 2,000 次、seed 20261002；native 按 recording session，构造流按旧方法并明确重复来源依赖未完全处理，区间近似且探索性。三种子区间同向排除零才称“一致提升”。包含零不等价于无差异或没有损失。C/Z/R 不是三个独立数据集。

## Task 6: 成功标准、解释与结束

**最终目标保持不变，只有同一个 R+B3 可以接受这一轮的候选称号：**

- selection：三种子平均 segment F1@50 ≥ `0.934409687184662 + 0.01`；
- calibration：三种子平均 ≥ `0.9721421709894332 + 0.01`，另报告对 `0.9738935784747763` 的敏感性比较；
- 每个种子、每个半区 boundary recall ≥.95；native macro-F1 selection ≥.973、calibration ≥.980；
- 没有可行 B3 设置的诊断行不能通过目标，不选择单个最好的 seed。

解释分别写，不能互相代替：

1. R 优于 Z，但未改善原始 C：只能说明在零初始化控制下直接路径有帮助，没有推进当前最佳读出。
2. Z 与 R 都优于 C，而 R−Z 无一致差异：效果可能主要与初始化有关，不能宣称残差路径是原因。
3. R 的 native 改善，但逐时刻/分段退步：属于识别与时序收益的取舍，不能宣布问题解决。
4. R 对原始 C 和 Z 均出现 native 改善，并保留目标时刻与分段收益：报告局部支持及全部区间；若未过上面的完整目标，仍不能称超过强基线。
5. 无论结果如何，本轮六次训练后停止。没有自动冻结编码器、加蒸馏/门控、调残差系数、增加训练步数或搜索新阈值。

- [ ] 用 `scripts/freeze_model.py` 的现有流程冻结全部六个新模型与摘要，即使结果不好也保留。
- [ ] README 用英文，开头回答：训练了哪六个模型；R−Z 与 R−C 各怎样；逐帧降噪与整段翻转是否改善；是否超过固定强基线并满足所有约束。写明六次新训练和三次旧控制的区别。
- [ ] 只提交本轮文件、不推送，不修改用户其他暂存文件；论文继续暂停。最终报告直接附逐种子主表、修好/新错来源数和实际成本。

## 当前状态

这是待执行的训练任务书。未在本次规划中新增模型实现、启动 GPU 或产生比较结果。

2026-10-06，独立计划审查通过，未发现实质性实施或比较设计缺口。审查覆盖 R−Z 与 R−C 的区别、零初始化梯度、目标索引、旧模型身份、评估和停止规则。代码实现审查与技术检查仍须在正式训练前完成。
