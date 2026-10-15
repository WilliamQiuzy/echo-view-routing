# A06-R1: Official FACT Fallback and Formal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Claude is the intended executor. The owner authorizes this explicit gate amendment, minimal implementation/revalidation, the original three formal runs and the unchanged staged evaluation/report within the cumulative A06 budget. Passing the revised checks authorizes immediate launch without waiting for the owner.

**Goal:** 修正一项超出官方系统正常输出所需的smoke内部状态要求，然后实际完成尚未执行的FACT视频特征比较。

**Architecture:** 官方GTEA FACT、冻结512维视频特征、三个初始态、训练批次、优化器、18750更新和官方解码全部沿用A06。只对单段序列允许官方全-null token回退，保持多段token要求；由旧check的可追溯证据和小规模新检查生成新的launch记录，不改原失败记录。

**Tech Stack:** Existing FACTHead and A06 training/report helpers, PyTorch, CPU-only saved-record validation, pytest, new revision configuration and provenance.

---

## 1. 负责人决定与理由

**A06 v1按原门槛正确停止，记录保持不变；它没有正式训练，没有FACT−ASRF结果。** 本文为 **A06-R1，看到训练smoke结果后的事后gate修订**，不是新模型假设、已预注册的原始计划、负结果翻案或额外性能搜索。不换名称假装新实验：科学问题、三次尚未执行的正式训练与A06相同。

原任务由负责人加上“每个参考段匹配token都必须正确分类”的启动要求。但[所固定的官方代码](https://github.com/ZijiaLewisLu/CVPR2024-FACT/blob/7bd81bda2b84618a3e23876a2637a82f24881122/models/blocks.py#L195-L205)中，`Block._eval`明确在所有token判null时返回帧分支argmax。这是官方的合法推理路径，不是我们新增的补救。A06的16条smoke序列已全部得到正确transcript、segment F1@50=1和一对一匹配；七条多段流的匹配token全部正确。未通过的四条都是单段且全-null，走官方回退仍输出正确。**要求每条单段都必须由token命名，并非测试完整官方系统的必要条件。** 这是修订依据，不说明token的优化问题已经解决，也不说明正式训练必然成功。

不延长2000步smoke、不换样本、不混批重训、不找一个新的“好检查点”、不改loss或null权重。直接使用已完成的**最终2000步**检查记录评估下列限定修订；不是从历次检查点选最有利者。多段流若未学会token，仍不放行。

前序：F02的延长smoke和E1图像输入只作历史，不沿用“多训就会过”的推断；A01、A02–A05、F03/F04和V02/V03的关闭范围保持原样。A06没有读development，所以这项修改发生在任何新模型开发结果之前，但它仍是事后程序修订，报告必须明示。

## 2. 不可变来源

本地核对的SHA256如下，配置先固定并提交；新计算先核验。

| 来源 | SHA256 |
|---|---|
| 原A06任务书 `docs/superpowers/plans/2026-10-15-fact-video-feature-training.md` | `cbd5f8e0d8a5bda0dc4afb27dcf29fdced8bef61b0948bfd1912ab9161fb49eb` |
| 原配置 `configs/experiments/fact_video_feature_v1.yaml` | `f9d3d51daeadcf9f9f5728bde45b8b8623bb2e07a21053c44fa657301a17f9a9` |
| 原prepare `docs/results/2026-10-15_fact_video_feature/preparation.json` | `94991e928ef99e1b00507ec9e5d4ccf9fe7730f9f20fdba84e3006e2b2f11c6b` |
| 原失败check `docs/results/2026-10-15_fact_video_feature/checks.json` | `9061032ef98c7527ef8eb46fe706f7ffe1dc82db66409e510db8c0eb00c859c1` |
| 原停止README `docs/results/2026-10-15_fact_video_feature/README.md` | `72089029ad49160d1a59a59e777cde3d0e46da0053b76ba01b8b7f125a947849` |

原prepare/check的server副本与以上文件也必须相同。继承原配置的8个来源pin、完整缓存hash、官方commit/30文件digest、环境、batch/initial state SHA和输入规则。查明旧check的实际training code digest `ae7531732dd0ea8be0a3aa09b9f94f236cac5006cf15163bb36de2d36d548546` 对应的源码版本，并和新执行的训练关键路径逐函数比较。严禁仅把旧`passed=false`改成true。

## 3. 只改这一项启动要求

全部其他checks照旧，旧check的失败项必须**仅为** `matched_classes_right`（及其聚合`all`、`ok`、`passed`）。在最终2000步的16个row上逐条检查：

```python
common = (len(rows) == 16 and all(
    r['transcript_exact'] and r['segment_f1_50'] == 1.0
    and r['match_one_to_one'] for r in rows))
def token_ok(r):
    if r['tokens_used'] == 0:
        return (r['segments'] == 1 and r['matched_token_class_correct'] == 0.0
                and r['transcript_exact'] and r['segment_f1_50'] == 1.0)
    return r['tokens_used'] > 0 and r['matched_token_class_correct'] == 1.0
gate_fit = common and all(token_ok(r) for r in rows)
```

- 仅允许**单段、全-null、官方输出正确**这一例外。单段有active token却匹配错类、多段全-null、多段匹配错类、transcript/F1/一对一失败，均拒绝。
- 全部单段/多段的token数及未匹配token情况仍完整报告；不要求每条单段恰好激活一个token，因为原检查也未要求。
- 在CPU以合成logits直接测试实际官方 `_eval`：all-null时等于frame argmax；存在active时走原官方融合公式。测试只能验证解码分支；旧记录提供这四条实际训练序列的正确输出，不声称新重跑了其模型或产生了新实测fallback logits。
- 旧padding、初始/过拟合态五假标签隔离、6步复现/3+3恢复及finite gradient记录必须逐字段核验，不能从全局“除了smoke都通过”一句话推导。
- 原梯度记录含三个已解释的zero/missing参数，保留具体签名和解释，不把继承记录改说成“所有参数都有非零梯度”；出现其他缺失或非有限梯度须定位停止。
- 单段fallback是在官方模型里定义的；限制例外只适用于单段是本修订的负责人决定，不声称官方只在单段允许fallback。

## 4. 文件、实现与证据继承

**新增：** `configs/experiments/fact_video_feature_v2.yaml`、`scripts/revalidate_fact_video_feature.py`、`tests/test_fact_video_feature_revision.py`；结果目录 `docs/results/2026-10-15_fact_video_feature_revised_gate/`，运行/检查点/冻结模型均用 `fact_video_feature_v2` 新namespace。原目录全部只读。

**允许最小修改：** 既有 `scripts/train_fact_video_feature.py`、`scripts/fact_video_feature_post.py` 和报告入口，使namespace/记录路径可从新配置获得，并显式支持一个经校验的revalidation记录。旧v1配置和tests必须继续保持原gate行为，默认不能跳过失败check。无需新框架或复制整套训练系统。新来源/identity覆盖修订配置、task SHA、原证据hash、相关执行代码及官方源码。

namespace改动须贯通`identity_payload`、queue/logs、checkpoint、freeze、train-fit和报告`verify_runs`，测试确保没有任何R1文件写入v1路径；不要靠临时修改模块全局NS伪装新身份。

### Task 1 — 配置、红测试与实现

- [ ] 先提交v2配置：保留v1原任务，明确本修订引用。`seeds/data/fact/monitor/evaluation/goal/readings`的**科学内容逐字段相等**；仅task/source provenance、namespace/paths、检查接受政策和账本汇总允许改变。测试拒绝修改lr、18750更新、initial state、输入或任何目标。
- [ ] 先提交新失败测试：原check仍失败；新限定gate接受已固定的四条单段例外；逐一改变segments、tokens_used、class correctness、transcript、F1、matching、isolation、code/hash均拒绝。新旧CLI及queue必须实际调用相应gate，不能只测试一个未用于启动的predicate。
- [ ] 实现revalidate入口，CPU核验old prepare/check/source/code/row内容和全部非smoke条件。保留原`passed=false`；在新记录中分列 `original_gate_passed=false`、新政策及逐项结果、证据来源、继承与新执行的检查，附原SHA。
- [ ] 对实际复用的model construction、loss/update、batch/RNG、optimizer/scheduler、epoch loop、官方decode，比较原版本和新版本的代码/AST与配置。只允许路径/身份/guard plumbing改变；数学/训练语义差异即停止，不能继承旧checks。
- [ ] 新prepare写新文件并重新验证全部source；可以从旧prepare继承已核验记录，但分别注明“本次重新检查”和“由不可变旧证据继承”，不得给旧证据伪造新的运行时间或source identity。
- [ ] `python -m pytest tests/test_fact_video_feature.py tests/test_fact_video_feature_revision.py tests/test_report_fact_video_feature.py -q`；独立实现审查及应修复核。审查须核对旧失败不是被篡改、异常只限单段、继承证据确实支持新路径、queue没有绕过guard。

### Task 2 — 小规模重新检查与启动

- [ ] `python scripts/revalidate_fact_video_feature.py --config configs/experiments/fact_video_feature_v2.yaml --prepare`，再 `--check`。只读训练数据和旧停止记录，不打开development。
- [ ] 新路径用原seed0未训练初始态做6更新重跑及3+3恢复（容差1e-6），并复验一个正式混合batch的padding和优化器绑定；这是小规模实现检查，不是延长或重复训练smoke，不用其权重初始化正式run。
- [ ] 新计时5warmup+50正式更新，使用旧投影和新投影的较大值估余下训练，保留实际monitor/save/post-training估算。全部新增GPU重新检查上限0.05小时；不得追加2000步overfit以期通过，旧最终smoke证据通过新predicate即可。
- [ ] 生成新的revalidation/launch记录并提交；只有来源/代码语义/新gate/新检查/累计预算均通过才允许queue。没通过就停止并报告具体原因，不再改gate。
- [ ] 通过后直接启动原定三个正式训练，seed0/1/2，每次从原F02未训练初始态开始；smoke/check权重绝不用作formal初始化。核对进程、日志、更新数后报告已启动。

### Task 3 — 完成原定比较、报告和关闭

- [ ] 原A06第4节Task3–4的half隔离、freeze、calibration、manifest提交proof屏障、一次selection、2000次配对bootstrap、六指标及FP/FN、源录像/session计数和独立核查全部照原文执行。三个新模型全部冻结后才读development。旧参考保持原参数。
- [ ] 主问题仍是FACT−A01 ASRF；必列Linear+B3、Frozen Ordered+B3和MS-TCN，旧目标/每seed约束不变。正式结果只属于R1修订后的执行；不叫原v1预注册方案通过。
- [ ] 每epoch在固定16序列的已有监测中，增加single/multi分层的active/null token数及匹配正确率。这只是用已执行monitor forward作日志汇总，不改变训练、选点、延长或触发新实验，也不要求额外全库推理。
- [ ] 英文README首段注明：v1因严格gate停止，负责人看到smoke结果后修订single-segment gate，原记录链接；为什么允许回退、未解释token优化、模型/配方/目标没有改变。保留继承检查和新检查之别、累计账本及任何停止。后续性能只能作为反复使用开发集上的结果。
- [ ] 独立核查覆盖gate例外、原/new源码和训练语义、全部模型/统计/half屏障及累计成本；仅本轮文件提交、不push。索引保留A06 v1停止条目，R1由pending换实际结果，不能删除或覆写原失败。

## 5. 累计预算与停止范围

**v1+R1合计仍≤12 GPU小时和4 CPU执行小时，不是另给一份预算。** 原成本在v1 append-only ledger，以其实际总和为准（报告约GPU0.34、CPU0.90小时），新ledger只记新发生的阶段。累计函数读两份账本，来源SHA/边界条目固定，不能漏旧成本或把旧行复制成新开销；并行/重叠不双计。

旧11.8109小时投影已含v1 check开销与1.25小时预留；新检查会减少剩余余量。队列前与每个run前重估：累计已花 + 全部剩余run保守估算 + 剩余post work + 1.25小时预留≤12。不足即停存进度，不缩run、不删控制、不加预算。

这是唯一一次本项gate修订，不增加模型、数据、epoch或搜索；新gate仍失败则停。正式报告后，沿用A06非互斥判读与end。无需负责人回来才执行已授权步骤；遇科学内容不一致、证据缺失/不兼容或预算不足，停止并如实报告。

test封存，Pediatric不读，公开标签不改，数据/私有编号/逐案例输出只留server。论文暂停，8个暂存理论文件及其他改动原样保留。旧A06 task/config/prepare/check/README和源码版本在git中可追溯，不改旧失败记录。

## Plan review record

2026-10-15完成独立计划审查，结论 **Approved，无阻断问题**。审查核对五个来源SHA、旧training code digest、最终smoke证据、官方回退、继承检查与新路径、namespace及累计预算。其建议保留三个已解释zero-gradient参数的签名并核查全套namespace路径，已写入本文。正式实现/新检查尚未执行；本审查不代表训练启动或性能结果。
