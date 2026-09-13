中文讨论摘要 · 2026年9月12日

# 1．建议先确定的研究方向

我们建议做“基于公开数据的心超视频可靠切面分段与分发”（selective temporal routing）：输入录像，输出每段的起止位置、切面名称，以及接受或暂缓判断。首批用户是整理心超资料的研究员和数据工程师。模型置信度只表示切面判断的把握，不代表图像一定适合测量，也不能直接推出探头应该往哪里移动。

主要研究问题是：模型能否区分“真正换了切面”和“画面突然变化”，并在减少错误分段的同时保留可用片段？例如，两段 A4C 视频来自不同患者，亮度和心脏外观明显不同，但在切面分类上没有变化；A4C 接到 PLAX 才是切面变化（semantic change）。人为拼接的位置不应自动等于语义边界。

已有工作已经覆盖切面识别、视频分类和未知切面拒绝。Jansen 将含多个切面的视频列入未知情况；EchoViewCLIP 已结合时间信息、切面识别与未知输入检测；STFM 提供公开多切面视频和时空模型；Gao 已用公开数据做切面分发与后续 EF 分析。因此，“分类器加置信度阈值”本身不足以构成新贡献。[^1] [^2] [^3] [^4]

我们的候选贡献是：建立能区分语义变化与外观变化的公开评估流程，并验证一个小型时间模型是否优于简单方法。创新目前是待验证假设。如果简单方法同样有效，应如实保留简单方案，而不是增加网络复杂度。

# 2．公开数据能支持到哪一步

首选 EV9V：公开说明包含 5,138 段视频、九个原始类别。初期暂定五类：PLAX、PSAX、A4C、A5C、剑突下四腔心。相关类别名称在发布材料中存在不一致，因此训练前必须完成标签核对；保留原始代码，不擅自纠正名称。[^5]

这是一套粗粒度分类：不同 PSAX 层面的切换，在五类任务中仍算同一类。论文必须说明这一点。A2C、IVC、胸骨上切面暂不承诺，后续有可靠数据再扩展。

我们已经核对公开清单的数量、类别和视频编号重叠，并成功解码三段训练视频。官方说明按患者划分，但清单没有患者对应表，我们只能独立确认视频编号不重复。当前完成的是数据可访问性和代码基础检查；还没有训练模型、测出准确率或验证临床效果。

完全不增加人工标注也能开始：用原有标签训练识别，用公开短视频构造带已知拼接位置的测试序列。但拼接序列不是实际探头扫查，短视频的类别标签也不是逐帧人工真值。真实长录像中的过渡效果不能由这些实验替代。

如果导师能安排医生复核，可在获准访问的 MIMIC 公开视频上新增标注。这不需要采集私有患者数据，但需要访问权限和专家时间。MIMIC 的多个文件不能直接当作连续录像，其元数据或机器预测也不能充当可靠真值。[^6]

# 3．方法和验证怎样落地

先训练轻量图像分类器，再固定视觉特征，比较逐帧预测、平滑、HMM 和小型时间卷积网络（TCN）。候选方法增加一个边界判断模块：比较相邻时间段的特征，判断是否发生切面变化。

训练与测试都包含四种组合：同类／异类，分别配合不加／加入温和外观扰动。这样模型不能只靠亮度跳变猜边界。增强必须保留可识别解剖；跨帧淡化等明显人工效果仅作为压力测试。

一个必须加入的对照是：利用实际存在的文件边界，对每个原始短视频分类一次，再接受或暂缓。如果它已经解决大部分整理需求，额外的时间分段就必须证明还有价值。构造实验可以隐藏拼接位置，但实际导出必须保留原始文件身份，不能把不同患者的视频合并成连续心超。

评价同时看四件事：切面分类是否正确；同类拼接是否被错误切开；真正的类别变化是否被发现；在错误分发比例相近时保留了多少视频时长。接受阈值只在验证集选择，测试前冻结。预设 5% 错误分发是研究工作点，不是临床标准或数学保证。

# 4．前两周的具体行动与决策

第一周：取得完整开发数据，核对五类映射和官方划分，检查视频解码、帧时间、文字标记及近重复内容；建立原始文件分类对照，记录各类错误。测试集只做结构核查，不用结果选择模型。

第二周：完成分类器和平滑方法的初步比较；建立同类与异类拼接的小型固定验证集；检查真实短视频中是否存在有意义的误判片段或不确定区间。同步确认是否能获得公开数据的专家复核和 MIMIC 访问。

两周后做继续／收缩决定（go/no-go）：如果时间处理能解决真实片段问题，或构造实验暴露的失败能在原始、外部视频中复现，再投入边界模型。否则把论文收缩为公开序列稳健性评估，或转向更有证据的数据整理问题，不继续承诺真实床旁长录像分段。

# 5．希望导师确认的三个问题

1\. 第一阶段能否以研究团队的数据整理为目标，并接受五类粗粒度切面范围？

2\. 是否能安排医生核对类别映射、复核少量公开视频？这是区分工程演示与可信临床评估的重要资源。

3\. 如果两周内不能证明时间分段优于逐文件分类，是否同意调整贡献方向，而不强行保留最初的“长床旁录像”表述？

# 参考来源

主要参考来源；完整文献评述和实验方案见英文提案。

1\. Gino E. Jansen et al. (2024). [<u>Automated echocardiography view classification and quality assessment with recognition of unknown views</u>](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364256/). Journal of Medical Imaging 11(5), 054002.

2\. Shanshan Song, Yi Qin, Honglong Yang, Taoran Huang, Hongwen Fei, and Xiaomeng Li (2025). [<u>EchoViewCLIP: Advancing Video Quality Control through High-performance View Recognition of Echocardiography</u>](https://papers.miccai.org/miccai-2025/paper/4443_paper.pdf). MICCAI 2025, pp. 181–191.

3\. Bo Gou, Jicheng Zhang, et al. (2026). [<u>Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos</u>](https://arxiv.org/html/2606.17437v1). arXiv:2606.17437v1. Preprint posted 16 June 2026; no journal acceptance established in this audit.

4\. Zhiyuan Gao, Dominic Yurk, and Yaser S. Abu-Mostafa (2026). [<u>Learning from Scarce Labels: Multi-View Echocardiography for Ejection Fraction Prediction</u>](https://arxiv.org/html/2609.02969). Machine Learning for Biomedical Imaging (MELBA), 2026:025. Published 27 August 2026; arXiv posted 2 September 2026. DOI: 10.59275/j.melba.2026-8194.

5\. Bo Gou, Jicheng Zhang, et al. (2026). [<u>Echocardiographic Videos of Nine Views (EV9V)</u>](https://huggingface.co/datasets/bgx666/EV9V/blob/main/README.md). Hugging Face dataset repository. Dataset card and original split manifests; accessed 12 September 2026.

6\. Brian Gow, Tom Pollard, Nathaniel Greenbaum, Benjamin Moody, Ahram Han, Jonathan W. Waks, Alistair Johnson, Elizabeth Herbst, Parastou Eslami, Ashish Chaudhari, Tanner Carbonati, Seth Berkowitz, Roger Mark, and Steven Horng (2026). [<u>MIMIC-IV-Echo: Echocardiogram Matched Subset</u>](https://physionet.org/content/mimic-iv-echo/1.0.1/). PhysioNet, version 1.0.1. Published 25 August 2026. DOI: 10.13026/307c-mr50; version-specific citation.

[^1]: Gino E. Jansen et al. (2024). [<u>Automated echocardiography view classification and quality assessment with recognition of unknown views</u>](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364256/)

[^2]: Shanshan Song, Yi Qin, Honglong Yang, Taoran Huang, Hongwen Fei, and Xiaomeng Li (2025). [<u>EchoViewCLIP: Advancing Video Quality Control through High-performance View Recognition of Echocardiography</u>](https://papers.miccai.org/miccai-2025/paper/4443_paper.pdf)

[^3]: Bo Gou, Jicheng Zhang, et al. (2026). [<u>Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos</u>](https://arxiv.org/html/2606.17437v1)

[^4]: Zhiyuan Gao, Dominic Yurk, and Yaser S. Abu-Mostafa (2026). [<u>Learning from Scarce Labels: Multi-View Echocardiography for Ejection Fraction Prediction</u>](https://arxiv.org/html/2609.02969)

[^5]: Bo Gou, Jicheng Zhang, et al. (2026). [<u>Echocardiographic Videos of Nine Views (EV9V)</u>](https://huggingface.co/datasets/bgx666/EV9V/blob/main/README.md)

[^6]: Brian Gow, Tom Pollard, Nathaniel Greenbaum, Benjamin Moody, Ahram Han, Jonathan W. Waks, Alistair Johnson, Elizabeth Herbst, Parastou Eslami, Ashish Chaudhari, Tanner Carbonati, Seth Berkowitz, Roger Mark, and Steven Horng (2026). [<u>MIMIC-IV-Echo: Echocardiogram Matched Subset</u>](https://physionet.org/content/mimic-iv-echo/1.0.1/)
