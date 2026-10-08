# FACT integration on frozen E1 features: implementation checks (2026-10-08)

This stage connected the official FACT (Frame-Action Cross-Attention Temporal Modeling; Lu and Elhamifar, CVPR 2024) to
the frozen E1 512-d per-frame features and checked the implementation on training data. FACT keeps frame features and
a small set of segment tokens, exchanges information between them by cross-attention in both directions, and trains
each token to cover one reference segment through a matching loss. FACT is prior work. Connecting it is not a
contribution of this project, and whether it helps on our task is still unknown.

**What kind of evidence this is.**
- Every check used the EV9V training split only. No development-set feature or metric was read, and no development
  label was used (section 6 notes one index file that lists them).
- No model was trained for comparison. The 16-sequence overfit shows that the implementation can fit training
  sequences. It says nothing about generalisation.
- Constructed streams are spliced from separate recordings and are not real scans. Native clips carry the public clip
  label, not expert frame labels.
- The test split and EchoNet-Pediatric were not read.

**Order of work.**
- The plan is `docs/superpowers/plans/2026-10-08-fact-integration.md`. 5ccc174 committed it with the configuration,
  the adapter, the check script and the tests before any check ran.
- The data step (CPU) and the model step (GPU) then ran once each on the server.
- After reading the model step, two diagnostics were added and committed before they ran (499ccda, 89f018a). Each ran
  once. They explain two observations (sections 4.4 and 4.5) and change no setting.

## Answers

1. **What was connected.**
   - **Code.** The official repository is used at commit `7bd81bda2b84618a3e23876a2637a82f24881122`, and it was never
     edited. Its pinned dependency, yacs 0.1.8, was added to the project environment.
   - **Adapter.** `echo_routing/temporal/fact_head.py` imports the authors' model, matching criterion, null-class weight
     rule and configuration defaults unchanged. It only feeds the data and decodes with dummy labels (section 2).
   - **Settings.** The base configuration is `breakfast.yaml`, the example in the official guide. Transcripts are off at
     training and inference (`trans=false`), and matching is one-to-one. There are 12 tokens, which is 1.5 times the
     longest training transcript of 8 segments. The null-class weight is 0.257, from the official rule.
2. **The four requested checks passed.**
   - **Alignment.** All 1,500 training streams (9,002 fragments) match the cache files read directly, sample by
     sample. All 3,364 training clips have increasing frame indices that agree between the two contrast variants. All
     375 batches of the seed-0 epoch-0 plan match their sequences after unpadding.
   - **Coverage.** The training bank holds every requested case. The fixed 16-sequence set covers single-view clips,
     same-view joins, A4C → A5C → A4C and one view in separate segments.
   - **Inference label isolation.** Five different label sequences were passed to the official forward for each of
     206 distinct sequences, with two models. Not one of the 864 comparisons per model changed a prediction, and the
     logits were identical.
   - **Overfit.** After 1,000 updates on the 16 sequences, every predicted transcript equalled the reference, and
     segment F1@50 was 1.0 for all 16. Each reference segment had its own token with the right view. Five sequences
     kept one wrong frame at a boundary.
3. **Gradient paths.** Every loss term reaches the 12 token queries and the branches it depends on, with the
   exceptions explained in section 4.4. Nine parameters get no gradient at initialisation.
   - **One weight matrix is fixed by the official design.** In the input block's first token self-attention, the query
     and key projection rows carried only rounding (4e-14) at updates 0 and 100. When transcripts are off, the values
     this layer averages are the same for every token, so its attention pattern cannot matter. The layer's output
     weight and norm still train.
4. **Size and cost.** FACT with the Breakfast base has 118.6 M parameters, against 1.2 M for ASRF. One update of the
   shared recipe takes 0.72 s, eight times ASRF's 0.09 s. One 18,750-update run would take about 3.8 h, against 0.5 h.
5. **Next.** `configs/experiments/fact_asrf_paired_v1.yaml` drafts the paired training of FACT and ASRF on the same E1
   features. Section 7 summarises it and lists four open items for the lead. Nothing in it has run.

## Terms

- A **transcript** is the list of label runs of a sequence. A same-view join inside a run does not start a new
  segment, so it adds nothing to the transcript.
- A **token** is one of FACT's 12 learned segment queries. Each token predicts one of the five views or the null
  class, which means "no segment".
- **f2a** is the token-to-frame cross-attention. Each token attends over the frames, or over predicted segments in the
  U blocks. **a2f** is the frame-to-token cross-attention. Each frame, or predicted segment, attends over the tokens.
- **Matching** pairs tokens with reference segments one to one by the Hungarian method, without gradient. Matched
  tokens learn their segment's view, and unmatched tokens learn the null class.
- The **soft IoU** of a token and a segment is the token's frame-to-token attention summed over the segment's frames,
  divided by the union of the two.
- **Segment F1@50** counts a predicted segment as correct when it has the reference view and an IoU of at least 0.5
  with a reference segment, one to one.
- The **halves** are the two development halves of the protocol (selection and calibration). This stage read neither.

## 1. Token flow, measured on one training stream

The stream has 130 samples (13 s) and six reference segments, in the order A4C, A5C, A4C, PSAX, PLAX and PSAX. The
shapes below were recorded from the official forward. The middle dimension 1 is FACT's one-video batch.

```text
E1 features (130, 1, 512)
  |
  v  block 0, InputBlock "i"
  |    frame branch: MS-TCN++ (10 layers, 512 maps) -> frame features (130, 1, 512), frame logits (130, 1, 5)
  |    12 tokens: learned queries, values start at zero
  |      -> 6 layers of token self-attention + token-to-frame cross-attention
  |      -> token features (12, 1, 512), token logits (12, 1, 6)   [5 views + null]
  v  block 1, UpdateBlock "u"
  |    f2a (12 x 130) -> token self-attention -> token logits (12, 1, 6)
  |    a2f (130 x 12) -> frame branch -> frame logits (130, 1, 5)
  v  blocks 2 and 3, UpdateBlockTDU "U"
  |    frames pooled into the segments of the current frame prediction
  |      (19 and 1 segments at initialisation; 6 and 6 after the overfit)
  |    f2a over segments -> token self-attention -> a2f -> segments back to frames -> frame branch
  v  prediction (official Block._eval of block 3)
       each frame takes the non-null token it attends to most; the token's class probabilities are
       mixed with the frame branch's at weight 0.1 for the frame branch
training only: matching of the 12 tokens to the reference segments, from block 3's token probabilities and a2f
```

The table gives the last block's frame-to-token assignment on this stream.

| Reference segment (samples) | View | Token at initialisation | Token after the overfit (samples) |
|---|---|---|---|
| 0–11 | A4C | token 3 (A4C) for all 130 samples | token 6, A4C (0–11) |
| 11–21 | A5C | | token 1, A5C (11–21) |
| 21–50 | A4C | | token 3, A4C (21–50) |
| 50–90 | PSAX | | token 0, PSAX (50–91) |
| 90–111 | PLAX | | token 5, PLAX (91–111) |
| 111–130 | PSAX | | token 11, PSAX (111–130) |

After the overfit, the two A4C segments use different tokens, and so do the two PSAX segments. One boundary is off by
one sample.

## 2. Official structure and what the adapter changes

| Part | Official | This integration |
|---|---|---|
| Model | `models/blocks.FACT`, blocks `iuUU` | imported unchanged |
| Matching and losses | `models/loss.MatchCriterion`, one-to-one | imported unchanged; no class weights, as in the official `train.py`; no background class, because the five views have none |
| Null-class weight | `utils/train_tools.compute_null_weight` | the official function, giving 0.257 |
| Configuration | `configs/default.py` plus a dataset YAML | defaults plus `breakfast.yaml`; dataset fields set as `utils/gen_config.py` computes them |
| Token count | `int(token_ratio x longest transcript)`, ratio 1.5 | 12, from the 1,500 training streams (longest 8); no development data |
| Average transcript length | mean over the dataset's label files | 2.654, over one epoch's training sequences (1,500 streams once, 1,500 native windows of one segment) |
| Data | `.npy` features and label files, 4 videos per batch | the shared batch plan (4 streams and 4 native windows per update), unpadded and passed one video at a time, as the official loader does |
| Inference | `forward(seq_list, label_list)` takes labels | a constant dummy label sequence of zeros; the isolation check verifies that labels have no effect |
| Evaluation and logging | official `Checkpoint` metrics, wandb | not used; the project's metrics come in the next stage |
| Environment | torch 1.12 | torch 2.14; `Dropout2d` still drops whole channels of a 3-D input, so no patch is needed |

The adapter also splits the official per-block loss into its terms for logging. Each block's weighted sum of terms is
checked against the official `compute_loss` of that block, and it must agree.

## 3. Loss definition (official, as configured)

For one video, each block computes the terms below. T is the number of samples and L a segment's length.
- **frame.** The cross-entropy of the frame logits against the reference view is averaged over frames.
- **token.** Each token's logits (five views and null) are scored by cross-entropy against its matched segment's view,
  or against null when it is unmatched. The null class has weight 0.257.
- **f2a.** For each matched token, minus the log of its attention over the frames is summed over every frame of its
  segment, then divided by T. A uniform attention inside the segment is optimal, so the term cannot fall below
  Σ L ln L / T.
- **a2f.** For each frame, the cross-entropy of its attention over the matched tokens is taken against its segment's
  token.
- **U blocks.** Both attention terms run over the predicted segments instead of frames. Their targets are each
  predicted segment's overlap shares with the reference segments, and they divide by the number of predicted segments.
- **seg** (U blocks only). The cross-entropy of segment-level logits is taken against the label share of each
  predicted segment.
- **smooth.** The clamped squared difference of adjacent log-probabilities enters at weight 5. In the u block it also
  covers both attention maps.

The blocks add their terms as follows.
- The **i block** adds frame + token + 5 × smooth.
- The **u block** adds frame + token + f2a + a2f + 5 × smooth.
- Each **U block** adds (frame + seg) / 2 + token + f2a + a2f + 5 × smooth.

The loss of a video is the mean over the four blocks. An update's loss is the mean over its eight videos. Matching
uses block 3's outputs. Its cost is −0.2 × the token's probability of the segment's view − 1.0 × the soft IoU between
the token and the segment.

## 4. Check results

### 4.1 Alignment (data step, 27 s)

- **Inputs.** The protocol and the E1 feature manifest match their sha256 values. The cache key, numeric frame order
  and feature size (512) match the configuration. Both contrast variants' index files match the manifest.
- **Streams.** The training bank, rebuilt from the protocol's recipe settings, matches the protocol's digest. It has
  1,500 streams, 9,002 fragments (4,496 orig and 4,506 gamma090) and 191,366 samples. For every fragment the source
  recording is a training clip, and the label equals that clip's label. The features equal the cache rows read
  directly with numpy. The frame indices and times increase, and the sample spacing is 0.1 s (0.099998 to 0.100002).
  The joins equal the recipe's.
- **Native clips.** All 3,364 training clips (502 sessions, 207,614 samples) have increasing frame indices that agree
  between the two contrast variants.
- **Batches.** The seed-0 epoch-0 plan has 375 batches with 1,500 streams and 1,500 native windows. Every native window
  equals its cache rows. Every batch, unpadded by the adapter, equals its sequences.
- **Transcripts.** For all 1,500 streams, both official copies of the transcript parser give the same runs as our
  segment code. A same-view join stays inside one segment, and a repeated view starts a new one.

### 4.2 Coverage

| Training-bank streams (1,500) | Count |
|---|---|
| a single view throughout | 22 |
| at least one same-view join | 1,266 |
| an A4C → A5C → A4C run | 49 |
| an A5C → A4C → A5C run | 48 |
| any X → Y → X run | 673 |
| one view in two or more separate segments | 934 |

- **Transcript lengths.** The streams have 1 to 8 segments, with counts 22, 135, 276, 397, 368, 214, 73 and 15.
  Every native clip is one segment.
- **Segment lengths.** Segments run from 10 to 199 samples. The quartiles are 16, 25 and 37.
- **The fixed 16-sequence set** follows the configuration's rule in bank order, without repeats.
  - It starts with five single-view native windows, one from the first training clip of each view (14 to 219
    samples).
  - It adds three streams with a same-view join, three with an A4C → A5C → A4C run, three more with one view in
    separate segments, and two with eight segments.
  - These eleven streams have 1 to 8 segments and 107 to 208 samples.

### 4.3 Inference label isolation

The official forward was called with five label sequences for the same input. These were the reference labels, all
zeros, uniform random labels, the reference reversed in time, and a single sample. The inputs were the 16 fixed
sequences and the first 200 bank streams. Ten of the eleven fixed streams are among those 200, so 206 sequences were
distinct and 40 comparisons per model repeat an input. The check ran in eval mode with deterministic algorithms.

| Model | Sequences (distinct) | Comparisons | Changed predictions | Largest logit difference | Adapter decode differs |
|---|---|---|---|---|---|
| initial | 216 (206) | 864 | 0 | 0 | 0 |
| after the overfit | 216 (206) | 864 | 0 | 0 | 0 |

With `trans=false`, the official code builds a transcript from the labels it receives but never uses it for the
prediction. The check confirms this for both models.

### 4.4 Gradient paths (initialisation, first overfit batch)

Each term was backpropagated alone. Every term reached the 12 token queries. The table lists the groups that got no
gradient.

| Term | Groups without gradient | Reason |
|---|---|---|
| frame, smooth | none | |
| token | block 3's a2f, frame branch and merge | computed after block 3's token logits |
| a2f | block 3's frame branch and merge | computed after block 3's a2f attention |
| seg | block 3's a2f, action branch, f2a, frame branch and merge | computed after block 3's segment logits |
| f2a | block 2's action branch, a2f, frame branch and merge; all of block 3 | block 3's parts after its f2a come after the term; every other listed group reaches the f2a terms only through block 3's f2a term, which is flat at initialisation |

Block 3's f2a term is flat because its softmax runs over the predicted segments. The token flow shows one predicted
segment in block 3 at initialisation, so that softmax has a single entry. The segment counts of the gradient batch
itself were not logged.

The official total left nine parameters without gradient at initialisation. The post-hoc diagnostic measured their
gradients again after replaying the first 100 overfit updates. The replay reproduced the logged evaluation loss exactly
(1.3581629991531372).

| Parameter | Update 0 | Update 100 | Reading |
|---|---|---|---|
| block 0, first token self-attention, query and key rows of `in_proj_weight` | 0 | 4.3e-14 | rounding only; the token values it averages are identical, so its attention pattern cannot matter |
| same layer, value rows | 0 | 0 | the token values start at zero |
| same layer, `out_proj.weight` | 0 | 3.7e-6 | small but real once the attention output (the value bias) is non-zero |
| same layer, `norm1.weight` | 0 | 5.7e-6 | small but real once the norm's input (the value and output biases) is non-zero |
| block 3, segment GRU recurrent weights (two directions) | 0 | 2.9e-4 and 3.3e-4 | normal size once block 3 sees more than one segment |
| block 3, f2a key and query weights, query bias | 0 | 3.7e-4, 3.4e-4, 1.3e-4 | normal size for the same reason |
| block 3, f2a key bias | 0 | 3.0e-10 | rounding only; a key bias adds the same amount to every logit of a softmax |

Each entry is the largest absolute gradient. The median over all parameters is 4.3e-3 at update 0 and 2.0e-4 at
update 100.
- **After the full overfit.** Every parameter had a non-zero gradient under an exact test. By the same argument, the
  query and key rows can only carry rounding there too. Their size after 1,000 updates was not measured.
- **Other key biases.** The key-bias argument covers every attention key bias in the model, not only block 3's. The
  exact test counts their rounding as gradient.

### 4.5 Overfit on the 16 fixed training sequences

The run used 1,000 updates in two fixed batches of 8 and AdamW (3e-4, weight decay 1e-4, clip 1.0, constant rate).
The official channel and temporal masking stayed on.
- The **training loss** is the mean of the 50 preceding updates, in training mode with masking.
- The other columns come from eval mode, without masking, on all 16 sequences. Each is that term's contribution to
  the official total.
- `overfit_curves.png` shows every 50 updates.

| Update | Training loss | Official total | f2a | frame | smooth | a2f | token | seg |
|---|---|---|---|---|---|---|---|---|
| 50 | 4.038 | 2.019 | 1.105 | 0.114 | 0.118 | 0.187 | 0.443 | 0.052 |
| 100 | 1.802 | 1.358 | 0.989 | 0.069 | 0.096 | 0.088 | 0.084 | 0.032 |
| 500 | 1.157 | 1.096 | 0.932 | 0.054 | 0.069 | 0.029 | 0.003 | 0.009 |
| 1,000 | 1.082 | 1.047 | 0.913 | 0.049 | 0.065 | 0.013 | 0.004 | 0.003 |

**Why the total stays near 1.05.**
- **The floor.** Most of the total is the u block's f2a term, which has the floor Σ L ln L / T (section 3). On these 16
  sequences that floor is 2.64 to 5.39 per video (median 3.41). Averaged over the four blocks, it adds 0.900 to the
  official total. The logged f2a contribution at update 1,000 is 0.913.
- **The rest.** The other terms together add 0.133, and the f2a term sits 0.014 above its floor.
- **The token term.** It rose four times, at updates 250, 400, 600 and 1,000, with the largest value 0.196 at update
  400. It fell back after the first three. At the last update it was 0.004, up from 0.0001 at update 950. The learning
  rate was constant.

**Outcome after 1,000 updates (eval mode, dummy labels).**

| Measure | Result |
|---|---|
| Segment F1@50 | 1.0 for all 16 |
| Transcript of the prediction equals the reference | 16 of 16 |
| Frame accuracy | mean 0.998; 11 sequences without error, 5 with one wrong frame each (lowest 0.992) |
| Matching one to one and covering every segment | 16 of 16 |
| Matched tokens predicting their segment's view | 100% |
| Unmatched tokens predicting null | 100% |
| Non-null tokens equal to the number of segments | 16 of 16 |

The A4C → A5C → A4C streams and the streams with a view in separate segments were fitted with one token per segment.

## 5. Size and measured cost

The cost was measured on the server's L40S with the seed-0 epoch-0 batches of the shared recipe. Each model ran 5
warm-up updates, then 100 timed updates, in fp32 with TF32 off and cuDNN deterministic. Decoding was timed on 200
training streams.

| Model | Parameters | Seconds per update | Peak memory allocated by torch | Decode per stream | Hours per 18,750 updates |
|---|---|---|---|---|---|
| FACT, `breakfast.yaml` base | 118,647,296 | 0.724 | 2,378 MB | 0.031 s | 3.8 |
| FACT, `gtea.yaml` base (not trained) | 10,215,424 | 0.551 | 368 MB | 0.019 s | 2.9 |
| ASRF, input 512 | 1,191,768 | 0.090 | 171 MB | 0.017 s | 0.5 |

- **The `gtea.yaml` alternative.** The help text of `utils/gen_config.py` suggests it for small datasets. It is a
  different architecture, not only a smaller one. It has three blocks (`iuU`), an MS-TCN frame branch with 128 maps,
  dropout 0.2, channel masking 0.5 and stronger temporal masking. It has about 12 times fewer parameters and takes
  24% less time per update. This stage did not profile where FACT's time goes.
- **The paired runs.** With the shared recipe and the Breakfast base, three seeds of both arms would take about 13 h of
  training on this GPU (11.3 h FACT, 1.4 h ASRF), plus smoke checks and one evaluation.
- **Where the parameters are.** 85 M of FACT's parameters sit in the four frame branches (MS-TCN++ with 512 maps).
  16 M sit in the input block's token decoder.

## 6. Deviations and limits

- **Post-hoc diagnostics.** The gradient-size and loss-floor steps were added after the model step, to explain two
  observations. They were committed before they ran, and they change no setting.
- **The cache index.** The plan said that no development label would be read. The checks read the feature cache's
  index file, which lists every clip with its label, including the development clips. Those rows were used only to
  confirm that the cache holds every protocol clip and no test clip, and as keys of the output id guard. No development
  label entered any check, count or model.
- **The overfit optimiser.** The overfit used AdamW with the shared recipe's values and a constant rate, not FACT's
  official optimiser (Adam 1e-4). The check concerns the gradient paths and the ability to fit, not the optimiser.
- **Sizes of the zero-gradient parameters.** The model step tested gradients only for being non-zero. Their sizes come
  from the diagnostic's replay of the first 100 updates, which reproduced the logged loss exactly. The sizes after
  the full 1,000 updates were not measured.
- **The fixed weight matrix.** With transcripts off and no attention dropout (as in `breakfast.yaml`), the input
  block's first token self-attention cannot learn an attention pattern on any data. With attention dropout, as in
  `gtea.yaml` (0.2), the dropout noise would give its query and key weights gradients.
- **What the checks do not show.**
  - They do not show generalisation. The overfit used 16 training sequences, and no development data was used.
  - They do not show whether one-to-one matching, the Breakfast base or 12 tokens suit this task. These follow the
    official guide, and nothing was compared.
  - They do not show behaviour on real continuous scans, because all streams are spliced.

## 7. The paired training configuration (draft, not run)

`configs/experiments/fact_asrf_paired_v1.yaml` drafts the next stage. It trains FACT and ASRF on the same E1 features,
three seeds each.
- **Data and budget.** Both arms use the same batch plan, 50 epochs of 375 updates, AdamW 3e-4, weight decay 1e-4,
  clip 1.0, two warm-up epochs then cosine, and the final weights.
- **Primary comparison.** FACT − ASRF is measured in multi-stream segment F1@50 in each half. A difference counts as
  consistent when all three seeds' 95% intervals exclude zero on the same side, as in the earlier rounds.
- **Secondary comparisons.** FACT is also compared with E1 + B3 and with the historical strongest, E0 + B3.
- **Mechanism measures.** These answer the lead's question and never decide. They are the recall of short reference
  segments (at most 16 samples, the training bank's lower quartile), the recall of X → Y → X middles, the class
  errors, and the false cuts.
- **Constraints and goal.** Both are unchanged. A draft decision table follows the form of the earlier rounds, and the
  lead confirms it.

**Open items for the lead.**
1. **Optimiser.** The choice is between the shared recipe and FACT's official Breakfast optimiser for the FACT arm
   (Adam 1e-4, no weight decay, clip 10, a tenfold decay at epoch 80 of 150, 4 videos per batch).
2. **Checkpoint.** The choice is between the final weights and the calibration-half feasibility rule of the L1 + ASRF
   round for both arms.
3. **Base configuration.** The choice is between `breakfast.yaml` (118.6 M parameters, four blocks) and `gtea.yaml`
   (10.2 M parameters, three blocks and different regularisation).
4. **Budget.** The six runs need about 13 h of GPU time with the shared recipe and the Breakfast base, plus smoke
   checks and one evaluation. Items 1 and 3 would change this figure.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/fact_integration_v1.yaml
python scripts/fact_integration.py --config $C --data              # CPU: alignment, coverage, token rule, the 16 sequences
python scripts/fact_integration.py --config $C --model             # GPU: isolation, gradient paths, overfit, token flow, cost
python scripts/fact_integration.py --config $C --gradient-detail   # GPU, post hoc: gradient sizes at updates 0 and 100
python scripts/fact_integration.py --config $C --loss-floor        # CPU, post hoc: the f2a floor on the 16 sequences
python -m pytest tests/test_fact_integration.py -q                 # 10 tests, also local with a clone of the pinned commit
```

## Files

- `data_checks.json` holds the input identities, the alignment and batch counts, coverage, the token rule and the
  composition of the 16-sequence set.
- `model_checks.json` holds the configuration, parameter counts, both isolation checks, the gradient paths, the overfit
  log and outcome, both token flows and the cost table.
- `gradient_detail.json` and `loss_floor.json` hold the two post-hoc diagnostics, and `overfit_curves.png` holds the
  overfit curves.
- None of these files contains recording or session ids. The ids of the 16 sequences stay on the server in
  `runs/fact_integration_v1/data_details.json`.
- No model was frozen in this stage, because nothing was trained for use.
