# Vision-Transformer family on EV9V — model designs (2026-09-25)

Question from the advisor: how do Vision Transformers perform on echo view recognition and temporal routing?
Five baselines are already frozen (STFM, EchoViewCLIP, EchoPrime, MS-TCN, ASFormer). Two of them already contain
transformers: EchoViewCLIP fine-tunes a CLIP **ViT-B/16** frame encoder with temporal pooling, and ASFormer is a temporal
Transformer on frozen ResNet-18 features. STFM (ResNet-18 + LSTM), MS-TCN (temporal convolutions) and EchoPrime's view
classifier (ConvNeXt-B) are convolutional/recurrent. The family below therefore asks *which* ViT design helps, at which
level (frame, clip, stream), and with which pre-training — each member is trained under the protocol of the baseline it
is compared with, on the official split, with the same evaluation code.

| id | model | level | task trained | compared with | borrowed from |
|---|---|---|---|---|---|
| V1 | ViT-S/16 frame classifier | frame | family5 | B0 ResNet-18 (identical data recipe), EchoPrime per frame | — (plain ViT) |
| V2 | Factorised video ViT (ViT-S frame encoder + temporal Transformer) | clip (1.6 s) | nine-code | STFM, EchoViewCLIP | STFM: spatial + temporal heads; EchoViewCLIP: ViT frame encoder |
| V2b | V2 + transition-aware training, share-of-frames targets (ours) | clip (1.6 s) | nine-code | V2, STFM, EchoViewCLIP | MS-TCN/ASFormer: train on constructed view changes |
| V2c | V2 + transition-aware training, centre-frame targets (ours) | clip (1.6 s) | nine-code | V2, V2b, STFM, EchoViewCLIP | as V2b; target matched to how windows are routed |
| V3 | MViTv2-S, Kinetics-400 init | clip (1.6 s) | nine-code | STFM, EchoViewCLIP | — (generic video ViT control) |
| V4 | Echo-MViTv2-S, EchoPrime init | clip (1.6 s) | nine-code | STFM, EchoViewCLIP, V3 | EchoPrime: echo-specific video pre-training |
| V4c | V4 + centre-supervised transition-aware training (ours) | clip (1.6 s) | nine-code | V4, STFM, EchoViewCLIP | V2c's fix on the stronger backbone |
| V5 | ViT-Router (ours) on frozen V1 features | stream | family5 | MS-TCN, ASFormer (same training streams) | ASFormer: local attention; MS-TCN: multi-stage refinement + smoothing loss |
| V6 | official MS-TCN on frozen V1 features | stream | family5 | MS-TCN on ResNet-18 features | encoder-swap control |

## Shared protocol

- Data: official EV9V split (3,683 / 567 / 888 cines; five-family subset 3,364 / 521 / 800). Frames are the released JPEGs
  (320×240), letterboxed to 224×224 with black padding (same geometry as `ingest.frames.letterbox`), decoded on the GPU
  (nvJPEG) because the server has 8 vCPUs.
- Sampling and augmentation follow the B0 recipe: every epoch draws one cine per training cine, class-balanced with
  replacement; the only appearance augmentation is the gamma edit U[0.9, 1.1]; class-weighted cross-entropy.
- Optimisation (standard ViT fine-tuning): AdamW, weight decay 0.05, 1 warm-up epoch then per-step cosine, gradient
  clipping 1.0, bf16 autocast; ViT-S trunks use layer-wise lr decay 0.75 and drop-path 0.1. At most 30 epochs; the
  checkpoint with the best validation cine macro-F1 is kept (patience 6), as for B0.
- Clip models see 16 frames taken every 3rd source frame (10 Hz), i.e. exactly the 1.6 s window with which STFM and
  EchoViewCLIP are applied in the unified evaluation; cines shorter than a clip repeat their last frame.
- Evaluation: the unified ladder (`scripts/run_baselines.py`) on the frozen `plans.json` streams — 800 untouched test clips,
  240 two-fragment streams (2×2 design), 120 multi-fragment streams; thresholds selected on validation and frozen.
  Nine-code models are mapped to family5 exactly like STFM/EchoViewCLIP (PMPALA mass dropped). Nine-code cine accuracy on
  the official test split is reported separately (the EV9V benchmark protocol used by STFM).

## V1 — ViT-S/16 frame classifier (plain ViT)

What: `vit_small_patch16_224.augreg_in21k_ft_in1k` (22 M parameters, ImageNet-21k pre-training), CLS embedding → linear
head, one frame in, one family label out; no temporal processing at inference (like B0 and EchoPrime per frame).

Why: the cleanest test of the architecture itself. Everything except the network (data, sampler, augmentation, loss,
selection rule, epochs) is B0's recipe, so V1 − B0 isolates *ViT vs CNN as the frame encoder*. Intuition for why a
ViT could help: a view is defined by the global arrangement of chambers, valves and the apex; self-attention relates
distant patches in the first layer, whereas a CNN builds that context only in deep layers. Intuition for why it may not:
ViTs lack locality/translation bias and usually need more data than ~3k cines; the ultrasound texture is far from
ImageNet. V1 is also the frame encoder for V5/V6.

## V2 — Factorised video ViT (ViViT factorised encoder + auxiliary frame head)

What: a shared ViT-S/16 encodes each of the 16 frames to a CLS token; a 2-layer temporal Transformer (6 heads, learned
temporal positions, its own CLS token) mixes the 16 tokens into a clip prediction. An auxiliary linear head on every
frame token is trained jointly (loss weight 0.5). Nine-code task.

Why: several EV9V views differ by motion rather than by a single frame — A5C vs A4C (the LVOT opens in systole), the
PSAX levels (valve leaflets vs papillary muscles moving through the plane). Factorising space and time keeps the cost
at 16 ViT-S frame passes plus a tiny temporal model, and it keeps the frame encoder frame-wise, so frame embeddings are
cached once and any window/stream can be scored cheaply. The auxiliary frame head borrows STFM's spatial + temporal
heads: it keeps every frame token individually discriminative, so the temporal layers learn motion cues on top of good
frame features instead of compensating for weak ones. Compared with EchoViewCLIP (CLIP ViT-B/16 + temporal *mean
pooling*), V2 is 4× smaller per frame but has real temporal attention.

## V2b — Factorised video ViT, transition-aware (ours)

What: V2's architecture and recipe, but half of the training clips are *mixed*: the first k frames of one training
cine's clip joined to the first 16 − k frames of a second, independently drawn training cine (k uniform in 1…15, one
gamma edit per fragment). The clip target is the share of frames per view (soft cross-entropy with the same class
weights, normalised so that single-view clips give exactly the ordinary loss); the auxiliary frame head keeps each
frame's own label. Only the official train split is used, as for the MS-TCN/ASFormer training streams.

Why (measured, not assumed): on the ladder, V2 lagged every PLAX→A4C view change by ≈ 0.6 s and missed 55% of
different-view boundaries in the two-fragment test streams, although its clip accuracy is fine. A probe on test cines
showed the cause: V2's attention read-out, trained only on single-view clips, is class-dominant on windows that
straddle a change — 2 PLAX frames among 16 A4C frames already drop P(A4C) from 0.97 to 0.48, and in the reverse
order PLAX keeps winning until 12 of 16 frames are A4C — while the mean of its per-frame probabilities falls
proportionally. EchoViewCLIP averages frame features and STFM sums per-key-frame evidence, so their windows switch at
the midpoint. A sliding window is exactly such a mixture near every view change, so the fix is to train on it: with
proportional soft targets the read-out learns to report the view that fills the window, and the window centre — whose
probabilities each sample inherits — flips when the change crosses it.

Outcome (seed 0): proportional soft targets did **not** fix routing — boundary F1 @0.5 s 0.576 (V2 0.579). The
bias flipped instead of vanishing: V2b switches 4–6 samples *before* a PLAX→A4C change, and the same probe now shows
A4C dominating mixed windows (8 PLAX + 8 A4C frames → P(A4C) = 0.74). With class-weighted soft targets the loss
rewards over-predicting high-weight classes in mixtures, so "share of frames" is not what the network learns to
report. Kept as a documented negative result.

## V2c — Factorised video ViT, centre-supervised transition-aware (ours)

What: V2b's mixed clips, but the target of a mixed clip is the label of its **centre frame** (window position 8), a
hard label with the usual class weights; single-view clips are unchanged.

Why: in sliding-window routing each sample inherits the probabilities of the window whose centre is nearest to it,
so the quantity a window must estimate is the view *at its centre*, using the ±0.8 s around it as context — not the
clip's majority and not its share of frames. Centre supervision trains exactly that decision, so the predicted change
lands within half a window stride (±0.25 s) of the true one whatever the classes involved; a hard label keeps class
weighting from biasing mixtures.

## V3 — MViTv2-S (Kinetics-400)

What: torchvision MViTv2-S (34 M), joint space-time attention with pooling (multiscale: resolution falls and width
grows over stages), Kinetics-400 video pre-training, new 9-way head.

Why: the strongest generic *video* transformer that fits the budget, and the control for V4: same architecture,
input and recipe, only the pre-training source differs. Joint space-time attention can model where a structure is
*and* how it moves in one operator, which a factorised model only approximates.

## V4 — Echo-MViTv2-S (EchoPrime echo-video initialisation)

What: identical to V3 but initialised from EchoPrime's released video encoder (MViTv2-S trained contrastively on about
12 M echocardiography videos with report text), with EchoPrime's input normalisation; EchoPrime's 512-d projection is
replaced by a 9-way head and the whole network is fine-tuned.

Why: borrows the key ingredient of the EchoPrime baseline — in-domain pre-training — without its fixed 11-view head
and per-frame use (which made EchoPrime flicker, 19 fragments/min). V4 − V3 measures what echo-specific pre-training
buys a video ViT on EV9V.

## V4c — Echo-MViTv2-S, centre-supervised transition-aware (ours)

What: V4 trained with V2c's recipe change (half of the clips mixed from two training cines, target = centre frame's
view). Same data, epochs, selection rule and evaluation as V4.

Why: MViTv2 also reads out through a class token, and V4 showed the same routing failure as V2 (boundary F1 0.765,
25% of different-view boundaries missed). V4 is the stronger recogniser, so V4c tests whether centre supervision can
fix routing without the accuracy cost seen on the small factorised ViT. It did: boundary F1 0.938 ± 0.049 and
5%-risk coverage 0.984 ± 0.010 over three seeds, with nine-code test accuracy 0.934 ± 0.007 (V4: 0.927 ± 0.001).

## V5 — ViT-Router (ours): stream-level temporal Transformer on ViT features

What: frozen V1 frame embeddings at 10 Hz → stage 1 of 6 local-attention blocks (window 5, 9, 17, 33, 65, 129 samples
= 0.5–13 s; each block = banded self-attention + dilated depthwise-conv feed-forward with dilation 1…32) → two
refinement stages (4 blocks each) that see only the previous stage's class probabilities → per-sample family
posteriors. Loss per stage: cross-entropy + 0.15 × truncated MSE between consecutive log-probabilities (τ = 4).
Input channel dropout 0.3. Trained on exactly the class-balanced multi-fragment training bank used for the official
MS-TCN and ASFormer (1,500 streams, 4–8 fragments, gamma edits on ~50% of fragments), batch 1, 50 epochs, final
epoch exported (their convention).

Why: routing is a *segmentation* problem over a stream, and the two segmentation baselines are the best at it
(boundary F1 0.98–0.99). V5 keeps what makes them work — ASFormer's local attention with a growing window (the right
inductive bias for piecewise-constant labels) and MS-TCN's refinement on probabilities with the smoothing loss (which
suppresses over-segmentation) — but replaces the ResNet-18 frame features with ViT features and uses attention
instead of convolutions in every stage. Relative position comes from the dilated convolutions, so there is no
absolute position encoding and any stream length works.

## V6 — official MS-TCN on ViT-S features (encoder-swap control)

What: the pinned official MS-TCN code, unchanged recipe (4 stages × 10 layers × 64 maps, 50 epochs, lr 5e-4, batch 1),
same training bank, only the frozen features differ (V1's 384-d CLS embedding, zero-padded to the code's 2048-d as for
the ResNet-18 features).

Why: V5 changes two things at once (features and temporal model). V6 changes only the features, so B4 → V6 is the
effect of the ViT encoder and V6 → V5 is the effect of the transformer temporal model.
