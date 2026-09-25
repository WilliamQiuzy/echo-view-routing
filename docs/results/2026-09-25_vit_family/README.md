# Vision-Transformer family on EV9V (2026-09-25)

Designs and rationale: [DESIGN.md](DESIGN.md). Full tables (all metrics, all rows): [vit_family.md](vit_family.md).
Every row below comes from one ladder run over the frozen `plans.json` streams, with thresholds chosen on validation;
the five baseline rows reproduce `2026-09-17_five_baselines` exactly.

## Findings

1. **Frame level: a plain ViT is not a better encoder than ResNet-18 on EV9V.** Under B0's identical recipe, ViT-S/16
   (ImageNet-21k) reaches the same frame accuracy (0.953 vs 0.951) but lower clip macro-F1 (0.935 vs 0.957) and balanced
   accuracy (0.929 vs 0.962): it is weaker on the minority views. The encoder swap under the unchanged official MS-TCN
   confirms it (clip macro-F1 0.956 → 0.927, boundary F1 0.992 → 0.960). With ~3.4k training cines, the convolutional
   bias still helps; the strongest transformer baseline, EchoViewCLIP, gets its edge from CLIP pre-training on 400 M
   image–text pairs, not from attention itself.
2. **Clip level: video ViTs reach STFM's accuracy, not EchoViewCLIP's.** On the nine-code benchmark (official test,
   888 cines) the centre-supervised Echo-MViT reaches 0.934 ± 0.007 accuracy / 0.898 ± 0.006 macro-F1 (3 seeds) and
   MViTv2-S (Kinetics) 0.935 ± 0.006 / 0.898 ± 0.008 (2 seeds) — level with STFM (0.938 ± 0.002 / 0.899 ± 0.003) and
   about 0.7 points below EchoViewCLIP (0.941 / 0.907, itself a CLIP ViT-B/16). The small factorised ViT-S is one
   point lower (0.928 ± 0.002 / 0.885 ± 0.009, 3 seeds). Errors concentrate on the pairs whose code expansions
   conflict in the published sources (PMASA↔PMPALA, PASA↔PPMLSA) and on A4C↔A5C; PLHLA is 275/275.
3. **Routing: a class-token read-out breaks sliding-window routing — and centre supervision fixes it.** Every clip
   ViT trained on single-view clips lags or skips view changes (boundary F1 @0.5 s 0.58–0.78 over 7 runs; 20–55% of
   different-view boundaries missed; 5%-risk coverage 0.70–0.93) although its clip accuracy is fine. A probe shows why:
   on a window that straddles a change the read-out is class-dominant (2 PLAX frames among 16 A4C frames halve P(A4C)),
   whereas EchoViewCLIP averages frames. Training on mixed two-view clips with the **centre frame's** label (what a
   routed window must report) restores boundary F1 to 0.975 (factorised ViT) and 0.938 ± 0.049 (Echo-MViT, 3 seeds),
   and 5%-risk coverage to 1.000 / 0.984 ± 0.010, at the same accuracy level for Echo-MViT (the factorised ViT-S lost
   accuracy). Proportional (share-of-frames) soft targets did not work (bias flipped; documented negative result).
4. **Stream level: the ViT-Router matches the segmentation baselines where the temporal model matters.** On the same
   ViT-S features it beats the official MS-TCN on boundaries (0.988 vs 0.960), fragmentation (0.86 vs 1.75 /min) and
   achieved risk (0.003 vs 0.007, the lowest of all rows), and is on par with MS-TCN/ASFormer on ResNet-18 features for
   boundaries; its clip macro-F1 (0.920) is capped by the ViT-S features (finding 1).
5. **Pre-training.** EchoPrime's echo-video initialisation makes MViT converge much faster (validation accuracy 0.947 vs
   0.801 after the second epoch) and scores higher on validation (0.956 vs 0.944), but at convergence it is *lower* on
   test than Kinetics-400 (0.927 ± 0.001 vs 0.935 ± 0.006, 2 seeds each). In-domain pre-training is not what limits
   these models on EV9V; label ambiguity between neighbouring views is.

## Caveats

- Rows are single training runs unless the seed column says otherwise (the frozen baselines are single runs except STFM);
  seed-level ladder numbers are in the seed-robustness table of vit_family.md.
  Validation-selected checkpoints disagree with test rankings by up to ~2 points, so single-seed differences below
  ~1 point are not meaningful.
- Early stopping (patience 6) interrupts the 30-epoch cosine schedule early for the ViT runs; clip models usually
  stopped at epoch 9–14.
- ViT-family frames are decoded and letterboxed on the GPU (nvJPEG + antialiased bilinear); geometry is identical to
  the PIL path, intensities differ by a few grey levels at edges. All ViT rows share this pipeline.
- Nine-code cine accuracy uses each model's own cine aggregation (STFM: authors' 10 key-frame test; EchoViewCLIP:
  authors' 16-frame test; ViT family: mean over 1.6 s windows); the ladder tables use one common aggregation.
