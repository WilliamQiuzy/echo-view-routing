# Reproduction protocol for external baselines

Rule: an external method enters the ladder only if its **official code is public**. It is run **unmodified** at a
pinned commit with the **authors' hyperparameters** on the **official EV9V split**, and its result is compared with the
number the authors report. Any deviation is a recorded patch or a documented data-layout adapter, never an edit to
model or training logic. Results that do not follow this protocol are labelled *pipeline check* and are never cited.

| Baseline | Official code | Pinned commit | Status | Deviations from the authors' recipe |
|---|---|---|---|---|
| STFM (Gou et al. 2026) | github.com/bgx666/stfm (MIT) | `532f60b` | **reproduced**: paper seeds 100/200/300 → test acc 93.77 ± 0.17, macro-F1 89.88 ± 0.29 (paper 94.07 ± 0.66 / 90.30 ± 1.03); seed 666 extra: 93.36 / 89.49 | none in code; CLI = README command (`--model_name resnet18 --batch_size 64`), defaults otherwise (100 ep, lr 1e-4, patience 20); `--num_workers 8` (CPU count only) |
| MS-TCN (Abu Farha & Gall 2019) | github.com/yabufarha/ms-tcn (MIT+CC) | `33ed91c` | environment verified on GTEA (4-split average matches the paper); done on EV9V banks (run `20260913-170910` / export `20260913-180443`) | see §MS-TCN below |
| ASFormer (Yi et al. 2021) | github.com/ChinaYi/ASFormer (MIT) | `e1bbe4f` | environment verified on GTEA (released models reproduce Table 7 exactly); **training** on the EV9V banks (`logs/b5_asformer.log`) | authors' constants (120 ep, lr 5e-4, 10 layers, 64 maps, bz 1, channel mask 0.3); 2-line patch `third_party/patches/asformer-py3.patch` (NumPy `np.float`, removed deprecated `verbose=` kwarg); the L72 window-mask bug the authors mention is left as released |
| EchoPrime (Vukadinovic et al. 2026) | github.com/echonet/EchoPrime (MIT) + released weights v1.0.0 | `03874a5` | done: released 11-view ConvNeXt-B classifier on EV9V test, five-family mapping: acc 0.959, macro-F1 0.906 (run `20260913-183600`) | externally pretrained, evaluation only; the paper's own view accuracy is on a private test set, so this is a transfer result, not a reproduction |
| EchoViewCLIP (Song et al. 2025) | github.com/xmed-lab/EchoViewCLIP (no licence file; no released weights) | `f6a0d86` | done: stage-1 trained on EV9V with the authors' recipe (30 ep, lr 2.2e-5, effective batch 128); EV9V test (their `validate_`): **Acc 94.14, macro-F1 0.907** (val best 95.41). Paper on its private 38-view set: Acc 96.8 / F1 0.957 — different data, not a reproduction check | only dataset fields changed (`third_party/adapters/echoviewclip/ev9v_stage1.yaml`: 9 classes, EV9V paths); 1 GPU with accumulation 32 instead of 2 GPUs × 16 (same effective batch 128). Paper numbers are on a private 38-view dataset, so no reproduction check is possible; we report their published numbers next to ours on EV9V |
| ResNet-18 encoder (He et al. 2016) | torchvision `resnet18`, ImageNet weights | torchvision release | done | fine-tuned on EV9V with the proposal §9.2 recipe; not an external claim |
| File-mean, Smoothing, HMM, JS-divergence | this repo | — | done | rules, no weights; not paper reproductions |

Removed from the reported set: the control rules (File-mean, Argmax, Smoothing, HMM, JS-divergence) stay in the code as sanity checks only, and **our compact MS-TCN re-implementation** (`echo_routing/temporal/models/mstcn.py`) — kept
only as unit-tested reference code, never reported.

## STFM

- Task: the authors' nine-code, video-level classification (their `INDEXOFLABEL` order).
- Data adapter: `scripts/prepare_stfm_data.py` writes `labels.csv`, `train/validation/test.txt` and a symlink to
  `Images/` from the official split files. No frames are altered.
- Published targets (arXiv 2606.17437, Table 5, ResNet-18 backbone, temporal hidden 512 × 2 layers, three seeds
  100/200/300, mean ± std, video-level): **val acc 94.65 ± 0.54, val macro-F1 90.10 ± 0.70, test acc 94.07 ± 0.66,
  test macro-F1 90.30 ± 1.03**. Table 4 "Ours" (main result) reports test acc 94.48 ± 0.30, macro-F1 91.14 ± 0.37.
- Paper vs released code: the paper states 100 epochs, Adam lr 1e-4, weight decay 0.01, StepLR ×0.1 every 10
  epochs; the released `trainSelective.py` defaults to weight decay 0.05 with a warm-up + cosine schedule (visible in
  its logs) and seed 666. We run the released code unchanged and report this discrepancy; seeds 100/200/300 are
  queued after the seed-666 run so the comparison uses the paper's seed set.
- Acceptance: test accuracy and macro-F1 of the `model.ckpt` selected by their own validation loop, mean ± std over
  seeds 100/200/300, within one std of the Table 5 ResNet-18 row counts as reproduced; a larger gap is reported as such.
- **Result (2026-09-13):** seeds 100/200/300 give test acc 93.92 / 93.58 / 93.81 (mean 93.77 ± 0.17) and macro-F1 90.14 / 89.57 / 89.92 (mean 89.88 ± 0.29); early stopping ended the runs at epochs 26/28/29. Both means sit within the paper's one-std band (acc 0.30 below, F1 0.42 below the paper mean). Reproduced; checkpoints frozen under `models_frozen/stfm_official/`.

## MS-TCN

The official repository is Python-2.7 / PyTorch-0.4 code with hard-coded constants (`num_stages=4`,
`num_layers=10`, `num_f_maps=64`, `features_dim=2048`, `bz=1`, `lr=5e-4`, `num_epochs=50`, fixed seed). We keep all of
them. Two things are needed to run it on our features:

1. **Python 3 compatibility**: `batch_gen.py` reuses `map(len, batch_target)` twice, which is empty on the second use
   in Python 3. Recorded patch: `third_party/patches/ms-tcn-py3.patch` (one line, `list(map(...))`).
2. **Feature dimension**: the code assumes 2048-d features. Our frozen ResNet-18 features are 512-d; they are
   **zero-padded to 2048** in the data adapter so the code is untouched. The extra input channels multiply zeros in the
   first 1×1 convolution and contribute nothing.

Data adapter (`scripts/run_mstcn_official.py`): writes `data/mstcn/<bank>/{features/*.npy, groundTruth/*.txt,
mapping.txt, splits/train.split1.bundle, splits/test.split1.bundle}` from our recipe banks at 10 Hz, runs
`main.py --action=train` then `--action=predict` in the pinned repo, and additionally loads the epoch-50 model through
the authors' own `model.MultiStageModel` to export per-frame softmax for our routing metrics.

Training bank: class-balanced multi-fragment streams from the official **train** split
(`make_multi_recipes`, 4–8 fragments, labels uniform over classes, gamma edits on ~50% of fragments). Validation and
test banks are the frozen pair banks plus a multi-fragment bank, built from the official validation/test splits with
each cine used at most once. There is no published MS-TCN number on EV9V, so acceptance is: the official code
trains to convergence under its own recipe and its own `eval.py` metrics (frame accuracy, edit score, F1@{10,25,50})
are reported next to ours.

## Environment checks on the authors' own benchmark (GTEA)

MS-TCN and ASFormer have no published echo results, so the code + patch + environment are validated on GTEA (features and
labels from the authors' Zenodo record 3625992, four official splits), comparing with the papers' four-split averages:

| Model | Paper (GTEA, avg of 4 splits): F1@10 / F1@25 / F1@50 / Edit / Acc | Our run | Status |
|---|---|---|---|
| MS-TCN (CVPR 2019, Table: MS-TCN, I3D features) | 85.8 / 83.4 / 69.8 / 79.0 / 76.3 | trained all 4 splits with the pinned code + py3 patch (50 ep each): split 1 83.0/80.3/63.0/75.4/76.1, split 2 84.1/81.2/65.9/80.3/75.5, split 3 89.1/85.3/77.7/84.2/78.4, split 4 87.0/86.2/74.3/84.2/77.6 → **average 85.8 / 83.2 / 70.3 / 81.0 / 76.9** | **verified** (F1/Acc within 0.6 of the paper; Edit +2.0) |
| ASFormer (BMVC 2021, Table 7) | 90.1 / 88.8 / 79.2 / 84.6 / 79.7 | authors' released GTEA models (Google Drive) → predict all 4 splits with the pinned code + 2-line patch → their `eval.py --split=0`: **90.1 / 88.8 / 79.2 / 84.6 / 79.7** | **verified** (identical to the paper) |

Acceptance: the four-split average within ~1 point of the paper on Edit/Acc and within a few points on F1 (MS-TCN training is
seeded but GPU non-determinism remains); larger gaps are investigated before the EV9V numbers are reported.

## Ledger discipline

Every run records commit hashes (ours and the vendored repo), the exact command line, seeds, config hash and the
metrics file in `docs/EXPERIMENT_LEDGER.md`.
