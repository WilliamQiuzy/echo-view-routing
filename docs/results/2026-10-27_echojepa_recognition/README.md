# A41: EchoJEPA against generic V-JEPA 2 ViT-L as a new recognition branch, same LoRA recipe (2026-10-27 UTC)

**Question.** Under one fixed adaptation recipe, does initialising a new recognition branch from EchoJEPA (E), a ViT-L
pretrained by latent prediction on large-scale echo video, give a transferable A5C correction over the same architecture
initialised from generic V-JEPA 2 (G)? It must also keep correct A4C / A5C and the complete segmentation.
- **Predecessors.** A31's last-block adaptation and error emphasis gave no held-out session net gain. A34 and A39
  recovered S1 0 / 103 from frozen A08 features or tokens. A40's masked-HOG auxiliary in a full-encoder A08 continuation
  learned its target, but H − R was 0 in every reported held-out aggregate (S1 1 / 103 for both).
- **The changed condition** is the source and backbone of the recognition representation. Earlier rounds changed the
  pooling, loss or A08 continuation. Here a separate ViT-L, cross-layer QKV LoRA and a new head are trained; no A08
  feature or probability is an input or teacher.
- **What the contrast isolates.** E − G compares the two named pretrained states under the same code, architecture,
  adaptation, batches and schedule. It cannot separate pretraining data volume, schedule, domain or other upstream
  conditions, and it is not evidence about the JEPA algorithm. Against A08 (Base), the backbone, pretraining and
  parameter count all differ at once.
- **Scope.** Source seed 0 on A31's two session folds; four models (G and E per fit fold). No development, check, test
  or Pediatric data. Labels are EV9V clip codes, not LVOT visibility or expert truth. A31's folds hold out sessions of
  this branch's training, not the whole system (A08 and B3 trained on these recordings).
- **Records.** Configuration `configs/experiments/video_echojepa_recognition_v1.yaml` (85da250, committed before any
  download or forward; amended in e74d7b2 after check attempt 1, before training). Code 5d04536 (+ e74d7b2). Task sheet
  `claude_prompts/2026-10-27-echojepa-a5c-recognition.md`.

## Sources (`sources.json`)

- **Code.** [bowang-lab/EchoJEPA](https://github.com/bowang-lab/EchoJEPA) at 77208e1a (2026-06-18), cloned unedited on
  the server. Only `src.models.vision_transformer` and its five imports are loaded; each equals the pinned blob, and
  `src` resolves to the clone only. The model files carry Meta's MIT header. The README names three Apache-2.0 files, and
  the repository includes `APACHE-LICENSE`.
  - The upstream `patch_embed.py` imports `einops` without using it. Neither environment had it, so the pinned
    einops 0.8.1 wheel (sha256 919387eb…) was installed with `--target` into a round-only directory. The shared
    `.venv` is unchanged.
- **Checkpoints**, downloaded on the server from the public endpoints:
  - **E:** `vitl-vmix22m-pt220-c55.pt` (Google Drive file 1T_ubAMp…, from the README folder), 5,127,744,286 bytes,
    sha256 c1f43dfb…. gdown did not capture the HTTP status or the final redirect URL; the identity is the file id, the
    byte count and the sha256. The README describes it as ViT-L initialised from V-JEPA 2 (VideoMix22M), then pretrained
    on echo video.
  - **G:** `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt`, HTTP 200, 5,127,726,842 bytes (equal to Codex's HEAD),
    sha256 5346856e….
- **Strict extraction.** Both files contain `target_encoder` (292 tensors, only a leading `module.` stripped). The
  common encoder takes all 292 keys with equal shapes: none missing, extra or dropped (no legacy `pos_embed`), and no
  random fill. Both are FP32 with 303,885,312 parameters.
  - Saved encoders: E sha256 40aa4b79…, G 76075f10…. E and G differ in all 292 tensors (relative L2 0.595).
  - After the sha256 manifest was committed, the two full downloaded checkpoints (10.3 GB) were deleted; they can be
    downloaded again.
- **Pretraining overlap with EV9V is unknown.** The exact pretraining sample lists are not available, so no claim is
  made that EV9V was unseen.

## Design

- **Encoder.** The pinned `vit_large` with 16 frames at 224, patch 16 and tubelet 2, RoPE, SDPA and activation
  checkpointing: 24 blocks, 1024-d, 16 heads. Under RoPE `pos_embed` is `None`, so `uniform_power` has no effect. The
  encoder returns final-norm tokens, 8 × 14 × 14 = 1,568.
- **LoRA.** Each block's fused `attn.qkv` computes W x + b + 2 B(A x), with r 8 and α 16. A is N(0, 0.02²) and B is 0;
  the base stays frozen in FP32.
- **Head.** Tokens are reshaped time-major to [8, 196]. The target vector is the spatial mean at token time ⌊m / 2⌋
  (m = A08's marker slot) and the global vector the mean over all tokens; Linear(2048, 5) reads both.
  - This is a time-aligned summary, not anatomical localisation. The marker only selects the readout time.
  - 786,432 LoRA + 10,245 head = 796,677 trainable parameters. Their initialisation comes from one local generator
    and is bit-identical for E and G.
- **Input.** A31's original request windows (stride-5 centres, 16 samples at 10 Hz, A08's crop / gamma, uint8). Each
  is divided by 255 and ImageNet-normalised, identically for E and G.
- **Batches and optimisation.** 900 real AdamW updates per model, each 8 fit-pool windows: 4 adjacent native centres
  and 4 adjacent constructed centres with a label change. The draws are uniform and stateless and differ from A40's.
  - Each window runs as microbatch 1 with CE / 8 accumulated, then one clip at 1 and one step. lr is 1e-4 for LoRA and
    1e-3 for the head, weight decay 1e-4, 90-update warm-up then cosine.
  - Encoder and LoRA run under BF16 autocast; the head and CE in FP32 with TF32 off. Final weights only; no KL, HOG,
    contrastive term or pseudo label.
- **Readout.** The unchanged Base P runs (frozen A08 + B3 cuts) take the argmax of the multiplicity-weighted mean
  softmax of the new model. Equal neighbours merge, and complete sequences are scored with the A31–A40 evaluator.
  Fit f is read on fold 1 − f, once, after the weights were committed.

## Checks (`checks.json`; attempt 1 failed and is kept as `checks_attempt1.json`)

- **Attempt 1 failed** (0.052 GPU h, booked).
  - With `use_deterministic_algorithms(warn_only=True)`, the SDPA backward was non-deterministic. The 6-update repeat
    differed by 0.030 in loss and 1.3e-3 in parameters, forced E = G by 0.015 / 2.3e-3, and 3 + 3 resume by
    0.017 / 2.2e-3. The fixed limits are 1e-3 / 1e-4.
  - The FP32 microbatch-1 against batch-8 gradient differed by 9.3e-3 × the gradient maximum (cosine 0.99998), over its
    1e-4 gate. The FP32 forward itself depends on the batch size: tokens differ by up to 1.17 of a maximum of 44, the
    same under math, efficient and default SDPA.
- **Diagnosis** (`diagnose_numerics*.json`, fit windows only).
  - Under `warn_only=False` two identical BF16 backward passes were bitwise equal; under `warn_only=True` LoRA B
    gradients differed by up to 6.5% of the group maximum.
  - In FP64, batch against single-window gradients agree to 2.7e-10 × the maximum, so there is no batch coupling.
    FP32 single-window tokens differ from FP64 by up to 0.13 in 1 of 3,136 token rows: finite-precision sensitivity,
    not a defect.
- **Repair before any training** (e74d7b2; configuration amended).
  - Strict deterministic algorithms (`warn_only=False`): the same official SDPA kernels with deterministic backward.
  - The batch-coupling gate moved to FP64 (≤ 1e-8 × the gradient maximum). The FP32 gate is not claimed to pass; its
    attempt-2 values stay recorded (9.3e-3 ×, BF16 cosine 0.9989).
  - The repeat / resume / forced-match limits, attention, training precision, batch and every scientific setting are
    unchanged.
- **Attempt 2 passed every gate of the amended set.** The original FP32 microbatch gate did not pass; it was replaced
  by the FP64 coupling gate, as above. The original repeat / resume / forced-match limits were met unchanged.
  - Upstream and pins: clone at 77208e1a and clean; the 6 loaded files equal the pinned blobs; `src` resolves to the
    clone only. All 153 loaded modules match the committed code or pins.
  - Zero-LoRA parity: the wrapped encoder equals a separately built official encoder for both E and G (BF16 and FP32
    token difference 0, same head argmax). The LoRA formula matches an independent W x + b + 2 B(A x) (2.8e-5 of a
    116 output); zero B equals the base exactly.
  - Tokens: patch tokens equal the Conv3d output flattened time-major, RoPE positions are time-major, and the readout
    equals an explicit loop. The marker slot is the only readout input besides the tokens.
  - Initialisation: identical for E and G, different between folds, B and head bias 0. Exactly 796,677 trainable
    parameters are bound; the base has no `requires_grad` and its digest is unchanged after updates.
  - Gradients: FP64 microbatch against batch, a maximum difference of 7.6e-12 at a gradient maximum of 12.9 (5.8e-13 ×;
    the 954d82c commit message loosely wrote "7.6e-12 × grad max"); checkpointing on / off 0 (FP32 and BF16). From the
    real zero-B start, A's first gradient is 0 and all 24 B gradients are nonzero; after 6 updates A, B and the head
    moved.
  - Mechanics: 6-update repeat, forced E = G (G's base through the E path) and 3 + 3 resume all 0 / 0.
  - Base: reproduced on every P run, sequence and A32 Base row of both folds. Batches follow A07's grid, and all
    900 × 2 draws are valid.
  - Launch gate: 2.17 s per real 8-window update and 23.7 ms per window for the online forward. The source-0
    projection of 4.21 GPU h (including the 0.75 h reserve) is within the 8 h cap.

**Training (fit-only monitors; `weights_s0.json`).** Each model ran 900 real AdamW steps (about 0.51 GPU h, 1.7 GB
peak), with no resume. The base digests were unchanged.
- **Fit batches** (updates 0–3, eval): CE G_f0 3.97 → 0.23, E_f0 2.08 → 0.14, G_f1 1.70 → 0.06, E_f1 2.56 → 0.27;
  accuracy ended at 0.94 / 0.94 / 0.97 / 0.91. Training CE over the first 90 updates was lower for E (0.93 / 1.16) than
  for G (1.87 / 1.87); over the last 90 it was similar (E 0.17 / 0.36, G 0.17 / 0.35). These are fit-only numbers, not
  evidence of transfer.
- **LoRA moved.** From the zero-B start, A's gradient was 0 at update 0 and nonzero from update 1. The final ‖2BA‖
  summed over blocks is 5.59 / 5.42 for G and 4.15 / 4.07 for E.

## Results (held-out folds; OOF = both folds summed)

**All held-out P runs.**
- Fold 0 has 3,384 runs (109 Base errors), and fold 1 has 3,404 (145 Base errors).
- A32's S1 runs (true A5C read as A4C by Base) are 34 / 69, from 12 / 26 A5C-coded recordings in 12 / 21 sessions.
- Base reproduces A32's rows and A40's Base cells in both folds.

| Model | Fixed / broken / net (f0; f1; OOF) | A4C read as A5C fixed | S1 fixed | Correct A5C kept | Correct A4C kept | Segment F1@50 (f0 / f1) | Native macro-F1 | Boundary R | Boundary P | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | 0 / 0 / 0 | 0 | 0 / 103 | 1,150 / 1,150 | 1,413 / 1,413 | 0.9639 / 0.9499 | 0.9776 / 0.9758 | 0.9676 / 0.9681 | 0.9556 / 0.9420 | 262 / 178 | 183 / 76 |
| G (generic V-JEPA 2) | 27/149/−122; 44/146/−102; **71/295/−224** | 52 | 17 / 103 | 973 / 1,150 | 1,320 / 1,413 | 0.9288 / 0.9219 | 0.9337 / 0.9416 | 0.9403 / 0.9491 | 0.9682 / 0.9610 | 339 / 159 | 420 / 181 |
| E (EchoJEPA) | 32/77/−45; 51/143/−92; **83/220/−137** | 66 | 12 / 103 | 982 / 1,150 | 1,364 / 1,413 | 0.9542 / 0.9197 | 0.9601 / 0.9504 | 0.9596 / 0.9471 | 0.9728 / 0.9639 | 290 / 109 | 351 / 117 |

FP, FN and their short parts are summed over the two folds.

| Contrast, fold | Net runs | S1 fixed | Correct A5C / A4C kept | Segment dF | Native macro-F1 | Boundary R | Boundary P | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|
| E − G, 0 | +77 | −5 | +34 / +24 | +0.0254 | +0.0265 | +0.0192 | +0.0046 | −49 / −31 | −79 / −49 |
| E − G, 1 | +10 | +0 | −25 / +20 | −0.0021 | +0.0088 | −0.0021 | +0.0029 | +0 / −19 | +10 / −15 |
| E − Base, 0 | −45 | +6 | −45 / −32 | −0.0097 | −0.0174 | −0.0081 | +0.0173 | −2 / −31 | +50 / +17 |
| E − Base, 1 | −92 | +6 | −123 / −17 | −0.0301 | −0.0254 | −0.0211 | +0.0219 | +30 / −38 | +118 / +24 |
| G − Base, 0 | −122 | +11 | −79 / −56 | −0.0351 | −0.0439 | −0.0273 | +0.0127 | +47 / +0 | +129 / +66 |
| G − Base, 1 | −102 | +6 | −98 / −37 | −0.0280 | −0.0342 | −0.0190 | +0.0190 | +30 / −19 | +108 / +39 |

- **Direction of the changes (OOF).**
  - E's fixes: 66 true A4C runs that Base read as A5C and 12 S1 runs (true A5C read as A4C). The other 5 are 2 true
    A5C and 2 true PLAX read as PSAX, and 1 true PLAX read as A4C.
  - E's breaks: 167 correct A5C into A4C and 42 A4C into A5C. The other 11 breaks are spread across classes.
  - G's fixes: 52 true A4C runs read as A5C, 17 S1 runs and 2 true PLAX runs.
  - G's breaks: 158 A5C into A4C and 74 A4C into A5C, plus 63 spread over PSAX / PLAX / SC4C (16 A4C → PSAX, 10 A5C →
    SC4C, 8 A5C → PSAX, 6 PSAX → PLAX, …).
  - Over all 1,255 true A5C runs, recall is 0.9163 for Base, 0.7888 for G and 0.7936 for E (read as A4C: 103, 242,
    257). Over all 1,554 true A4C runs it is 0.9093, 0.8829 and 0.9202 (read as A5C: 135, 151, 111).
  - A32's S2 runs kept as A5C: Base 84 / 84 and 157 / 157; G 60 / 84 and 97 / 157; E 61 / 84 and 77 / 157.
- **Sources of the S1 fixes** (only A5C-coded sources inside the fixed runs are counted).
  - E: 6 + 6 runs from 5 + 3 recordings (8 in total) in 5 + 3 sessions; at most 2 uses per recording.
  - G: 11 + 6 runs from 3 + 3 recordings (6) in 3 + 2 sessions. In fold 0, one recording carries 63% of G's S1 weight
    (up to 7 uses).
  - All fixes come from 74 recordings in 62 sessions for E (G: 56 / 48), and breaks from 194 / 143 (G: 264 / 172). A
    run counts all its source recordings. Session counts are descriptive; sessions are not confirmed patients, and
    constructed streams share sources.
- **Cuts.** The 2,044 / 2,057 internal Base boundaries never move. Relabel and merge delete 62 / 115 of them for E and
  99 / 97 for G.
- **E and G differ** in 205 / 211 per-run labels, counted in the readout process from the private cache
  (`readout_cache_s0.npz`, sha256 b114cad7…: per-window logits and per-run labels, server only).

**Source-2 rule (fixed before any output): not met.**

| Part | Requirement | Result | Met |
|---|---|---|---|
| Target increment | E − G S1 fixed ≥ +1 per fold and ≥ +5 in total | −5 / 0; total −5 (E 12, G 17) | No |
| A5C signal | E S1 fixed ≥ 11 of 103, from ≥ 3 A5C-coded recordings and ≥ 1 per fold | 12; 8 recordings (5 / 3) | Yes |
| Retention | correct A5C ≥ 1,127 of 1,150 and ≥ G − 3; correct A4C ≥ 1,385 of 1,413 and ≥ G − 3 | 982 (G 973); 1,364 (G 1,320) | No |
| Net | E − Base ≥ 0 and E − G ≥ 0 per fold | −45 / −92; +77 / +10 | No (E − G yes) |
| System | segment dF (E − Base) ≥ 0 and (E − G) ≥ 0 per fold | −0.0097 / −0.0301; +0.0254 / −0.0021 | No |
| | native (E − Base) ≥ −0.003; boundary recall ≥ −0.005 per fold; missed short ≤ Base + 3 | −0.0174 / −0.0254; −0.0081 / −0.0211; 117 against 76 (limit 79) | No |

## Answers

1. **The EchoJEPA initialisation did not give a transferable A5C correction over the generic one under this recipe.**
   E recovers fewer S1 runs than G (12 against 17; −5 / 0 per fold). E reaches the 11-run signal level, but only
   alongside a large loss of correct A5C: 982 of 1,150 kept, with 1,127 needed. The main question is answered
   negatively for this configuration.
2. **E is the less harmful branch, but not consistently.** E − G is net +77 / +10, with 75 fewer breaks over both
   folds and 44 more correct A4C kept. In fold 0 it also gains +0.0254 in segment F1. In fold 1, E keeps 25 fewer
   correct A5C and its segment F1 is −0.0021 against G. The echo-pretrained state gives a somewhat better general
   recogniser in one fold, not a targeted A5C gain. That is not equivalence, and not a ranking of the two pretraining
   methods.
3. **Both new branches are much weaker than Base as recognisers of these runs.**
   - E is net −45 / −92 against Base, and G −122 / −102. Segment F1 falls by 0.010–0.035, and missed short true
     segments rise from 76 to 117 (E) and 181 (G).
   - Most errors are at the A4C / A5C boundary in both directions (E: 167 correct A5C into A4C, 42 A4C into A5C).
   - Each branch is a standalone recogniser fitted on 900 × 8 fit windows through r8 LoRA, while A08 was trained on
     all these recordings. This explains why the comparison against Base mixes backbone, pretraining, capacity and
     training exposure. No longer schedule, other rank or full fine-tuning was run, so the cause is not identified.
4. **Closing scope.**
   - Under the fixed rule this configuration closes at source seed 0. There is no source 2, and no rank, lr, head,
     pooling, input or model variant.
   - The result covers only these two pretrained states with r8 QKV LoRA, the target / global head and 900 updates of
     this batch. It does not speak for JEPA pretraining in general, other foundations, full fine-tuning or longer
     schedules.
   - Pretraining overlap with EV9V is unknown. A08 + B3 stays the working reference.
5. **Cost of deployment.** The branch is one frozen ViT-L forward per window: 23.9 ms GPU (BF16, batch 32) against
   16.5 ms for Base's A31 pass, so (16.5 + 23.9) / 16.5 ≈ 2.45 × Base's encoder time, sharing a 2.5 ms decode. It adds 303.9M frozen
   and 0.8M trained parameters to A08's 34.6M.

## Cost, timeline and files

- **GPU 2.87 of 8 h (ledger):**
  - check attempt 1 (failed, 0.052 h), two diagnostics (0.015 h) and check attempt 2 (0.055 h);
  - four trainings (2.038 h: 0.509 / 0.508 / 0.511 / 0.511);
  - the readout (0.710 h).
  - Source 2 would have been projected at a further 3.34 h; it was not triggered.
- **CPU 0.76 of 4 h, as of 2026-10-27 17:58 UTC:**
  - jobs (hashing, extraction, freeze): 0.020 h;
  - the agent's active wall clock: 0.739 h in seven spans. Idle waits during the extract / check / diagnostic jobs
    (−0.134 h, a correction entry), training (15:11:40–15:36:55, 15:38:05–16:07:08, 16:07:40–16:37:52,
    16:38:40–17:08:23) and the readout (17:10:30–17:52:07) are excluded.
  - Network download time (0.036 h) is booked separately, never as GPU. Codex's two read-only helpers (history,
    backbone inventory) ran outside this ledger. All of it is in `runs/video_echojepa_recognition_v1/ledger.jsonl` on
    the server, and a short closing span is appended after the result commit.
- **New disk:** 2.48 GB of 20 GB now (the two FP32 encoders 2.43 GB, clone 32 MB, einops 0.7 MB, run directory
  15 MB). The peak was 12.7 GB, before the downloaded checkpoints were deleted. No pixel, token or feature cache was
  written, and no resume file remains. Free space is 144 GB.
- **Timeline (UTC, 2026-10-27).**
  - Dispatch 14:27, configuration 14:33 (85da250), downloads 14:34–14:37, code 14:49 (5d04536), extraction 14:50.
  - Check attempt 1 failed 14:55; diagnostics 14:57–15:00; repair and amendment 15:01 (e74d7b2); attempt 2 passed
    15:05. Sources and checks were committed at 15:05:44 (954d82c).
  - Training 15:06–17:08. Weights committed 17:09:03 (e980fa3), freeze 17:09:22, readout 17:09–17:52.
  - Codex's review file was read before training (its one blocking item, a valid attempt 2 under the original limits,
    was satisfied), at each training poll and before the freeze; it had no blocking finding at those points. No helper
    agent was used by this worker.
- **Records.** `sources.json`, `checks_attempt1.json`, `diagnose_numerics.json`, `diagnose_numerics_fp64.json`,
  `checks.json`, `weights_s0.json` (manifests with fit-only monitors and LoRA movement) and `readout_s0.json`.
- **Code.** `echo_routing/temporal/echojepa_lora.py`, `scripts/train_echojepa_recognition.py`,
  `scripts/acquire_echojepa_sources.py`, `scripts/diagnose_echojepa_numerics.py` and `tests/test_echojepa_lora.py`
  (9 tests).
- **Server only.** Encoders, LoRA / head weights, per-update logs, job logs, the private readout cache and all
  identifiers.
