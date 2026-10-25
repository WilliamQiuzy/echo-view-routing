# A4C/A5C expert pilot: blind reading tool and supervision interface (2026-10-25)

**What this is.** Annotation preparation after A32, under namespace `a5c_expert_pilot_v1`.
- **Not a model experiment.** It is not an A33 result and makes no performance claim.
- **What it delivers.**
  - A private, blind reading tool for A32's 24 fixed cases.
  - An audit of the source videos behind them.
  - An export → overlay interface that a later training round could use.
- **What was not done.** No model was run or trained, and no development, check, test or Pediatric data was read. GPU
  use was 0 h.
- Configuration: `configs/experiments/a5c_expert_pilot_v1.yaml` (8169e89, committed before implementation). Task sheet:
  `claude_prompts/2026-10-25-a5c-expert-pilot-readiness.md`.

**Status: 0 real expert records.** No reader has used the tool yet. What is missing is human judgement, not engineering.
Every record that exists is synthetic or technical, flagged as such, and stored in a separate tree.

## The 24 cases and their source videos

- **Recordings.** The 24 A32 cases (6 per set S1–S4) come from **24 unique source recordings in 23 sessions**. One
  session contributes two recordings, in different sets. No recording appears in more than one set.
- **Stream types.** Twenty cases are runs in constructed streams and four are native recordings.
- **Unit of reading.** The unit is the whole source recording. A constructed stream used only part of a recording (10–40
  of its 12–290 cached samples), but the tool always shows the full recording. The original case, its stream and its
  target run stay mapped in the private key, for review after the blind pass.
- **Media audit** (`media_audit_summary.json`, aggregates only; the per-recording audit stays on the server):
  - **Originals.** All 24 recordings have their original video: mpeg4, 320 × 240, nominal and average rate 30/1, time
    base 1/15360.
  - **Timing.** Every recording's PTS is monotonic with a constant 1/30 s step, so none has a variable frame rate. Length
    is 34–868 frames (median 147), 4,598 frames in all.
  - **Original vs extracted frames.** The decoded frame count equals the extracted JPEG count for all 24. Each decoded
    frame differs from the JPEG of the same index by 0.36–0.56 grey levels on average (at most 0.62). The same index is
    the closest match for all 4,598 frames.
  - **Mapping of A08's references.** The 10 Hz cache takes every third original frame. Every A32 reference of these
    recordings lies on that cached axis, and none points past the last frame.
  - So a 10 Hz sample maps exactly to one original frame and its PTS time. The "30 fps" that A32's package assumed holds
    for these 24 recordings.
- **Fallback paths.** The tool also supports a recording with only the 10 Hz cache (one step = one 10 Hz sample) and a
  variable-rate axis (times from the PTS). Both paths were tested on synthetic media only, since no real case needs them.

## The private tool

**Where everything lives.** Server only, under `runs/a5c_expert_pilot_v1/`.
- `package/media/<id>/` holds the decoded original frames as PNG (233 MB). `package/cases.json` is the blind manifest.
- `package/reveal.json` (mode 600) is the unblinding layer.
- `private_key.json` (mode 600) maps the neutral ids to the A32 cases, recordings and sessions.
- `readers.json` (mode 600) holds the reader tokens.
- Real records go to `annotations/real/<reader>/<id>.json`.
- The A32 package is read only and was not changed.

**Access.** The server binds 127.0.0.1 only. Start it on the server:

```
cd /home/william/echo-view-routing && .venv/bin/python scripts/a5c_pilot_server.py \
  --package runs/a5c_expert_pilot_v1/package --annotations runs/a5c_expert_pilot_v1/annotations/real \
  --readers runs/a5c_expert_pilot_v1/readers.json --port 8765
```

Then each reader tunnels from their own machine, and opens their personal link:

```
ssh -N -L 8765:127.0.0.1:8765 echo-l40s
http://127.0.0.1:8765/?t=<that reader's token from readers.json>
```

- No public port, permission change or account was created, and nobody was contacted.
- Frames reach the reader's browser only for display. Nothing is saved on their machine.

**Blind first pass.**
- **What the reader sees.** Neutral ids P01–P24 come from a fixed permutation, in a fixed order. The page and the API
  show only the frames and the time axis.
  - Before a reader submits the blind record of a recording, no response carries a set name, public label, model
    output, probability trace, file name or session. The blind case response is limited to an allowlist of keys.
- **Navigation.** The reader can play at the PTS timing, pause, step one frame forward or back (buttons or arrow keys),
  jump 10 frames, drag a slider, click the timeline and zoom 1–4×. Every position shows its frame index and PTS time.
- **Fields.** These are kept separate:
  1. Whole-recording view (A4C, A5C, transitional / off-axis, other, cannot judge) and confidence.
  2. Optional view intervals where the view changes. Frames outside every interval stay unannotated.
  3. LVOT and aortic valve, each as visible / uncertain intervals, "not visible anywhere" or "cannot judge".
  4. Image quality with a reason, plus comments.
  - "Unanswered" is distinct from "not visible" and "cannot judge".
  - A structure state never changes the view field.
- **Unblinding.** It needs a submitted blind record and an explicit confirmation. The reader then sees the public clip
  code and, for each A32 use of the recording, these outputs on its own frames:
  - the target run's frames;
  - the run's reference label;
  - the labels from A08 + B3, the four A31 branches and both external rules;
  - the A4C / A5C probabilities of A08, EchoViewCLIP and EchoPrime.
- **After unblinding.** Label fit (yes / partly / no / cannot judge) and a review comment are added as new versions.
  Every version is append-only, and the submitted blind version can never be edited.
- **Two readers.** R1 and R2 each have their own token and directory, with entries for all 24 recordings, S1
  included. A token reaches only its own records. A second reader never sees the first reader's fields.
  - `double_reading.csv` is the pairwise comparison format: field equality and per-frame overlap of submitted blind
    versions. No agreement statistic is computed or reported until real double reading exists.
- **Exports.** Each reader exports JSON (the primary record) and CSV from the page.
  - Records carry the neutral id, reader id, frame-axis sha256, frame count, time base, axis source, versions with
    phase, submission flag and UTC time, and the unblinding flag and time.
- **Synthetic separation.** Synthetic and technical records exist only under `runs/a5c_expert_pilot_v1_synthetic/`
  with `synthetic: true`. The synthetic package shows a red "SYNTHETIC ENGINEERING DATA" banner.

**Reader instructions** for the first expert are in [READER_GUIDE.md](READER_GUIDE.md).

## Training interface (overlay), not used for training here

`scripts/a5c_expert_pilot.py --stage overlay` turns submitted records into a private overlay bound to each recording's
frame-axis sha256. The rules are fixed now, before any annotation exists.
- **Labels.** Public labels are not an input and are never written.
- **Frame codes.** Per original frame, the view and structure codes keep these states distinct: unknown (unannotated),
  each view, not visible, visible, uncertain, cannot judge, and reader disagreement.
- **No inference across fields.** A structure that is not visible never becomes A4C, and unannotated frames are never
  negatives. The whole-recording view stays a recording-level field and is not spread over frames.
- **What reaches training.** Only submitted records. Drafts never enter, and real and synthetic records cannot be mixed.
- **10 Hz samples.** Each sample takes the code of its own original frame, so there is no interpolation and no evidence
  from outside the recording.
- **Folds.** Each recording carries its A31 session fold, which is also its pool, so later training can keep every
  source and its window context inside one fold.
- `check_overlay` verifies the axis binding and the codes, and reports frames asserted, unknown and disagreed.

## Engineering checks

| Check | Result |
|---|---|
| Unit tests on synthetic data (`tests/test_a5c_pilot.py`, 15; local and server) | pass |
| Interval validation: beyond the last frame, reversed, overlapping, float frames, intervals with a contradictory status, unblinded fields in the blind phase | all refused |
| JSON ↔ CSV round trip (unicode, quotes, commas, newlines, all versions) | identical |
| Synthetic flow through the HTTP API (two readers × four recordings: constant 30 fps, variable rate, 3 frames, 10 Hz only) | as expected (below) |
| Overlay on synthetic records | correct codes for agreement, disagreement and unknown frames; checks pass |
| Browser run of the page on synthetic data (local headless Chromium) | blind pass, stepping and clamping, marking, save, submit, lock, unblind, review version, second-reader isolation, bad token; no page error |
| 2 real recordings, chosen by `default_rng([20261025, 33, 3])` (94 and 208 frames) | all checks pass (below) |
| Real records after all checks | 0 |

- **Synthetic API flow.** Drafts and blind submissions were accepted, blind edits after submission were refused,
  unblinding and the post-unblinding submission worked, and the other reader's reveal was refused.
- **Two real recordings.** The checks covered the following, without filling any medical field:
  - PTS and frame count equal the decode, and the served first and last frames are pixel-identical to the decode.
  - A frame past the end is refused, and the blind response keys are allowlisted with no leak scan hit.
  - The reveal layer is refused before unblinding, and out-of-range and reversed intervals are refused.
  - A technical draft saves, the other reader sees nothing, and the export round trip is identical.
  - Their technical records are flagged synthetic and stored in the synthetic tree.

**Not verified.**
- The page has not been rendered with real frames in a browser. The server has no browser, and real frames are not
  copied off the server. The real-frame HTTP layer and the page logic were checked separately.
- No real reader has used the tool, so reading time, field usability and repeatability are unknown.

## What the first expert would do

Read the 24 recordings in order (P01–P24), in the blind pass first. For each recording:
1. Fill the whole-recording view and confidence.
2. Mark view intervals only where the view changes.
3. Mark the LVOT and aortic-valve intervals, or state "not visible anywhere" or "cannot judge".
4. Rate image quality and give a reason, then submit.

Then, if willing, unblind each recording and record label fit and a comment. A second reader does the same blind pass
on their own link, at least for the S1 recordings. The key holder can find which these are; readers cannot. How long
this takes is unknown, because nobody has read yet.

## Conditional training draft (not started)

The pilot has to answer two questions first:
- Can the fields be completed consistently? Report "cannot judge" rates and, where two readers exist, repeatability.
  The repeatability method is fixed once the real denominators and reading design are known.
- Does expert evidence differ from the clip code in a local, operational way? For example: frames of A5C-coded
  recordings judged transitional or A4C, or LVOT / aortic valve not visible over the frames that A08 reads as A4C,
  while the controls do not show this.

**If both answers are yes**, a later round could test one question. On the same annotated recordings, does supervision
from the expert overlay correct A5C runs read as A4C without breaking correct A5C runs, compared with supervision from
the clip code?

**Design constraints.**
- **Matched arms.** Both arms start from the same A08 point. They use the same inputs, batches and compute, and the same
  annotated samples, so a gain cannot come from extra sample count.
- **Supervision rules, fixed before training.** Unknown, cannot-judge and disagreed frames carry no loss in the overlay
  arm. How to treat transitional frames is decided in advance as well.
- **Cuts.** The original A08 + B3 cuts stay fixed, and runs are only relabelled.
- **Folds.** A31's session folds are kept. Readout is on held-out-fold annotated recordings plus full sequences, with no
  development data used to choose anything.

**If the pilot shows no local difference, or fields that cannot be completed**, there is no supervision change to test.
- The 123-recording extension proposed in A32 is then not justified by this evidence.
- That number was an enrichment scope, not a computed minimum.

**Scope limits.**
- The 24 cases were enriched by error set, so they cannot estimate EV9V's label-error rate or any performance gain.
- The expert field definitions do not change after any model result.

## Cost and files

- **Cost.** GPU 0 h. CPU about 0.35 of 2 h: 0.03 h of stages plus the agent's active wall clock, booked in
  `runs/a5c_expert_pilot_v1/ledger.jsonl` on the server.
- **Code** (ddde2f4).
  - `scripts/a5c_expert_pilot.py`, stages audit / build / readers / synthetic / techcheck / status / overlay.
  - `scripts/a5c_pilot_server.py`, `scripts/a5c_pilot_records.py`, `scripts/a5c_pilot_ui.html`.
  - `tests/test_a5c_pilot.py`.
- **Records here.** `media_audit_summary.json`, `techcheck.json`, `status.json` (aggregates, no identifiers, no media).
