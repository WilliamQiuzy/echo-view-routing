# Post-A37 cached diagnostic: sources of the S1 fixes, an eW → e224 substitution, and the extra missed short segments (2026-10-26)

**Question.** A37's W head fixed 8 S1 runs (true A5C read as A4C by Base), but it broke 40 runs and missed 17 more
short true segments than Base. This diagnostic asks two things before any new training is discussed.
- Are the 8 fixes spread over different sources, and do they depend on the added W feature?
- Through which relabels or merges of runs do the extra missed short true segments arise?

**Scope.** This is not a training round and has no A-number. GPU 0 h. It reads A31's two session-fold training pools,
source seed 0's frozen Base / S / W outputs, the four heads committed in f55c1ed, the cached e224 / eW and the
reference labels. There was no new feature, training, threshold or ratio sweep, router or variant. No development,
check, test or Pediatric data was read. Scope 771f501, with a clarification committed before any computation
(825cb5a: every undo restores one original Base run only). Code cf36357. Task sheet
`claude_prompts/2026-10-26-post-a37-context-evidence-audit.md`.

**Data limits.** A08 trained on these recordings. Labels are EV9V clip codes, not expert LVOT judgements. Constructed
streams reuse source recordings, so streams, repeats and seeds are not independent patients.

## Reproduction

Base, S and W were recomputed through A37's own table code before any new number. Every cell of A37's
`readout_s0.json` (sha256 54273e6a…, equal to the committed file) matched: runs, changes, correct A5C, S1, counts,
sequence rows and coverage, with 0 mismatches in both folds. The P runs equal Base's run-length runs in every
sequence, and each run's pooled label equals the sequence relabel.

## 1. Sets and sources

Units are A31 P runs of the held-out folds. A run counts every one of its source recordings.

| Set | Units | Source recordings / sessions | Units sharing source frames with another unit of the set | Mean streams holding the same frames |
|---|---|---|---|---|
| R (S1 fixed by W) | 8 | 14 / 14 | 0 | 9.4 |
| S1 not fixed by W | 95 | 66 / 59 | 79 | 8.4 |
| D5 (correct A5C broken by W) | 32 | 36 / 32 | 17 | 9.7 |
| D4 (correct A4C turned into A5C by W) | 8 | 12 / 12 | 0 | 4.6 |
| L (short true segments W loses) | 22 | 20 / 19 | 2 | 5.9 |

- **R is spread out.** Counting only A5C-coded sources, R's 8 runs come from 8 recordings in 8 sessions, one run
  each. The largest single recording holds 14% of R's A5C-coded weight. Three R runs are mixed and add 6 more source
  recordings.
- **The controls are concentrated.** The 95 unfixed S1 runs come from 37 A5C-coded recordings, with up to 12 runs from
  one recording. 79 of them share source frames with another unfixed S1 run in a different stream.
- **Pool reuse.** The frames of every R run also occur in other constructed streams of the pool (9.4 on average, at
  most 23). No other R run uses them.

**Correct A5C losses (the 9 / 11 question).** S keeps 1,139 of 1,150 correct A5C runs. Its 11 losses go to A4C (9),
PLAX (1) and PSAX (1). A37's README table ("1,139") and its prose "S breaks 9 correct A5C runs" are both right. The 9 counts
only the A4C destination. W's 32 losses all go to A4C.
- 9 of W's 32 are also broken by S, and 23 are W's alone. Two runs are broken by S only.
- W's 40 breaks use only two transitions: A5C→A4C (32) and A4C→A5C (8). S's 16 breaks spread over eight transitions.

## 2. One input substitution through the frozen W heads

Each held-out fold used the frozen W head of the other fold. Its auxiliary input eW was replaced once by e224 from the
same request window. The weights, runs and decoding were unchanged:
- original: q + g_W(concat(e224, eW));
- substitute: q + g_W(concat(e224, e224)).

| Set | Changes W made | Kept under the substitute | Back to Base | Other |
|---|---|---|---|---|
| R (S1 fixes) | 8 | 1 | 7 | 0 |
| A4C read as A5C, fixed by W | 44 | 28 | 16 | 0 |
| D5, broken by S and W | 9 | 7 | 1 | 1 to PLAX |
| D5, broken by W only | 23 | 8 | 15 | 0 |
| D4 | 8 | 0 | 8 | 0 |
| L (lost short segments) | 22 | 6 still unmatched | 16 matched again | – |

- **Whole sequences under the substitute.**
  - Segment F1@50 is 0.9652 / 0.9510, against W's 0.9651 / 0.9513 and Base's 0.9639 / 0.9499.
  - It has 85 missed short true segments (W 93, Base 76) and 157 short false segments (W 137, Base 178).
  - It keeps 1,129 correct A5C runs and fixes 1 S1 run.
  - It also makes 15 new changes on runs W had left at Base.
- **The eW-dependent part nets out.** 23 of W's 52 fixes return to Base (7 S1 and 16 A4C read as A5C), and so do 24 of
  its 40 breaks (15 W-only D5, 8 D4 and 1 shared D5). This matches A37's W − S of about zero.
- **Reading limits.**
  - The substitution changes the joint input distribution the head saw in training.
  - Seven vanished S1 fixes show that they depend on this replacement of eW. They do not show that distant frames hold
    generalisable anatomy.
  - A kept change does not show that W was unused in training.
  - The substitute is not the S system and not a candidate. Nothing is selected from it.

## 3. The extra missed short true segments

The trace used the evaluator's own one-to-one matching (IoU ≥ 0.5, same label) and run-length segments. The chain
classes were fixed in the scope before any computation, with priority relabel, then merge, then rematch, then other.

| Arm | Lost | Gained | Net | Relabel | Merge | Rematch | Other | Restored by one single-run undo |
|---|---|---|---|---|---|---|---|---|
| W | 22 | 5 | +17 (f0 12 − 3, f1 10 − 2) | 19 | 3 | 0 | 0 | 22 of 22 |
| S | 8 | 5 | +3 | 8 | 0 | 0 | 0 | 8 of 8 |

- **The net matches the counts.** Lost minus gained equals A37's missed-short difference in every fold (93 − 76 = 17
  for W, 79 − 76 = 3 for S).
- **Relabel (19).** W relabels the very Base run that matched the short true segment. 14 go A5C→A4C and 5 go A4C→A5C,
  and 13 of these runs are absorbed into a neighbouring segment of the new class. These runs are W's breaks: 14 are D5
  and 5 are D4.
- **Merge (3).** W keeps the short segment's label but relabels one neighbouring run into it. The merged segment no
  longer overlaps the true segment enough. The neighbouring runs are 2 D4 and 1 D5.
- **Overlap with the run sets.** 15 of the 32 D5 runs and 7 of the 8 D4 runs appear in these chains.
- **Single-run undo.** Restoring Base's label on one original run brings back every one of the 22 matches. No unit
  needed more than one run. This is a reference-assisted check, not a deployable intervention.
- **What this means.** The extra missed short segments are the same A4C/A5C relabel events as D5 / D4, seen at the
  segment level. They are not lost cuts, since the cuts are fixed. Boundary recall falls because merged runs remove
  predicted label changes.
- **Gains.** W's 5 gains are all A4C segments from relabels. They come from W's fixes of A4C read as A5C. S also has 5
  gains, all A4C relabels.
- **Sources.** The 22 lost segments come from 20 recordings in 19 sessions. One recording holds 3.

**What the inputs contain** (reference labels, described after the fact; the model never saw them; correlation is
not cause, and no mask or rule follows).

| Set | W slots with another label that S lacks | S window crosses a label change | S window crosses a Base cut | Median length (samples) | Pure |
|---|---|---|---|---|---|
| R | 0.15 | 0.46 | 0.85 | 10 | 0.63 |
| S1 not fixed | 0.15 | 0.41 | 0.79 | 15 | 0.74 |
| All P runs | 0.17 | 0.41 | 0.44 | 27 | 0.59 |
| D5 | 0.26 | 0.70 | 0.78 | 15 | 0.59 |
| D4 | 0.36 | 0.94 | 1.00 | 10 | 0.63 |
| L (Base run that matched) | 0.38 | 0.96 | 0.96 | 10.5 | 0.50 |

- The harms sit in short runs whose S windows already cross a label change, and W adds more other-label frames there.
- R and the unfixed S1 runs have the same share of added other-label slots (0.15). R's 8 runs are only somewhat
  shorter and less often pure.

## Answers

1. **R's 8 fixes come from 8 different A5C-coded recordings in 8 sessions**, one run each. The largest single
   recording holds 14%, and no two R runs share source frames. With eW replaced by e224, 7 of the 8 return to A4C.
2. **W's 40 breaks are 32 A5C→A4C and 8 A4C→A5C.** S's 11 correct-A5C losses are 9 to A4C, 1 to PLAX and 1 to PSAX.
   The 9 and the 11 differ only by destination. W's breaks move only along the A4C/A5C boundary. Nine of its 32
   A5C→A4C breaks are shared with S, and these mostly stay under the substitute (7 of 9). The W-only breaks mostly
   vanish (15 of 23).
3. **The extra 17 missed short true segments are 22 lost minus 5 gained.** The main chain is a relabel of the matched
   Base run (19, mostly A5C→A4C absorbed into a neighbour), plus 3 merges and no rematches. A single-run undo restores
   every one.
4. **There is not yet enough evidence for a specific new training condition.** The strongest lead is W's input-dependent
   S1 signal.
   - **Evidence.** The 8 fixes come from distinct sources, and 7 of them depend on eW under this substitution.
   - **Against.** The eW-dependent harms are as large as the eW-dependent fixes. 24 breaks return to Base under the
     substitute against 23 fixes, and 16 lost short segments are matched again. For the target, 7 eW-dependent S1 fixes
     face 24 eW-dependent breaks (16 of correct A5C, 8 of correct A4C).
   - **Unknowns.**
     - No input field we described separates R from the unfixed S1 runs (added other-label slots 0.15 in both). A deployable signal that separates the
       fixes from the harms is therefore missing.
     - Span and sampling density were not separated.
     - Whether the frames that help R lie inside or beyond Base's predicted run was not measured. R runs are short
       (median 10 samples), and 85% of their S windows already cross a Base cut.
     - Whether the clip codes of these short runs match a visible LVOT is unknown; the reader pilot has 0 records.
   - This diagnostic does not stop the research line. A08 + B3 stays the working reference, A37's closing decision is
     unchanged, and nothing here is a model.

## Cost and files

- **GPU 0 h.**
- **CPU about 0.4 of 0.75 h.** The job took 0.007 h (24 s wall, 28 s process CPU). The rest is the agent's active
  time: 0.33 h through this report, plus the closing commits. The ledger
  `runs/post_a37_context_evidence_audit_v1/ledger.jsonl` on the server books all of it.
- **Disk 76 KB.**
- **Records.** `diagnostic.json` holds id-free aggregates. Per-unit traces and source maps stay on the server under
  `runs/post_a37_context_evidence_audit_v1/private/`.
- **Code.** `scripts/audit_post_a37_context_evidence.py`; scope `configs/experiments/post_a37_context_evidence_audit_v1.yaml`.
