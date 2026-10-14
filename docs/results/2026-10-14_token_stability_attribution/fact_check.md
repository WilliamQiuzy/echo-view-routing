# Fact-check of the token stability attribution round

An independent fact-checker, a separate agent, checked this round on 2026-10-14 after the single run. It worked
read-only on the repository and the server, ran no stage of the round, loaded no model and printed no recording,
session, stream or recipe id. It used its own code (numpy, pandas and yaml only) except where noted. It used about 0.045
CPU hours, recorded in the round's ledger.

**Verdict.** No error in the results. The README had three medium and twelve low wording errors, all corrected before
publication. Three notes on the records were handled as described below.

## What was verified

1. **Integrity and order.**
   - **Pins.** All 44 pins match: the task sheet, A03's configuration, summary, README, choices and six saved outputs,
     A02's configuration, summary and README, the protocol, the clip index, the window plans, the 18 prediction files
     and the nine final weights.
   - **Identities and decoder values.** The 18 prediction files carry A02's training identities. The fixed decoder
     values equal A02's stored aligned CLS settings.
   - **Code and configuration.**
     - The checker recomputed the code digest e861fc8d with its own hashing over the 85 imported modules. It is the same
       on the server, in the working tree and at e4ee52e and 27d44aa.
     - The configuration sha256 f8b7898c matches e4ee52e. The only change from ae15344 (e42dec3c) is the
       receptive-field wording, which no code reads.
     - The preparation, check and run all carry both identities.
   - **Order.** The commits run 99b4571, ae15344, 35b626a, 3855d7a, e4ee52e and 27d44aa. The records commit precedes
     the run's ledger start.
   - **Tests.** At 35b626a the tests fail at collection, as intended. At e4ee52e all 21 round tests pass.
   - **Run directory.** It is complete. The run summary equals the results summary apart from its cost and run
     entries.
2. **Segment counts.**
   - **Rebuilt chain.** The checker rebuilt both halves' windows from the allowlist metadata, with its own selector,
     softmax, nearest-centre mapping, B3 decoder and matching. It reproduced A03's anchors, token logits and decoded
     outputs, and A03's stored CC and TT rows.
   - **Comparisons.** Over all six seed-and-half cells, 10,648 comparisons gave 0 mismatches. They covered the actual
     systems, the fixed decoder and the four cells with B3 and with argmax:
     - TP, FP, FN and F1 per population, and segment status;
     - newly matched, lost and unmatched predicted segments by class, length, X-Y-X middle, recording and session;
     - the probability-only changes, the switches and the counts over seeds.
   - **Row metrics.** These agree within 1e-12.
3. **Contrasts.**
   - An independent bootstrap replayed the draw sequence: 2,000 draws, seed 20261002, sessions then streams, and the
     same draws for every contrast.
   - It reproduced all 288 differences and intervals within 1e-12.
   - Re-applying the three-seed rule reproduced all 96 verdicts.
4. **Windows and readings.**
   - The request-distance bins, the changed requests and their sources, and the status of the segment holding each
     corrected request all reproduce in all six cells.
   - So do the local shares and sessions, the anchor disagreements and sessions, the flags and the applied readings.
5. **Overlapping anchors.**
   - **Pairs.** A separately written pair construction gives the same 35,429 (selection) and 32,050 (calibration)
     pairs. Every attribute is identical: completeness, sample and marker triples, main subset, selected anchors, centre
     distances, positions, population, context, label and distance bin.
   - **Weights.** The group weights sum to 1 per group within every subset, population and context.
   - **Rates.** The disagreement rates, divergences, correctness counts, roles, unions and per-pair flags of all nine
     arm-and-seed combinations reproduce.
   - **The selected anchor in both windows.** No pair holds it, and this follows from the geometry.
     - A selected anchor holds the request or the sample before it.
     - Requests of a sequence are at least 3 samples apart.
     - Two selected anchors could share a sample only if their requests were at most 1 sample apart.
6. **Stem geometry.** Read from the library source: time kernel 3, stride 2 and padding 1, time stride 1 in all 16
   blocks, and a marker stem that copies the patch embedding.
7. **Ids.** No recording or session id appears in any results file, in the working tree or at the round's commits.

## Corrections made to the README

**Medium**
1. **The seed-0 note under table 2.** It said both decodings give the same counts. They differ by one matched segment
   on calibration (512 / 13 / 6 against 511 / 13 / 7).
2. **Counts on fewer than 10 sessions.** Many were not flagged. The README now flags:
   - the per-status columns of table 1;
   - the per-seed newly matched and lost counts;
   - the native window counts;
   - the changed segments of native clips and pair streams.
3. **Output anchors.** The README said the observed instability concerns side anchors only. Now it says:
   - 79% to 88% of the disagreeing equal-marker positions lie 4 or 6 slots from the marker;
   - 44 to 64 disagreeing main-subset pairs per seed and half compare an output anchor with another window's side
     anchor;
   - the measured instability is not evidence about the cause of A03's output switches.

**Low**
1. **Wording.** The opening says "where" rather than "why".
2. **Order of events.** The review verification followed the repeated preparation and check.
3. **Anchor pairs.** These are 10 samples apart in all but 5 to 10 pairs per seed and half, which are 15 apart at
   sequence ends.
4. **The comparator.** The share within 5 samples of a change is 32% for the equal-marker multi pairs (30% for the
   whole main subset).
5. **The controls.** The comparison with the controls is quantified (about half in pair streams, 0.34 to 0.47 times in
   multi streams) and moved to the observations. Native clips show similar rates.
6. **Section 5.** It says "mostly", gives 19 of 220, and merges the two lists of observations.
7. **Pure-window fragments.** They are qualified: more native switches in all six cells, consistent on calibration
   only.
8. **Argmax TT − CC.** Its consistent calibration result was added.
9. **Changed-marker pairs.** The selected-anchor share is 72% to 75%.
10. **Deviations.** They now describe the test changes after the failing commit and the exact configuration wording.
11. **Files.** This file now exists, as the README's file list states.

## Notes on the records

1. **Ledger timestamp.** The hand-recorded verification line gives a finishing time of 22:20, which is impossible
   because the verification ended before 22:07. A zero-hour correction note was appended; the line's 0.15 hours stand.
2. **Test-time estimate.** A full run of the round's tests took about 59 s on the server, so the first 0.06-hour
   estimate was low. A supplementary 0.06-hour line was appended.
3. **Test fixture names.** Two synthetic recipe names in `tests/test_token_stability_attribution.py` formally matched
   positional stream names. Such names are a bank name and an index, with no recording or session information. They
   were renamed after the run. Tests are not part of the code digest.
