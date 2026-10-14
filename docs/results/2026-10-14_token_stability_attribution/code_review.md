# Implementation review of the token stability attribution round

An independent reviewer, a separate agent with read-only access, reviewed the round's code before the diagnosis ran. It
reviewed commit 3855d7a against the task sheet and the configuration (ae15344), with the tests committed before the
modules (35b626a), and then verified the fixes in commit e4ee52e. It ran no stage of the round, computed no diagnostic
count from the development data and printed no recording, session, stream or recipe id. Its probes ran the tests in
isolated copies, recomputed the code digest and drove the run stage end to end on synthetic data. They used about
0.11 CPU hours for the review and about 0.15 hours for the verification, both recorded in the round's
ledger.

**Verdict.** Pass with fixes. No finding was blocking. The four should-fix findings and the consequential notes were
fixed in e4ee52e. The reviewer verified the fixes: the 58 tests pass in isolated copies, the recomputed digest
equals the one recorded by the repeated preparation and check, and its synthetic harness exercised every fix, including
five reruns of the publish step.

## What the reviewer confirmed before the fixes

- **Provenance.** The server's script, tests and configuration equal the commit. The recomputed code digest equals the
  one recorded by the preparation and the check, so both stages ran the reviewed code.
- **Sources.** Every pin and the A02 run identities are checked, and A03's allowlist adapter is used. No training
  feature, test or EchoNet-Pediatric file, raw frame, non-allowlisted label, validation entry, Inputs class or A02
  evaluate entry is read. No model is built, loaded or run.
- **Stem geometry.** The source-code check finds time kernel 3, stride 2 and padding 1, and no later temporal
  downsampling, and it stops on a mismatch.
- **Reproduction.** A03's aligned token and CLS rows and its saved token logits, anchors, offsets, per-sample
  probabilities and decoded outputs are reproduced bit for bit.
- **The four cells.** The source is chosen per window before the mapping, native windows are pure, CC and TT equal the
  two readouts, and the decoder is A02's stored numeric setting without refit.
- **Matching and accounting.** The greedy order and the populations follow the task sheet. In the reviewer's own
  brute-force checks, the matching, segment status, X-Y-X middles and request distances had zero mismatches on 3,000 and
  2,000 random sequences. Its independent pairing code agreed with the round's on 40 random window geometries.
- **Bootstraps.** All contrasts use the same draws, and the interaction equals the algebra of the three contrasts.
- **Outputs.** The id guards cannot trip falsely (the shortest of 1,095 ids has 15 characters). Private data stay
  under the server's ignored `runs/` directory, the run is written atomically, and a finished run is never overwritten.

## Should-fix findings and their fixes

1. **The disagreeing anchor pairs were not broken down by role.** Role distributions existed only for the whole main
   subset, so the task sheet's "changed roles" would have needed a post hoc addition. Now every arm, seed and half
   reports, for its disagreeing pairs, the anchor pairs, marker-relative positions, marker-triple pairs, centre
   distances, boundary-distance bins and A03's selected output anchors. The per-pair disagreement flags stay on the
   server.
2. **The few-session flag was too strict and not traceable.** Sessions were counted per population, and the local
   reading's sessions counted even when the reading did not apply. Now the anchor reading's sessions are counted per
   seed and half over all populations, the flag considers only readings that apply, and the summary records the
   session counts and which condition fired.
3. **A crash between the run directory and the results copies could lose the copies.** Now a reproduced rerun writes
   missing or differing results copies again, and the tables of the results copy are checked for ids too.
4. **The configuration was not tied across stages.** Now the preparation and the check record the configuration's
   sha256, every later stage requires it to be unchanged, and the run checks the task-sheet pin, the stem geometry and
   the bootstrap settings again.

## Notes and their fixes

- The decoder flag is derived from the two settings instead of being fixed to true.
- The configuration's stem wording now says no later temporal downsampling, since later blocks still pool time with
  stride 1. Equal local inputs are a statement about the stem only. This is a wording fix, not a definition change.
- The stem check also requires the source encoder to be built with the default constructor, and torchvision is recorded
  and checked.
- The check compares the CLS per-sample probabilities with A02's saved ones.
- Predicted segments carry no X-Y-X flag (not applicable). Probability-only changes are split by population. The
  unions over seeds include the changed windows and the unmatched predicted segments. The actual systems' counts are
  checked against the evaluator's tables, and the bootstrap parity check covers both decodings.
- New tests cover the marker-plane and builder failure branches, the configuration-hash precondition, the publish and
  rerun paths, and the readings' flags.
- **The configuration's identity changed.** The wording fix moved the configuration's sha256 from e42dec3c (ae15344) to
  f8b7898c (e4ee52e). No code reads the edited text, and no definition, threshold or budget changed. The final
  preparation, check and run all carry f8b7898c.
- Two observations were kept for the report. A pair can hold A03's selected output anchor in both windows only if
  their centres differ by at most 1, while stride-5 centres are at least 3 apart, so zero such pairs is a design
  property. Commit 3855d7a also changed the red tests, which the report states.
