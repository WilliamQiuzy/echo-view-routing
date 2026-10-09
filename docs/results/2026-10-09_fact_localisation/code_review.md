# Code review before the localisation run

This is the independent code review that the task sheet requires before any diagnostic is run
(`docs/superpowers/plans/2026-10-09-fact-localisation.md`, Task 4). An independent code-review agent read commit 89bac43
against the task sheet and the pinned official FACT source (`third_party/fact` at 7bd81bd). It also ran the local CPU
tests (21 passed, none skipped). Nothing had been run on the server at the time of the review.

## Verdict

The verdict was "pass with fixes". There was one medium finding and one low finding, and no critical or high finding.
Both findings were fixed before the run, in the commit that adds this record.

## What the review confirmed

- **Only the partition changes.** The override of `temporal_downsample` matches the official method line for line,
  except that it builds the segment list from the given intervals. It is set on the last block's instance only, and it
  is removed on exit, also after an error. The inputs to the U block are checked bit for bit between A, B and C. A is
  checked against the ordinary official forward, and the order A, B, C, A is checked on every sequence. The official
  checkout and the formal adapter are unchanged.
- **No reference label enters the network.** Segment actions come from the u block's argmax. Tokens come from the
  learned queries, because `FACT.trans` is false. Reference labels only decide where B places its cuts.
- **Denominators and directions.** Pooled segment F1@50 is 2 x matched / (predicted runs + reference runs), and the
  per-sequence distribution is reported beside it. Subset tables report only matched runs, totals, recall and fixed or
  new runs, with no subset F1 or precision. In a comparison "x - y", a fixed run is matched under x but not under y, and a
  new error is matched under y but not under x. A test covers both directions.
- **Allowlist before labels.** The label column is parsed only for the allowlisted training rows, after the filter. The
  bank and the batches come only from those rows. The previous round's `Ctx` is never instantiated.
- **Native windows.** A native window has one reference label, so B and C equal A. The run reuses A's output for these
  windows. The check executes B and C on the five native windows of the fixed 16 and compares the records.
- **Measures, sources and ids.** The per-run matching is checked against the shared `segment_f1` rule. Every count
  carries its distinct recordings and sessions. Repo-bound outputs pass two id guards.

## Findings and fixes

1. **Medium: the run identity did not cover all code that shapes the saved records.** The identity left out
   `echo_routing/temporal/segments.py`, which builds the A partition and the B cut positions. It also left out the
   constants that the inference code imports from the tables module. A later edit to either could have let stale
   records be reused under an unchanged identity.
   *Fix.* `segments.py` and `dev_run.py` are now in the hashed code list. The inference constants are now part of the
   identity payload. A test guards both.
2. **Low: a stop for the budget wrote no summary.** If the budget stopped `--run` before a seed, the finished seeds were
   kept on the server, but no partial summary was written.
   *Fix.* A stop for the budget now writes `summary.json` and `report_tables.md` for the finished seeds, marked as
   partial, and then exits with the reason.
