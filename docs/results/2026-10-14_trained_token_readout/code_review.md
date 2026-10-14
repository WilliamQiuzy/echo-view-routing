# Implementation review of the trained token readout round

An independent reviewer, a separate agent with read-only access, reviewed the round's code on 2026-10-14 before any
token metric existed. It reviewed commit 783af48 (selector, source checks, metadata adapter, prepare and check stages)
and commit 95b56e4 (tables module, selection stage, dry run and their tests). It then verified the fixes in commit
0e62879. It ran no calibration, dry run or selection, computed no token metric, printed no recording, session or stream
id and wrote nothing on the server. Its read-only probes used about 3 minutes of server CPU outside the stage ledger.

**Verdict.** Pass with fixes. No finding was blocking. All seven should-fix findings and the nine notes were fixed in
0e62879, and the reviewer verified the fixes.

## Should-fix findings and their fixes

1. **The choices were not write-once, and selection was not tied to the committed record.** `--calibrate` could
   overwrite an earlier record, and `--selection` only checked that the record agreed with itself. Now `--calibrate`
   refuses when an existing record differs. `--selection` needs `--choices-sha256`, the sha256 of the committed file,
   and both `done.json` and the summary's identity block carry it.
2. **A choice was never re-derived from its own grid.** A record that named another setting than the rule's choice
   would have passed. Now the calibration rule is applied again to each record's own 40 settings. The status, setting
   and feasible count must agree, an infeasible record must show the 9 / 0.95 diagnostic setting, and each short choice
   must equal its full record.
3. **Per-window outputs were written before the repeat-run guard.** Now a run computes everything in memory and compares
   it with any earlier summary before it writes anything.
4. **A repeat run rewrote the results summary and `done.json`.** Now a repeat after a finished run must reproduce the
   stored summary exactly and writes nothing. The cost and stage times go only into the results copy.
5. **The fixed readings had no note for native or recall losses against CLS.** The task sheet asks for these losses to
   be stated separately. Now the readings carry a note when Aligned token − Aligned CLS is consistent negative in
   native macro-F1 or boundary recall, and a test covers it.
6. **The clip index was not pinned after the preparation.** Now every read of a half checks its sha256.
7. **The results summary carried no run identity.** Now it holds the code digest, the configuration, choices,
   prediction, Ordered output, protocol, clip index and window plan digests, which half each reported half was read from,
   and the earlier stages' ledger times.

## Notes and their fixes

1. The code digest now covers this file, the listed helpers and every repository module imported with it (83 modules),
   including the segment and boundary metric modules.
2. Calibration, the dry run and selection now need a passed check made by the same code. The check needs the
   preparation made by the same code.
3. Every stage checks the pinned environment, and torch must equal the version recorded at preparation.
4. Error messages name a recipe or sequence position instead of an id.
5. The per-sample offset split counts samples, not windows.
6. The dry run writes only under its own run directory, marks its summary as not a result and prints no reading.
7. The tests were strengthened. The dry run reads the calibration half exactly once, and a finished run is never
   overwritten.
8. Comparisons made at an infeasible system's diagnostic B3 setting are flagged.
9. Smaller points.
   - The check records the real largest probability difference.
   - The stored-row checks also compare the native confusion matrix and the A4C/A5C counts.
   - Each window plan must belong to its half's bank.
   - In the final run, the recomputed CLS and Ordered goals must equal A02's stored goals.

## What the reviewer confirmed

- **Sources.**
  - The preparation and the check ran exactly the reviewed code.
  - The pinned A02 summary, README and configuration equal the files at 05a5228 and 13f286c.
  - Every saved prediction file's identity equals A02's run record, and every final-weight pin equals A02's recorded
    weight digest.
- **Label isolation.**
  - A file-open trace of the calibration path, stopped before any token computation, opened only the protocol, the clip
    index, the window plans, A02's summary, the nine calibration prediction files and the nine weight files. The weight
    files are hashed but never loaded.
  - No selection file, feature cache, test file or EchoNet-Pediatric file was opened.
  - The clip index holds training and validation rows only. Its training rows are read for id, split and length,
    without labels.
- **The selector and the time mapping.**
  - The selector implements the task sheet's rule exactly.
  - Token j is the time logit of slot 2j, as A02 trained it.
  - The prediction stays at the request, and the probabilities go through A02's own mapping.
- **Positions.**
  - Offsets are 0 or −1 only. Every non-exact window has an odd marker slot and reads the slot just before it.
  - Non-exact windows occur only at the clamped start (the second window of each sequence), at the clamped end and in
    the 28 selection and 19 calibration native clips shorter than 16 samples.
  - Every recipe fragment is 10 to 40 samples long, so no view change falls between a request and its anchor.
- **Reproduction and decoding.**
  - Every A02 CLS row and every frozen Ordered row reproduces exactly in both halves.
  - The B3 grid and choice rule equal A02's, and selection applies the committed numeric thresholds.
- **Comparisons and readings.**
  - The comparisons, bootstrap settings, consistency rule and classification subsets match the task sheet.
  - A stream-sensitivity conflict makes a half uncertain, and only all-mixed accuracy enters the readings.
- **Ids.** The id guard holds 1,095 ids, the shortest 15 characters long. None of them occurs in the fixed text of the
  tables, so the guard cannot trip falsely. Per-window outputs stay under the server's ignored `runs/` directory.
- **Tests.** 36 of 36 pass on the final code, and ruff finds no undefined or unused names.
