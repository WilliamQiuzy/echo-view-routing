# Public code release — 2026-11-07

The owner requested publication of the latest committed mainline and explicitly chose to defer the still-uncommitted full-loader migration. Source snapshot: `29858620d501adff95cb481c697311db73d73fc1` (`helena-default-code`), which includes the committed EX7/Helena input line and EX1–EX3 integration. The public parent is `fa38246e9a73275f523fce9c5a7d2917bdf07f60`.

This is a new public commit based on public history, not a merge or push of private development history. Source working directories, their staged and unstaged changes, and server jobs are not modified. The uncommitted `echo_routing/input/` migration is excluded from this release.

The update carries committed library, experiment-script, configuration and test files, the EX5–EX7 result summaries and their pre-result plans. Public-only training-log export functionality and existing website files are preserved. The public experiment index receives the committed Helena decision/EX5–EX7 update without replacing its historical body.

Packaging explicitly includes `helena_input`, `exam_level` and `open_routing`, so these imported project modules are included in an installed wheel as well as a source checkout. The additions to `.gitignore` keep H frame arrays and cache roots out of commits.

No datasets, model/optimizer weights, `.env` credentials, provider repositories, private per-case outputs or remote-operating helpers are part of this publication. Existing scientific results retain their dated input definitions and limitations; publication is not new training, model parity evidence or clinical validation.

## Publication verification and compatibility

The source snapshot also needs its committed public protocol/approval documents and aggregate profiles for the historical risk-code tests; these are shipped byte-for-byte, without model or per-case cache payloads. The immutable `readonly_context` documents retain their original relative references as historical snapshots, not a new standalone directory layout.

Synthetic child-process tests use a test-only future window. Production budget/deadline constants and every historical experiment gate remain unchanged. The public mirror's existing skip for a historical code object that is absent after history rewriting is retained; private development history is not imported merely to satisfy that test. The complete CPU suite, wheel import check and static publication scan are recorded in the local release receipts.

The historical A63 graph test is brought up to date with the already-committed read-only FIT adoption and FIT score stages, while retaining exact stage membership and the coverage/calibration dependency barriers. This changes the stale test expectation, not the production graph or accepted scientific matrix.
