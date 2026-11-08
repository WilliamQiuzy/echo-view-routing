# Research history and experiment decisions

For echocardiographic view recognition and temporal segmentation, use the repository records rather than chat memory as the source of truth.

Before proposing, changing or starting a modeling experiment:

1. Read `docs/EXPERIMENT_INDEX.md` and the linked predecessor reports. Consult `docs/EXPERIMENT_LEDGER.md` for historical runs; its dated table stops at 2026-10-01.
2. Identify the previous question, representation/checkpoint, supervision, data/protocol, controls, metrics, result and stopping decision. Distinguish an invalidated run, diagnostic, untrained hypothesis and completed comparison.
3. In the new task, cite the predecessor record IDs and explain the substantive changed condition. Do not repeat a closed configuration under a new name or claim a whole model family was disproved by one negative configuration.
4. Respect the active task's fixed scope, budget and stopping rules. The owner authorizes routine implementation, repair, checks, training, evaluation and round-only commits within the specified task without waiting for their return. A history check is not a new approval gate for already-authorized work.

After each completed round, update its entry in `docs/EXPERIMENT_INDEX.md` with the scoped result, constraint/goal outcome, source report, configuration/frozen-model identities, result commits and closed scope. Keep negative and uncertain results. A pending entry becomes a result only after verified execution and the completed report.

Corrections must remain traceable. Use current corrected reports for numbers; preserve the old ledger and superseded records. No consistent difference is not equivalence. Source sessions are not confirmed patients. Constructed streams and reference-assisted diagnoses are not clinical continuous-scan evidence.

Keep the current test and Pediatric restrictions, public labels and source-model provenance as specified by the task. Recording/session identifiers, raw media and private per-case outputs stay on the server. Preserve unrelated working-tree and staged changes; commit only the authorized files and do not push unless explicitly requested.

Do not rewrite an active experiment's configuration or results merely to reconcile documentation. Updating history must not change its model, data, metrics, thresholds, budget or run identity.

## Input preprocessing (from 2026-11-06)

On 2026-11-06 the project owner decided that every new experiment uses Helena's preprocessing. This is the input of EX3 arm H: the rendered `ev9v_processed_oct16` frames in `~/hli-echo-segmentation` on the server, which stay read-only. `helena_input.pixels.HelenaPixels` reads them from the frame cache that `scripts/ex3_h_cache.py` rebuilds. The code is on branch `ex3-helena-input` and public `main` fa38246e. The reference window classifier on this input is EX3 H, frozen as `models_frozen/ex3_helena_input_v1/H_s0`, `H_s1` and `H_s2`.

Earlier results on A08's centre crop stay valid for that input and are not rewritten. A new experiment that still uses A08's crop must say why. Valuable earlier results are rerun on the new input under EX5. Closed A-rounds are not rerun by default.

Helena's geometry falls back on about 23% of EV9V recordings (1,204 of 5,131), so screen text can remain in those frames. External datasets have no provider output. EX3's copy of her code falls back on 95.6% of CAMUS recordings.
