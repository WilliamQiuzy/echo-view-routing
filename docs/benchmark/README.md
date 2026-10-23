# Frozen EV9V test benchmark: how to run it and how to add a model

The benchmark scores frozen systems on the historical EV9V test, with H09's own definitions. It runs on the server.
Benchmark id `ev9v-test-ladder-v1`, metrics version `ev9v-benchmark-metrics-v1`, first run A26 (2026-10-23).

**What the benchmark is.**
- **Items.**
  - The 800 five-family native test clips. The 888 official test clips minus the 88 PMPALA clips.
  - H09's 240 two-fragment and 120 four-to-eight-fragment test streams (ladder recipe seed 1234).
- **Axis.** Each item's own 10 Hz sample grid, with numeric frame order (cache_v2).
- **Primary endpoint.** Multi-stream class-aware segment F1@50, pooled over streams.
- **Not a fresh holdout.** H01–H09 evaluated many systems on this test. A new score here is a same-benchmark comparison,
  not an unseen-data confirmation.
- **Repeated use is recorded.** Every run is recorded under its run id, so the history of benchmark feedback stays
  traceable. Never tune, select or train on its output.

**Files.**

| File | Role |
|---|---|
| `configs/experiments/frozen_external_benchmark_v1.yaml` | Fixed benchmark definition, metrics, budget and outputs |
| `configs/benchmarks/ev9v_test_registry_v1.yaml` | Model registry: one entry per frozen system |
| `scripts/benchmark_manifest.py` | Builds the private manifest once and checks it against H09's banks and the frame references |
| `scripts/benchmark_adapters.py` | Adapters: `cached_probs`, `a08_b3`, `a14_router` |
| `scripts/benchmark_evaluate.py` | Evaluator: H09's `evaluate_spec` and bootstrap, plus the added endpoints |
| `scripts/run_benchmark.py` | Command: `dev-check`, `predict`, `evaluate`, `report` |
| `scripts/benchmark_provenance.py` | Training-data overlap audit (counts and evidence paths) |

**Commands** (server, repository root, `.venv/bin/python`).

```bash
C=configs/experiments/frozen_external_benchmark_v1.yaml
python scripts/benchmark_manifest.py --config $C            # once; refuses to rebuild
python scripts/run_benchmark.py --config $C --stage dev-check  # no test read
python scripts/run_benchmark.py --config $C --stage predict  --run-id v1 [--systems A ...]
python scripts/run_benchmark.py --config $C --stage evaluate --run-id v1 [--systems A ...]
python scripts/run_benchmark.py --config $C --stage report   --run-id v1
```

- **Outputs.** `runs/frozen_external_benchmark_v1/runs/<run id>/` (predictions, `predictions.json`, `evaluation.json`)
  and `docs/results/2026-10-23_frozen_external_benchmark/` (`evaluation_<run id>.json`, `values_<run id>.csv`,
  `comparison_<run id>_zh.md`).
- **A finished run id is never overwritten.**

**Adding a new frozen model.**
1. **Add a registry entry** under a new name with these fields:
   - `adapter`: an existing one, or a new function in `scripts/benchmark_adapters.py` that returns `{item id: int labels
     of length n}`.
   - `weights`: every file the system reads, each with its sha256.
   - `decoder`: the decoder and its saved numeric setting. A setting defined by a quantile keeps its saved number. It is
     never re-chosen on this test.
   - `seed`, and `provenance`: the training data, the pretraining and any earlier test use.
   - `cache`: only for reused historical outputs.
2. **Write the adapter.** It receives items with `refs` (recording, frame, gamma) and a placeholder length only. It must
   not read labels, joins, sources or file identity as inputs.
3. **Check on development data first.** Reproduce the model's stored development rows through the adapter, as
   `dev-check` does for A08 and A14.
4. **Run predict and evaluate under a new run id** with `--systems` listing the new entry and the comparators.
5. **Extend the provenance audit** with the model's actual training list. Report external pretraining without a list
   as unknown.

**Caching.**
- Online A08 outputs are stored per chunk of items, under a key of benchmark digest, weights, adapter version, code
  and precision. A store whose key or windows differ is refused.
- Historical caches are used only after their H09 rows reproduce exactly through the evaluator.
- Cache reads are evaluation cost, not model speed.
