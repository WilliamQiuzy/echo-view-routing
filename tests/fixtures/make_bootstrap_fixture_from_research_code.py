"""Writes bootstrap_parity.json: made-up counts per unit and the intervals the research code's bootstrap gives them.

Run inside the research repository (branch robust-upgrade) with its environment:

    ECHO_INPUT=centre_crop_legacy python make_bootstrap_fixture_from_research_code.py <research root> <output folder>

Every count is a random number. No recording, label or prediction of any dataset is read. The tables have the columns
the research bootstrap (scripts/benchmark_evaluate.py) sums; the fixture keeps the ones this package uses: the
recording-level confusion counts by examination and the segment counts by stream.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "scripts"))
import benchmark_evaluate as research  # noqa: E402

rng = np.random.default_rng(20261120)
EXAMINATIONS, NATIVES, MULTI, PAIRS = 23, 160, 31, 47

names = sorted({f"exam{int(rng.integers(0, 10 ** 6)):06d}" for _ in range(4 * EXAMINATIONS)})
examinations = [names[i] for i in rng.permutation(len(names))[:EXAMINATIONS]]
natives = []
for k in range(NATIVES):
    exam = examinations[k] if k < EXAMINATIONS else examinations[int(rng.integers(0, EXAMINATIONS))]  # every examination at least once
    label = int(rng.integers(0, 4))  # view 4 never occurs as a reference: one class has no recording
    majority = label if rng.random() > 0.2 else int(rng.integers(0, 5))
    natives.append({"video_id": f"{exam}_{k}", "group": exam, "label": label, "n": int(rng.integers(20, 90)), "majority": majority,
                    "cuts": int(rng.integers(0, 2)), "seg_matched": int(majority == label), "seg_pred": 1, "seg_true": 1})


def streams(prefix: str, count: int) -> tuple[list[dict], list[dict]]:
    segments, fragments = [], []
    for k in rng.permutation(count):  # the rows are not in the order of their names
        true = int(rng.integers(2, 9))
        predicted = max(1, true + int(rng.integers(-2, 3)))
        matched = int(rng.integers(0, min(true, predicted) + 1))
        name = f"{prefix}_{int(k):04d}"
        segments.append({"stream": name, "seg_matched": matched, "seg_pred": predicted, "seg_true": true, "fp": predicted - matched,
                         "fn": true - matched, "fp_short": 0, "fn_short": 0, "true_short": 0})
        for piece in range(true):
            fragments.append({"stream": name, "video_id": f"{examinations[int(rng.integers(0, EXAMINATIONS))]}_{piece}", "t_true": 1,
                              "t_matched": int(rng.integers(0, 2)), "p_pred": int(rng.integers(0, 3)), "same_join": int(rng.integers(0, 2)),
                              "false_split": 0})
    return segments, fragments


multi_segments, multi_fragments = streams("multi_test", MULTI)
pair_segments, pair_fragments = streams("pairs_test_x", PAIRS)
evaluation = {"natives": pd.DataFrame(natives),
              "banks": {"multi": {"segments": pd.DataFrame(multi_segments), "fragments": pd.DataFrame(multi_fragments)},
                        "pairs": {"segments": pd.DataFrame(pair_segments), "fragments": pd.DataFrame(pair_fragments)}}}
result = research.bootstrap({"system": evaluation}, "system")["system"]

rank = {name: k for k, name in enumerate(sorted(examinations))}
KEPT = ("clip macro-F1", "clip accuracy", "segment F1@50, multi", "segment F1@50, pairs")
fixture = {
    "note": "made-up counts; the expected values are the output of the research bootstrap (2,000 draws, seed 20261001)",
    "numpy": np.__version__,
    "natives": [[rank[row["group"]], row["label"], row["majority"]] for row in natives],  # examination number, reference, decided
    "multi": [[row["stream"], row["seg_matched"], row["seg_pred"], row["seg_true"]] for row in multi_segments],
    "pairs": [[row["stream"], row["seg_matched"], row["seg_pred"], row["seg_true"]] for row in pair_segments],
    "expected": {name: {"value": result[name]["value"], "ci95": result[name]["ci95"]} for name in KEPT},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "bootstrap_parity.json").write_text(json.dumps(fixture, indent=1) + "\n")
print(json.dumps(fixture["expected"]))
