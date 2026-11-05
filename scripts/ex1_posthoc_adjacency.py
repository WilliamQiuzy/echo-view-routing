#!/usr/bin/env python3
"""EX1 post-hoc, added after reading E1: A4C/A5C adjacency in each bank and where A5C frame errors fall. CPU, aggregates only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from exam_level import io, streams as st
from exam_level.decode import decode_b3
A4C, A5C = 2, 3
BANKS = ("exam", "shuffled", "rewired")
def clip_labels(q):
    return [int(q["labels"][a]) for _, a, _ in q["clips"]]
out = []
for half in io.HALVES:
    for bank in BANKS:
        seqs = np.load(io.RUN / "streams" / f"{bank}_{half}.npy", allow_pickle=True).tolist()
        pair = fives = fives_adj = 0
        for q in seqs:
            y = clip_labels(q)
            pair += sum({a, b} == {A4C, A5C} for a, b in zip(y, y[1:]))
            for i, v in enumerate(y):
                if v == A5C:
                    fives += 1
                    fives_adj += any(0 <= j < len(y) and y[j] == A4C for j in (i - 1, i + 1))
        row = {"half": half, "bank": bank, "a4c_a5c_joins": pair, "a5c_clips": fives, "a5c_clips_next_to_a4c": fives_adj, "seeds": {}}
        for seed in io.SEEDS:
            params = io.b3_params(half, seed)
            with np.load(io.RUN / "forward" / f"{bank}_{half}_s{seed}.npz") as z:
                edges = np.cumsum([0, *z["position_counts"]])
                probs = [z["probs"][a:b] for a, b in zip(edges, edges[1:])]
            adj = {"frames": 0, "to_a4c": 0, "clips_mostly_a4c": 0}
            lone = {"frames": 0, "to_a4c": 0, "clips_mostly_a4c": 0}
            for q, p in zip(seqs, probs):
                pred, y = decode_b3(p, params), clip_labels(q)
                for i, (_, a, b) in enumerate(q["clips"]):
                    if y[i] != A5C:
                        continue
                    g = adj if any(0 <= j < len(y) and y[j] == A4C for j in (i - 1, i + 1)) else lone
                    wrong = int((pred[a:b] == A4C).sum())
                    g["frames"] += b - a
                    g["to_a4c"] += wrong
                    g["clips_mostly_a4c"] += wrong * 2 > b - a
            row["seeds"][f"s{seed}"] = {"next_to_a4c": adj, "not_next_to_a4c": lone}
        out.append(row)
io.write_json(io.RUN / "public/posthoc_adjacency.json", out)
for r in out:
    print(r["half"][:3], r["bank"], "joins", r["a4c_a5c_joins"], "a5c", r["a5c_clips"], "adj", r["a5c_clips_next_to_a4c"], {k: (v["next_to_a4c"]["to_a4c"], v["next_to_a4c"]["frames"], v["next_to_a4c"]["clips_mostly_a4c"], v["not_next_to_a4c"]["to_a4c"], v["not_next_to_a4c"]["frames"], v["not_next_to_a4c"]["clips_mostly_a4c"]) for k, v in r["seeds"].items()})
