#!/usr/bin/env python3
"""EX1 post-hoc, added after reading E1: how few exam streams hold the segment errors. CPU, aggregates only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from exam_level import io, metrics as mx
from exam_level.decode import decode_b3
out = []
for half in io.HALVES:
    seqs = np.load(io.RUN / "streams" / f"exam_{half}.npy", allow_pickle=True).tolist()
    for seed in io.SEEDS:
        params = io.b3_params(half, seed)
        with np.load(io.RUN / "forward" / f"exam_{half}_s{seed}.npz") as z:
            edges = np.cumsum([0, *z["position_counts"]])
            probs = [z["probs"][a:b] for a, b in zip(edges, edges[1:])]
        errors, sizes = [], []
        for q, p in zip(seqs, probs):
            c = mx.stream_counts(q["labels"], decode_b3(p, params))
            errors.append(int(c["T"] + c["P"] - 2 * c["M"]))
            sizes.append(len(q["clips"]))
        e = np.array(errors)
        ranked = np.sort(e)[::-1]
        out.append({"half": half, "seed": seed, "streams": len(e), "streams_with_errors": int((e > 0).sum()), "total_errors": int(e.sum()),
                    "top1": int(ranked[0]), "top3": int(ranked[:3].sum()), "largest_stream_clips": max(sizes), "median_stream_clips": float(np.median(sizes))})
io.write_json(io.RUN / "public/posthoc_concentration.json", out)
for r in out:
    print(r)
