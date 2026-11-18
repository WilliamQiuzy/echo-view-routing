"""Writes research_parity.npz: inputs and outputs of the research code's window and decoder functions.

The release package re-implements these functions without the research repository's dependencies. The tests feed the
stored inputs to the package and require the stored outputs, so the package cannot drift from the code that produced
the published numbers. This script is the record of how the file was made; it runs only inside the research
repository (branch robust-upgrade, commit 2ea5e9f5), which is not part of this release:

    cd <research repository> && python <this file> <output .npz>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from echo_routing.evaluate.window_target import nearest_centre, window_frames
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores, detect_boundaries, labels_from_boundaries
from echo_routing.temporal.windowed import window_centres
from echo_routing.vit.target_mvit import marker_slot

LENGTHS = [*range(1, 121), 150, 199, 200, 201, 257, 600, 1503]
out = {"window_lengths": np.array(LENGTHS)}
for n in LENGTHS:
    centres = window_centres(n, 5)
    frames = np.stack([window_frames(int(c), n, 16, 16) for c in centres])
    slots = np.array([marker_slot(s, int(c)) for s, c in zip(frames, centres)])
    out[f"w{n}_centres"], out[f"w{n}_frames"], out[f"w{n}_slots"] = centres, frames, slots
    out[f"w{n}_nearest"] = nearest_centre(n, centres)

rng = np.random.default_rng(20261117)
cases = []
for case in range(60):
    n = int(rng.integers(1, 400))
    pieces = int(rng.integers(1, 7))
    cuts = np.sort(rng.integers(0, n + 1, pieces - 1))
    views = rng.integers(0, 5, pieces)
    truth = np.zeros(n, int)
    for a, b, v in zip([0, *cuts], [*cuts, n], views):
        truth[a:b] = v
    logits = rng.normal(0.0, 1.0, (n, 5)) * float(rng.choice([0.3, 1.0, 2.5]))
    logits[np.arange(n), truth] += float(rng.choice([1.0, 3.0, 6.0]))
    prob = np.exp(logits - logits.max(-1, keepdims=True))
    prob /= prob.sum(-1, keepdims=True)
    half = int(rng.choice([3, 5, 7, 9, 11, 15]))
    threshold = float(rng.choice([0.02, 0.1, 0.1737918038235492, 0.35]))
    labels = labels_from_boundaries(prob, detect_boundaries(prob, half, threshold, 5))
    out[f"d{case}_prob"], out[f"d{case}_labels"] = prob, labels
    out[f"d{case}_scores"] = boundary_scores(prob, half)
    cases.append((half, threshold, 5))
out["decoder_settings"] = np.array(cases, dtype=float)
np.savez_compressed(sys.argv[1], **out)
print(f"{len(LENGTHS)} window lengths, {len(cases)} decoder cases -> {sys.argv[1]}")
