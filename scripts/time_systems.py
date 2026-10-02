#!/usr/bin/env python
"""Same-hardware timing of E1 systems (round 2, Task 6).

Every system processes one sequence at a time (batch = 1) after warm-up, with CUDA synchronisation around each timed
stage, on the same list of selection-half sequences grouped by length. Reported per sample:
- head: the temporal head's forward and its default decoding on precomputed features;
- system: JPEG decoding and preprocessing on the CPU (one process, no workers), the frozen E1 image encoder (fp32, all
  frames of the sequence in one batch), the head and its decoding; for B3 the decoder replaces the head.
Batched throughput (all sequences of a length group at once through the head) is listed separately.
The official windowed EchoViewCLIP is timed by scripts/time_echoviewclip_windowed.py on the same sequence list.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402
from echo_routing.temporal.dev_data import Seq, collate  # noqa: E402
from echo_routing.temporal.dev_heads import head_from_manifest  # noqa: E402
from echo_routing.temporal.dev_run import Inputs  # noqa: E402
from echo_routing.temporal.joint_visual import VisualEncoder, load_frame_u8, load_vision_transformer  # noqa: E402
from echo_routing.temporal.windowed import SampleRef  # noqa: E402

GROUPS = ((1, 60), (61, 120), (121, 400))


def sync() -> None:
    torch.cuda.synchronize()


def sequences(plans: dict, ids: list[str]) -> list[tuple[str, list[SampleRef]]]:
    return [(i, [SampleRef(v, int(f), float(g)) for v, f, g in plans["streams"][i]["refs"]]) for i in ids]


def pick(plans: dict, per_group: int, seed: int = 0) -> dict[str, list[str]]:
    """The same selection-half sequences for every system: up to `per_group` per length group."""
    rng = np.random.default_rng(seed); out = {}
    ids = sorted(k for k, v in plans["streams"].items() if v["bank"].endswith("_selection"))
    for lo, hi in GROUPS:
        cand = [i for i in ids if lo <= plans["streams"][i]["n"] <= hi]
        out[f"{lo}-{hi}"] = sorted(rng.choice(cand, size=min(per_group, len(cand)), replace=False).tolist())
    return out


def timed(fn, repeats: int = 1):
    sync(); t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    sync()
    return out, (time.perf_counter() - t0) / repeats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plans", required=True); ap.add_argument("--feature-manifest", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--visual-export", required=True); ap.add_argument("--runs", required=True, help="name=run_dir;... (E1 heads)")
    ap.add_argument("--per-group", type=int, default=20); ap.add_argument("--out", required=True)
    ap.add_argument("--b3-selected", required=True, help="selected.json of the E1 simple-decoder run (B3 setting to time)")
    ap.add_argument("-c", "--config", default="configs/experiments/model_development_v2.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); device = torch.device("cuda")
    # full fp32: with TF32 the encoder departs from the E1 features by up to 4e-3; without it the match is about 3e-6
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    plans = json.loads(Path(a.plans).read_text()); groups = pick(plans, a.per_group)
    inputs = Inputs(cfg, Path(a.protocol), Path(a.feature_manifest), paths.cache_root)
    man = load_manifest(paths.manifest_path).set_index("video_id"); fp_cache = {}

    def frame_paths(v):
        if v not in fp_cache:
            fp_cache[v] = list_frame_paths(Path(man.loc[v, "frames_dir"]))
        return fp_cache[v]

    vt, _ = load_vision_transformer(paths.repo_root / "third_party" / "EchoViewCLIP", Path(a.visual_export))
    enc = VisualEncoder(vt, trainable_blocks=1, chunk=512).to(device).eval()
    probe = torch.load(inputs.fman["encoder"]["checkpoint"], map_location="cpu", weights_only=False)
    probe_lin = torch.nn.Linear(probe["in_dim"], 5); probe_lin.load_state_dict(probe["state_dict"]); probe_lin = probe_lin.to(device).eval()
    heads = {}
    for item in a.runs.split(";"):
        name, _, rd = item.partition("="); m = json.loads((Path(rd) / "run_manifest.json").read_text())
        h = head_from_manifest(m, paths.repo_root, inputs.feature_dim, device); model = h.build().to(device)
        model.load_state_dict(torch.load(m["checkpoints"]["best"], map_location=device)); model.eval(); heads[name] = (h, model)
    b3 = json.loads(Path(a.b3_selected).read_text())["selected"]["b3"]["main"]
    results = {"groups": {}, "note": "batch = 1 sequence; fp32; CPU preprocessing in one process; warm-up excluded"}
    with torch.no_grad():
        warm = sequences(plans, groups[next(iter(groups))][:2])
        for _, refs in warm:   # warm-up of every path
            u8 = torch.from_numpy(np.stack([load_frame_u8(frame_paths(r.video_id)[min(r.frame_idx, len(frame_paths(r.video_id)) - 1)], r.gamma) for r in refs]))
            f = enc(u8.to(device)).float()
            for h, model in heads.values():
                h.decode(model, [Seq("w", False, f.cpu().numpy(), np.zeros(len(refs), int), np.zeros(0, int))], device)
        for g, ids in groups.items():
            seqs = sequences(plans, ids); n_samples = sum(len(r) for _, r in seqs); t = {"preprocess": 0.0, "encoder": 0.0}
            feats = []
            for _, refs in seqs:
                t0 = time.perf_counter()
                u8 = torch.from_numpy(np.stack([load_frame_u8(frame_paths(r.video_id)[min(r.frame_idx, len(frame_paths(r.video_id)) - 1)], r.gamma)
                                                for r in refs]))
                t["preprocess"] += time.perf_counter() - t0
                f, dt = timed(lambda: enc(u8.to(device)).float()); t["encoder"] += dt; feats.append(f)
            row = {"n_sequences": len(seqs), "n_samples": n_samples,
                   "preprocess_ms_per_sample": 1000 * t["preprocess"] / n_samples, "encoder_ms_per_sample": 1000 * t["encoder"] / n_samples}
            for name, (h, model) in heads.items():
                th = 0.0
                for f, (sid, refs) in zip(feats, seqs):
                    s = Seq(sid, False, f.cpu().numpy(), np.zeros(len(refs), int), np.zeros(0, int))
                    _, dt = timed(lambda: h.decode(model, [s], device)); th += dt
                row[f"{name}_head_ms_per_sample"] = 1000 * th / n_samples
                row[f"{name}_system_ms_per_sample"] = row["preprocess_ms_per_sample"] + row["encoder_ms_per_sample"] + row[f"{name}_head_ms_per_sample"]
                ss = [Seq(sid, False, f.cpu().numpy(), np.zeros(len(r), int), np.zeros(0, int)) for f, (sid, r) in zip(feats, seqs)]
                _, dt = timed(lambda: h.decode(model, ss, device)); row[f"{name}_head_batched_ms_per_sample"] = 1000 * dt / n_samples
            tp = 0.0   # B3 on the probe's probabilities (CPU decoder)
            for f in feats:
                prob = torch.softmax(probe_lin(f), 1).cpu().numpy()
                t0 = time.perf_counter()
                decode_with_prob("b3", prob, {"decode": {"b3": b3}})
                tp += time.perf_counter() - t0
            row["B3_decoder_ms_per_sample"] = 1000 * tp / n_samples
            row["B3_system_ms_per_sample"] = row["preprocess_ms_per_sample"] + row["encoder_ms_per_sample"] + row["B3_decoder_ms_per_sample"]
            results["groups"][g] = row; print(g, json.dumps(row), flush=True)
    results["sequences"] = groups; results["device"] = torch.cuda.get_device_name(0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
