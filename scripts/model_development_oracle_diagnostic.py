#!/usr/bin/env python
"""Limited error attribution on constructed streams (round 2, Task 2). ORACLE_DIAGNOSTIC: not a deployable result.

For B0 (the encoder's own per-sample probabilities) and finished learned runs (ASRF, M3p, M5, any seed), on both
validation halves:
- native-clip classification (majority vote of the default decoding) and constructed-stream per-sample error, kept
  apart, because a per-sample advantage on streams does not imply better native-clip classification;
- ORACLE_DIAGNOSTIC: every multi-fragment stream is cut at its reference class changes (adjacent fragments of the same
  class form one segment; a same-class join is not a boundary), the method's class probabilities are averaged inside
  each reference segment and the segment takes the argmax; adjacent segments with the same output then merge. It asks
  how much this classifier still gets wrong once the segmentation is given. It is neither deployable nor a bound, the
  labels are proxy labels from the source clips, and the true boundaries never enter the normal inference path;
- the default predictions' error runs (split at every join, so each run lies in one fragment), typed as timing
  (within 5 samples of a true change), whole-fragment (at least 80% of the fragment) or interior runs;
- per source recording: each wrong reference segment's source clips are counted once, with how often they occur.
M3p and M5 pool their output over predicted segments, so oracle rows on that output still contain the model's mixing;
they are also given the diagnostic pre-pooling probabilities, which never replace the default output.
Writes oracle.json and oracle.md into a new run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.stream_assembler import Stream  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.temporal.dev_heads import head_from_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402

LABEL = "ORACLE_DIAGNOSTIC"
NUM_CLASSES = 5
TIMING_SAMPLES = 5


def reference_segments(stream: Stream) -> list[dict]:
    """Runs of fragments with the same class: (start, end, label, source video ids)."""
    segs, pos = [], 0
    for fr in stream.recipe.fragments:
        n = fr.end - fr.start
        if segs and segs[-1]["label"] == fr.label:
            segs[-1]["end"] += n; segs[-1]["videos"].append(fr.video_id)
        else:
            segs.append({"start": pos, "end": pos + n, "label": int(fr.label), "videos": [fr.video_id]})
        pos += n
    return segs


def oracle_labels(prob: np.ndarray, segs: list[dict]) -> tuple[np.ndarray, list[int]]:
    """Per-sample labels when each reference segment takes the argmax of its mean probability; per-segment outputs."""
    out = np.empty(prob.shape[0], int); seg_pred = []
    for s in segs:
        k = int(np.asarray(prob[s["start"]:s["end"]], float).mean(0).argmax()); seg_pred.append(k)
        out[s["start"]:s["end"]] = k
    return out, seg_pred


def oracle_summary(probs: dict[str, np.ndarray], streams: list[Stream]) -> dict:
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), int); wrong_samples = n_samples = 0
    wrong_videos: dict[str, int] = {}; merged_segments = 0
    for s in streams:
        segs = reference_segments(s); lab, seg_pred = oracle_labels(probs[s.recipe.recipe_id], segs)
        merged_segments += 1 + int(np.count_nonzero(np.diff(lab)))
        for seg, k in zip(segs, seg_pred):
            conf[seg["label"], k] += 1
            if k != seg["label"]:
                wrong_samples += seg["end"] - seg["start"]
                for v in seg["videos"]:
                    wrong_videos[v] = wrong_videos.get(v, 0) + 1
        n_samples += lab.size
    n_seg = int(conf.sum()); n_wrong = int(n_seg - np.trace(conf))
    return {"label": LABEL, "reference_segments": n_seg, "wrong_segments": n_wrong, "segment_error_rate": n_wrong / n_seg,
            "sample_error_rate": wrong_samples / n_samples, "output_segments_after_merge": merged_segments,
            "confusion": conf.tolist(), "wrong_source_videos": len(wrong_videos),
            "wrong_source_occurrences": int(sum(wrong_videos.values())), "wrong_videos": dict(sorted(wrong_videos.items()))}


def error_runs(labels: dict[str, np.ndarray], streams: list[Stream]) -> dict:
    """Default predictions' wrong runs on streams by type, and the source clips of whole-fragment errors."""
    cats = dict.fromkeys(("timing", "whole_fragment", "interior_short", "interior_long"), 0); samples = dict(cats)
    whole_videos: dict[str, int] = {}
    for s in streams:
        lab = np.asarray(labels[s.recipe.recipe_id]); wrong = lab != s.labels
        bounds = np.concatenate([[0], s.joins, [lab.size]]); sem = s.semantic_boundaries
        runs = []   # wrong runs split at every join, so each run lies inside one fragment
        for fi, (f0, f1) in enumerate(zip(bounds[:-1], bounds[1:])):
            e = np.diff(np.concatenate([[0], wrong[f0:f1].astype(int), [0]]))
            runs += [(f0 + a, f0 + b, fi) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]
        for a, b, fi in runs:
            flen = bounds[fi + 1] - bounds[fi]
            if b - a >= 0.8 * flen:
                c = "whole_fragment"; v = s.recipe.fragments[fi].video_id; whole_videos[v] = whole_videos.get(v, 0) + 1
            elif sem.size and (np.min(np.abs(sem - a)) <= TIMING_SAMPLES or np.min(np.abs(sem - b)) <= TIMING_SAMPLES):
                c = "timing"
            else:
                c = "interior_short" if b - a <= 3 else "interior_long"
            cats[c] += 1; samples[c] += int(b - a)
    return {"runs": cats, "samples": samples, "whole_fragment_source_videos": len(whole_videos),
            "whole_fragment_occurrences": int(sum(whole_videos.values()))}


def native_summary(labels: dict[str, np.ndarray], natives) -> dict:
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), int); wrong = []
    for c in natives:
        y = int(c.labels[0]); p = int(np.bincount(np.asarray(labels[c.seq_id]), minlength=NUM_CLASSES).argmax())
        conf[y, p] += 1
        if p != y:
            wrong.append(c.seq_id)
    tp = np.diag(conf); den = 2 * tp + (conf.sum(0) - tp) + (conf.sum(1) - tp)
    return {"macro_f1": float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean()),
            "recall": {str(k): [int(tp[k]), int(conf[k].sum())] for k in range(NUM_CLASSES)}, "wrong_clips": wrong,
            "confusion": conf.tolist()}


def stream_sample_error(labels: dict[str, np.ndarray], streams: list[Stream]) -> float:
    wrong = sum(int((np.asarray(labels[s.recipe.recipe_id]) != s.labels).sum()) for s in streams)
    return wrong / sum(s.labels.size for s in streams)


def summarise(name: str, ev, probs: dict, labels: dict, prepool: dict | None) -> dict:
    row = {"method": name, "native": native_summary(labels, ev.natives),
           "multi_sample_error_rate": stream_sample_error(labels, ev.multi), "error_runs": error_runs(labels, ev.multi),
           "oracle": oracle_summary(probs, ev.multi)}
    if prepool is not None:
        row["oracle_prepool"] = oracle_summary(prepool, ev.multi)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="", help="row=run_dir;row=run_dir (finished learned runs)")
    ap.add_argument("--feature-manifest", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--tag", default="E0"); ap.add_argument("-c", "--config", default="configs/experiments/boundary_context_v1.yaml")
    args = ap.parse_args(); args.set = []
    cfg = load_cfg(args); paths = load_paths()
    inputs = Inputs(cfg, Path(args.protocol), Path(args.feature_manifest), paths.cache_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {h: [] for h in HALVES}
    for h in HALVES:  # B0: the encoder's own probabilities (v3 head for E0, the linear probe for E1)
        ev = inputs.halves[h]
        probs = {**{s.seq_id: inputs.store.get(s.seq_id, "orig").prob for s in ev.natives}, **{s.recipe.recipe_id: s.prob for s in ev.multi}}
        rows[h].append(summarise(f"B0 ({args.tag})", ev, probs, {k: v.argmax(1) for k, v in probs.items()}, None))
    for item in filter(None, args.runs.split(";")):
        name, _, rd = item.partition("="); rd = Path(rd.strip())
        man = json.loads((rd / "run_manifest.json").read_text())
        head = head_from_manifest(man, paths.repo_root, inputs.feature_dim, device)
        model = head.build().to(device); model.load_state_dict(torch.load(man["checkpoints"]["best"], map_location=device))
        for h in HALVES:
            ev = inputs.halves[h]
            dec = head.decode(model, ev.sequences(), device)
            probs = {k: v["prob"] for k, v in dec.items()}; labels = {k: v["labels"] for k, v in dec.items()}
            pre = {k: v["prepool_prob"] for k, v in dec.items()} if man.get("model_cfg", {}).get("segment_pool") else None
            rows[h].append(summarise(name.strip(), ev, probs, labels, pre))
        print(f"{name.strip()} done", flush=True)
    run_dir = new_run_dir(cfg, f"md_oracle_diagnostic_{args.tag}")
    (run_dir / "oracle.json").write_text(json.dumps({"label": LABEL, "tag": args.tag, "feature_manifest": args.feature_manifest,
                                                     "protocol": args.protocol, "rows": rows}, indent=1))
    md = [f"# {LABEL} ({args.tag}): classification versus segmentation errors on multi-fragment streams", "",
          "Reference segmentation is given only in the oracle columns. Native and stream columns use the default decoding.", ""]
    for h in HALVES:
        md += [f"## {h} half", "", "| Method | native macro-F1 | native clips wrong | A5C clips right / all | stream sample error | "
               "oracle wrong segments | oracle segment error | oracle sample error | oracle (pre-pooling) segment error | "
               "default runs: timing / whole / interior ≤3 / interior >3 | wrong source clips (occurrences) |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for r in rows[h]:
            o = r["oracle"]; er = r["error_runs"]["runs"]; a5 = r["native"]["recall"]["3"]
            pre = "" if "oracle_prepool" not in r else f"{r['oracle_prepool']['segment_error_rate']:.4f}"
            md.append(f"| {r['method']} | {r['native']['macro_f1']:.3f} | {len(r['native']['wrong_clips'])} | {a5[0]}/{a5[1]} | "
                      f"{r['multi_sample_error_rate']:.4f} | {o['wrong_segments']}/{o['reference_segments']} | {o['segment_error_rate']:.4f} | "
                      f"{o['sample_error_rate']:.4f} | {pre} | "
                      f"{er['timing']} / {er['whole_fragment']} / {er['interior_short']} / {er['interior_long']} | "
                      f"{o['wrong_source_videos']} ({o['wrong_source_occurrences']}) |")
        md.append("")
    (run_dir / "oracle.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
