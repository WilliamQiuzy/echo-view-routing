#!/usr/bin/env python
"""Round 3, step 3: do neighbouring fragments change M5's classification of a fragment? (diagnostic only)

For every fragment that the E1 probe labels right (fragment-mean argmax) but an E1 M5 run gets wrong on at least 80%
of its samples, the fragment's input embeddings are kept fixed and only its surroundings change:
- original: the whole original stream (normal inference);
- isolated: the fragment on its own;
- same-class neighbours: the previous and next fragments replaced by fragments of the target's class from other clips;
- other-clip neighbours: the previous and next fragments replaced by fragments of the same classes as the originals,
  taken from other clips.
For each variant the target's samples are summarised at four stages of M5: the local (convolution) classifier, the
context (attention) classifier, the gated mix before pooling and the pooled output. The reference fragment limits are
used only to build these inputs; normal inference never sees them. Counts are also given once per source recording.
Writes neighbor_check.json and neighbor_check.md into a new run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.temporal.dev_data import stream_seq  # noqa: E402
from echo_routing.temporal.dev_heads import head_from_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402

CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
STAGES = (("local", "local_logits"), ("context", "context_logits"), ("pre_pool", "mixed_logits"), ("output", "class_logits"))


def stage_summary(model, feat: np.ndarray, a: int, b: int, device) -> dict:
    """Forward one sequence; mean probability over samples a..b-1 at each stage, and its argmax."""
    x = torch.from_numpy(feat.T.astype(np.float32))[None].to(device); mask = torch.ones(1, 1, feat.shape[0], device=device)
    with torch.no_grad():
        out = model(x, mask)
    res = {}
    for name, key in STAGES:
        p = torch.softmax(out[key][0, :, a:b], dim=0).mean(1).cpu().numpy()
        res[name] = {"argmax": int(p.argmax()), "prob": [round(float(v), 4) for v in p]}
    bp = torch.sigmoid(out["boundary_logits"][0, 0]).cpu().numpy()
    res["boundary_at_edges"] = [round(float(bp[a]), 4) if a > 0 else None, round(float(bp[b]), 4) if b < len(bp) else None]
    return res


def donors(inputs: Inputs, half: str) -> dict[int, list[tuple[str, np.ndarray]]]:
    out: dict[int, list] = {}
    for s in inputs.halves[half].natives:
        out.setdefault(int(s.labels[0]), []).append((s.seq_id, inputs.store.get(s.seq_id, "orig").feat))
    return out


def take(pool: list, length: int, exclude: set[str], rng: np.random.Generator) -> tuple[str, np.ndarray] | None:
    """A block of `length` samples from a clip of the pool not in `exclude` (start drawn uniformly)."""
    cand = [(v, f) for v, f in pool if v not in exclude and f.shape[0] >= length]
    if not cand:
        return None
    v, f = cand[int(rng.integers(len(cand)))]; st = int(rng.integers(0, f.shape[0] - length + 1))
    return v, f[st:st + length]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="M5_s0=dir;M5_s1=dir;M5_s2=dir (frozen E1 M5 runs)")
    ap.add_argument("--feature-manifest", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--case-b", default="M5_s0|selection|multi_selection_0004|126|136")
    ap.add_argument("-c", "--config", default="configs/experiments/model_development_v3.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = Inputs(cfg, Path(a.protocol), Path(a.feature_manifest), paths.cache_root)
    pools = {h: donors(inputs, h) for h in HALVES}; rows = []; case_b = None
    for item in a.runs.split(";"):
        name, _, rd = item.partition("="); man = json.loads((Path(rd.strip()) / "run_manifest.json").read_text())
        head = head_from_manifest(man, paths.repo_root, inputs.feature_dim, device)
        model = head.build().to(device); model.load_state_dict(torch.load(man["checkpoints"]["best"], map_location=device)); model.eval()
        for h in HALVES:
            dec = head.decode(model, [stream_seq(s) for s in inputs.halves[h].multi], device)
            for s in inputs.halves[h].multi:
                sid = s.recipe.recipe_id; lab = dec[sid]["labels"]; pos = 0; spans = []
                for fr in s.recipe.fragments:
                    spans.append((pos, pos + fr.end - fr.start, fr)); pos += fr.end - fr.start
                for i, (st, en, fr) in enumerate(spans):
                    if int(s.prob[st:en].mean(0).argmax()) != fr.label or (lab[st:en] != fr.label).mean() < 0.8:
                        continue
                    rng = np.random.default_rng(int(hashlib.sha256(f"{name}|{sid}|{st}".encode()).hexdigest()[:8], 16))   # stable donor choice
                    prev = spans[i - 1] if i > 0 else None; nxt = spans[i + 1] if i + 1 < len(spans) else None
                    target = s.feat[st:en]; used = {fr.video_id} | {x[2].video_id for x in (prev, nxt) if x}
                    variants = {"original": (s.feat, st, en), "isolated": (target, 0, en - st)}
                    for tag, cls_of in (("same_class_neighbours", lambda x: fr.label), ("other_clip_neighbours", lambda x: x[2].label)):
                        blocks = []
                        for x in (prev, nxt):
                            if x is None:
                                blocks.append(None); continue
                            got = take(pools[h].get(cls_of(x), []), x[1] - x[0], used, rng)
                            blocks.append(got[1] if got else None)
                        if all(b is not None or x is None for b, x in zip(blocks, (prev, nxt))):
                            parts = [b for b in (blocks[0],) if b is not None] + [target] + [b for b in (blocks[1],) if b is not None]
                            off = 0 if blocks[0] is None else blocks[0].shape[0]
                            variants[tag] = (np.concatenate(parts), off, off + en - st)
                    res = {k: stage_summary(model, f, aa, bb, device) for k, (f, aa, bb) in variants.items()}
                    row = {"run": name, "half": h, "stream": sid, "start": st, "end": en, "label": int(fr.label), "video_id": fr.video_id,
                           "neighbour_labels": [prev[2].label if prev else None, nxt[2].label if nxt else None], "variants": res}
                    rows.append(row)
                    if f"{name}|{h}|{sid}|{st}|{en}" == a.case_b:
                        case_b = row
    run_dir = new_run_dir(cfg, "md3_neighbor_check")
    summary = {}
    for tag in ("original", "isolated", "same_class_neighbours", "other_clip_neighbours"):
        have = [r for r in rows if tag in r["variants"]]
        summary[tag] = {"n": len(have)}
        for stage, _ in STAGES:
            right = [r for r in have if r["variants"][tag][stage]["argmax"] == r["label"]]
            summary[tag][stage] = {"right": len(right), "of": len(have), "source_clips_right": len({r["video_id"] for r in right}),
                                   "source_clips": len({r["video_id"] for r in have})}
    (run_dir / "neighbor_check.json").write_text(json.dumps({"summary": summary, "case_b": case_b, "rows": rows}, indent=1))
    md = ["# Neighbour-swap check (round 3, step 3; diagnostic only)", "",
          "Fragments the E1 probe labels right and an E1 M5 run gets wrong on at least 80% of their samples. The fragment's",
          "embeddings are fixed; only the surroundings change. A cell counts fragments whose mean probability at that stage",
          "picks the reference class; source clips are counted once.", "",
          "| Surroundings | fragments | local (convolutions) | context (attention) | before pooling | output |", "|---|---:|---|---|---|---|"]
    for tag, s in summary.items():
        cells = [f"{s[st]['right']}/{s[st]['of']} ({s[st]['source_clips_right']}/{s[st]['source_clips']} clips)" for st, _ in STAGES]
        md.append(f"| {tag.replace('_', ' ')} | {s['n']} | " + " | ".join(cells) + " |")
    if case_b:
        md += ["", f"## Case B: {case_b['stream']} {case_b['start']}-{case_b['end']} ({CLASSES[case_b['label']]}), {case_b['run']}", "",
               "| Surroundings | local | context | before pooling | output | boundary at the two edges |", "|---|---|---|---|---|---|"]
        for tag, v in case_b["variants"].items():
            cells = [f"{CLASSES[v[st]['argmax']]} (A4C {v[st]['prob'][2]:.2f}, A5C {v[st]['prob'][3]:.2f})" for st, _ in STAGES]
            md.append(f"| {tag.replace('_', ' ')} | " + " | ".join(cells) + f" | {v['boundary_at_edges']} |")
    (run_dir / "neighbor_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
