#!/usr/bin/env python
"""Round 5, step 1: can an A4C/A5C decision on a base's spans reach the success criterion at all?
(configs/experiments/evidence_pooling_v1.yaml, sections reachability and sensitivity). Diagnostic only; nothing trains.

For every base (E0 and E1 ASRF seeds 0-2; E0 and E1 B3 tuned for reference) and both validation halves:
- the final labels of the current decoding (ASRF must reproduce its stored predictions exactly) and the candidate
  spans: ASRF's accepted boundary peaks before its class vote merges equal neighbours (the official argrelmax at
  boundary_th 0.5), and B3's detected boundaries;
- the A4C/A5C oracle on the final segments and on the candidate spans, the recall of the candidate cuts themselves
  (one-to-one within 0.5 s), and the success-test metrics: segment F1@50, boundary recall and native macro-F1;
- the same metrics with the ten sensitivity clips removed (the same list for every system, nothing re-tuned), also for
  B7 with B3, the fixed system that is strongest on the calibration half.
Writes reach.json and reach.md into a new run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.evaluate.failure_units import match_changes  # noqa: E402
from echo_routing.temporal.baselines.b3_js_divergence import detect_boundaries, labels_from_boundaries  # noqa: E402
from echo_routing.temporal.dev_data import EvalSet  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402

A4C, A5C, NUM_CLASSES = 2, 3, 5
FLAT = ("segment_f1_50_multi", "boundary_recall_multi", "native_macro_f1")


def without(ev: EvalSet, clips: set[str]) -> EvalSet:
    """The half with the sensitivity clips removed: from the natives, and every stream with a fragment from them."""
    keep = lambda s: not any(f.video_id in clips for f in s.recipe.fragments)  # noqa: E731
    return EvalSet(ev.name, tuple(n for n in ev.natives if n.seq_id not in clips), tuple(s for s in ev.pairs if keep(s)),
                   tuple(s for s in ev.multi if keep(s)))


def inputs_of(ev: EvalSet, store) -> dict[str, tuple]:
    """Sequence id -> (features, frozen classifier probabilities, reference labels) for natives, pairs and streams."""
    out = {n.seq_id: (n.feat, store.get(n.seq_id, "orig").prob, n.labels.astype(int)) for n in ev.natives}
    # streams go through the same float16 cast as stream_seq, so ASRF sees exactly the inputs of its stored decode
    out.update({s.recipe.recipe_id: (s.feat.astype(np.float16), s.prob, s.labels.astype(int)) for s in (*ev.pairs, *ev.multi)})
    return out


@torch.no_grad()
def asrf_decode(head, model, feat: np.ndarray, device) -> tuple[np.ndarray, list[int]]:
    """The official decode (as ASRFHead.decode) plus its candidate cuts: accepted peaks before the class vote."""
    from libs.metric import argrelmax   # the pinned official repository, on sys.path once the head exists
    x = torch.from_numpy(feat.T.astype(np.float32))[None].to(device)
    cls, bnd = model(x); cls = cls.cpu().numpy(); bprob = torch.sigmoid(bnd).cpu().numpy()
    final = head.post(cls, boundaries=bprob.copy(), masks=np.ones((1, 1, feat.shape[0]), bool))[0].astype(int)
    cuts = [int(i) for i in argrelmax(bprob[0, 0].copy(), threshold=head.boundary_th) if i > 0]
    return final, cuts


def b3_decode(prob: np.ndarray, p: dict) -> tuple[np.ndarray, list[int]]:
    b = detect_boundaries(prob, int(p["half_window"]), float(p["threshold"]), int(p["nms_radius"]))
    return labels_from_boundaries(prob, b).astype(int), [int(i) for i in b]


def oracle(labels: np.ndarray, cuts: list[int], ref: np.ndarray) -> np.ndarray:
    """Every span labelled A4C or A5C takes its reference majority class when that is A4C or A5C."""
    out = labels.copy(); edges = [0, *sorted(cuts), labels.size]
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a and len(set(labels[a:b].tolist())) != 1:
            raise SystemExit(f"labels vary inside a span {a}-{b}: the spans do not match the decoding")
        if b > a and out[a] in (A4C, A5C):
            m = int(np.bincount(ref[a:b], minlength=NUM_CLASSES).argmax())
            out[a:b] = m if m in (A4C, A5C) else out[a]
    return out


def candidate_recall(cuts: dict[str, list[int]], ev: EvalSet, tol: int) -> float:
    hit = tot = 0
    for s in ev.multi:
        true = boundaries_from_labels(s.labels); tot += true.size
        hit += len(match_changes(np.array(cuts[s.recipe.recipe_id], int), true, tol))
    return hit / tot


def score(preds: dict, ev: EvalSet, hz: float, tol_s: float) -> dict:
    m, _ = evaluate_predictions(preds, ev, hz, tol_s)
    return {k: m[k] for k in FLAT}


def base_rows(name: str, decode, seqs: dict, ev: EvalSet, red: EvalSet, hz: float, tol_s: float) -> dict:
    final, cuts = {}, {}
    for sid, (feat, prob, _) in seqs.items():
        final[sid], cuts[sid] = decode(feat, prob)
    fcuts = {sid: boundaries_from_labels(v).tolist() for sid, v in final.items()}
    out = {"default": score(final, ev, hz, tol_s), "default_without_clips": score(final, red, hz, tol_s),
           "candidate_cut_recall": candidate_recall(cuts, ev, int(round(tol_s * hz))),
           "final_cut_recall": candidate_recall(fcuts, ev, int(round(tol_s * hz)))}
    for span, cc in (("final", fcuts), ("candidate", cuts)):
        o = {sid: oracle(final[sid], cc[sid], seqs[sid][2]) for sid in seqs}
        out[f"oracle_{span}"] = score(o, ev, hz, tol_s); out[f"oracle_{span}_without_clips"] = score(o, red, hz, tol_s)
    return {**out, "final": final}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("-c", "--config", default="configs/experiments/evidence_pooling_v1.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); spec = json.loads(Path(a.spec).read_text())
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); sc = cfg["selection"]
    hz, tol_s = float(cfg["protocol"]["hz"]), float(sc["boundary_tolerance_s"]); clips = set(cfg["sensitivity"]["exclude_clips"])
    b7 = json.loads((Path(spec["b7_run"]) / "posterior_decoders.json").read_text())["results"]["B7"]
    res = {}
    for enc, e in spec["encoders"].items():
        inputs = Inputs(cfg, Path(a.protocol), Path(e["feature_manifest"]), paths.cache_root)
        b3_params = json.loads((Path(e["simple_run"]) / "selected.json").read_text())["selected"]["b3"]["main"]
        asrf = {f"ASRF_s{k}": (Path(rd), *load_asrf(Path(rd), paths.repo_root, inputs.feature_dim, device)) for k, rd in sorted(e["asrf"].items())}
        for h in HALVES:
            ev = inputs.halves[h]; red = without(ev, clips); seqs = inputs_of(ev, inputs.store)
            full_floor = inputs.frame_majority(ev, hz, tol_s)["native_macro_f1"] - float(sc["native_f1_tolerance"])
            res[f"{enc}|{h}|counts"] = {"natives": [len(ev.natives), len(red.natives)], "pairs": [len(ev.pairs), len(red.pairs)],
                                        "multi": [len(ev.multi), len(red.multi)], "floor": full_floor,
                                        "floor_without_clips": inputs.frame_majority(red, hz, tol_s)["native_macro_f1"] - float(sc["native_f1_tolerance"])}
            r = base_rows("B3", lambda f, p: b3_decode(p, b3_params), seqs, ev, red, hz, tol_s); r.pop("final")
            res[f"{enc}|{h}|B3"] = r
            for name, (rd, head, model) in asrf.items():
                r = base_rows(name, lambda f, p: asrf_decode(head, model, f, device), seqs, ev, red, hz, tol_s)
                stored = np.load(rd / f"predictions_{h}.npz")
                if any(not np.array_equal(r["final"][sid], stored[sid].astype(int)) for sid in seqs):
                    raise SystemExit(f"{enc} {name} {h}: decoding does not reproduce the stored predictions")
                r.pop("final"); res[f"{enc}|{h}|{name}"] = r
            if enc == "E1":   # B7 with B3 does not depend on the input; scored once
                bp = {sid: b3_decode(np.load(Path(b7["run_dir"]) / f"{sid}.npz")["prob"], b7["families"]["b3"]["params"])[0] for sid in seqs}
                res[f"system|{h}|B7_B3"] = {"default": score(bp, ev, hz, tol_s), "default_without_clips": score(bp, red, hz, tol_s)}
            print(f"{enc} {h} done", flush=True)
        del inputs, asrf; torch.cuda.empty_cache()
    run_dir = new_run_dir(cfg, "md5_reachability"); (run_dir / "reach.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:4000]); print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
