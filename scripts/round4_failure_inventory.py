#!/usr/bin/env python
"""Round 4: per-unit failure inventory of the strong baselines from existing predictions (diagnostic only).

configs/experiments/failure_analysis_v1.yaml fixes every definition. For each visual input (E0, E1) and validation
half:
- predictions of B3 tuned (recomputed from the simple-decoder run's selected setting), the three ASRF seeds (the
  stored default predictions, which a fresh decode must reproduce) and B7 with B3 (recomputed from the stored B7
  posteriors); each system's pooled segment counts must equal its reported rows, or the script stops;
- per unit (reference segment of a multi-fragment stream): length, class, neighbours, source clips and sessions, and
  per system the outcome and cause (echo_routing.evaluate.failure_units);
- per unit, with the same frames and contrast variant as in the stream: the frozen classifier's mean probability over
  each whole source clip and over the unit's own samples, and the unit's samples alone through B3 and every ASRF seed;
- spurious predicted segments with their host units, and the native clips each system gets wrong or cuts.
Writes units.csv, spurious.csv, predictions.npz (reference labels, classifier probabilities, system labels and ASRF
posteriors per stream) and inventory.json into a new run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.failure_units import segment_counts, session_of, stream_units, unit_outcomes  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402
from echo_routing.temporal.dev_data import Seq, stream_seq  # noqa: E402
from echo_routing.temporal.dev_heads import head_from_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402

NUM_CLASSES = 5


def b3(prob: np.ndarray, params: dict) -> np.ndarray:
    return decode_with_prob("b3", prob, {"decode": {"b3": params}})[0].astype(int)


def majority(labels: np.ndarray) -> int:
    return int(np.bincount(np.asarray(labels, int), minlength=NUM_CLASSES).argmax())


def check_counts(name: str, half: str, preds: dict, units: dict, rows_csv: Path) -> dict:
    """Pooled (matched, predicted, reference) segments must equal the system's reported multi-stream rows."""
    got = np.sum([segment_counts(preds[sid], units[sid]) for sid in units], axis=0)
    ref = pd.read_csv(rows_csv)[["seg_matched", "seg_pred", "seg_true"]].sum().to_numpy()
    if not np.array_equal(got, ref):
        raise SystemExit(f"{name} {half}: recomputed segments {got.tolist()} != reported {ref.tolist()} ({rows_csv})")
    return {"matched": int(got[0]), "pred": int(got[1]), "true": int(got[2]), "segment_f1_50": float(2 * got[0] / (got[1] + got[2]))}


def load_asrf(run_dir: Path, repo_root: Path, in_dim: int, device) -> tuple:
    man = json.loads((run_dir / "run_manifest.json").read_text())
    head = head_from_manifest(man, repo_root, in_dim, device)
    model = head.build().to(device); model.load_state_dict(torch.load(man["checkpoints"]["best"], map_location=device)); model.eval()
    return head, model


def unit_checks(u, s, store, prob: np.ndarray) -> dict:
    """Visual checks of one unit with the same frames and variant as in the stream (the frozen classifier only)."""
    crop = prob[u.start:u.end].mean(0); other = np.delete(crop, u.label)
    full, full_major, frag = [], [], []
    for vid, variant, a, b in u.sources:
        vp = store.get(vid, variant).prob
        full.append(int(vp.mean(0).argmax())); full_major.append(majority(vp.argmax(1))); frag.append(int(vp[a:b].mean(0).argmax()))
    return {"crop_class": int(crop.argmax()), "crop_p_label": float(crop[u.label]), "crop_margin": float(crop[u.label] - other.max()),
            "full_clip_right_all": all(c == u.label for c in full), "full_clip_majority_right_all": all(c == u.label for c in full_major),
            "fragments_crop_right_all": all(c == u.label for c in frag), "full_clip_classes": ";".join(map(str, full))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="JSON: encoders{E0,E1:{feature_manifest, simple_run, asrf{seed:run_dir}}}, b7_run")
    ap.add_argument("--protocol", required=True); ap.add_argument("-c", "--config", default="configs/experiments/failure_analysis_v1.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); spec = json.loads(Path(a.spec).read_text())
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hit, near, iou = int(cfg["causes"]["hit"]), int(cfg["causes"]["near"]), float(cfg["units"]["iou"])
    b7 = json.loads((Path(spec["b7_run"]) / "posterior_decoders.json").read_text())["results"]["B7"]
    b7_dir, b7_params = Path(b7["run_dir"]), b7["families"]["b3"]["params"]
    rows, spurious, arrays, verify, natives = [], [], {}, {}, {}
    for enc, e in spec["encoders"].items():
        inputs = Inputs(cfg, Path(a.protocol), Path(e["feature_manifest"]), paths.cache_root)
        b3_params = json.loads((Path(e["simple_run"]) / "selected.json").read_text())["selected"]["b3"]["main"]
        asrf = {f"ASRF_s{k}": (Path(rd), *load_asrf(Path(rd), paths.repo_root, inputs.feature_dim, device)) for k, rd in sorted(e["asrf"].items())}
        for h in HALVES:
            ev = inputs.halves[h]; streams = {s.recipe.recipe_id: s for s in ev.multi}
            units = {sid: stream_units(s.recipe) for sid, s in streams.items()}
            preds = {"B3": {sid: b3(s.prob, b3_params) for sid, s in streams.items()}}
            verify[f"{enc}|{h}|B3"] = check_counts(f"{enc} B3", h, preds["B3"], units, Path(e["simple_run"]) / f"rows_b3_tuned_main_{h}_multi_seg.csv")
            seqs = [stream_seq(s) for s in streams.values()]
            for name, (rd, head, model) in asrf.items():
                stored = np.load(rd / f"predictions_{h}.npz")
                dec = head.decode(model, seqs, device)
                for sid in streams:
                    if not np.array_equal(dec[sid]["labels"], stored[sid].astype(int)):
                        raise SystemExit(f"{enc} {name} {h} {sid}: a fresh decode differs from the stored predictions")
                    arrays[f"{enc}|{h}|{name}|prob|{sid}"] = dec[sid]["prob"].astype(np.float32)
                preds[name] = {sid: stored[sid].astype(int) for sid in streams}
                verify[f"{enc}|{h}|{name}"] = check_counts(f"{enc} {name}", h, preds[name], units, rd / f"rows_{h}_multi_seg.csv")
            preds["B7_B3"] = {sid: b3(np.load(b7_dir / f"{sid}.npz")["prob"], b7_params) for sid in streams}
            verify[f"{enc}|{h}|B7_B3"] = check_counts("B7 with B3", h, preds["B7_B3"], units, Path(spec["b7_run"]) / f"rows_B7_b3_{h}_multi_seg.csv")
            iso = [Seq(f"{sid}#{u.index}", False, streams[sid].feat[u.start:u.end].astype(np.float16), streams[sid].labels[u.start:u.end].astype(np.int64),
                       np.zeros(0, int), streams[sid].prob[u.start:u.end].astype(np.float32)) for sid in streams for u in units[sid]]
            iso_out = {name: head.decode(model, iso, device) for name, (_, head, model) in asrf.items()}
            for sid, s in streams.items():
                arrays[f"{h}|ref|{sid}"] = s.labels.astype(np.int8); arrays[f"{enc}|{h}|clf|{sid}"] = s.prob.astype(np.float32)
                outcomes = {}
                for name, p in preds.items():
                    arrays[f"{enc}|{h}|{name}|labels|{sid}"] = p[sid].astype(np.int8)
                    outcomes[name], sp = unit_outcomes(p[sid], units[sid], iou, hit, near)
                    spurious += [{"enc": enc, "half": h, "system": name, "stream": sid, **x} for x in sp]
                for u in units[sid]:
                    row = {"enc": enc, "half": h, "stream": sid, "unit": u.index, "start": u.start, "end": u.end, "length": u.length,
                           "label": u.label, "left": u.left, "right": u.right, "n_sources": len(u.sources),
                           "sources": ";".join(v for v, *_ in u.sources), "variants": ";".join(var for _, var, *_ in u.sources),
                           "sessions": ";".join(sorted({session_of(v) for v, *_ in u.sources})),
                           **unit_checks(u, s, inputs.store, s.prob), "iso_B3": majority(b3(s.prob[u.start:u.end], b3_params))}
                    for name in asrf:
                        row[f"iso_{name}"] = majority(iso_out[name][f"{sid}#{u.index}"]["labels"])
                    for name in preds:
                        o = outcomes[name][u.index]
                        row.update({f"{name}_matched": o["matched"], f"{name}_cause": o["cause"], f"{name}_majority": o["majority"],
                                    f"{name}_frac_label": o["frac_label"], f"{name}_best_iou": o["best_iou"], f"{name}_best_label": o["best_label"],
                                    f"{name}_edge_dist": json.dumps(o["edge_dist"]), f"{name}_extra_cuts": o["extra_cuts"]})
                    rows.append(row)
            nat = {"B3": {c.seq_id: b3(c.prob, b3_params) for c in ev.natives}}
            for name, (rd, *_) in asrf.items():
                stored = np.load(rd / f"predictions_{h}.npz"); nat[name] = {c.seq_id: stored[c.seq_id].astype(int) for c in ev.natives}
            for name, p in nat.items():
                natives[f"{enc}|{h}|{name}"] = {"wrong": [c.seq_id for c in ev.natives if majority(p[c.seq_id]) != int(c.labels[0])],
                                                "cut": [c.seq_id for c in ev.natives if boundaries_from_labels(p[c.seq_id]).size],
                                                "n": len(ev.natives)}
            print(f"{enc} {h}: " + "; ".join(f"{k.split('|')[-1]} {v['segment_f1_50']:.4f}" for k, v in verify.items() if k.startswith(f"{enc}|{h}|")), flush=True)
        del inputs, asrf; torch.cuda.empty_cache()
    run_dir = new_run_dir(cfg, "md4_failure_inventory")
    pd.DataFrame(rows).to_csv(run_dir / "units.csv", index=False); pd.DataFrame(spurious).to_csv(run_dir / "spurious.csv", index=False)
    np.savez_compressed(run_dir / "predictions.npz", **arrays)
    (run_dir / "inventory.json").write_text(json.dumps({"verified_against_reported_rows": verify, "natives": natives, "spec": spec,
                                                        "b7_b3_params": b7_params, "causes": cfg["causes"], "units_cfg": cfg["units"]}, indent=1))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
