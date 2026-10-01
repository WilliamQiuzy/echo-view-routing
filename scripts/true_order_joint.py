#!/usr/bin/env python
"""Joint true-order comparison on EV9V (deliverable 3 of the 2026-10-01 round).

All methods read the same corrected test inputs (cache_v2; recipes of the ladder config) in acquisition order:
clip macro-F1 (native clips, decoded majority), class-aware segmental F1 at IoU 0.5, boundary recall / F1 at 0.5 s,
internal cuts of native clips and the same-class join cut rate, plus paired differences to a reference method with
the same resampling draws (native clips by presumed patient group, pair streams by stream, 4-8 fragment streams by
cine for boundary counts and by stream for segment counts). Timing is measured separately for B0 feature extraction
(frame decoding + preprocessing on CPU, forward pass on GPU) and for decoding (CPU, one thread; MS-TCN forward
included for B4). Windowed end-to-end models (B6-B8) have no separate decoder; their cost is reported from run logs.
Writes <out>/joint.json and <out>/joint.md (aggregates only).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402
from scripts.permutation_mechanism import Frozen, assemble, natives  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.compose.banks import multi_bank, pair_bank  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.ladder import constructed_metrics  # noqa: E402
from echo_routing.evaluate.metrics_classification import classification_summary  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import segment_f1  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.realvideo.screening import patient_of  # noqa: E402
from echo_routing.temporal.baselines.registry import AUDITED_B3, decode_with_prob  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels, labels_to_segments  # noqa: E402

N_BOOT, BOOT_SEED = 2000, 20261001
ENC = "79f4a41af6e3"


def extra_b4(arg: str | None) -> list[dict]:
    """Extra official-MS-TCN prediction dirs, "label=dir;label=dir" (e.g. the seeded matrix models in true order)."""
    out = []
    for item in filter(None, (arg or "").split(";")):
        label, _, pred = item.partition("=")
        out.append({"name": label.strip(), "method": "b4", "encoder": "b0", "set": {"mstcn_official": {"pred_dir": pred.strip()}}})
    return out


def specs(paths, ep_b3a: dict) -> list[dict]:
    c = paths.cache_root
    out = [{"name": "B0 frame classifier", "method": "b0", "encoder": "b0"},
           {"name": "B1 smoothing", "method": "b1", "encoder": "b0"},
           {"name": "B2 HMM", "method": "b2", "encoder": "b0"},
           {"name": "B3 change points, default", "method": "b3", "encoder": "b0"},
           {"name": "B3 change points, audited", "method": "b3a", "encoder": "b0"},
           {"name": "B4 MS-TCN v2", "method": "b4", "encoder": "b0", "set": {"mstcn_official": {"pred_dir": str(c / "mstcn_official" / ENC)}}},
           {"name": "B5 ASFormer v2", "method": "b5", "encoder": "b0", "set": {"asformer_official": {"pred_dir": str(c / "asformer_official" / ENC)}}},
           {"name": "B6 STFM windowed, numeric-order model", "method": "b6", "encoder": "b0",
            "set": {"stfm_windowed": {"pred_dir": str(c / "stfm_windowed" / f"{ENC}_numeric_s100")}}},
           {"name": "B7 EchoViewCLIP windowed", "method": "b7", "encoder": "b0",
            "set": {"echoviewclip_windowed": {"pred_dir": str(c / "echoviewclip_windowed" / ENC)}}},
           {"name": "B8 EchoPrime per frame", "method": "b0", "encoder": "echoprime"},
           {"name": "B8 + HMM", "method": "b2", "encoder": "echoprime"},
           {"name": "B8 + audited B3 (EchoPrime setting)", "method": "b3a", "encoder": "echoprime", "set": {"decode": {"b3a": ep_b3a}}}]
    return [s for s in out if s["method"] != "b5" or (c / "asformer_official" / ENC / "DONE").exists()
            or any((c / "asformer_official" / ENC).glob("*.npz"))]


def merged(cfg: dict, extra: dict | None) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in (extra or {}).items():
        out[k] = {**(out.get(k) or {}), **v}
    return out


def evaluate_spec(spec: dict, cfg: dict, inputs: dict, hz: float) -> dict:
    c = merged(cfg, spec.get("set")); m = spec["method"]; tol = cfg["policy"]["boundary_tolerances_s"][1] * hz
    nat, pairs, multi = inputs[spec["encoder"]]
    nat_rows = []
    for x in nat:
        lab = decode_with_prob(m, x.prob, c, x.feat, x.video_id)[0]
        seg = segment_f1(labels_to_segments(lab), labels_to_segments(np.full(lab.size, x.label)), 0.5)
        nat_rows.append({"video_id": x.video_id, "group": patient_of(x.video_id), "label": x.label, "n": lab.size,
                         "majority": int(np.bincount(lab, minlength=5).argmax()), "cuts": int(boundaries_from_labels(lab).size),
                         "seg_matched": seg["matched"], "seg_pred": len(labels_to_segments(lab)), "seg_true": 1})
    banks = {}
    for bank, streams in (("pairs", pairs), ("multi", multi)):
        res = constructed_metrics(m, streams, hz, c, tau=None, taus={})
        seg_rows = []
        for s in streams:
            lab = decode_with_prob(m, s.prob, c, s.feat, s.recipe.recipe_id)[0]
            ps, ts = labels_to_segments(lab), labels_to_segments(s.labels)
            seg_rows.append({"stream": s.recipe.recipe_id, "seg_matched": segment_f1(ps, ts, 0.5)["matched"], "seg_pred": len(ps), "seg_true": len(ts)})
        banks[bank] = {"fragments": pd.DataFrame(res["_fragments"]), "segments": pd.DataFrame(seg_rows)}
    return {"natives": pd.DataFrame(nat_rows), "banks": banks}


CONF = [f"c{i}{j}" for i in range(5) for j in range(5)]


def macro_f1_from_confusion(c: np.ndarray) -> np.ndarray:
    """Macro-F1 over the five classes from summed confusion counts (leading axis = the 25 cells, true-major)."""
    m = c.reshape((5, 5) + c.shape[1:]); tp = np.stack([m[i, i] for i in range(5)])
    fp = m.sum(0) - tp; fn = m.sum(1) - tp; den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0).mean(0)


def endpoint_values(ev: dict) -> dict[str, tuple]:
    """endpoint -> (unit table name, unit column, numerator cols, function of summed columns)."""
    return {
        "clip macro-F1": ("nat", "group", CONF, macro_f1_from_confusion),
        "segment F1@50, multi": ("multi_seg", "stream", ["seg_matched", "seg_pred", "seg_true"], lambda c: 2 * c[0] / (c[1] + c[2])),
        "segment F1@50, pairs": ("pairs_seg", "stream", ["seg_matched", "seg_pred", "seg_true"], lambda c: 2 * c[0] / (c[1] + c[2])),
        "segment F1@50, native": ("nat", "group", ["seg_matched", "seg_pred", "seg_true"], lambda c: 2 * c[0] / (c[1] + c[2])),
        "boundary recall, multi": ("multi_frag", "video_id", ["t_matched", "t_true"], lambda c: c[0] / c[1]),
        "boundary F1, multi": ("multi_frag", "video_id", ["t_matched", "p_pred", "t_true"], lambda c: 2 * c[0] / (c[1] + c[2])),
        "boundary F1, pairs": ("pairs_frag", "stream", ["t_matched", "p_pred", "t_true"], lambda c: 2 * c[0] / (c[1] + c[2])),
        "same-class cut rate, multi": ("multi_frag", "video_id", ["false_split", "same_join"], lambda c: c[0] / c[1]),
        "native cuts per clip": ("nat", "group", ["cuts", "one"], lambda c: c[0] / c[1]),
        "native cuts per min": ("nat", "group", ["cuts", "minutes"], lambda c: c[0] / c[1]),
    }


def tables(ev: dict) -> dict[str, pd.DataFrame]:
    n = ev["natives"].copy(); n["one"] = 1.0; n["minutes"] = n["n"] / 600.0
    for i in range(5):
        for j in range(5):
            n[f"c{i}{j}"] = ((n["label"] == i) & (n["majority"] == j)).astype(float)
    return {"nat": n, "multi_seg": ev["banks"]["multi"]["segments"], "pairs_seg": ev["banks"]["pairs"]["segments"],
            "multi_frag": ev["banks"]["multi"]["fragments"], "pairs_frag": ev["banks"]["pairs"]["fragments"]}


def bootstrap(evals: dict[str, dict], reference: str) -> dict:
    rng = np.random.default_rng(BOOT_SEED); tabs = {k: tables(v) for k, v in evals.items()}
    ref_tabs = tabs[reference]; spec = endpoint_values(None); W: dict = {}; out = {k: {} for k in evals}
    for ep, (tab, unit, cols, fn) in spec.items():
        units = pd.Index(sorted(ref_tabs[tab][unit].unique()))
        if (tab, unit) not in W:
            d = rng.integers(0, len(units), size=(N_BOOT, len(units)))
            W[(tab, unit)] = np.stack([np.bincount(x, minlength=len(units)) for x in d]).astype(float)
        w = W[(tab, unit)]
        reps, pts = {}, {}
        for k, t in tabs.items():
            s = t[tab].groupby(unit)[cols].sum().reindex(units, fill_value=0).to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                pts[k] = float(fn(s.sum(0))); reps[k] = fn(np.moveaxis(w @ s, -1, 0))
        for k in tabs:
            e = {"value": pts[k], "ci95": np.nanpercentile(reps[k], [2.5, 97.5]).tolist()}
            if k != reference:
                e["diff"] = pts[k] - pts[reference]; e["diff_ci95"] = np.nanpercentile(reps[k] - reps[reference], [2.5, 97.5]).tolist()
            out[k][ep] = e
    return out


def clip_f1(ev: dict) -> float:
    n = ev["natives"]
    return float(classification_summary(n["label"].to_numpy(), n["majority"].to_numpy(), 5)["macro_f1"])


def time_decoders(evals_specs: list[dict], cfg: dict, inputs: dict, paths) -> dict:
    """CPU, one thread: decoding of the true-order test inputs; B4 includes the MS-TCN forward pass."""
    import torch
    torch.set_num_threads(1)
    out = {}
    nat, pairs, multi = inputs["b0"]
    n_samples = sum(x.prob.shape[0] for x in nat) + sum(s.prob.shape[0] for s in pairs + multi)
    for s in evals_specs:
        if s["encoder"] != "b0" or s["method"] not in ("b0", "b1", "b2", "b3", "b3a"):
            continue
        c = merged(cfg, s.get("set")); t0 = time.perf_counter()
        for x in nat:
            decode_with_prob(s["method"], x.prob, c)
        for st in pairs + multi:
            decode_with_prob(s["method"], st.prob, c)
        out[s["name"]] = 1e6 * (time.perf_counter() - t0) / n_samples
    sys.path.insert(0, str(paths.repo_root / "third_party" / "ms-tcn"))
    from model import MultiStageModel
    model = MultiStageModel(4, 10, 64, 2048, 5); model.load_state_dict(torch.load(paths.repo_root / "models_frozen/mstcn_official/v2_frameorder/epoch-50.model", map_location="cpu")); model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for f in [x.feat for x in nat] + [st.feat for st in pairs + multi]:
            x = np.zeros((2048, f.shape[0]), dtype=np.float32); x[: f.shape[1]] = f.T
            xt = torch.from_numpy(x).unsqueeze(0)
            torch.softmax(model(xt, torch.ones(1, 5, xt.shape[2]))[-1], dim=1).argmax(1)
    out["B4 MS-TCN v2"] = 1e6 * (time.perf_counter() - t0) / n_samples
    return {"unit": "microseconds per 10 Hz sample (CPU, 1 thread)", "samples": int(n_samples), "per_method": out}


def time_b0_extraction(paths, cfg: dict, n_clips: int = 100) -> dict:
    """B0 on the first n test clips: CPU frame decoding + letterbox, then the GPU forward pass, timed separately."""
    import torch
    from echo_routing.features.encoder import load_checkpoint
    from echo_routing.features.sampling import stride_indices
    from echo_routing.ingest.frames import list_frame_paths, load_frame
    from echo_routing.manifest import load_manifest
    meta = json.loads((variant_dir(paths.cache_root, ENC, "orig") / "meta.json").read_text())
    model, _ = load_checkpoint(paths.repo_root / meta["ckpt"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    df = load_manifest(paths.manifest_path); df = df[(df["split"] == "test") & df["frames_dir"].notna()].head(n_clips)
    t_io = t_fw = 0.0; n = 0
    with torch.no_grad():
        for r in df.itertuples(index=False):
            t0 = time.perf_counter()
            fr = list_frame_paths(Path(r.frames_dir)); idx = stride_indices(int(r.n_frames), float(r.fps_playback), cfg["sampling"]["target_hz"])
            x = torch.from_numpy(np.stack([load_frame(fr[min(i, len(fr) - 1)], int(meta["image_size"])) for i in idx]))
            t1 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model.forward_with_features(x.to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fw += time.perf_counter() - t1; t_io += t1 - t0; n += len(idx)
    return {"clips": int(len(df)), "samples": n, "frame_io_us_per_sample": 1e6 * t_io / n, "forward_us_per_sample": 1e6 * t_fw / n,
            "device": str(device), "note": "shared GPU (ASFormer training running); one clip per batch"}


def main() -> None:
    ap = base_parser("Joint true-order comparison"); ap.add_argument("--out", required=True); ap.add_argument("--reference", default="B4 MS-TCN v2")
    ap.add_argument("--ep-b3a", default="runs/_demo/permutation_mechanism_r2/echoprime_b3a.json"); ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--extra-b4", default=None, help='extra MS-TCN prediction dirs as "label=dir;label=dir"')
    args = ap.parse_args(); cfg = load_cfg(args); cfg["decode"].setdefault("b3a", dict(AUDITED_B3)); paths = load_paths(); out = Path(args.out)
    hz = cfg["sampling"]["target_hz"]
    ep = json.loads(Path(args.ep_b3a).read_text()); ep_b3a = {k: ep[k] for k in ("half_window", "threshold", "nms_radius")}
    idx = read_index(variant_dir(paths.cache_root, ENC, "orig")); fz = Frozen(paths, ENC)
    bank_p, bank_m = pair_bank(idx, "test", cfg), multi_bank(idx, "test", cfg)
    inputs = {e: (natives(fz, idx, "test", "true", 0, None, e)[0], [assemble(fz, r, "true", 0, None, e)[0] for r in bank_p.recipes],
                  [assemble(fz, r, "true", 0, None, e)[0] for r in bank_m.recipes]) for e in ("b0", "echoprime")}
    sp = specs(paths, ep_b3a) + extra_b4(args.extra_b4)
    evals = {s["name"]: evaluate_spec(s, cfg, inputs, hz) for s in sp}
    boot = bootstrap(evals, args.reference)
    res = {"methods": [s["name"] for s in sp], "reference": args.reference, "clip_macro_f1": {k: clip_f1(v) for k, v in evals.items()},
           "endpoints": boot, "units": {"native": "presumed patient group", "pairs": "stream", "multi boundary": "cine", "multi segments": "stream"}}
    if not args.skip_timing:
        res["timing_decoding"] = time_decoders(sp, cfg, inputs, paths); res["timing_b0_extraction"] = time_b0_extraction(paths, cfg)
    out.mkdir(parents=True, exist_ok=True); (out / "joint.json").write_text(json.dumps(res, indent=1, default=float))
    eps = list(endpoint_values(None))
    md = ["# Joint true-order comparison (EV9V test, corrected inputs)", "", f"Paired differences are to {args.reference}.", "",
          "| method | " + " | ".join(eps) + " |", "|---|" + "---|" * len(eps)]
    for k in res["methods"]:
        cells = []
        for e in eps:
            x = boot[k][e]; lo, hi = x["ci95"]
            cell = f"{x['value']:.3f} [{lo:.3f}, {hi:.3f}]"
            if "diff" in x:
                dl, dh = x["diff_ci95"]; cell += f"<br>Δ {x['diff']:+.3f} [{dl:+.3f}, {dh:+.3f}]"
            cells.append(cell)
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    if "timing_decoding" in res:
        md += ["", f"Decoding cost ({res['timing_decoding']['unit']}): " + json.dumps({k: round(v, 2) for k, v in res["timing_decoding"]["per_method"].items()}),
               "", "B0 extraction: " + json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res["timing_b0_extraction"].items()})]
    (out / "joint.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
