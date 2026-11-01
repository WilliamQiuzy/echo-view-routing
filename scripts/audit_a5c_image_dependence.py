#!/usr/bin/env python3
"""Independent cache/membership/pixel-subset recount; no model forward or training."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/a5c_image_dependence_v1"
PUB = ROOT / "docs/results/2026-10-31_a5c_image_dependence"
NAMES = ("original", "periphery", "periphery_control", "chroma", "chroma_control", "static_bright",
         "static_bright_control", "grayscale", "target_repeat", "full_letterbox", "retained_letterbox")
MODELS = ("A08", "C", "N")
GROUPS = ("A5C_recovered", "A5C_missed", "A4C_harmed", "A4C_retained")


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while b := f.read(1048576):
            h.update(b)
    return h.hexdigest()


def reference_run_scores(row_ids, scores, members, weights):
    ids = list(map(int, row_ids)); weights = list(map(float, weights))
    if len(set(ids)) != len(ids) or len(members) != len(weights) or min(weights) < 0 or abs(sum(weights) - 1) > 1e-12:
        raise ValueError("invalid membership/weight mass")
    ans = np.zeros(scores.shape[1:], np.float64)
    for member, weight in zip(members, weights):
        ans += weight * scores[ids.index(int(member))].astype(np.float64)
    return ans


def stats(x):
    a = np.asarray([v for v in x if v is not None], np.float64)
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "q05": None, "q95": None}
    if not np.isfinite(a).all():
        raise ValueError("nonfinite statistic")
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "q05": float(np.quantile(a, .05)), "q95": float(np.quantile(a, .95))}


def effect_counts(original, changed):
    x, y = np.asarray(original), np.asarray(changed)
    return {"n": len(x), "original_positive": int((x > 0).sum()), "positive_after": int((y > 0).sum()),
            "positive_to_nonpositive": int(((x > 0) & (y <= 0)).sum()),
            "nonpositive_to_positive": int(((x <= 0) & (y > 0)).sum()),
            "score_delta": stats(y - x)}


def correlation(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    x = x - x.mean(); y = y - y.mean()
    den = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.clip(np.dot(x, y) / den, -1, 1)) if den > 1e-12 else None


def assert_original_parity(observed, expected):
    a, b = np.asarray(observed), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("bad original score array")
    err = float(np.max(np.abs(a - b)))
    if err > 1e-4:
        raise ValueError("original score cache alignment/parity failed")
    return err


def reference_pixels(F, row):
    from PIL import Image
    seq = F.seqs[int(F.W["seq"][row])]
    xs, fs, rs = [], [], []
    for pos in F.W["s"][row]:
        v, f, g = seq["refs"][int(pos)]; g = round(float(g), 6)
        paths = F.fp(v)
        with Image.open(paths[min(int(f), len(paths) - 1)]) as image:
            im = image.convert("RGB")
        if abs(g - 1) > 1e-9:
            im = im.point([int(255 * (i / 255.) ** g + .5) for i in range(256)] * 3)
        w, h = im.size
        rw, rh = max(224, round(w * 256 / min(w, h))), max(224, round(h * 256 / min(w, h)))
        left, top = (rw - 224) // 2, (rh - 224) // 2
        big = im.resize((rw, rh), Image.Resampling.BILINEAR)
        xs.append(np.asarray(big.crop((left, top, left + 224, top + 224))))
        nw, nh = max(1, round(w * 224 / max(w, h))), max(1, round(h * 224 / max(w, h)))
        xx, yy = (224 - nw) // 2, (224 - nh) // 2
        full = np.zeros((224, 224, 3), np.uint8)
        full[yy:yy + nh, xx:xx + nw] = im.resize((nw, nh), Image.Resampling.BILINEAR)
        footprint = np.zeros((rh, rw), np.uint8); footprint[top:top + 224, left:left + 224] = 255
        fp = Image.fromarray(footprint).resize((w, h), Image.Resampling.NEAREST).resize((nw, nh), Image.Resampling.NEAREST)
        keep = np.zeros((224, 224), bool); keep[yy:yy + nh, xx:xx + nw] = np.asarray(fp) > 0
        retained = full.copy(); retained[~keep] = 0
        fs.append(full); rs.append(retained)
    return np.stack(xs), np.stack(fs), np.stack(rs)


def reference_variants(x, full, retained, marker, seed):
    from PIL import Image
    out = {"original": x}
    band = np.ones((224, 224), bool); band[16:208, 16:208] = False
    gray = x.astype(np.float32).mean(-1)
    masks = [band, ((x.max(-1) - x.min(-1) >= 25) & (x.max(-1) >= 60)).any(0),
             (gray.mean(0) >= 180) & (gray.std(0) <= 3)]
    fill = np.rint(x.mean((1, 2), keepdims=True)).astype(np.uint8)
    for j, (name, mask) in enumerate(zip(("periphery", "chroma", "static_bright"), masks)):
        out[name] = np.where(mask[None, ..., None], fill, x)
        ctl = np.zeros(224 * 224, bool)
        ctl[np.random.default_rng([seed, j]).permutation(224 * 224)[:int(mask.sum())]] = True
        out[name + "_control"] = np.where(ctl.reshape(1, 224, 224, 1), fill, x)
    out["grayscale"] = np.stack([np.repeat(np.asarray(Image.fromarray(f).convert("L"))[..., None], 3, -1) for f in x])
    out["target_repeat"] = np.stack([x[marker]] * 16)
    out["full_letterbox"], out["retained_letterbox"] = full, retained
    return out


def audit_geometry(c, baseline, cases):
    from PIL import Image, ImageFilter
    import scripts.train_risk_constrained_a5c as a
    complete = json.loads((OUT / "geometry_complete.json").read_text())
    assert digest(OUT / "geometry_scores.npz") == complete["scores_sha256"]
    assert digest(OUT / "geometry_input_hashes.json") == complete["input_hashes_sha256"]
    assert digest(OUT / "geometry_mask_metadata.json") == complete["mask_metadata_sha256"]
    assert complete["forward_windows"] == complete["expected_windows"]
    z = np.load(OUT / "geometry_scores.npz")
    hashes = json.loads((OUT / "geometry_input_hashes.json").read_text())
    geometry = json.loads((OUT / "geometry_mask_metadata.json").read_text())
    summary, checks, repeat_max = {}, 0, 0.
    for h in (0, 1):
        rows = z[f"rows_f{h}"]; assert np.array_equal(rows, baseline[f"rows_f{h}"])
        cc = [q for q in cases if q["fold"] == h]
        for row in geometry[str(h)].values():
            for props in row.values():
                assert len({p["area_pixels"] for p in props}) == 1
                assert len({p["periodic_boundary_edges"] for p in props}) == 1
        for model in MODELS:
            S = z[f"geometry_{model}_f{h}"]; G = baseline[f"global_{model}_f{h}"]
            assert S.shape == (len(rows), 3, 2, 4, 2) and np.isfinite(S).all()
            for ni, oldindex in enumerate((1, 3, 5)):
                repeat_max = max(repeat_max, assert_original_parity(S[:, ni, 0, 0, 0], G[:, oldindex, 0]))
            weighted = np.stack([reference_run_scores(rows, S, q["rows"], q["weights"]) for q in cc])
            original = np.stack([reference_run_scores(rows, G, q["rows"], q["weights"])[0] for q in cc])
            for group in GROUPS:
                at = [j for j, q in enumerate(cc) if q["group"] == group]
                key = f"{model}_fold{h}_{group}"; summary[key] = {}
                for ni, name in enumerate(("periphery", "chroma", "static_bright")):
                    for fi, fill in enumerate(("mean", "blur")):
                        data = weighted[at, ni, fi, :, 0]
                        mask_rec = {f"location_{ki}": effect_counts(original[at, 0], data[:, ki]) for ki in range(4)}
                        mask_rec["actual_drop_excess_over_translated_mean"] = stats(data[:, 1:].mean(1) - data[:, 0])
                        mask_rec["actual_drop_larger_than_all_translations"] = int((data[:, 0] < data[:, 1:].min(1)).sum())
                        mask_rec["mean_shifted_score"] = stats(data[:, 1:].mean(1))
                        summary[key][name + "_" + fill] = mask_rec
        F = a.Fold(c, 0, h)
        for rowkey, expected in hashes[str(h)].items():
            row = int(rowkey); x, _, _ = reference_pixels(F, row)
            gray = x.astype(np.float32).mean(-1)
            band = np.ones((224, 224), bool); band[16:208, 16:208] = False
            masks = {"periphery": band,
                     "chroma": ((x.max(-1) - x.min(-1) >= 25) & (x.max(-1) >= 60)).any(0),
                     "static_bright": (gray.mean(0) >= 180) & (gray.std(0) <= 3)}
            for name, mask in masks.items():
                for fi, fill in enumerate(("mean", "blur")):
                    rep = np.broadcast_to(np.rint(x.mean((1, 2), keepdims=True)).astype(np.uint8), x.shape) if fill == "mean" else np.stack([np.asarray(Image.fromarray(v).filter(ImageFilter.BoxBlur(8))) for v in x])
                    for ki, shift in enumerate(((0, 0), (0, 112), (112, 0), (112, 112))):
                        moved = np.roll(mask, shift, axis=(0, 1))
                        v = np.where(moved[None, ..., None], rep, x)
                        assert hashlib.sha256(v.tobytes()).hexdigest() == expected[f"{name}_{fill}_{ki}"]
                        prop = geometry[str(h)][rowkey][name][ki]
                        contact = int(moved[0].sum() + moved[-1].sum() + moved[:, 0].sum() + moved[:, -1].sum())
                        planar = int((moved[1:] != moved[:-1]).sum() + (moved[:, 1:] != moved[:, :-1]).sum()) + contact
                        assert prop["area_pixels"] == int(mask.sum()) and prop["planar_boundary_edges"] == planar
                        checks += 1
    receipt = {"independent_geometry_audit_pass": True, "pixel_hashes_and_geometry_checked": checks,
               "all_row_actual_mask_replay_max_abs": repeat_max, "new_baseline_forwards": 0,
               "new_model_forwards": 0, "scores_sha256": complete["scores_sha256"],
               "limits": "Toroidal controls preserve periodic geometry and area, but retain planar seam/contact/contiguity confounds. No anatomical truth."}
    (PUB / "geometry_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (PUB / "geometry_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    from PIL import Image, ImageFilter
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT))
    import scripts.train_risk_constrained_a5c as a
    start, cpu = time.time(), time.process_time()
    complete = json.loads((OUT / "run_complete.json").read_text())
    manifest = json.loads((OUT / "input_freeze.json").read_text())
    cases = json.loads((OUT / "selection.json").read_text())["cases"]
    metadata = json.loads((OUT / "pixel_metadata.json").read_text())
    assert complete["scores_sha256"] == digest(OUT / "scores.npz")
    assert manifest["selection_sha256"] == digest(OUT / "selection.json")
    assert complete["forward_windows"] == complete["expected_windows"]
    for p, h in manifest["pinned_files"].items():
        assert digest(ROOT / p) == h
    cfg = "configs/experiments/video_risk_constrained_a5c_v1.yaml"
    c = a.Ctx(a.load_cfg(SimpleNamespace(config=cfg, set=[])), cfg)
    old, z = np.load(c.out / "readout_cache_s0.npz"), np.load(OUT / "scores.npz")
    summary = {"scope": "fixed outcome-conditioned descriptive training-source audit; no new efficacy", "groups": manifest["groups"],
               "global_interventions": {}, "spatial_temporal": {}, "input_proxies": {}, "model_training_updates": 0}
    summary["interpretation_limits"] = [
        "Equal-area scattered controls do not preserve mask contiguity; their difference alone does not isolate spatial location.",
        "The 49 equal-square interventions provide shape-matched spatial comparisons with two replacements.",
        "Chroma/static-bright masks are not verified text/calipers; grid patches are not anatomical annotations.",
        "Readout-weight permutation is a limited head check, not complete network randomization.",
        "More full-frame context can contain both anatomy and non-anatomical acquisition cues.",
        "All original heldout folds here are historically inspected training-source folds, not a new independent test."]
    checks, parity, pixel_checks = 0, {}, 0
    spatial_border = np.array([i // 7 in (0, 6) or i % 7 in (0, 6) for i in range(49)])
    for h in (0, 1):
        F = a.Fold(c, 0, h)
        cc = [q for q in cases if q["fold"] == h]
        rows, anchors = z[f"rows_f{h}"], z[f"anchors_f{h}"]
        assert rows.tolist() == sorted({r for q in cc for r in q["rows"]})
        assert anchors.tolist() == sorted({q["anchor"] for q in cc})
        rpos, apos = {int(v): j for j, v in enumerate(rows)}, {int(v): j for j, v in enumerate(anchors)}
        pixel_meta = {q["row"]: q for q in metadata[str(h)]["rows"]}
        anchor_meta = {q["row"]: q for q in metadata[str(h)]["anchors"]}
        y, base, pred = (old[f"{k}_heldout_f{h}"] for k in ("run_label", "run_pred_Base", "run_pred_N"))
        # Reconstruct sample-level nearest-centre membership independently of stored run offsets/weights.
        for q in cc:
            i = q["run"]; seq = int(F.pk["seq"][i]); lo, hi = int(F.pk["a"][i]), int(F.pk["b"][i])
            rr = np.flatnonzero(F.W["seq"] == seq); centers = F.W["centre"][rr]
            choices = [int(rr[np.argmin(abs(centers - t))]) for t in range(lo, hi)]
            counts = Counter(choices)
            assert sorted(counts) == q["rows"]
            assert np.allclose([counts[r] / (hi - lo) for r in q["rows"]], q["weights"], atol=1e-12, rtol=0)
            expected = ("A5C_recovered" if pred[i] == 3 else "A5C_missed") if y[i] == 3 else ("A4C_harmed" if pred[i] == 3 else "A4C_retained")
            assert base[i] == 2 and q["group"] == expected
            checks += 1
        # Reproduce metadata-only cohort order using a separately implemented preference loop.
        sources, sessions = [], []
        for i in range(len(base)):
            sq = F.seqs[int(F.pk["seq"][i])]
            ct = Counter(r[0] for r in sq["refs"][int(F.pk["a"][i]):int(F.pk["b"][i])])
            source = min(ct, key=lambda s: (-ct[s], s))
            sources.append(source); sessions.append(a.session_of(source))
        for gi, group in enumerate(GROUPS):
            is5, ispos = group.startswith("A5C"), group in ("A5C_recovered", "A4C_harmed")
            eligible = np.flatnonzero((base == 2) & (y == (3 if is5 else 2)) & ((pred > 2) == ispos))
            assert len(eligible) == manifest["groups"][str(h)][group]["denominator"]
            perm = np.random.default_rng([20261031, 51, h, gi]).permutation(eligible).tolist()
            selected = []
            for level in ("session", "recording", "remaining"):
                for i in perm:
                    if len(selected) >= 8: break
                    if i in selected: continue
                    field = sessions if level == "session" else sources
                    if level != "remaining" and field[i] in {field[v] for v in selected}: continue
                    selected.append(i)
            assert selected == [q["run"] for q in cc if q["group"] == group]
        for model in MODELS:
            g = z[f"global_{model}_f{h}"]; maps = z[f"maps_{model}_f{h}"]; temporal = z[f"temporal_{model}_f{h}"]
            assert g.shape == (len(rows), 11, 2) and maps.shape == (len(anchors), 2, 49, 2)
            assert temporal.shape == (len(anchors), 4, 2)
            assert np.isfinite(g).all() and np.isfinite(maps).all() and np.isfinite(temporal).all()
            if model == "A08":
                O = a.t31.outputs(c.a31, 0, h); expected = O["logits"][rows, 3] - O["logits"][rows, 2]
            else:
                prior_rows = old[f"rows_heldout_f{h}"].tolist()
                expected = old[f"scores_{model}_heldout_f{h}"][[prior_rows.index(int(r)) for r in rows]]
            parity[f"{model}_f{h}"] = assert_original_parity(g[:, 0, 0], expected)
            weighted = np.stack([reference_run_scores(rows, g, q["rows"], q["weights"]) for q in cc])
            for group in GROUPS:
                at = [j for j, q in enumerate(cc) if q["group"] == group]
                key = f"{model}_fold{h}_{group}"
                summary["global_interventions"][key] = {name: effect_counts(weighted[at, 0, 0], weighted[at, j, 0]) for j, name in enumerate(NAMES)}
                summary["global_interventions"][key]["original_scores"] = stats(weighted[at, 0, 0])
                summary["global_interventions"][key]["full_minus_retained_same_scale"] = stats(weighted[at, 9, 0] - weighted[at, 10, 0])
                for j, name in ((1, "periphery"), (3, "chroma"), (5, "static_bright")):
                    summary["global_interventions"][key][name + "_drop_excess_over_control"] = stats(weighted[at, j + 1, 0] - weighted[at, j, 0])
                features = []
                for j in at:
                    q = cc[j]; pos = apos[q["anchor"]]; rp = rpos[q["anchor"]]
                    drop = g[rp, 0][None, None, :] - maps[pos]
                    td = g[rp, 0, 0] - temporal[pos, :, 0]
                    top = np.argsort(abs(drop[0, :, 0]))[-5:]
                    top_blur = np.argsort(abs(drop[1, :, 0]))[-5:]
                    features.append({"fill_correlation": correlation(drop[0, :, 0], drop[1, :, 0]),
                                     "head_permutation_correlation_mean": correlation(drop[0, :, 0], drop[0, :, 1]),
                                     "head_permutation_correlation_blur": correlation(drop[1, :, 0], drop[1, :, 1]),
                                     "top5_fill_overlap": len(set(top) & set(top_blur)) / 5,
                                     "mean_fill_outer_abs_drop": float(abs(drop[0, spatial_border, 0]).mean()),
                                     "mean_fill_inner_abs_drop": float(abs(drop[0, ~spatial_border, 0]).mean()),
                                     "blur_fill_outer_abs_drop": float(abs(drop[1, spatial_border, 0]).mean()),
                                     "blur_fill_inner_abs_drop": float(abs(drop[1, ~spatial_border, 0]).mean()),
                                     "top_mean_patch_outer": bool(spatial_border[abs(drop[0, :, 0]).argmax()]),
                                     "temporal_max_abs_drop": float(abs(td).max()),
                                     "strongest_temporal_block_contains_marker": int(abs(td).argmax()) == anchor_meta[q["anchor"]]["marker"] // 4})
                summary["spatial_temporal"][key] = {name: stats([f[name] for f in features]) for name in features[0]}
        for group in GROUPS:
            qq = [q for q in cc if q["group"] == group]
            fields = [k for k, v in next(iter(pixel_meta.values())).items() if isinstance(v, (float, int)) and k != "row"]
            summary["input_proxies"][f"fold{h}_{group}"] = {field: stats([sum(w * pixel_meta[r][field] for r, w in zip(q["rows"], q["weights"])) for q in qq]) for field in fields}
            summary["input_proxies"][f"fold{h}_{group}"]["composition"] = {
                "native_runs": sum(F.seqs[int(F.pk["seq"][q["run"]])]["kind"] == "native" for q in qq),
                "reference_pure_runs": sum(float(F.pk["purity"][q["run"]]) >= 1 for q in qq),
                "duration_samples": stats([int(F.pk["b"][q["run"]] - F.pk["a"][q["run"]]) for q in qq]),
                "distinct_majority_sessions": len({q["session"] for q in qq})}
        # Fixed subset independent pixel reconstruction; no repeated model inference.
        audit_global = {int(rows[0]), int(rows[len(rows) // 2]), int(rows[-1])}
        for row in sorted(audit_global | set(map(int, anchors))):
            x, full, retained = reference_pixels(F, row)
            assert hashlib.sha256(x.tobytes()).hexdigest() == pixel_meta[row]["pixel_sha256"]
            pixel_checks += 1
            if row in audit_global:
                vv = reference_variants(x, full, retained, int(F.W["marker"][row]), row + 100000 * h)
                assert tuple(vv) == NAMES
                for name, v in vv.items():
                    assert hashlib.sha256(v.tobytes()).hexdigest() == pixel_meta[row]["global_input_hashes"][name]
                    pixel_checks += 1
            if row in anchor_meta:
                for fill in ("mean", "blur"):
                    replacement = np.broadcast_to(np.rint(x.mean((1, 2), keepdims=True)).astype(np.uint8), x.shape) if fill == "mean" else np.stack([np.asarray(Image.fromarray(v).filter(ImageFilter.BoxBlur(8))) for v in x])
                    for cell in (0, 24, 48):
                        yy, xx = divmod(cell, 7); v = x.copy()
                        v[:, yy * 32:(yy + 1) * 32, xx * 32:(xx + 1) * 32] = replacement[:, yy * 32:(yy + 1) * 32, xx * 32:(xx + 1) * 32]
                        assert hashlib.sha256(v.tobytes()).hexdigest() == anchor_meta[row]["selected_spatial_input_hashes"][f"{fill}_{cell}"]
                        pixel_checks += 1
        # Local-only, explicitly unadjudicated contact sheets and sensitivity maps.
        for j, row in enumerate(anchors):
            canvas = Image.new("RGB", (3 * 224, 2 * 248), "black")
            from PIL import ImageDraw
            draw = ImageDraw.Draw(canvas)
            for mi, model in enumerate(MODELS):
                rp = rpos[int(row)]
                drop = z[f"global_{model}_f{h}"][rp, 0, 0] - z[f"maps_{model}_f{h}"][j, :, :, 0]
                scale = max(float(abs(drop).max()), 1e-8)
                for fi, fill in enumerate(("mean", "blur")):
                    d = (drop[fi].reshape(7, 7) / scale).clip(-1, 1)
                    pic = np.zeros((7, 7, 3), np.uint8)
                    pic[..., 0] = np.rint(np.maximum(d, 0) * 255).astype(np.uint8)
                    pic[..., 2] = np.rint(np.maximum(-d, 0) * 255).astype(np.uint8)
                    canvas.paste(Image.fromarray(pic).resize((224, 224), Image.Resampling.NEAREST), (224 * mi, 248 * fi + 24))
                    draw.text((224 * mi + 4, 248 * fi + 4), model + " " + fill + "; red=score drop", fill="white")
            canvas.save(OUT / "private_review" / f"fold{h}_anchor{j:02d}_maps.png")
    summary["cohort_sources"] = {"runs": len(cases), "distinct_majority_recordings": len({q["source"] for q in cases}),
                                 "distinct_majority_sessions": len({q["session"] for q in cases}),
                                 "source_overlap_between_groups_possible": True, "sessions_are_not_verified_patients": True}
    html = ["<!doctype html><meta charset='utf-8'><title>Private unadjudicated image-dependence review</title>",
            "<style>body{background:#181818;color:#eee;font:16px system-ui}article{border:1px solid #555;padding:16px;margin:16px}img{max-width:48%;vertical-align:top}</style>",
            "<h1>Private exploratory input-dependence review</h1><p>Source identifiers, frames and per-case outputs must remain on this server. No expert image-quality, LVOT or A5C visibility adjudication has occurred. Maps are score sensitivity, not anatomical truth.</p>"]
    for h in (0, 1):
        aa = z[f"anchors_f{h}"].tolist()
        for j, row in enumerate(aa):
            related = [q["group"] for q in cases if q["fold"] == h and q["anchor"] == row]
            html.append(f"<article><h2>Fold {h}; anchor {j}; {', '.join(related)}</h2><img src='fold{h}_case{j:02d}.jpg'><img src='fold{h}_anchor{j:02d}_maps.png'></article>")
    (OUT / "private_review/index.html").write_text("\n".join(html))
    geometry_receipt = audit_geometry(c, z, cases) if (OUT / "geometry_complete.json").exists() else None
    receipt = {"independent_cache_audit_pass": True, "runs_checked": checks, "parity_max_abs": parity,
               "pixel_reconstructions_checked": pixel_checks, "new_model_forwards": 0,
               "scores_sha256": digest(OUT / "scores.npz"), "all_pinned_files_unchanged": True,
               "scope": "all membership/cohort/score summaries; fixed original/global/spatial pixel subset; no anatomical or clinical-quality adjudication",
               "geometry_supplement": geometry_receipt,
               "wall_seconds": time.time() - start, "process_CPU_seconds": time.process_time() - cpu}
    PUB.mkdir(parents=True, exist_ok=True)
    for p, obj in [(PUB / "summary.json", summary), (PUB / "independent_audit.json", receipt), (OUT / "independent_audit.json", receipt)]:
        p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
