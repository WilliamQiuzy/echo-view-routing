#!/usr/bin/env python
"""a5c_expert_pilot_v1 (configs/experiments/a5c_expert_pilot_v1.yaml): blind expert pilot on A32's 24 fixed cases.

  --stage audit      per source recording: ffprobe / decoded PTS, frame counts, decoded mp4 vs extracted Images frames,
                     A32 10 Hz references -> original frames (server; private audit + id-free summary)
  --stage build      new private package: decoded original frames, blind case manifest, reveal layer, private key, links
  --stage readers    one random token per anonymous reader (R1, R2) for the real store; prints the tunnel / URL recipe
  --stage synthetic  synthetic media package (constant, variable-rate, 3-frame and 10 Hz-only recordings) + its tokens
  --stage techcheck  2 real recordings by a fixed rule: timeline, served frames, leak scan, isolation and export, with a
                     technical record in the synthetic tree only (no medical field filled)
  --stage status     record counts per store and reader (no content)
  --stage overlay    real records -> private overlay (validated); with --synthetic, the synthetic store
The A32 package is read only; nothing here trains or runs a model.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # no GPU in this round

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import secrets  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import a5c_pilot_records as rec  # noqa: E402
import a5c_pilot_server as srv  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

A32_RUN = ROOT / "runs/a5c_source_external_evidence_v1"
A32_KEY = A32_RUN / "review_package/private_key.json"
FAMILY5 = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]


class Ctx:
    def __init__(self, config: str) -> None:
        self.v = yaml.safe_load((ROOT / config).read_text())["a5c_expert_pilot"]; o = self.v["outputs"]
        self.out, self.syn, self.results, self.ledger = ROOT / o["root"], ROOT / o["synthetic_root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.out.mkdir(parents=True, exist_ok=True)


def ledger(ctx: Ctx, job: str, started: str, sec: float, ok: bool, note: str) -> None:
    with open(ctx.ledger, "a") as fh:
        fh.write(json.dumps({"job": job, "kind": "cpu", "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hours": sec / 3600, "ok": ok, "note": note}) + "\n")


def staged(ctx: Ctx, job: str, fn) -> None:
    t0, started, ok, note = time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), False, ""
    try:
        note = fn() or ""; ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger(ctx, job, started, time.time() - t0, ok, note)


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def blind_ids(recordings: list[str]) -> dict:
    rs = sorted(recordings); perm = np.random.default_rng([20261025, 33, 1]).permutation(len(rs)); return {rs[i]: f"P{j + 1:02d}" for j, i in enumerate(perm)}


def a32_cases() -> dict:
    return json.loads(A32_KEY.read_text())


# ----------------------------------------------------------------------------------------------- media
def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,nb_frames,duration,start_time",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


def decode(path: Path) -> tuple[list, list, str]:
    """Every decoded frame (RGB) with its PTS time; the stream time base as text."""
    import av  # noqa: E402
    frames, times = [], []
    with av.open(str(path)) as c:
        s = c.streams.video[0]; tb = f"{s.time_base.numerator}/{s.time_base.denominator}"
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24")); times.append(float(f.pts * s.time_base) if f.pts is not None else float("nan"))
    return frames, times, tb


def axis_sha(times: list, frames: list) -> str:
    h = hashlib.sha256(json.dumps([round(t, 6) for t in times]).encode())
    for f in frames:
        h.update(hashlib.md5(f.tobytes()).digest())
    return h.hexdigest()


def rate_info(times: list) -> dict:
    d = np.diff(np.asarray(times)) if len(times) > 1 else np.asarray([np.nan])
    ok = bool(len(times) > 1 and np.isfinite(d).all() and (d > 0).all())
    return {"fps_from_pts": float(round(1 / np.median(d), 4)) if ok else None, "constant_rate": bool(ok and np.ptp(d) < 1e-6), "monotonic": ok, "dt_min": float(np.min(d)) if ok else None, "dt_max": float(np.max(d)) if ok else None}


def images_compare(frames: list, img_paths: list) -> dict:
    """Mean absolute difference of each decoded frame against the extracted JPEG of the same index and of its neighbours."""
    n = min(len(frames), len(img_paths)); same, best, worst = [], 0, 0.0
    for k in range(n):
        im = np.asarray(Image.open(img_paths[k]).convert("RGB"), np.float32)
        d = {j: float(np.abs(np.asarray(frames[j], np.float32) - im).mean()) for j in (k - 1, k, k + 1) if 0 <= j < len(frames) and frames[j].shape == im.shape}
        if k in d:
            same.append(d[k]); worst = max(worst, d[k]); best += int(min(d, key=d.get) == k or d[k] <= min(d.values()) + 1e-6)
    return {"compared": n, "mae_same_index_mean": float(np.mean(same)) if same else None, "mae_same_index_max": worst if same else None, "same_index_is_best_match": int(best)}


def recordings_of(cases: dict) -> dict:
    by = {}
    for cid, c in cases.items():
        by.setdefault(c["recording"], []).append(cid)
    return by


def audit(ctx: Ctx) -> str:
    from echo_routing.config import load_paths  # noqa: E402
    from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
    import scripts.diagnose_a5c_evidence as d32  # noqa: E402
    paths = load_paths(); cases = a32_cases(); by = recordings_of(cases); pid = blind_ids(list(by)); data = d32.t31.vpr.training_data(_a31_ctx().a15.a10.a07p)
    priv, summ = {}, {"cases": len(cases), "unique_recordings": len(by), "sessions": len({c["session"] for c in cases.values()}), "recordings_in_several_sets": sum(len({cases[c]["set"] for c in v}) > 1 for v in by.values()),
                      "per_set_cases": dict(Counter(c["set"] for c in cases.values())), "stream_types": {}, "recordings": {}}
    pools = {}
    for v, cids in sorted(by.items()):
        mp4 = paths.ev9v_images.parent / "Videos" / f"{v}.mp4"; imgs = list_frame_paths(paths.ev9v_images / v); r = {"cases": cids, "has_original": mp4.is_file(), "images": len(imgs)}
        if mp4.is_file():
            pr = ffprobe(mp4); frames, times, tb = decode(mp4)
            r.update({"ffprobe": pr, "decoded_frames": len(frames), "time_base": tb, **rate_info(times), "first_pts_time": times[0], "last_pts_time": times[-1], "compare": images_compare(frames, imgs), "axis_sha256": axis_sha(times, frames)})
        fidx = np.asarray(data.frame_idx(v)); r["cached_10hz_samples"] = int(len(fidx)); r["cached_frame_step"] = sorted(Counter(np.diff(fidx).tolist()).items()) if len(fidx) > 1 else []
        uses = []
        for cid in cids:
            c = cases[cid]; key = (c["seed"], c["heldout_fold"])
            if key[1] not in pools:
                pools[key[1]] = {q["id"]: q for q in d32.t31.load_pool(_a31_ctx(), key[1])}
            q = pools[key[1]][c["stream"]]; a, b = c["run"]; fr = [int(x[1]) for x in q["refs"] if x[0] == v]; fr_run = [int(x[1]) for x in q["refs"][a:b] if x[0] == v]
            uses.append({"case": cid, "set": c["set"], "stream_type": q["kind"], "stream_samples": len(q["refs"]), "recording_samples_in_stream": len(fr), "run_samples": b - a, "run_samples_from_recording": len(fr_run),
                         "refs_beyond_last_image": int(sum(f >= len(imgs) for f in fr)), "refs_on_cached_axis": bool(set(fr) <= set(fidx.tolist())), "run_source_frames": [min(fr_run), max(fr_run)] if fr_run else None})
        r["uses"] = uses; priv[v] = {"pid": pid[v], **r}
        summ["recordings"][pid[v]] = {k: r.get(k) for k in ("has_original", "images", "decoded_frames", "time_base", "fps_from_pts", "constant_rate", "monotonic", "dt_min", "dt_max", "cached_10hz_samples", "cached_frame_step")} | {
            "nominal_fps": r.get("ffprobe", {}).get("r_frame_rate"), "avg_fps": r.get("ffprobe", {}).get("avg_frame_rate"), "codec": r.get("ffprobe", {}).get("codec_name"), "size": f"{r.get('ffprobe', {}).get('width')}x{r.get('ffprobe', {}).get('height')}",
            "compare": r.get("compare"), "uses": [{k: u[k] for k in ("stream_type", "recording_samples_in_stream", "run_samples", "run_samples_from_recording", "refs_beyond_last_image", "refs_on_cached_axis")} for u in uses]}
    summ["stream_types"] = dict(Counter(u["stream_type"] for r in priv.values() for u in r["uses"]))
    write_json(ctx.out / "audit_private.json", priv); flat = json.loads(json.dumps(summ, default=float)); _no_ids(flat, priv); write_json(ctx.out / "media_audit.json", flat)
    agg = audit_summary(flat); _no_ids(agg, priv); write_json(ctx.results / "media_audit_summary.json", agg)       # the repo gets aggregates only
    print(json.dumps({p: {k: x[k] for k in ("has_original", "images", "decoded_frames", "fps_from_pts", "constant_rate")} for p, x in summ["recordings"].items()}, indent=0)); return f"{len(by)} recordings"


def audit_summary(a: dict) -> dict:
    import statistics as st  # noqa: E402
    R = list(a["recordings"].values()); fr = [x["decoded_frames"] for x in R]; s10 = [x["cached_10hz_samples"] for x in R]; uses = [u for x in R for u in x["uses"]]; con = [u for u in uses if u["stream_type"] == "multi"]
    return {"cases": a["cases"], "unique_recordings": a["unique_recordings"], "sessions": a["sessions"], "recordings_in_several_sets": a["recordings_in_several_sets"], "per_set_cases": a["per_set_cases"],
            "stream_types_of_cases": a["stream_types"], "with_original_video": sum(x["has_original"] for x in R), "codec": sorted({x["codec"] for x in R}), "size": sorted({x["size"] for x in R}),
            "nominal_fps": sorted({x["nominal_fps"] for x in R}), "avg_fps": sorted({x["avg_fps"] for x in R}), "time_base": sorted({x["time_base"] for x in R}), "fps_from_pts": sorted({x["fps_from_pts"] for x in R}),
            "constant_rate": sum(x["constant_rate"] for x in R), "monotonic_pts": sum(x["monotonic"] for x in R), "decoded_equals_extracted_count": sum(x["images"] == x["decoded_frames"] for x in R),
            "frames": {"min": min(fr), "median": st.median(fr), "max": max(fr), "total": sum(fr)}, "cached_10hz_samples": {"min": min(s10), "max": max(s10)},
            "cached_step_original_frames": sorted({k for x in R for k, _ in x["cached_frame_step"]}),
            "decoded_vs_extracted_mae": {"per_recording_mean_min": round(min(x["compare"]["mae_same_index_mean"] for x in R), 3), "per_recording_mean_max": round(max(x["compare"]["mae_same_index_mean"] for x in R), 3),
                                         "per_frame_max": round(max(x["compare"]["mae_same_index_max"] for x in R), 3)},
            "same_index_best_match": f"{sum(x['compare']['same_index_is_best_match'] for x in R)}/{sum(x['compare']['compared'] for x in R)}",
            "a32_refs_on_cached_axis": f"{sum(u['refs_on_cached_axis'] for u in uses)}/{len(uses)}", "a32_refs_beyond_last_frame": sum(u["refs_beyond_last_image"] for u in uses),
            "constructed_samples_used_from_recording": {"min": min(u["recording_samples_in_stream"] for u in con), "max": max(u["recording_samples_in_stream"] for u in con)}}


_A31 = {}


def _a31_ctx():
    if not _A31:
        import scripts.diagnose_a5c_evidence as d32  # noqa: E402
        from scripts._common import load_cfg  # noqa: E402
        from types import SimpleNamespace  # noqa: E402
        _A31["c"] = d32.t31.Ctx(load_cfg(SimpleNamespace(config=d32.A31_CONFIG, set=[])), d32.A31_CONFIG)
    return _A31["c"]


def _no_ids(obj, priv: dict) -> None:
    s = json.dumps(obj); bad = [v for v in priv if v in s] + [x for x in {v.rsplit("_", 1)[0] for v in priv} if x in s]
    if bad:
        raise SystemExit("a repo-bound summary carries a private id")


# ----------------------------------------------------------------------------------------------- package
def build(ctx: Ctx) -> str:
    import scripts.diagnose_a5c_evidence as d32  # noqa: E402
    import scripts.report_raw_video_context as rr  # noqa: E402
    from echo_routing.config import load_paths  # noqa: E402
    priv = json.loads((ctx.out / "audit_private.json").read_text()); cases = a32_cases(); paths = load_paths(); pkg = ctx.out / "package"
    if pkg.exists():
        raise SystemExit("the package exists; it is built once (remove it deliberately to rebuild)")
    data = d32.t31.vpr.training_data(_a31_ctx().a15.a10.a07p); man = {"synthetic": False, "order": sorted(x["pid"] for x in priv.values()), "cases": {}}; reveal, key, links = {}, {}, {}
    folds = {}; ro = np.load(A32_RUN / "readout_private.npy", allow_pickle=True).item(); dctx = d32.Ctx(d32.load_cfg(_ns("configs/experiments/a5c_source_external_evidence_v1.yaml")), "configs/experiments/a5c_source_external_evidence_v1.yaml")
    for v, r in sorted(priv.items()):
        pid = r["pid"]; mdir = pkg / "media" / pid; mdir.mkdir(parents=True)
        if r["has_original"]:
            frames, times, tb = decode(paths.ev9v_images.parent / "Videos" / f"{v}.mp4")
            if axis_sha(times, frames) != r["axis_sha256"]:
                raise SystemExit("decoded frames differ from the audit")
            for k, f in enumerate(frames):
                Image.fromarray(f).save(mdir / f"{k:06d}.png")
            src = "original mp4 decode"; note = f"one step = one original frame ({r['ffprobe']['r_frame_rate']} nominal)"
        else:
            raise SystemExit("a pilot recording has no original video; the 10 Hz-only path is not wired for real cases")
        man["cases"][pid] = {"pid": pid, "n_frames": len(frames), "time_base": tb, "pts_time": [round(t, 6) for t in times], "fps_nominal": r["ffprobe"]["r_frame_rate"], "fps_from_pts": r["fps_from_pts"],
                             "constant_rate": r["constant_rate"], "source": src, "sha256": r["axis_sha256"], "resolution_note": note, "frame_ext": "png"}
        uses = []
        for u in r["uses"]:
            c = cases[u["case"]]; s, h = c["seed"], c["heldout_fold"]
            if (s, h) not in folds:
                D = d32.fold_data(dctx, s, h); W = d32.t31.pool_W(D["seqs"]); O = d32.t31.outputs(dctx.a31, s, h); folds[(s, h)] = (D, rr.per_sample(W, O["logits"], D["seqs"]), d32.external_probs(dctx, D["seqs"], h))
            D, p08, ext = folds[(s, h)]; pk = D["pk"]; i = next(j for j, q in enumerate(D["seqs"]) if q["id"] == c["stream"]); a, b = c["run"]
            k = next(j for j, kk in enumerate(D["key"]) if kk == (i, a, b)); q = D["seqs"][i]; own = [j for j, x in enumerate(q["refs"]) if x[0] == v]
            fr = [int(q["refs"][j][1]) for j in own]
            prob = {"source_frame": fr, **{m: {"A4C": [round(float(P[q["id"]][j][2]), 4) for j in own], "A5C": [round(float(P[q["id"]][j][3]), 4) for j in own]} for m, P in (("A08 + B3 (A08 probabilities)", p08), ("EchoViewCLIP", ext["EchoViewCLIP"]), ("EchoPrime", ext["EchoPrime"]))}}
            run_fr = [int(x[1]) for x in q["refs"][a:b] if x[0] == v]
            uses.append({"stream_type": q["kind"], "model_seed": s, "target_run_source_frames": [min(run_fr), max(run_fr)] if run_fr else None, "run_reference_label": FAMILY5[int(pk["labels"][k])],
                         "A08_B3": FAMILY5[int(pk["base"][k])], **{f"A31_{x}": FAMILY5[int(D["arms"][x][k])] for x in d32.ARMS}, **{f"{m}_rule": FAMILY5[int(ro[(s, h)][m][k])] for m in d32.MODELS},
                         "probabilities_on_this_recording": prob, "model_window_note": "the model saw 16 samples around each position; the run and windows are for review after the blind pass, not truth"})
            key.setdefault(pid, {"recording": v, "session": v.rsplit("_", 1)[0], "a32_cases": [], "fold": h})["a32_cases"].append({"case": u["case"], "set": c["set"], "seed": s, "stream": c["stream"], "run": c["run"]})
        reveal[pid] = {"public_clip_code": FAMILY5[int(data.label[v])], "label_nature": "one clip-level code for the whole recording (EV9V); no frame-level or structure annotation exists", "uses_in_A32": uses}
        links[pid] = {"fold": key[pid]["fold"], "sample_frames": [int(x) for x in data.frame_idx(v)]}
    write_json(pkg / "cases.json", man); write_json(pkg / "reveal.json", reveal); write_json(ctx.out / "private_key.json", key); write_json(ctx.out / "links.json", links)
    for f in (pkg / "reveal.json", ctx.out / "private_key.json", ctx.out / "links.json"):
        os.chmod(f, 0o600)
    return f"{len(man['order'])} recordings, {sum(c['n_frames'] for c in man['cases'].values())} frames"


def _ns(config: str):
    from types import SimpleNamespace  # noqa: E402
    return SimpleNamespace(config=config, set=[])


def readers(ctx: Ctx, root: Path, synthetic: bool) -> dict:
    p = root / "readers.json"
    if p.is_file():
        return json.loads(p.read_text())
    toks = {"tokens": {secrets.token_urlsafe(18): r for r in ("R1", "R2")}, "synthetic": synthetic}; write_json(p, toks); os.chmod(p, 0o600); return toks


def access_recipe(root: Path, ann: Path, synthetic: bool, port: int) -> str:
    rel = lambda x: x.relative_to(ROOT)  # noqa: E731
    return (f"server: cd {ROOT} && .venv/bin/python scripts/a5c_pilot_server.py --package {rel(root / 'package')} --annotations {rel(ann)} --readers {rel(root / 'readers.json')}{' --synthetic' if synthetic else ''} --port {port}\n"
            f"local:  ssh -N -L {port}:127.0.0.1:{port} echo-l40s   then open http://127.0.0.1:{port}/?t=<reader token from {rel(root / 'readers.json')}>")


# ----------------------------------------------------------------------------------------------- synthetic
def synthetic(ctx: Ctx) -> str:
    """Numbered frames on known axes: constant 30 fps (40 frames), variable rate (25 frames), 3 frames, 10 Hz-only (12)."""
    from PIL import ImageDraw  # noqa: E402
    root = ctx.syn; pkg = root / "package"; specs = {"Z01": ("constant", 40), "Z02": ("variable", 25), "Z03": ("constant", 3), "Z04": ("tenhz", 12)}
    man = {"synthetic": True, "order": list(specs), "cases": {}}; links = {}
    for pid, (kind, n) in specs.items():
        d = pkg / "media" / pid; d.mkdir(parents=True, exist_ok=True); frames = []
        for k in range(n):
            im = Image.new("RGB", (160, 120), (20 + 5 * k % 200, 40, 60)); ImageDraw.Draw(im).text((10, 50), f"SYNTHETIC {pid} {k}", fill=(255, 255, 255)); im.save(d / f"{k:06d}.png"); frames.append(np.asarray(im))
        times = [k / 30 for k in range(n)] if kind == "constant" else ([sum(1 / 30 * (1 + (j % 3 == 0)) for j in range(k)) for k in range(n)] if kind == "variable" else [k / 10 for k in range(n)])
        ri = rate_info(times)
        man["cases"][pid] = {"pid": pid, "n_frames": n, "time_base": "1/15360" if kind != "tenhz" else "1/10", "pts_time": [round(t, 6) for t in times], "fps_nominal": "30/1" if kind != "tenhz" else "10/1",
                             "fps_from_pts": ri["fps_from_pts"], "constant_rate": ri["constant_rate"], "source": "10 Hz cache only" if kind == "tenhz" else "synthetic decode", "sha256": axis_sha(times, frames),
                             "resolution_note": "10 Hz cache only: one step = one 10 Hz sample" if kind == "tenhz" else "one step = one frame", "frame_ext": "png"}
        links[pid] = {"fold": 0, "sample_frames": list(range(0, n, 3)) if kind != "tenhz" else list(range(n))}
    write_json(pkg / "cases.json", man); write_json(pkg / "reveal.json", {p: {"public_clip_code": "SYNTHETIC", "uses_in_A32": []} for p in specs}); write_json(root / "links.json", links)
    tok = readers(ctx, root, True); fill = synthetic_fill(root, tok); print(access_recipe(root, root / "annotations", True, 8766)); print(json.dumps(fill, indent=1))
    return f"{len(specs)} synthetic recordings; {fill['records']} synthetic records"


def synthetic_fill(root: Path, tok: dict, port: int = 18766) -> dict:
    """Two synthetic readers through the HTTP API: drafts, blind submit, a refused blind edit, unblinding, a post-unblinding
    version, and readers who disagree; every record is flagged synthetic and lives only under the synthetic root."""
    app = srv.App(root / "package", root / "annotations", root / "readers.json", True); httpd = srv.serve(app, port); threading.Thread(target=httpd.serve_forever, daemon=True).start()
    t = {r: k for k, r in tok["tokens"].items()}; out = {"steps": {}}
    try:
        for reader, view, lv in (("R1", "A5C", [{"start": 2, "end": 9, "state": "visible"}]), ("R2", "A4C", [{"start": 4, "end": 7, "state": "uncertain"}])):
            for pid in ("Z01", "Z02", "Z03", "Z04"):
                n = app.cases[pid]["n_frames"]; ivs = [x for x in lv if x["end"] < n]
                f = {"overall_view": view, "overall_confidence": "low", "view_intervals": [{"start": 0, "end": n - 1, "state": "transitional_or_off_axis" if pid == "Z02" else view}],
                     "lvot": {"status": "intervals", "intervals": ivs} if ivs else {"status": "not_visible_anywhere", "intervals": []}, "aortic_valve": {"status": "cannot_judge", "intervals": []},
                     "image_quality": "adequate", "quality_reason": "synthetic", "comments": "SYNTHETIC engineering record"}
                st1, _ = http(port, f"/api/save/{pid}", t[reader], {"phase": "blind", "submit": False, "fields": {"comments": "draft"}})
                st2, _ = http(port, f"/api/save/{pid}", t[reader], {"phase": "blind", "submit": True, "fields": f})
                st3, _ = http(port, f"/api/save/{pid}", t[reader], {"phase": "blind", "submit": False, "fields": f})
                out["steps"][f"{reader}/{pid}"] = [st1, st2, st3]
        st4, _ = http(port, "/api/unblind/Z01", t["R1"], {}); st5, b5 = http(port, "/api/reveal/Z01", t["R1"]); st6, _ = http(port, "/api/reveal/Z01", t["R2"])
        f2 = json.loads(http(port, "/api/case/Z01", t["R1"])[1])["record"]["versions"][-1]["fields"]; f2.update({"label_fit": "partly", "review_comment": "SYNTHETIC post-unblinding"})
        st7, _ = http(port, "/api/save/Z01", t["R1"], {"phase": "unblinded", "submit": True, "fields": f2})
        out["unblind"] = {"unblind": st4, "reveal_R1": st5, "reveal_R2_refused": st6, "post_unblinding_submit": st7}
    finally:
        httpd.shutdown()
    recs = [json.loads(p.read_text()) for p in sorted((root / "annotations").rglob("*.json"))]; out["records"] = len(recs)
    out["all_flagged_synthetic"] = all(r["synthetic"] for r in recs); out["blind_versions_kept"] = all(rec.blind_version(r) is not None for r in recs); return out


# ----------------------------------------------------------------------------------------------- technical check on 2 real recordings
def http(port: int, path: str, token: str, body: dict | None = None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}{'&' if '?' in path else '?'}t={token}", data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def techcheck(ctx: Ctx, port: int = 18765) -> str:
    from echo_routing.config import load_paths  # noqa: E402
    key = json.loads((ctx.out / "private_key.json").read_text()); man = json.loads((ctx.out / "package/cases.json").read_text())
    pick = [man["order"][i] for i in sorted(np.random.default_rng([20261025, 33, 3]).choice(len(man["order"]), 2, replace=False))]
    tdir = ctx.syn / "techcheck_real_axes"; tok = {"tokens": {secrets.token_urlsafe(12): "TC", secrets.token_urlsafe(12): "TD"}}; write_json(tdir / "readers.json", tok)
    app = srv.App(ctx.out / "package", tdir / "annotations", tdir / "readers.json", False); app.store = rec.Store(tdir / "annotations", True)    # real axes; technical records flagged synthetic, outside the real tree
    httpd = srv.serve(app, port); th = threading.Thread(target=httpd.serve_forever, daemon=True); th.start(); res = {"recordings_checked": len(pick), "checks": {}}
    t1, t2 = list(tok["tokens"]); forbidden = set()
    for p, k in key.items():
        forbidden |= {k["recording"], k["session"], *[x["case"] for x in k["a32_cases"]], *[x["stream"] for x in k["a32_cases"]], *[x["set"] for x in k["a32_cases"]]}
    try:
        st, body = http(port, "/", ""); ui = body.decode(); res["checks"]["ui_has_no_private_or_set_strings"] = not any(f in ui for f in forbidden) and "S1" not in ui and "S2" not in ui
        st, body = http(port, "/api/me", t1); me = json.loads(body); res["checks"]["me_lists_neutral_ids_only"] = set(me["order"]) == set(man["order"]) and not any(f in body.decode() for f in forbidden)
        for p in pick:
            v = key[p]["recording"]; c = man["cases"][p]; st, body = http(port, f"/api/case/{p}", t1); j = json.loads(body); txt = body.decode()
            ok_keys = set(j["case"]) <= set(srv.BLIND_KEYS); leak = any(f in txt for f in forbidden) or any(w in txt for w in ("public_clip_code", "A08", "EchoPrime", "EchoViewCLIP", "probab"))
            frames, times, _ = decode(load_paths().ev9v_images.parent / "Videos" / f"{v}.mp4"); same = []
            for kk in (0, c["n_frames"] - 1):
                st2, img = http(port, f"/media/{p}/{kk:06d}.png", t1)
                same.append(st2 == 200 and np.array_equal(np.asarray(Image.open(io.BytesIO(img)).convert("RGB")), frames[kk]))
            beyond, _ = http(port, f"/media/{p}/{c['n_frames']:06d}.png", t1)
            rv, _ = http(port, f"/api/reveal/{p}", t1)
            bad_iv, _ = http(port, f"/api/save/{p}", t1, {"phase": "blind", "submit": False, "fields": {"lvot": {"status": "intervals", "intervals": [{"start": 0, "end": c["n_frames"], "state": "visible"}]}}})
            rev_iv, _ = http(port, f"/api/save/{p}", t1, {"phase": "blind", "submit": False, "fields": {"view_intervals": [{"start": 5, "end": 2, "state": "A4C"}]}})
            st3, body3 = http(port, f"/api/save/{p}", t1, {"phase": "blind", "submit": False, "fields": {"comments": "technical check only; no medical judgement"}})
            other, body_o = http(port, f"/api/case/{p}", t2); iso = json.loads(body_o)["record"]["versions"] == []
            _, ej = http(port, "/api/export?fmt=json", t1); _, ec = http(port, "/api/export?fmt=csv", t1); rj = json.loads(ej)["records"]; rt = rec.canonical(rec.from_csv(ec.decode())) == rec.canonical(rj)
            res["checks"][f"recording_{len(res['checks'])}"] = {"frames": c["n_frames"], "pts_frames_match_decode": len(c["pts_time"]) == len(frames) and np.allclose(c["pts_time"], [round(t, 6) for t in times]),
                                                                 "first_and_last_served_frame_equal_decode": all(same), "frame_beyond_end_refused": beyond == 404, "blind_keys_allowlisted": ok_keys, "blind_response_leak": leak,
                                                                 "reveal_refused_before_unblind": rv == 403, "out_of_range_refused": bad_iv == 400, "reversed_refused": rev_iv == 400, "technical_draft_saved": st3 == 200,
                                                                 "other_reader_sees_nothing": iso, "export_json_csv_round_trip": rt, "fps_from_pts": c["fps_from_pts"], "constant_rate": c["constant_rate"]}
    finally:
        httpd.shutdown()
    res["real_store_records"] = sum(1 for _ in (ctx.out / "annotations/real").rglob("*.json")) if (ctx.out / "annotations/real").exists() else 0
    write_json(ctx.out / "techcheck.json", res); write_json(ctx.results / "techcheck.json", res); print(json.dumps(res, indent=1, default=str)); return "2 recordings"


def status(ctx: Ctx) -> str:
    out = {}
    for name, root in (("real", ctx.out / "annotations/real"), ("synthetic", ctx.syn / "annotations"), ("technical_on_real_axes", ctx.syn / "techcheck_real_axes/annotations")):
        recs = [json.loads(p.read_text()) for p in sorted(root.rglob("*.json"))] if root.exists() else []
        out[name] = {"records": len(recs), "by_reader": dict(Counter(r["reader"] for r in recs)), "by_state": dict(Counter(rec.state_of(r) for r in recs)), "flag_synthetic": dict(Counter(bool(r["synthetic"]) for r in recs))}
    write_json(ctx.out / "status.json", out); write_json(ctx.results / "status.json", out); print(json.dumps(out, indent=1)); return json.dumps({k: v["records"] for k, v in out.items()})


def overlay(ctx: Ctx, synthetic_store: bool) -> str:
    root = ctx.syn if synthetic_store else ctx.out; man = json.loads((root / "package/cases.json").read_text()); links = json.loads((root / "links.json").read_text())
    store = root / "annotations" if synthetic_store else ctx.out / "annotations/real"
    recs = [json.loads(p.read_text()) for p in sorted(store.rglob("*.json"))] if store.exists() else []
    ov = rec.build_overlay(recs, man["cases"], links); cov = rec.check_overlay(ov, man["cases"]); write_json(root / "overlay.json", ov)
    (root / "double_reading.csv").write_text(rec.rows_csv(rec.disagreement_rows(recs, man["cases"]), rec.DISAGREE_COLUMNS))
    print(json.dumps({"records": len(recs), "recordings_in_overlay": len(ov["recordings"]), "coverage": cov}, indent=1)); return f"{len(ov['recordings'])} recordings in the overlay"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/a5c_expert_pilot_v1.yaml"); ap.add_argument("--stage", required=True, choices=("audit", "build", "readers", "synthetic", "techcheck", "status", "overlay"))
    ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args(); ctx = Ctx(a.config)
    if a.stage == "readers":
        readers(ctx, ctx.out, False); print(access_recipe(ctx.out, ctx.out / "annotations/real", False, 8765)); return
    fn = {"audit": lambda: audit(ctx), "build": lambda: build(ctx), "synthetic": lambda: synthetic(ctx), "techcheck": lambda: techcheck(ctx), "status": lambda: status(ctx), "overlay": lambda: overlay(ctx, a.synthetic)}[a.stage]
    staged(ctx, f"{a.stage} (CPU)", fn)


if __name__ == "__main__":
    main()
