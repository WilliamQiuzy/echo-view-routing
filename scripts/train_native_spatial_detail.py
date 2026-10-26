#!/usr/bin/env python
"""A36 (configs/experiments/video_native_spatial_detail_v1.yaml): native spatial detail recognition screen.

  --stage check    input / encoder / head checks, 224 parity, 320-grid equivalence, Init = Base, timing and disk (GPU)
  --stage extract  --seed s: frozen A08 512-d embeddings of every request window of both pools for H and L (GPU)
  --stage train    --seed s: residual heads H_f0, L_f0, H_f1, L_f1 on A31's uniform plans (CPU, single thread)
  --stage readout  --seed s: after the heads are committed, one held-out readout (+ the source-2 rule for seed 0) (CPU)
  --stage coverage --seed s: recording / session coverage of the readout's fixed and broken runs (descriptive) (CPU)
The A08 + B3 path supplies q, e224, Base and the runs; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # A27's CPU-only trainer import would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.diagnose_a5c_evidence as d32  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_native_window_recognition as t29  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.temporal.spatial_detail as sd  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

A31_CONFIG, A32_CONFIG = "configs/experiments/video_session_correction_v1.yaml", "configs/experiments/a5c_source_external_evidence_v1.yaml"
FOLDS, ARMS, U, WARM, CHUNK_SEQS, BATCH = (0, 1), ("H", "L"), 1350, 135, 40, 16
now, write_json, ledger_append = t31.now, t31.write_json, t31.ledger_append


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["native_spatial_detail"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG); self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        note = fn() or ""; ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, job, kind, started, time.time() - t0, ok, note)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    flat = json.loads(json.dumps(rec, default=float)); d32.assert_no_ids(flat, d32.private_ids(ctx.d), name)
    for p in (ctx.out / name, ctx.results / name):
        p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)


def code_key() -> str:
    h = hashlib.sha256()
    for p in (Path(sd.__file__), Path(tm.__file__), Path(__file__)):
        h.update(p.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- frames and features
class PairFrames(Dataset):
    def __init__(self, refs: list, fp, chunk: int = 64) -> None:
        self.refs, self.fp, self.chunk = refs, fp, chunk

    def __len__(self) -> int:
        return (len(self.refs) + self.chunk - 1) // self.chunk

    def __getitem__(self, i: int):
        H, L = [], []
        for v, f, g in self.refs[i * self.chunk:(i + 1) * self.chunk]:
            p = self.fp(v); h, l_ = sd.load_pair(p[min(int(f), len(p) - 1)], g); H.append(h); L.append(l_)
        return torch.from_numpy(np.stack(H)), torch.from_numpy(np.stack(L))


def decode_pairs(refs: list, fp, workers: int) -> dict:
    out = {a: np.empty((len(refs), sd.CANVAS, sd.CANVAS, 3), np.uint8) for a in ARMS}; k = 0
    for h, l_ in DataLoader(PairFrames(refs, fp), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None):
        out["H"][k:k + len(h)] = h.numpy(); out["L"][k:k + len(h)] = l_.numpy(); k += len(h)
    if k != len(refs):
        raise SystemExit(f"decoded {k} of {len(refs)} frames")
    return out


def embed(m, frames: np.ndarray, W: dict, ks: np.ndarray, device) -> torch.Tensor:
    with torch.no_grad():
        return m.embed(tm.to_rgb(torch.from_numpy(frames[W["ref"][ks]]).to(device)), tm.marker_planes(torch.from_numpy(W["marker"][ks]).long().to(device), size=sd.CANVAS)).float()


def extract(ctx: Ctx, s: int, device, workers: int) -> str:
    m = sd.encoder320(ctx.a31.a08(s, device)); d0 = tet.state_digest(m); fdir = ctx.out / "feats"; fdir.mkdir(parents=True, exist_ok=True); acc = {"gpu_seconds": 0.0, "decode_seconds": 0.0}; rec = {}
    for f in FOLDS:
        seqs = t31.load_pool(ctx.a31, f); W = t31.pool_W(seqs); n = len(W["seq"]); fp = t28.frame_paths(seqs); E = {a: np.zeros((n, 512), np.float32) for a in ARMS}; row = 0
        for c0 in range(0, len(seqs), CHUNK_SEQS):
            ch = seqs[c0:c0 + CHUNK_SEQS]; Wc = t31.pool_W(ch); refs = sorted({tuple(r) for q in ch for r in q["refs"]}); a = time.time(); fr = decode_pairs(refs, fp, workers); acc["decode_seconds"] += time.time() - a
            for arm in ARMS:
                for b in range(0, len(Wc["seq"]), BATCH):
                    ks = np.arange(b, min(b + BATCH, len(Wc["seq"]))); torch.cuda.synchronize(); a = time.time(); e = embed(m, fr[arm], Wc, ks, device); torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a
                    E[arm][row + ks] = e.cpu().numpy()
            if not (np.array_equal(Wc["seq"] + c0, W["seq"][row:row + len(Wc["seq"])]) and np.array_equal(Wc["centre"], W["centre"][row:row + len(Wc["seq"])])):
                raise SystemExit("chunk windows differ from the pool's")
            row += len(Wc["seq"]); del fr
            print(f"seed {s} fold {f}: {row}/{n} windows (H and L); GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s; device {device}", flush=True)
        for arm in ARMS:
            np.save(fdir / f"e320_s{s}_{arm}_f{f}.npy", E[arm])
        rec[str(f)] = {"windows": n, **{f"{arm}_sha256": sha256_file(fdir / f"e320_s{s}_{arm}_f{f}.npy") for arm in ARMS}, "windows_seq_sha256": hashlib.sha256(W["seq"].tobytes() + W["centre"].tobytes()).hexdigest()}
    if tet.state_digest(m) != d0:
        raise SystemExit("the encoder changed during extraction")
    rec.update({"seed": s, "key": code_key(), "a08": t31.A08[s], "grid": list(sd.GRID), "time": now(), **acc}); write_json(fdir / f"record_s{s}.json", rec); return f"{sum(rec[str(f)]['windows'] for f in FOLDS)} windows x 2 arms"


def feats(ctx: Ctx, s: int, pinned: bool = False) -> dict:
    """pinned: accept the cache by the feature record committed with the heads (weights_s{s}.json) instead of the code key."""
    rec = json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text())
    if (json.loads((ctx.results / f"weights_s{s}.json").read_text())["feature_record"] != rec) if pinned else (rec["key"] != code_key()):
        raise SystemExit("the feature cache belongs to other code")
    out = {"q": [], "e224": [], "H": [], "L": [], "offset": {}}; o = 0
    for f in FOLDS:
        O = t31.outputs(ctx.a31, s, f); out["q"].append(O["logits"]); out["e224"].append(O["emb"]); out["offset"][f] = o; o += len(O["logits"])
        if int(O["offset"]) != out["offset"][f]:
            raise SystemExit("fold offsets differ from A31's")
        for arm in ARMS:
            p = ctx.out / "feats" / f"e320_s{s}_{arm}_f{f}.npy"
            if sha256_file(p) != rec[str(f)][f"{arm}_sha256"]:
                raise SystemExit("a feature file differs from its record")
            out[arm].append(np.load(p))
    return {k: (torch.from_numpy(np.concatenate(v)) if isinstance(v, list) else v) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------- heads
def head_and_opt(f: int):
    g = sd.Residual(20261026 + 36 + f); return g, t31.opt_of([("g", list(g.parameters()), 1e-3)])


def step(g, opt, X: dict, arm: str, R: dict, u: int) -> dict:
    t31.sched(opt, u, {"updates": U, "warmup": WARM}); rows, w, ix, y = t29.batch_rows(R, R["bt"][u], R["bu"][u])
    loss = t29.pooled_ce(g(X["q"][rows], X["e224"][rows], X[arm][rows]), w, ix, y).mean(); opt.zero_grad(set_to_none=True); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(g.parameters(), t31.CLIP)); opt.step(); return {"u": int(u), "loss": float(loss.detach()), "grad_norm": gn}


def fit_monitor(g, X: dict, arm: str, R: dict) -> dict:
    pk = R["P"]; idx = np.arange(len(pk["labels"]))
    with torch.no_grad():
        z = g(X["q"], X["e224"], X[arm]); pp = t29.pooled_probs(z, pk, idx); rows, w, ix, y = t29.batch_rows(R, np.arange(len(R["T"]["labels"])), idx)
        ce = float(t29.pooled_ce(z[rows], w, ix, y).mean())
    pred = pp.argmax(1); wr = pk["base"] != pk["labels"]
    return {"pooled_ce_T_and_P": ce, "base_wrong": int(wr.sum()), "fixed": int((wr & (pred == pk["labels"])).sum()), "broken": int((~wr & (pred != pk["labels"])).sum())}


def train(ctx: Ctx, s: int) -> None:
    torch.set_num_threads(1); X = feats(ctx, s)
    for f in FOLDS:
        R = t31.runs(ctx.a31, s, f)
        for arm in ARMS:
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; rd.mkdir(parents=True, exist_ok=True); g, o = head_and_opt(f); before = fit_monitor(g, X, arm, R); log = []
                for u in range(U):
                    log.append(step(g, o, X, arm, R, u))
                    if (u + 1) % 270 == 0:
                        print(f"[{arm}_s{s}_f{f}] update {u + 1}/{U}: loss {np.mean([x['loss'] for x in log[-270:]]):.4f}  {time.time() - t0:.1f} s", flush=True)
                after = fit_monitor(g, X, arm, R); torch.save({k: v.detach().clone() for k, v in g.state_dict().items()}, rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                man = {"arm": arm, "seed": s, "fit_fold": f, "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "fit_before": before, "fit_after": after, "seconds": time.time() - t0}
                write_json(rd / "run_manifest.json", man); print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "final_sha256")}), flush=True); ok = True
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {arm}_s{s}_f{f} (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- check
def check(ctx: Ctx, device, workers: int) -> str:
    res = {"config_sha256": sha256_file(ROOT / ctx.config)}; s = 0; a08 = ctx.a31.a08(s, device); d0 = tet.state_digest(a08); seqs = t31.load_pool(ctx.a31, 0); W = t31.pool_W(seqs); fp = t28.frame_paths(seqs)
    O = t31.outputs(ctx.a31, s, 0); pick = np.random.default_rng([20261026, 36, 1]).choice(len(W["seq"]), 24, replace=False)
    # inputs on real training frames (technical only)
    refs = sorted({tuple(seqs[int(W["seq"][j])]["refs"][k]) for j in pick[:4] for k in W["s"][j]}); fr = decode_pairs(refs, fp, 0); sizes = set(); hbit = True; ldiff = []
    from PIL import Image
    for (v, fi, gm), h, l_ in zip(refs, fr["H"], fr["L"]):
        p = fp(v)[min(int(fi), len(fp(v)) - 1)]
        with Image.open(p) as im:
            sizes.add(im.size); ref_img = np.asarray(sd.gamma_rgb(im, round(float(gm), 6)))
        if im.size == (320, 240):
            hbit &= bool(np.array_equal(h[40:280], ref_img)); ldiff.append(float(np.abs(h.astype(int) - l_.astype(int)).mean()))
    res["inputs"] = {"frames_checked": len(refs), "sizes": sorted(sizes), "H_content_equals_gamma_original": hbit, "L_minus_H_mean_abs": float(np.mean(ldiff)) if ldiff else None,
                     "canvas_fill_equal": bool((fr["H"][:, :40] == fr["L"][:, :40]).all())}
    # 224 parity: A08's own path against A31's cached logits / embeddings
    refs16 = {int(j): [tuple(seqs[int(W["seq"][j])]["refs"][k]) for k in W["s"][j]] for j in pick}; uniq = sorted({r for v in refs16.values() for r in v}); f224 = rr.decode_frames(uniq, fp, 0); at = {r: i for i, r in enumerate(uniq)}
    with torch.no_grad():
        x = tm.to_rgb(torch.from_numpy(f224[np.asarray([[at[r] for r in refs16[int(j)]] for j in pick])]).to(device)); mk = tm.marker_planes(torch.from_numpy(W["marker"][pick]).long().to(device))
        e = a08.embed(x, mk); z = a08.classifier(e)
    res["parity_224"] = {"logits_max_abs": float(np.abs(z.float().cpu().numpy() - O["logits"][pick]).max()), "emb_max_abs": float(np.abs(e.float().cpu().numpy() - O["emb"][pick]).max())}
    # 320 grid, equivalence, repeat and batch split
    m320 = sd.encoder320(a08); refs8 = sorted({tuple(seqs[int(W["seq"][j])]["refs"][k]) for j in pick[:8] for k in W["s"][j]}); fr8 = decode_pairs(refs8, fp, 0); at8 = {r: i for i, r in enumerate(refs8)}
    Wp = {"ref": np.asarray([[at8[tuple(seqs[int(W["seq"][j])]["refs"][k])] for k in W["s"][j]] for j in pick[:8]]), "marker": W["marker"][pick[:8]]}; ks = np.arange(8)
    with torch.no_grad():
        xs = tm.to_rgb(torch.from_numpy(fr8["H"][Wp["ref"]]).to(device)); mk = tm.marker_planes(torch.from_numpy(Wp["marker"]).long().to(device), size=sd.CANVAS)
        grids = sd.block_grids(m320, xs[:1], mk[:1]); mat, changed = sd.materialised320(a08); e1 = embed(m320, fr8["H"], Wp, ks, device); e2 = embed(m320, fr8["H"], Wp, ks, device)
        e3 = torch.cat([embed(m320, fr8["H"], Wp, ks[:4], device), embed(m320, fr8["H"], Wp, ks[4:], device)]); em = mat.embed(xs, mk).float()
    res["grid_320"] = {"stem_and_blocks": [list(g) for g in grids], "materialised_tables": len(changed), "runtime_vs_materialised_max_abs": float((e1 - em).abs().max()), "repeat_max_abs": float((e1 - e2).abs().max()),
                       "batch_split_max_abs": float((e1 - e3).abs().max()), "H_vs_L_embedding_max_abs": float((e1 - embed(m320, fr8["L"], Wp, ks, device)).abs().max())}
    res["weights_unchanged"] = tet.state_digest(a08) == d0 and tet.state_digest(m320) == d0
    # Init = Base, forced equality, gradients, resume (placeholder e320 = e224: no new features needed)
    X, offs = {"q": [], "e224": []}, {}
    for f in FOLDS:
        Of = t31.outputs(ctx.a31, s, f); X["q"].append(Of["logits"]); X["e224"].append(Of["emb"]); offs[f] = int(Of["offset"])
    X = {k: torch.from_numpy(np.concatenate(v)) for k, v in X.items()}; X["H"] = X["e224"].clone(); X["L"] = X["e224"].clone(); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); init = {}
    for f in FOLDS:
        g, _ = head_and_opt(f); D = d32.fold_data(ctx.d, s, f); pk = D["pk"]; off = offs[f]; Wf = t31.pool_W(D["seqs"])
        with torch.no_grad():
            z = g(X["q"], X["e224"], X["H"])
        pp = t29.pooled_probs(z, pk, np.arange(len(pk["labels"]))); zf = z[off:off + len(Wf["seq"])].numpy(); ps = rr.per_sample(Wf, zf, D["seqs"])
        seq_eq = all(np.array_equal(ir.relabel(D["base_preds"][q["id"]], ps[q["id"]]), D["base_preds"][q["id"]]) for q in D["seqs"])
        P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; sr = t31.seq_rows(a8.cfg, P, {q["id"]: ir.relabel(D["base_preds"][q["id"]], ps[q["id"]]) for q in D["seqs"]}, bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{f}"))
        old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"][f"s{s}_heldout_f{f}"]["A08 Base"]
        init[str(f)] = {"z_equals_q": bool(torch.equal(z, X["q"])), "runs_equal_base": bool(np.array_equal(pp.argmax(1), pk["base"])), "sequences_equal_base": bool(seq_eq), "rows_equal_A32_base": bool(sr["row"] == old["sequence_row"] and rsh.counts_of(D["seqs"], sr) == old["counts"])}
    res["init_equals_base"] = init; R = t31.runs(ctx.a31, s, 0); torch.set_num_threads(1)
    gh, oh = head_and_opt(0); gl, ol = head_and_opt(0); res["identical_init"] = all(torch.equal(a, b) for a, b in zip(gh.state_dict().values(), gl.state_dict().values()))
    lh = [step(gh, oh, X, "H", R, u) for u in range(3)]; ll = [step(gl, ol, X, "L", R, u) for u in range(3)]
    res["forced_equal"] = {"loss_max_abs": max(abs(a["loss"] - b["loss"]) for a, b in zip(lh, ll)), "params_max_abs": float(max((a - b).abs().max() for a, b in zip(gh.state_dict().values(), gl.state_dict().values())))}
    g1, o1 = head_and_opt(0); t31.sched(o1, 0, {"updates": U, "warmup": WARM}); rows, w, ix, y = t29.batch_rows(R, R["bt"][0], R["bu"][0]); t29.pooled_ce(g1(X["q"][rows], X["e224"][rows], X["H"][rows]), w, ix, y).mean().backward()
    first = {"fc1_grad_abs_sum_step1": float(g1.fc1.weight.grad.abs().sum()), "fc2_grad_abs_sum_step1": float(g1.fc2.weight.grad.abs().sum())}
    g2, o2 = head_and_opt(0); step(g2, o2, X, "H", R, 0); o2.zero_grad(); rows, w, ix, y = t29.batch_rows(R, R["bt"][1], R["bu"][1]); t29.pooled_ce(g2(X["q"][rows], X["e224"][rows], X["H"][rows]), w, ix, y).mean().backward()
    first["fc1_grad_abs_sum_step2"] = float(g2.fc1.weight.grad.abs().sum()); res["gradients"] = first
    res["optimizer_binds_exactly"] = sorted(id(p) for gr in oh.param_groups for p in gr["params"]) == sorted(id(p) for p in gh.parameters())
    ga, oa = head_and_opt(0); [step(ga, oa, X, "H", R, u) for u in range(6)]; gb, ob = head_and_opt(0); [step(gb, ob, X, "H", R, u) for u in range(3)]; sv = ({k: v.clone() for k, v in gb.state_dict().items()}, ob.state_dict())
    gc, oc = head_and_opt(0); gc.load_state_dict(sv[0]); oc.load_state_dict(sv[1]); [step(gc, oc, X, "H", R, u) for u in range(3, 6)]
    res["resume_3_plus_3_vs_6_max_abs"] = float(max((a - b).abs().max() for a, b in zip(ga.state_dict().values(), gc.state_dict().values())))
    sp = json.loads((ctx.a31.out / "split.json").read_text()); res["isolation"] = sp["isolation"]; res["windows"] = {str(f): int(len(t31.pool_W(t31.load_pool(ctx.a31, f))["seq"])) for f in FOLDS}
    # timing and disk
    torch.cuda.synchronize(); t0 = time.time(); refs_t = sorted({tuple(r) for q in seqs[:6] for r in q["refs"]}); frt = decode_pairs(refs_t, fp, workers); td = time.time() - t0; Wt = t31.pool_W(seqs[:6]); nt = len(Wt["seq"]); t1 = time.time()
    for arm in ARMS:
        for b in range(0, nt, BATCH):
            embed(m320, frt[arm], Wt, np.arange(b, min(b + BATCH, nt)), device)
    torch.cuda.synchronize(); tg = time.time() - t1; nw = sum(res["windows"].values()); per = (td + tg) / nt
    res["timing"] = {"windows": nt, "decode_s": td, "gpu_s_two_arms": tg, "seconds_per_window_two_arms": per, "projected_hours_per_seed": nw * per / 3600 * 1.15, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    res["disk"] = {"projected_feature_gb_per_seed": nw * 512 * 4 * 2 / 1e9, "free_gb": shutil.disk_usage(ctx.out).free / 1e9}
    c = res["inputs"]; g_ = res["grid_320"]
    res["ok"] = bool(c["H_content_equals_gamma_original"] and (c["L_minus_H_mean_abs"] or 0) > 0 and c["canvas_fill_equal"] and res["parity_224"]["logits_max_abs"] <= 1e-4 and res["parity_224"]["emb_max_abs"] <= 1e-4
                     and g_["stem_and_blocks"][1] == [8, 80, 80] and g_["runtime_vs_materialised_max_abs"] <= 1e-4 and g_["repeat_max_abs"] <= 1e-5 and g_["batch_split_max_abs"] <= 1e-4 and g_["H_vs_L_embedding_max_abs"] > 0
                     and res["weights_unchanged"] and all(all(v.values()) for v in init.values()) and res["identical_init"] and res["forced_equal"]["loss_max_abs"] == 0 and res["forced_equal"]["params_max_abs"] == 0
                     and first["fc1_grad_abs_sum_step1"] == 0 and first["fc2_grad_abs_sum_step1"] > 0 and first["fc1_grad_abs_sum_step2"] > 0 and res["optimizer_binds_exactly"] and res["resume_3_plus_3_vs_6_max_abs"] == 0
                     and not any(sp["isolation"].values()) and res["timing"]["projected_hours_per_seed"] <= 1.6 and res["disk"]["projected_feature_gb_per_seed"] < 2.5)
    res["finished"] = now(); publish(ctx, "checks.json", res); print(json.dumps(res, indent=1, default=str)[:6000], flush=True)
    if not res["ok"]:
        raise SystemExit("a check failed")
    return f"{per:.3f} s/window, {res['timing']['projected_hours_per_seed']:.2f} h/seed"


# ----------------------------------------------------------------------------------------------- readout
def load_head(ctx: Ctx, arm: str, s: int, f: int):
    rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text())
    if sha256_file(rd / "final.pt") != man["final_sha256"] or com.get(f"{arm}_s{s}_f{f}") != man["final_sha256"]:
        raise SystemExit(f"{arm}_s{s}_f{f}: weights not committed or changed")
    g, _ = head_and_opt(f); g.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True)); return g.eval()


def tables(z: torch.Tensor, D: dict, off: int, a8cfg, P: dict, ev) -> dict:
    pk, seqs = D["pk"], D["seqs"]; pp = t29.pooled_probs(z, pk, np.arange(len(pk["labels"]))); pred = pp.argmax(1); y, b = pk["labels"], pk["base"]; M = D["M"]; ok5 = (y == 3) & (b == 3)
    W = t31.pool_W(seqs); zf = z[off:off + len(W["seq"])].numpy(); ps = rr.per_sample(W, zf, seqs); preds = {q["id"]: ir.relabel(D["base_preds"][q["id"]], ps[q["id"]]) for q in seqs}
    sr = t31.seq_rows(a8cfg, P, preds, ev)
    return {"runs": t31.run_table(pp, pk, seqs, D["sessions"]), "changes": d32.change_table(y, b, pred), "correct_A5C_kept": {"runs": int(ok5.sum()), "kept": int((ok5 & (pred == 3)).sum())},
            "S1_fixed": {"runs": int(M["S1"].sum()), "fixed": int((M["S1"] & (pred == 3)).sum())}, "S2_kept": {"runs": int(M["S2"].sum()), "kept": int((M["S2"] & (pred == 3)).sum())},
            "sequence_row": sr["row"], "counts": rsh.counts_of(seqs, sr), "pred": pred}


def readout(ctx: Ctx, s: int) -> str:
    name = f"readout_s{s}.json"
    if (ctx.out / name).exists():
        raise SystemExit("this seed's readout exists; it runs once")
    X = feats(ctx, s); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"seed": s, "cells": {}}; acc = {m: [] for m in ("Base", "H", "L")}; ys, bs = [], []
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); off = X["offset"][h]; P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        cell = {}
        with torch.no_grad():
            for arm in ARMS:
                g = load_head(ctx, arm, s, f); t = tables(g(X["q"], X["e224"], X[arm]), D, off, a8.cfg, P, ev); acc[arm].append(t.pop("pred")); cell[arm] = t
            t = tables(X["q"], D, off, a8.cfg, P, ev); acc["Base"].append(t.pop("pred")); cell["Base"] = t
        mans = {arm: json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for arm in ARMS}; cell["fit"] = {arm: {k: mans[arm][k] for k in ("fit_before", "fit_after")} for arm in ARMS}
        row = lambda m_: cell[m_]["sequence_row"]  # noqa: E731
        cell["comparisons"] = {f"{a} - {bb}": {"net_fixed": cell[a]["runs"]["all"]["net_fixed"] - (cell[bb]["runs"]["all"]["net_fixed"] if bb != "Base" else 0), **{k: row(a)[k] - row(bb)[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")},
                                                 **{k: cell[a]["counts"][k] - cell[bb]["counts"][k] for k in ("fp", "fp_short", "fn", "fn_short")}} for a, bb in (("H", "L"), ("H", "Base"), ("L", "Base"))}
        rec["cells"][f"heldout_f{h}"] = cell; ys.append(D["pk"]["labels"]); bs.append(D["pk"]["base"])
    y, b = np.concatenate(ys), np.concatenate(bs); wr = b != y; rec["oof"] = {}
    for m_ in ("H", "L"):
        p = np.concatenate(acc[m_]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); rec["oof"][m_] = {"runs": int(len(y)), "fixed": fx, "broken": br, "net_fixed": fx - br}
    c = rec["cells"]; dF = lambda a, bb, h: c[f"heldout_f{h}"][a]["sequence_row"]["segment_f1_50_multi"] - c[f"heldout_f{h}"][bb]["sequence_row"]["segment_f1_50_multi"]  # noqa: E731
    net = lambda a, h: c[f"heldout_f{h}"][a]["runs"]["all"]["net_fixed"]  # noqa: E731
    s1 = sum(c[f"heldout_f{h}"]["H"]["S1_fixed"]["fixed"] for h in FOLDS); n1 = sum(c[f"heldout_f{h}"]["H"]["S1_fixed"]["runs"] for h in FOLDS)
    k5 = sum(c[f"heldout_f{h}"]["H"]["correct_A5C_kept"]["kept"] for h in FOLDS); n5 = sum(c[f"heldout_f{h}"]["H"]["correct_A5C_kept"]["runs"] for h in FOLDS)
    rule = {"detail_net_each_fold_ge_5": all(net("H", h) - net("L", h) >= 5 for h in FOLDS), "detail_dF_mean_ge_0.003": float(np.mean([dF("H", "L", h) for h in FOLDS])) >= 0.003, "detail_dF_each_fold_ge_0": all(dF("H", "L", h) >= 0 for h in FOLDS),
            "useful_net_each_fold_ge_0": all(net("H", h) >= 0 for h in FOLDS), "useful_dF_each_fold_ge_0": all(dF("H", "Base", h) >= 0 for h in FOLDS), "useful_dF_mean_ge_0.003": float(np.mean([dF("H", "Base", h) for h in FOLDS])) >= 0.003,
            "S1_recovered": f"{s1}/{n1}", "S1_ok": s1 >= math.ceil(0.10 * n1), "correct_A5C_kept": f"{k5}/{n5}", "A5C_ok": k5 >= math.ceil(0.98 * n5),
            "native_each_fold_ge_-0.003": all(c[f"heldout_f{h}"]["H"]["sequence_row"]["native_macro_f1"] - c[f"heldout_f{h}"]["Base"]["sequence_row"]["native_macro_f1"] >= -0.003 for h in FOLDS),
            "missed_short_le_base_plus_3": sum(c[f"heldout_f{h}"]["H"]["counts"]["fn_short"] for h in FOLDS) <= sum(c[f"heldout_f{h}"]["Base"]["counts"]["fn_short"] for h in FOLDS) + 3,
            "boundary_recall_each_fold_ge_-0.005": all(c[f"heldout_f{h}"]["H"]["sequence_row"]["boundary_recall_multi"] - c[f"heldout_f{h}"]["Base"]["sequence_row"]["boundary_recall_multi"] >= -0.005 for h in FOLDS)}
    rule["met"] = all(v for k, v in rule.items() if isinstance(v, bool)); rec["seed2_rule"] = rule
    rec["summary"] = {f"heldout_f{h}": {m_: {"net": net(m_, h) if m_ != "Base" else 0, "F1": c[f"heldout_f{h}"][m_]["sequence_row"]["segment_f1_50_multi"]} for m_ in ("Base", "H", "L")} for h in FOLDS}
    fr = json.loads((ctx.out / "feats" / f"record_s{s}.json").read_text()); pr = json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text()); nw = sum(fr[str(f)]["windows"] for f in FOLDS)
    g = load_head(ctx, "H", s, 0); torch.set_num_threads(1); t0 = time.time()
    with torch.no_grad():
        g(X["q"], X["e224"], X["H"])
    rec["deployment_cost"] = {"gpu_ms_per_window_224_A31_pass": 1e3 * pr["gpu_seconds"] / sum(pr["rows"].values()), "gpu_ms_per_window_320_one_arm": 1e3 * fr["gpu_seconds"] / (2 * nw),
                              "cpu_decode_ms_per_window_both_arms": 1e3 * fr["decode_seconds"] / nw, "head_cpu_ms_per_window": 1e3 * (time.time() - t0) / len(X["q"]),
                              "note": "the 224 pass also wrote A31's prefix cache; the 320 figure is the frozen encoder only; the head is one CPU thread over all windows"}
    publish(ctx, name, rec); print(json.dumps({"summary": rec["summary"], "oof": rec["oof"], "rule": rule}, indent=1)); return f"rule met: {rule['met']}"


def coverage(ctx: Ctx, s: int) -> str:
    """Recording / session coverage of fixed and broken runs (descriptive, after the readout; same committed heads).
    A run's recordings are all its source recordings (true-class sources separately); the readout's counts must reproduce."""
    name = f"coverage_s{s}.json"; rd = json.loads((ctx.out / f"readout_s{s}.json").read_text()); X = feats(ctx, s, pinned=True); rec = {"seed": s, "cells": {}, "oof": {}}; acc = {}
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); y, b = D["pk"]["labels"], D["pk"]["base"]; wr = b != y; cell = {}
        for arm in ARMS:
            with torch.no_grad():
                z = load_head(ctx, arm, s, f)(X["q"], X["e224"], X[arm])
            p = t29.pooled_probs(z, D["pk"], np.arange(len(y))).argmax(1); fx, br = wr & (p == y), ~wr & (p != y)
            if (int(fx.sum()), int(br.sum())) != (rd["cells"][f"heldout_f{h}"][arm]["runs"]["all"]["fixed"], rd["cells"][f"heldout_f{h}"][arm]["runs"]["all"]["broken"]):
                raise SystemExit("coverage does not reproduce the readout")
            sets = {}
            for k, m in (("fixed", fx), ("broken", br)):
                ix = np.flatnonzero(m); recs = {v for i in ix for v in D["src"][i]}; trec = {v for i in ix for v in D["true_src"][i]}
                sets[k] = (recs, trec); cell[f"{arm}_{k}"] = {"runs": int(m.sum()), "recordings": len(recs), "true_class_recordings": len(trec), "sessions": len({session_of(v) for v in recs})}
            acc.setdefault(arm, []).append(sets)
        rec["cells"][f"heldout_f{h}"] = cell
    for arm in ARMS:
        for k in ("fixed", "broken"):
            recs = set().union(*[c[k][0] for c in acc[arm]]); rec["oof"][f"{arm}_{k}"] = {"recordings": len(recs), "true_class_recordings": len(set().union(*[c[k][1] for c in acc[arm]])), "sessions": len({session_of(v) for v in recs})}
    publish(ctx, name, rec); print(json.dumps(rec, indent=1)); return "coverage reproduced the readout counts"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("check", "extract", "train", "readout", "coverage"), required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        return train(ctx, a.seed)
    if a.stage in ("readout", "coverage"):
        return staged(ctx, f"{a.stage} s{a.seed} (CPU)", "cpu", lambda: (readout if a.stage == "readout" else coverage)(ctx, a.seed))
    t31.vsh.determinism(); device = torch.device("cuda")
    staged(ctx, f"{a.stage}{' s' + str(a.seed) if a.stage == 'extract' else ''} (GPU job wall time)", "gpu", {"check": lambda: check(ctx, device, a.workers), "extract": lambda: extract(ctx, a.seed, device, a.workers)}[a.stage])


if __name__ == "__main__":
    main()
