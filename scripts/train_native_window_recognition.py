#!/usr/bin/env python
"""A29 (configs/experiments/video_native_window_recognition_v1.yaml): interval recognition on A08's own request windows.
Both arms pool the nearest-centre window probabilities inside each interval and train on the pooled CE; F trains the
copied classifier on the cached 512-d embeddings, T also the last block, norm and projection through the real windows
(frozen prefix cached per seed, verified against A15's outputs). Zero update = Base.

  --stage prepare   pins, monitoring groups and the training window set
  --stage cache     T's frozen prefix for every training window of a seed (verified; deleted after the seed's training)
  --stage check     window/pixel/marker, zero-update Base reproduction, objective, microbatch, forced equality, resume, timing
  --stage train     F and T for the seeds whose cache exists, with monitoring and the final training readout
  --stage freeze    freeze both arms; training summary
  --stage evaluate  once: development halves and the A15 check pool
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # this process only; importing A27's CPU-only trainer would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.b3_coverage_evaluate as bce  # noqa: E402
import scripts.report_hard_replay as rhr  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_recognition as t27  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.interval_clips as ic  # noqa: E402
import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

A27_CONFIG = "configs/experiments/video_interval_recognition_v1.yaml"
SEEDS, ARMS, HALVES = (0, 2), ("F", "T"), ("selection", "calibration")
UPDATES, WARM, PER = 1350, 135, 16
LR_HEAD, LR_REP, WD, CLIP = 1e-4, 1e-5, 1e-4, 1.0
TOKENS, DIM = 393, 768
MONITOR_AT = (0, 270, 540, 810, 1080, 1350)
CMPS = {"T - F": ("T", "F"), "F - Base": ("F", "Base"), "T - Base": ("T", "Base")}
now, write_json, ledger_append = vsh.now, vsh.write_json, vsh.ledger_append


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vn = cfg, config, cfg["video_native_window_recognition"]; self.pin = self.vn["pinned"]; o = self.vn["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.vn["budget"]["ledger"]
        for k in ("A08_choices", "A27_config"):
            if sha256_file(ROOT / self.pin[k]["path"]) != self.pin[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")
        self.a27 = t27.Ctx(load_cfg(SimpleNamespace(config=A27_CONFIG, set=[])), A27_CONFIG); self.t28 = SimpleNamespace(pin={"A08_weights": self.pin["A08_weights"]})

    def a08(self, s: int, device) -> tm.TargetMViT:
        return t28.Ctx.a08(self.t28, s, device)

    def intervals(self, s: int) -> dict:
        p = ROOT / self.pin["A27_intervals_path"].format(seed=s)
        if sha256_file(p) != self.pin["A27_intervals"][s]:
            raise SystemExit(f"seed {s}: A27's interval file differs from its pin")
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        return {"T": {k[2:]: v for k, v in d.items() if k.startswith("T_")}, "P": {k[2:]: v for k, v in d.items() if k.startswith("P_")}, "bt": d["batch_T"], "bp": d["batch_P"]}


def pooled_ce(z: torch.Tensor, w: torch.Tensor, ix: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-interval -log(sum_t w_t softmax(z_t)[y]) by logsumexp; z, w, ix over gathered rows; y per interval."""
    n = int(y.shape[0]); s = torch.log(w) + torch.log_softmax(z, -1).gather(1, y[ix].unsqueeze(1)).squeeze(1)
    m = torch.full((n,), -torch.inf, dtype=z.dtype, device=z.device).scatter_reduce(0, ix, s, reduce="amax", include_self=True)
    return -(m + torch.log(torch.zeros(n, dtype=z.dtype, device=z.device).index_add(0, ix, torch.exp(s - m[ix]))))


def batch_rows(I: dict, idx_T: np.ndarray, idx_P: np.ndarray):
    rt, wt, it, yt = ir.gather(I["T"], idx_T); rp, wp, ip, yp = ir.gather(I["P"], idx_P)
    return torch.cat([rt, rp]), torch.cat([wt, wp]), torch.cat([it, ip + len(idx_T)]), torch.cat([yt, yp])


def code_digest() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(ic.__file__), Path(ir.__file__), Path(tm.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- prepare
def monitor_groups(I: dict, D: dict, P: dict, s: int) -> dict:
    """Base-wrong P runs (all) and a fixed 1,000 Base-correct P runs; descriptors pure / mixed, single / multi recording."""
    fit = bc.members(P, "large"); seqs = P["seqs"]; runs = [(i, a, b, c) for i in fit for a, b, c in ir.runs(D["base"][i])]
    if len(runs) != len(I["P"]["labels"]):
        raise SystemExit("the Base runs differ from A27's P list")
    wrong = np.asarray([k for k, r in enumerate(runs) if r[3] != I["P"]["labels"][k]]); right = np.asarray([k for k, r in enumerate(runs) if r[3] == I["P"]["labels"][k]])
    pick = np.sort(np.random.default_rng([20261024, 29, s]).choice(right, size=1000, replace=False)); idx = np.concatenate([wrong, pick])
    multi = np.asarray([len({x[0] for x in seqs[runs[k][0]]["refs"][runs[k][1]:runs[k][2]]}) > 1 for k in idx]); pure = I["P"]["purity"][idx] >= 1.0
    return {"idx": idx, "base_wrong": np.isin(idx, wrong), "pure": pure, "multi_recording": multi, "base_class": np.asarray([runs[k][3] for k in idx]),
            "sessions": [sorted({session_of(x[0]) for x in seqs[runs[k][0]]["refs"][runs[k][1]:runs[k][2]]}) for k in idx]}


def prepare(ctx: Ctx) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "seeds": {}}
    try:
        ddir = ctx.out / "data"; ddir.mkdir(parents=True, exist_ok=True); P = ctx.a27.pool()
        for s in SEEDS:
            I = ctx.intervals(s); D = t27.pool_data(ctx.a27, s); g = monitor_groups(I, D, P, s); rows = np.union1d(I["T"]["rows"], I["P"]["rows"]).astype(np.int64)
            np.savez(ddir / f"monitor_s{s}.npz", **{k: np.asarray(v) for k, v in g.items() if k != "sessions"}); np.save(ddir / f"cache_rows_s{s}.npy", rows)
            per = [int(np.diff(I["T"]["offsets"])[I["bt"][u]].sum() + np.diff(I["P"]["offsets"])[I["bp"][u]].sum()) for u in range(UPDATES)]
            rec["seeds"][str(s)] = {"training_windows": int(rows.size), "rows_per_update_mean": float(np.mean(per)), "rows_per_update_max": int(max(per)),
                                    "monitor": {"base_wrong": int(g["base_wrong"].sum()), "base_correct_sample": int((~g["base_wrong"]).sum()),
                                                "wrong_mixed": int((g["base_wrong"] & ~g["pure"]).sum()), "wrong_multi_recording": int((g["base_wrong"] & g["multi_recording"]).sum()),
                                                "wrong_sessions": len({x for k, v in zip(g["base_wrong"], g["sessions"]) if k for x in v})},
                                    "cache_bytes": int(rows.size) * TOKENS * DIM * 4, "monitor_sha256": sha256_file(ddir / f"monitor_s{s}.npz"), "cache_rows_sha256": sha256_file(ddir / f"cache_rows_s{s}.npy")}
            print(json.dumps(rec["seeds"][str(s)]), flush=True)
        rec["finished"] = now()
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare (CPU)", "cpu", started, time.time() - t0, ok, note)


def prep_rec(ctx: Ctx, s: int) -> dict:
    return json.loads((ctx.out / "preparation.json").read_text())["seeds"][str(s)]


def cache_rows(ctx: Ctx, s: int) -> np.ndarray:
    p = ctx.out / "data" / f"cache_rows_s{s}.npy"
    if sha256_file(p) != prep_rec(ctx, s)["cache_rows_sha256"]:
        raise SystemExit("cache rows differ from preparation.json")
    return np.load(p)


def monitor(ctx: Ctx, s: int) -> dict:
    p = ctx.out / "data" / f"monitor_s{s}.npz"
    if sha256_file(p) != prep_rec(ctx, s)["monitor_sha256"]:
        raise SystemExit("monitor groups differ from preparation.json")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------------- prefix cache (T)
def cache_key(ctx: Ctx, s: int) -> str:
    return hashlib.sha256(json.dumps({"a08": ctx.pin["A08_weights"][s], "rows": prep_rec(ctx, s)["cache_rows_sha256"], "pool": "5ab124d4", "code": code_digest(), "precision": "fp32 eval",
                                      "layer": "input of encoder.blocks.15"}, sort_keys=True).encode()).hexdigest()


def pool_windows(P: dict, seqs_idx: list[int]) -> tuple[dict, list, int]:
    """Request windows of consecutive pool sequences [c0, c1), rebuilt as A15 did, with their global row offset."""
    seqs = [P["seqs"][i] for i in seqs_idx]; refs = sorted({tuple(r) for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    first = int(np.flatnonzero(P["W"]["seq"] == seqs_idx[0])[0]); g = slice(first, first + len(W["seq"]))
    if not (np.array_equal(P["W"]["seq"][g] - seqs_idx[0], W["seq"]) and np.array_equal(P["W"]["centre"][g], W["centre"]) and np.array_equal(P["W"]["marker"][g], W["marker"])):
        raise SystemExit("rebuilt windows differ from A15's pool windows")
    return W, refs, first


def build_cache(ctx: Ctx, s: int, device, workers: int) -> None:
    t0, started, ok, note, acc = time.time(), now(), False, "", {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    try:
        P = ctx.a27.pool(); D = t27.pool_data(ctx.a27, s); rows = cache_rows(ctx, s); pos = {int(r): k for k, r in enumerate(rows)}; cdir = ctx.out / "cache"; cdir.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(cdir / f"prefix_s{s}.npy", mode="w+", dtype=np.float32, shape=(rows.size, TOKENS, DIM)); m = ic.branch(ctx.a08(s, device), True); d0 = tet.state_digest(m)
        got = np.zeros(rows.size, bool); lg_err, mism = 0.0, 0; fp = t28.frame_paths(P["seqs"]); fit = bc.members(P, "large"); CH = 150
        for c0 in range(0, len(fit), CH):
            ids = fit[c0:c0 + CH]; W, refs, first = pool_windows(P, ids); need = [k for k in range(len(W["seq"])) if first + k in pos]
            if not need:
                continue
            a = time.time(); frames = rr.decode_frames(refs, fp, workers); acc["decode_seconds"] += time.time() - a
            for b in range(0, len(need), 32):
                ks = need[b:b + 32]; u8 = torch.from_numpy(frames[W["ref"][ks]]).to(device); mk = tm.marker_planes(torch.from_numpy(W["marker"][ks]).long().to(device))
                torch.cuda.synchronize(); a = time.time()
                with torch.no_grad():
                    x, thw = ic.prefix(m, tm.to_rgb(u8), mk); z = m.classifier(ic.suffix(m, x, thw))
                torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a; gr = [first + k for k in ks]; ci = [pos[g] for g in gr]
                mm[ci] = x.float().cpu().numpy(); got[ci] = True; ref = D["logits"][gr]; zz = z.float().cpu().numpy()
                lg_err = max(lg_err, float(np.abs(zz - ref).max())); mism += int((zz.argmax(1) != ref.argmax(1)).sum())
            del frames; print(f"seed {s}: {int(got.sum())}/{rows.size} windows cached; GPU {acc['gpu_seconds']:.0f} s", flush=True)
        mm.flush(); del mm
        if not got.all() or tet.state_digest(m) != d0:
            raise SystemExit("the prefix cache is incomplete or the branch changed")
        rec = {"key": cache_key(ctx, s), "windows": int(rows.size), "logits_max_abs_against_A15": lg_err, "argmax_mismatches": mism, "file_bytes": (cdir / f"prefix_s{s}.npy").stat().st_size, "time": now(), **acc}
        write_json(cdir / f"prefix_s{s}.json", rec); print(json.dumps(rec), flush=True)
        if lg_err > 1e-4 or mism:
            raise SystemExit("cached prefix + original suffix does not reproduce A15's logits")
        ok = True; note = f"{rows.size} windows, max |dlogit| {lg_err:.2e}"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, f"prefix cache seed {s} (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


class Prefix:
    def __init__(self, ctx: Ctx, s: int) -> None:
        cdir = ctx.out / "cache"; rec = json.loads((cdir / f"prefix_s{s}.json").read_text())
        if rec["key"] != cache_key(ctx, s):
            raise SystemExit(f"seed {s}: the prefix cache belongs to another key")
        self.mm = np.load(cdir / f"prefix_s{s}.npy", mmap_mode="r"); rows = cache_rows(ctx, s); self.pos = np.full(int(rows.max()) + 1, -1, np.int64); self.pos[rows] = np.arange(rows.size)

    def get(self, rows: np.ndarray) -> torch.Tensor:
        ci = self.pos[rows]
        if (ci < 0).any():
            raise SystemExit("a requested window is not in the prefix cache")
        order = np.argsort(ci); out = np.empty((len(ci), TOKENS, DIM), np.float32); out[order] = self.mm[ci[order]]; return torch.from_numpy(out)


class UpdateRows(Dataset):
    """Per update: the unique windows of the batch (prefix tokens), their gather map, weights, interval index and labels."""

    def __init__(self, ctx: Ctx, s: int, I: dict, us) -> None:
        self.ctx, self.s, self.I, self.us, self.px = ctx, s, I, list(us), None

    def __len__(self) -> int:
        return len(self.us)

    def __getitem__(self, i: int):
        if self.px is None:
            self.px = Prefix(self.ctx, self.s)
        u = self.us[i]; rows, w, ix, y = batch_rows(self.I, self.I["bt"][u], self.I["bp"][u]); uniq, inv = np.unique(rows.numpy(), return_inverse=True)
        return u, self.px.get(uniq), torch.from_numpy(inv), w, ix, y


# ----------------------------------------------------------------------------------------------- training
def t_opt(m, rep_lr: float = LR_REP):
    rep = [p for n, p in ic.trainable(m) if not n.startswith("classifier.")]; head = [p for n, p in ic.trainable(m) if n.startswith("classifier.")]
    return torch.optim.AdamW([{"params": rep, "lr": 0.0, "base": rep_lr, "name": "representation"}, {"params": head, "lr": 0.0, "base": LR_HEAD, "name": "classifier"}], betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)


def f_opt(head):
    return torch.optim.AdamW([{"params": list(head.parameters()), "lr": 0.0, "base": LR_HEAD, "name": "classifier"}], betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)


def schedule(opt, u: int) -> None:
    for g in opt.param_groups:
        g["lr"] = g["base"] * lr_factor(int(u), WARM, UPDATES)


def f_step(head, opt, emb: torch.Tensor, I: dict, u: int) -> dict:
    schedule(opt, u); rows, w, ix, y = batch_rows(I, I["bt"][u], I["bp"][u]); loss = pooled_ce(head(emb[rows]), w, ix, y).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)); opt.step(); return {"u": int(u), "loss": float(loss.detach()), "grad_norm": gn}


def t_forward(m, tok: torch.Tensor, device) -> torch.Tensor:
    thw = (m.encoder.pos_encoding.temporal_size,) + m.encoder.pos_encoding.spatial_size; return m.classifier(ic.suffix(m, tok.to(device, non_blocking=True), thw))


def t_step(m, opt, item, device, micro: int | None = None) -> dict:
    u, tok, inv, w, ix, y = item; u = int(u); schedule(opt, u); params = [p for _, p in ic.trainable(m)]; opt.zero_grad(set_to_none=True); inv, w, ix, y = (x.to(device) for x in (inv, w, ix, y))
    if micro is None:
        loss = pooled_ce(t_forward(m, tok, device)[inv], w, ix, y).mean(); loss.backward(); lv = float(loss.detach())
    else:                                   # interval microbatches: each keeps all its windows; the sum of parts equals the batch mean
        n = int(y.shape[0]); lv = 0.0
        for a in range(0, n, micro):
            sel = (ix >= a) & (ix < a + micro); r = torch.unique(inv[sel]); loc = torch.searchsorted(r, inv[sel])
            part = pooled_ce(t_forward(m, tok[r.cpu()], device)[loc], w[sel], ix[sel] - a, y[a:a + micro]).sum() / n; part.backward(); lv += float(part.detach())
    gn = float(torch.nn.utils.clip_grad_norm_(params, CLIP)); opt.step(); return {"u": u, "loss": lv, "grad_norm": gn}


def pooled_probs(z_rows: torch.Tensor, packed: dict, idx: np.ndarray) -> np.ndarray:
    """Pooled class probabilities of the chosen intervals from per-window logits indexed by global row."""
    off = packed["offsets"]; out = np.empty((len(idx), 5)); p = torch.softmax(z_rows, -1).numpy()
    for j, i in enumerate(np.asarray(idx, int)):
        a, b = off[i], off[i + 1]; out[j] = (packed["weights"][a:b, None] * p[packed["rows"][a:b]]).sum(0)
    return out


def window_logits(arm: str, model, rows: np.ndarray, emb: np.ndarray, px: Prefix | None, device) -> torch.Tensor:
    """Per-window logits of the arm at the given global rows (a dense array over max row + 1, others zero)."""
    full = torch.zeros(int(rows.max()) + 1, 5)
    with torch.no_grad():
        if arm == "F":
            full[rows] = model(torch.from_numpy(emb[rows]))
        else:
            for b in range(0, len(rows), 256):
                r = rows[b:b + 256]; full[r] = t_forward(model, px.get(r), device).float().cpu()
    return full


def scores(pp: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled CE, true-class margin (log p_y - max other log p) and predicted class."""
    lp = np.log(np.clip(pp, 1e-30, None)); k = np.arange(len(y)); ce = -lp[k, y]; other = lp.copy(); other[k, y] = -np.inf
    return ce, lp[k, y] - other.max(1), pp.argmax(1)


def list_probs(arm: str, model, pk: dict, idx: np.ndarray, emb: np.ndarray, px, device) -> np.ndarray:
    rows = np.unique(np.concatenate([pk["rows"][pk["offsets"][i]:pk["offsets"][i + 1]] for i in idx])); return pooled_probs(window_logits(arm, model, rows, emb, px, device), pk, idx)


def readout(arm: str, model, I: dict, emb: np.ndarray, px, device, g: dict) -> dict:
    """Monitoring on the fixed P groups: CE, true-class margin, fixed (Base-wrong now right) / broken (Base-right now wrong)."""
    pk = I["P"]; idx = g["idx"]; y = pk["labels"][idx]; ce, margin, pred = scores(list_probs(arm, model, pk, idx, emb, px, device), y); wrong = g["base_wrong"]; out = {"changed_from_base": int((pred != g["base_class"]).sum())}
    for key, mask in {"base_wrong": wrong, "base_correct": ~wrong, "base_wrong_mixed": wrong & ~g["pure"], "base_wrong_pure": wrong & g["pure"],
                      "base_wrong_multi_recording": wrong & g["multi_recording"], "base_wrong_single_recording": wrong & ~g["multi_recording"]}.items():
        hit = (pred == y) if key.startswith("base_wrong") else (pred != y)
        out[key] = {"n": int(mask.sum()), "ce": float(ce[mask].mean()) if mask.any() else None, "margin": float(margin[mask].mean()) if mask.any() else None,
                    ("fixed" if key.startswith("base_wrong") else "broken"): int(hit[mask].sum())}
    return out


def full_readout(arm: str, model, I: dict, emb: np.ndarray, px, device, base_cls: np.ndarray) -> dict:
    """The full training lists: T accuracy / CE; P accuracy, fixed / broken / changed against the Base run class, CE and margin by Base correctness."""
    out = {}
    for name in ("T", "P"):
        pk = I[name]; idx = np.arange(len(pk["labels"])); y = pk["labels"]; ce, margin, pred = scores(list_probs(arm, model, pk, idx, emb, px, device), y); r = {"intervals": int(len(idx)), "accuracy": float((pred == y).mean()), "ce": float(ce.mean())}
        if name == "P":
            wrong = base_cls != y
            r.update({"base_accuracy": float((~wrong).mean()), "base_wrong": int(wrong.sum()), "fixed": int((wrong & (pred == y)).sum()), "broken": int((~wrong & (pred != y)).sum()), "changed": int((pred != base_cls).sum()),
                      "ce_base_wrong": float(ce[wrong].mean()), "ce_base_correct": float(ce[~wrong].mean()), "margin_base_wrong": float(margin[wrong].mean()), "margin_base_correct": float(margin[~wrong].mean())})
        out[name] = r
    return out


def train(ctx: Ctx, seeds: list[int], device, workers: int) -> None:
    for s in seeds:
        I = ctx.intervals(s); D = t27.pool_data(ctx.a27, s); emb = D["emb"]; g = monitor(ctx, s); P = ctx.a27.pool(); fit = bc.members(P, "large")
        base_cls = np.asarray([c for i in fit for _, _, c in ir.runs(D["base"][i])]); init = ctx.a27.init(s); px = Prefix(ctx, s)
        for arm in ARMS:
            t0, started, ok, note, gsec = time.time(), now(), False, "", 0.0
            try:
                rd = ctx.out / "train" / f"{arm}_s{s}"; rd.mkdir(parents=True, exist_ok=True); log, mon = [], {}
                if arm == "F":
                    torch.set_num_threads(1); model = ir.copied_head(*init); opt = f_opt(model); e = torch.from_numpy(emb)
                    for u in range(UPDATES):
                        if u in MONITOR_AT:
                            mon[u] = readout("F", model, I, emb, None, device, g)
                        log.append(f_step(model, opt, e, I, u))
                    mon[UPDATES] = readout("F", model, I, emb, None, device, g); state = {f"classifier.{k}": v.detach().clone() for k, v in model.state_dict().items()}; pfx_ok = True
                else:
                    torch.set_num_threads(8); model = ic.branch(ctx.a08(s, device), True); pd0 = t28.prefix_digest(model); opt = t_opt(model)
                    for item in DataLoader(UpdateRows(ctx, s, I, range(UPDATES)), batch_size=None, num_workers=workers, prefetch_factor=2, pin_memory=True):
                        u = int(item[0])
                        if u in MONITOR_AT:
                            mon[u] = readout("T", model, I, emb, px, device, g)
                        torch.cuda.synchronize(); a = time.time(); log.append(t_step(model, opt, item, device)); torch.cuda.synchronize(); gsec += time.time() - a
                        if (u + 1) % WARM == 0:
                            seg = log[-WARM:]; print(f"  [T_s{s}] update {u + 1}/{UPDATES}: loss {np.mean([x['loss'] for x in seg]):.4f}", flush=True)
                    mon[UPDATES] = readout("T", model, I, emb, px, device, g); pfx_ok = t28.prefix_digest(model) == pd0
                    if not pfx_ok:
                        raise SystemExit("a frozen prefix parameter changed")
                    state = {n: p.detach().cpu().clone() for n, p in ic.trainable(model)}
                fr = full_readout(arm, model, I, emb, px if arm == "T" else None, device, base_cls)
                torch.save(state, rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                man = {"arm": arm, "seed": s, "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "trainable_parameters": int(sum(v.numel() for v in state.values())), "frozen_prefix_unchanged": pfx_ok,
                       "loss": {"first 135": float(np.mean([x["loss"] for x in log[:WARM]])), "last 270": float(np.mean([x["loss"] for x in log[-270:]]))}, "monitor": {str(k): v for k, v in mon.items()},
                       "training_readout": fr, "seconds": time.time() - t0, "gpu_seconds": gsec, "a08_sha256": ctx.pin["A08_weights"][s], "prefix_cache_key": cache_key(ctx, s) if arm == "T" else None}
                write_json(rd / "run_manifest.json", json.loads(json.dumps(man, default=str)))
                print(f"[{arm}_s{s}] {len(log)} updates in {man['seconds']:.0f} s; last-270 loss {man['loss']['last 270']:.4f}; P fixed {fr['P']['fixed']} broken {fr['P']['broken']} of {fr['P']['base_wrong']} wrong; final {man['final_sha256'][:12]}", flush=True); ok = True
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {arm}_s{s} ({'CPU' if arm == 'F' else 'GPU job wall time'})", "cpu" if arm == "F" else "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- check
def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note, res = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config)}
    try:
        s = 0; I = ctx.intervals(s); D = t27.pool_data(ctx.a27, s); P = ctx.a27.pool(); emb = D["emb"]; px = Prefix(ctx, s); init = ctx.a27.init(s); a08_file = sha256_file(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt")
        crec = json.loads((ctx.out / "cache" / f"prefix_s{s}.json").read_text()); res["prefix_cache"] = {k: crec[k] for k in ("windows", "logits_max_abs_against_A15", "argmax_mismatches", "file_bytes")}
        fit = bc.members(P, "large"); seqs = P["seqs"]; W, refs, first = pool_windows(P, fit[:3]); fp = t28.frame_paths(P["seqs"]); frames = rr.decode_frames(refs, fp, 0)
        win = W["ref"][0]; res["windows"] = {"first_window_samples": W["s"][0].tolist(), "marker": int(W["marker"][0]), "frames_from_refs": bool(np.array_equal(frames[win], rr.decode_frames([refs[k] for k in win], fp, 0)))}
        m0 = ic.branch(ctx.a08(s, device), True); head0 = ir.copied_head(*init); pk = I["P"]; idx = np.arange(len(pk["labels"])); rows = np.unique(pk["rows"])
        zf = window_logits("F", head0, rows, emb, None, device); zt = window_logits("T", m0, rows, emb, px, device); zq = torch.zeros_like(zf); zq[rows] = torch.from_numpy(D["logits"][rows])
        base_cls = np.asarray([c for i in fit for _, _, c in ir.runs(D["base"][i])])
        res["zero_update"] = {"F_logits_max_abs": float((zf[rows] - zq[rows]).abs().max()), "T_logits_max_abs": float((zt[rows] - zq[rows]).abs().max()),
                              "F_argmax_equal": bool(torch.equal(zf[rows].argmax(1), zq[rows].argmax(1))), "T_argmax_equal": bool(torch.equal(zt[rows].argmax(1), zq[rows].argmax(1))),
                              "F_run_labels_equal_base": bool(np.array_equal(pooled_probs(zf, pk, idx).argmax(1), base_cls)), "T_run_labels_equal_base": bool(np.array_equal(pooled_probs(zt, pk, idx).argmax(1), base_cls))}
        pz = rr.per_sample(P["W"], head_probs_all(head0, emb), seqs); res["zero_update"]["F_pool_sequences_equal_base"] = int(sum(np.array_equal(ir.relabel(D["base"][i], pz[q["id"]]), D["base"][i]) for i, q in enumerate(seqs)))
        res["zero_update"]["pool_sequences"] = len(seqs); res["base_rows_and_dev_zero_update"] = t27.evaluate(ctx.a27, list(SEEDS), zero_only=True)
        rng = np.random.default_rng(0); z = torch.randn(9, 5, dtype=torch.float64); w = torch.tensor([.5, .3, .2, 1., .25, .25, .25, .25, 1.], dtype=torch.float64); ix = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 3]); y = torch.tensor([1, 4, 0, 2])
        p = torch.softmax(z, -1).numpy(); ref = [-np.log(sum(w[k].item() * p[k, y[j]] for k in range(9) if ix[k] == j)) for j in range(4)]; _ = rng
        res["pooled_ce_against_numpy_max_abs"] = float(np.abs(pooled_ce(z, w, ix, y).numpy() - np.asarray(ref)).max())
        item = UpdateRows(ctx, s, I, [WARM + 5])[0]; ma = ic.branch(ctx.a08(s, device), True); oa = t_opt(ma); mb = ic.branch(ctx.a08(s, device), True); ob = t_opt(mb)
        schedule(oa, WARM + 5); schedule(ob, WARM + 5); ga, gb = {}, {}
        for mm_, micro, gd in ((ma, None, ga), (mb, 7, gb)):
            u_, tok, inv, w_, ix_, y_ = item; inv, w_, ix_, y_ = (x.to(device) for x in (inv, w_, ix_, y_)); mm_.zero_grad(set_to_none=True)
            if micro is None:
                pooled_ce(t_forward(mm_, tok, device)[inv], w_, ix_, y_).mean().backward()
            else:
                n = int(y_.shape[0])
                for a in range(0, n, micro):
                    sel = (ix_ >= a) & (ix_ < a + micro); r = torch.unique(inv[sel]); loc = torch.searchsorted(r, inv[sel])
                    (pooled_ce(t_forward(mm_, tok[r.cpu()], device)[loc], w_[sel], ix_[sel] - a, y_[a:a + micro]).sum() / n).backward()
            gd.update({n_: p_.grad.detach().clone() for n_, p_ in ic.trainable(mm_)})
        res["microbatch_grad_max_abs"] = float(max((ga[k] - gb[k]).abs().max() for k in ga)); res["grad_max"] = float(max(v.abs().max() for v in ga.values()))
        res["grads_only_trainable"] = all(p_.grad is None for n_, p_ in ma.named_parameters() if not p_.requires_grad)
        us = range(WARM, WARM + 10); hf = ir.copied_head(*init); of = f_opt(hf); e = torch.from_numpy(emb)
        for u in us:
            f_step(hf, of, e, I, u)
        mt = ic.branch(ctx.a08(s, device), True); ot = t_opt(mt, rep_lr=0.0); rep0 = {n: p_.detach().clone() for n, p_ in ic.trainable(mt) if not n.startswith("classifier.")}; pd0 = t28.prefix_digest(mt)
        for it in DataLoader(UpdateRows(ctx, s, I, us), batch_size=None, num_workers=2):
            t_step(mt, ot, it, device)
        res["forced_equal"] = {"classifier_max_abs": float(max((mt.classifier.weight.detach().cpu() - hf.weight.detach()).abs().max(), (mt.classifier.bias.detach().cpu() - hf.bias.detach()).abs().max())),
                               "classifier_moved_max_abs": float((hf.weight.detach() - init[0]).abs().max()), "representation_unchanged": all(torch.equal(p_.detach(), rep0[n]) for n, p_ in ic.trainable(mt) if not n.startswith("classifier.")),
                               "prefix_unchanged": t28.prefix_digest(mt) == pd0}

        def run(us_, m=None, o=None):
            if m is None:
                m = ic.branch(ctx.a08(s, device), True); o = t_opt(m)
            for it in DataLoader(UpdateRows(ctx, s, I, us_), batch_size=None, num_workers=2):
                t_step(m, o, it, device)
            return m, o
        a, _ = run(range(WARM - 3, WARM + 3)); b, bo = run(range(WARM - 3, WARM)); st = {"t": {n: p_.detach().clone() for n, p_ in ic.trainable(b)}, "o": bo.state_dict()}
        c = ic.branch(ctx.a08(s, device), True); co = t_opt(c)
        with torch.no_grad():
            for n, p_ in ic.trainable(c):
                p_.copy_(st["t"][n])
        co.load_state_dict(st["o"]); run(range(WARM, WARM + 3), c, co); pa = dict(ic.trainable(a)); res["resume_params_max_abs"] = float(max((pa[n] - p_).detach().abs().max() for n, p_ in ic.trainable(c)))
        tr = ic.trainable(m0); oo = t_opt(m0); res["trainable"] = {"parameters": int(sum(p_.numel() for _, p_ in tr)), "binds_exactly": sorted(id(p_) for g in oo.param_groups for p_ in g["params"]) == sorted(id(p_) for _, p_ in tr)}
        tm0 = time.time(); g0 = 0; run(range(WARM + 10, WARM + 40)); spu = (time.time() - tm0) / 30; _ = g0
        crec2 = {s_: json.loads((ctx.out / "cache" / f"prefix_s{s_}.json").read_text()) if (ctx.out / "cache" / f"prefix_s{s_}.json").exists() else None for s_ in SEEDS}
        cache_h = crec["gpu_seconds"] / 3600; spent = vsh.ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600; eval_h = len(SEEDS) * 32862 * 0.0165 / 3600
        proj = {"remaining_prefix_caches": sum(cache_h * 1.15 for s_ in SEEDS if crec2[s_] is None), "T_training": len(SEEDS) * UPDATES * spu / 3600, "T_evaluation_online": eval_h}
        res["timing"] = {"T_seconds_per_update_wall": spu, "prefix_cache_gpu_hours_seed0": cache_h}
        res["budget"] = {"gpu_spent_incl_this_check": spent, "projection_two_seeds": spent + sum(proj.values()), "parts": proj, "seed1_reserve": cache_h * 1.15 + 0.35 + UPDATES * spu / 3600 + eval_h / 2, "cap": 6.0}
        res["budget"]["fits_with_seed1"] = bool(res["budget"]["projection_two_seeds"] + res["budget"]["seed1_reserve"] <= 6.0); res["a08_file_unchanged"] = sha256_file(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt") == a08_file
        zu = res["zero_update"]
        res["ok"] = bool(res["prefix_cache"]["argmax_mismatches"] == 0 and res["windows"]["frames_from_refs"] and zu["F_logits_max_abs"] <= 1e-4 and zu["T_logits_max_abs"] <= 1e-4 and zu["F_argmax_equal"] and zu["T_argmax_equal"]
                         and zu["F_run_labels_equal_base"] and zu["T_run_labels_equal_base"] and zu["F_pool_sequences_equal_base"] == zu["pool_sequences"] and res["base_rows_and_dev_zero_update"]["ok"]
                         and res["pooled_ce_against_numpy_max_abs"] < 1e-10 and res["microbatch_grad_max_abs"] <= 1e-6 * max(1.0, res["grad_max"]) * 10 and res["grads_only_trainable"]
                         and res["forced_equal"]["classifier_max_abs"] <= 1e-4 and res["forced_equal"]["representation_unchanged"] and res["forced_equal"]["prefix_unchanged"] and res["resume_params_max_abs"] <= 1e-5
                         and res["trainable"]["parameters"] == 7497989 and res["trainable"]["binds_exactly"] and res["a08_file_unchanged"] and res["budget"]["fits_with_seed1"])
        res["finished"] = now()
        for p_ in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p_.parent.mkdir(parents=True, exist_ok=True); write_json(p_, json.loads(json.dumps(res, default=str)))
        print(json.dumps(res, indent=1, default=str)[:6000], flush=True)
        if not res["ok"]:
            raise SystemExit("a check failed")
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def head_probs_all(head, emb: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return head(torch.from_numpy(emb)).numpy()


# ----------------------------------------------------------------------------------------------- freeze
def freeze(ctx: Ctx, seeds: list[int]) -> None:
    t0, started, ok, note, out = time.time(), now(), False, "", {}
    try:
        for s in seeds:
            for arm in ARMS:
                rd = ctx.out / "train" / f"{arm}_s{s}"; man = json.loads((rd / "run_manifest.json").read_text()); dest = ROOT / "models_frozen/video_native_window_recognition_v1" / f"{arm}_s{s}"
                if not dest.exists():
                    subprocess.run([sys.executable, "scripts/freeze_model.py", "video_native_window_recognition_v1", f"{arm}_s{s}", "--file", str(rd / "final.pt"), "--file", str(rd / "run_manifest.json"),
                                    "--meta", f"a08_base={man['a08_sha256']}", "--notes", f"A29 {arm}_s{s} recognition branch trainable tensors (apply onto A08 all_s{s}); final update 1350"], cwd=ROOT, check=True, capture_output=True)
                if sha256_file(dest / "final.pt") != man["final_sha256"]:
                    raise SystemExit(f"{arm}_s{s}: the frozen copy differs")
                out[f"{arm}_s{s}"] = {k: man[k] for k in ("final_sha256", "trainable_parameters", "frozen_prefix_unchanged", "loss", "training_readout", "monitor", "seconds", "gpu_seconds", "updates")}
        rec = {"frozen": out, "config_sha256": sha256_file(ROOT / ctx.config), "time": now()}
        for p in (ctx.out / "training_summary.json", ctx.results / "training_summary.json"):
            write_json(p, rec)
        print(json.dumps({k: {x: v[x] for x in ("final_sha256", "trainable_parameters", "loss")} for k, v in out.items()}, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "freeze (CPU)", "cpu", started, time.time() - t0, ok, note)


def load_arm(ctx: Ctx, s: int, arm: str, device):
    summ = json.loads((ctx.out / "training_summary.json").read_text())["frozen"][f"{arm}_s{s}"]["final_sha256"]; p = ROOT / "models_frozen/video_native_window_recognition_v1" / f"{arm}_s{s}" / "final.pt"
    if sha256_file(p) != summ:
        raise SystemExit(f"{arm}_s{s}: frozen weights differ from the training summary")
    st = torch.load(p, map_location="cpu", weights_only=True)
    if arm == "F":
        h = torch.nn.Linear(512, 5); h.load_state_dict({k.split(".", 1)[1]: v for k, v in st.items()}); return h.eval()
    m = ic.branch(ctx.a08(s, device), True)
    if set(st) != {n for n, _ in ic.trainable(m)}:
        raise SystemExit(f"T_s{s}: stored tensors do not match the trainable set")
    with torch.no_grad():
        for n, p_ in ic.trainable(m):
            p_.copy_(st[n].to(p_.device))
    return m.eval()


# ----------------------------------------------------------------------------------------------- evaluate
def online_logits(model, seqs: list[dict], device, workers: int, acc: dict) -> tuple[np.ndarray, dict]:
    refs = sorted({tuple(r) for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5); orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); out = []
    CH = 150
    for c0 in range(0, len(seqs), CH):
        ch = seqs[c0:c0 + CH]; rf = sorted({tuple(r) for q in ch for r in q["refs"]}); a_ = {r: i for i, r in enumerate(rf)}; Wc = rr.half_windows(ch, a_, 5); oc = np.broadcast_to(np.arange(16, dtype=np.int64), Wc["s"].shape).copy()
        frames = rr.decode_frames(rf, t28.frame_paths(ch), workers); lg, _, sec = rsh.infer(model, Wc, oc, frames, device); acc["gpu_seconds"] += sec; acc["windows"] += len(Wc["seq"]); out.append(lg.astype(np.float32)); del frames
    _ = orders
    return np.concatenate(out), W


def evaluate(ctx: Ctx, seeds: list[int], device, workers: int) -> None:
    t0, started, ok, note, acc = time.time(), now(), False, "", {"gpu_seconds": 0.0, "windows": 0}
    try:
        if (ctx.out / "screen.json").exists():
            raise SystemExit("A29 was evaluated already; evaluation runs once")
        heads = {f"{a}_s{s}": json.loads((ctx.out / "training_summary.json").read_text())["frozen"][f"{a}_s{s}"]["final_sha256"] for s in seeds for a in ARMS}; a8 = rsh.a08_ctx(); systems, itab, seqs_of = {}, {}, {}
        for t in (*HALVES, "check"):
            for s in seeds:
                ev, seqs, base, brow = t28.dev_set(ctx, a8, t, s) if t != "check" else t28.check_set(ctx, a8, s); seqs_of[t] = seqs
                if t != "check":
                    z = t27.pinned_npz(ROOT / f"runs/video_frozen_switch_v1/evaluation/{t}/a08_s{s}.npz", ctx.a27.pin["dev_caches"][t][s]); emb = z["emb"].astype(np.float32); refs = sorted({tuple(r) for q in seqs for r in q["refs"]})
                    W = rr.half_windows(seqs, {r: i for i, r in enumerate(refs)}, 5)
                else:
                    P = ctx.a27.pool(); cidx = bc.members(P, "check"); keep = np.isin(P["W"]["seq"], cidx); remap = {g: k for k, g in enumerate(cidx)}; D = t27.pool_data(ctx.a27, s); emb = D["emb"][keep]
                    W = {"seq": np.asarray([remap[int(x)] for x in P["W"]["seq"][keep]]), "centre": P["W"]["centre"][keep]}
                F_ = load_arm(ctx, s, "F", device); lf = head_probs_all(F_, emb); T_ = load_arm(ctx, s, "T", device); lt, Wt = online_logits(T_, seqs, device, workers, acc)
                if not (np.array_equal(Wt["seq"], W["seq"]) and np.array_equal(Wt["centre"], W["centre"])):
                    raise SystemExit("T's online windows differ from the cached windows")
                preds = {"Base": base, "F": {k: ir.relabel(base[k], v) for k, v in rr.per_sample(W, lf, seqs).items()}, "T": {k: ir.relabel(base[k], v) for k, v in rr.per_sample(W, lt, seqs).items()}}
                runs_of = {q["id"]: [((a, e, c), 0) for a, e, c in ir.runs(base[q["id"]])] for q in seqs}
                for name, pr in preds.items():
                    if name == "Base":
                        sysd = brow
                    elif t == "check":
                        m_, tb = bc.evaluate_pool(a8.cfg, pr, ev); sysd = {"row": bce.check_row(m_), "tables": tb, "preds": pr}
                    else:
                        sysd = rso.row_of(a8.cfg, pr, ev)
                    systems.setdefault(f"{name} s{s}", {})[t] = sysd
                for name, v in t28.interval_table(seqs, base, runs_of, preds).items():
                    itab.setdefault(f"{name} s{s}", {})[t] = v
                print(f"{t} s{s}: done ({acc['windows']} T windows so far)", flush=True)
        rows = {k: {t: v[t]["row"] for t in v} for k, v in systems.items()}; counts = {k: {t: rsh.counts_of(seqs_of[t], v[t]) for t in v} for k, v in systems.items()}; cmp_, acc_ = {}, {}
        for t in (*HALVES, "check"):
            for s in seeds:
                rhr.compare(seqs_of[t], systems, t, s, CMPS, cmp_, acc_)
        a27 = json.loads((ROOT / "docs/results/2026-10-23_interval_recognition/screen.json").read_text())["systems"]; a28 = json.loads((ROOT / "docs/results/2026-10-24_interval_visual_adaptation/screen.json").read_text())["systems"]
        context = {**{f"A27 {a} s{s}": a27[f"{a} s{s}"] for a in ("W", "I") for s in seeds}, **{f"A28 {a} s{s}": a28[f"{a} s{s}"] for a in ("Init", "Frozen", "Adapted") for s in seeds}}
        res = {"stage": "evaluate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "systems": rows, "segment_counts": counts, "interval_table": itab, "comparisons": cmp_, "accounting": acc_,
               "context_rows": context, "gate": gate(rows, counts), "heads": heads,
               "cost": {"T_branch_gpu_seconds_per_window": acc["gpu_seconds"] / max(1, acc["windows"]), "T_windows": acc["windows"], "F_extra": "one 512x5 linear layer on A08's own embeddings",
                        "T_extra": "a second suffix (last block, norm, projection, classifier) per request window; shared frozen prefix possible, but this evaluation ran a full copy"}, "time": now()}
        ids = {q["id"] for t in seqs_of for q in seqs_of[t] if q["kind"] == "native"} | {r[0] for t in seqs_of for q in seqs_of[t] for r in q["refs"]}
        flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "screen")
        for p in (ctx.out / "screen.json", ctx.results / "screen.json"):
            write_json(p, flat)
        print(json.dumps({k: {x: flat["gate"][k][x] for x in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "calibration_feasible", "passes")} for k in ARMS}, indent=1), flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "evaluate once (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def gate(rows: dict, counts: dict) -> dict:
    out = t28.gate({k.replace("F s", "Frozen s").replace("T s", "Adapted s") if k.split(" ")[0] in ("F", "T") else k: v for k, v in rows.items()},
                   {k.replace("F s", "Frozen s").replace("T s", "Adapted s") if k.split(" ")[0] in ("F", "T") else k: v for k, v in counts.items()})
    return {"F": out["Frozen"], "T": out["Adapted"], "train_seed1": out["train_seed1"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("prepare", "cache", "check", "train", "freeze", "evaluate"), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS)); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.stage in ("prepare", "freeze"):
        return prepare(ctx) if a.stage == "prepare" else freeze(ctx, a.seeds)
    vsh.determinism(); device = torch.device("cuda")
    if a.stage == "cache":
        for s in a.seeds:
            build_cache(ctx, s, device, a.workers)
        return
    {"check": lambda: check(ctx, device, a.workers), "train": lambda: train(ctx, a.seeds, device, a.workers), "evaluate": lambda: evaluate(ctx, a.seeds, device, a.workers)}[a.stage]()


if __name__ == "__main__":
    main()
