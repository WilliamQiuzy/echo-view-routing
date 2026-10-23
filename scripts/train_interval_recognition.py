#!/usr/bin/env python
"""A27 (configs/experiments/video_interval_recognition_v1.yaml): an independent copy of A08's 512->5 classifier trained on
A08's frozen cached embeddings with a window (W) or interval (I) objective plus the same own-A08 retention KL; deployed by
relabelling the fixed Base + B3 runs (echo_routing/temporal/interval_recognition.py). CPU only for seeds 0 and 2.

  --stage prepare   rebuild A15's pinned pool, check sources and caches, build the T / P interval lists and the batches
  --stage check     copied-head parity, zero-update Base reproduction (pool and four stored development rows), objectives,
                    gradient isolation, original-q immutability, 3 + 3 resume, timing
  --stage train     W and I per seed (1,350 updates each)
  --stage freeze    freeze the final heads and write the training summary
  --stage evaluate  once: development halves and the A15 check pool; rows, counts, comparisons, gate
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_hard_replay as rhr  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

B3C_CONFIG = "configs/experiments/video_b3_coverage_v1.yaml"
SEEDS, ARMS, HALVES = (0, 2), ("W", "I"), ("selection", "calibration")
UPDATES, WARM, PER_LIST = 1350, 135, 16
OPT = {"lr": 1e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "clip": 1.0}
CMPS = {"I - W": ("I", "W"), "W - Base": ("W", "Base"), "I - Base": ("I", "Base")}
now, write_json, ledger_append = vsh.now, vsh.write_json, vsh.ledger_append


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vi = cfg, config, cfg["video_interval_recognition"]; self.pin = self.vi["pinned"]; o = self.vi["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.vi["budget"]["ledger"]; self._b, self._P = None, None
        for k in ("A08_choices", "A15_config", "A15_pool_manifest", "A15_preparation", "check_rows"):
            if sha256_file(ROOT / self.pin[k]["path"]) != self.pin[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")

    def bctx(self):
        if self._b is None:
            self._b = bc.Ctx(load_cfg(SimpleNamespace(config=B3C_CONFIG, set=[])), B3C_CONFIG)
        return self._b

    def pool(self) -> dict:
        if self._P is None:
            self._P = bc.with_index(self.bctx(), bc.pool(self.bctx()))
        return self._P

    def params(self, s: int) -> dict:
        return json.loads((ROOT / self.pin["A08_choices"]["path"]).read_text())["choices"][f"all_s{s}"]["params"]

    def init(self, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt"
        if sha256_file(p) != self.pin["A08_weights"][s]:
            raise SystemExit(f"A08 all_s{s}: weights differ from the pin")
        sd = torch.load(p, map_location="cpu", weights_only=True); return sd["classifier.weight"].float(), sd["classifier.bias"].float()


def pinned_npz(path: Path, sha: str) -> dict:
    if sha256_file(path) != sha:
        raise SystemExit(f"{path}: differs from its pin")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def nearest_rows(W: dict, seqs: list[dict]) -> list[np.ndarray]:
    """Per sequence, the window row each sample maps to (the deployed nearest-centre rule)."""
    out = []
    for qi, q in enumerate(seqs):
        rows = np.flatnonzero(W["seq"] == qi); out.append(rows[rr.nearest_centre(len(q["labels"]), W["centre"][rows])])
    return out


def pool_data(ctx: Ctx, s: int) -> dict:
    P = ctx.pool(); win = pinned_npz(ROOT / "runs/video_router_data_v1/features" / f"windows_s{s}.npz", ctx.pin["A15_windows"][s])
    if not (np.array_equal(win["seq"], P["W"]["seq"]) and np.array_equal(win["centre"], P["W"]["centre"])):
        raise SystemExit(f"seed {s}: A15's windows differ from the rebuilt pool's")
    pr = pinned_npz(ROOT / "runs/video_router_data_v1/features" / f"preds_s{s}.npz", ctx.pin["A15_preds"][s]); off = pr["offsets"]
    base = [pr["base"][off[i]:off[i + 1]].astype(np.int64) for i in range(len(off) - 1)]; labels = [pr["labels"][off[i]:off[i + 1]].astype(np.int64) for i in range(len(off) - 1)]
    return {"P": P, "logits": win["logits"].astype(np.float32), "emb": win["emb"].astype(np.float32), "base": base, "labels": labels, "nearest": nearest_rows(P["W"], P["seqs"])}


# ----------------------------------------------------------------------------------------------- prepare
def prepare(ctx: Ctx) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "seeds": {}}
    try:
        P = ctx.pool(); fit, chk = bc.members(P, "large"), bc.members(P, "check"); seqs = P["seqs"]
        recs = {g: {r[0] for i in m for r in seqs[i]["refs"]} for g, m in (("fit", fit), ("check", chk))}; allow = set(prv.allowlist(ctx.bctx().a15.a10.rv, ROOT))
        rec["pool"] = {"digest_equals_pin": True, "fit_sequences": len(fit), "check_sequences": len(chk), "fit_recordings": len(recs["fit"]), "check_recordings": len(recs["check"]),
                       "recordings_shared": len(recs["fit"] & recs["check"]), "sessions_shared": len({session_of(v) for v in recs["fit"]} & {session_of(v) for v in recs["check"]}),
                       "all_protocol_train": bool((recs["fit"] | recs["check"]) <= allow), "fit_kinds": {k: sum(seqs[i]["kind"] == k for i in fit) for k in ("multi", "native")}}
        if rec["pool"]["recordings_shared"] or rec["pool"]["sessions_shared"] or not rec["pool"]["all_protocol_train"]:
            raise SystemExit(f"pool separation or source check failed: {rec['pool']}")
        ddir = ctx.out / "data"; ddir.mkdir(parents=True, exist_ok=True)
        for s in SEEDS:
            D = pool_data(ctx, s); probs = rr.per_sample(P["W"], D["logits"], seqs); rb = decode(ctx.params(s), probs)
            lab_ok = all(np.array_equal(D["labels"][i], np.asarray(q["labels"])) for i, q in enumerate(seqs)); base_ok = all(np.array_equal(D["base"][i], rb[q["id"]]) for i, q in enumerate(seqs))
            if not (lab_ok and base_ok):
                raise SystemExit(f"seed {s}: cached labels or Base differ from the rebuilt ones")
            T, Pl = [], []
            for i in fit:
                d = ir.build_intervals(D["labels"][i], D["base"][i], D["nearest"][i]); T += d["T"]; Pl += d["P"]
            pt, pp = ir.pack(T), ir.pack(Pl); rng = np.random.default_rng([20261023, 27, s]); bt, bp = [], []
            for _ in range(UPDATES):
                bt.append(rng.integers(0, len(T), PER_LIST)); bp.append(rng.integers(0, len(Pl), PER_LIST))
            np.savez(ddir / f"intervals_s{s}.npz", **{f"T_{k}": v for k, v in pt.items()}, **{f"P_{k}": v for k, v in pp.items()}, batch_T=np.asarray(bt), batch_P=np.asarray(bp))
            def summ(x, items):
                n = np.diff(x["offsets"]); return {"intervals": len(items), "per_class": np.bincount(x["labels"], minlength=5).tolist(), "length_le16": int((x["lengths"] <= 16).sum()),
                                                  "requests_1": int((n == 1).sum()), "requests_2_3": int(((n >= 2) & (n <= 3)).sum()), "requests_ge4": int((n >= 4).sum()),
                                                  "mixed": int((x["purity"] < 1).sum()), "mean_purity": float(x["purity"].mean())}
            rec["seeds"][str(s)] = {"T": summ(pt, T), "P": summ(pp, Pl), "base_recomputed_equal": base_ok, "labels_equal": lab_ok}
            rec["seeds"][str(s)]["P_label_differs_from_base_class"] = int(sum(1 for i in fit for a, b, c in ir.runs(D["base"][i]) if ir.majority(D["labels"][i][a:b])[0] != c))
            rec["seeds"][str(s)]["intervals_sha256"] = sha256_file(ddir / f"intervals_s{s}.npz")
            print(json.dumps(rec["seeds"][str(s)]), flush=True)
        rec["finished"] = now(); ids = {q["id"] for q in seqs if q["kind"] == "native"} | recs["fit"] | recs["check"]; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare (CPU)", "cpu", started, time.time() - t0, ok, note)


def load_intervals(ctx: Ctx, s: int) -> dict:
    p = ctx.out / "data" / f"intervals_s{s}.npz"; prep = json.loads((ctx.out / "preparation.json").read_text())
    if sha256_file(p) != prep["seeds"][str(s)]["intervals_sha256"]:
        raise SystemExit(f"seed {s}: the interval file differs from preparation.json")
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    return {"T": {k[2:]: v for k, v in d.items() if k.startswith("T_")}, "P": {k[2:]: v for k, v in d.items() if k.startswith("P_")}, "bt": d["batch_T"], "bp": d["batch_P"]}


# ----------------------------------------------------------------------------------------------- training
def batch_of(I: dict, u: int):
    rt, wt, it, yt = ir.gather(I["T"], I["bt"][u]); rp, wp, ip, yp = ir.gather(I["P"], I["bp"][u])
    return torch.cat([rt, rp]), torch.cat([wt, wp]), torch.cat([it, ip + PER_LIST]), torch.cat([yt, yp])


def make(init: tuple) -> tuple:
    head = ir.copied_head(*init); opt = torch.optim.AdamW(head.parameters(), lr=OPT["lr"], betas=OPT["betas"], eps=OPT["eps"], weight_decay=OPT["weight_decay"]); return head, opt


def steps(head, opt, arm: str, emb: torch.Tensor, ql: torch.Tensor, I: dict, us, log: list | None = None) -> None:
    for u in us:
        for g in opt.param_groups:
            g["lr"] = OPT["lr"] * lr_factor(int(u), WARM, UPDATES)
        rows, w, ix, y = batch_of(I, u); o = ir.objective(arm, head(emb[rows]), ql[rows], w, ix, y)
        opt.zero_grad(set_to_none=True); o["loss"].backward(); gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), OPT["clip"])); opt.step()
        if log is not None:
            log.append({"u": int(u), "loss": float(o["loss"]), "cls": float(o["cls"]), "kl": float(o["kl"]), "grad_norm": gn})


def train(ctx: Ctx, seeds: list[int]) -> None:
    torch.set_num_threads(1)
    for s in seeds:
        D = pool_data(ctx, s); I = load_intervals(ctx, s); emb, ql = torch.from_numpy(D["emb"]), torch.from_numpy(D["logits"]); init = ctx.init(s)
        for arm in ARMS:
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                head, opt = make(init); log = []; steps(head, opt, arm, emb, ql, I, range(UPDATES), log); rd = ctx.out / "train" / f"{arm}_s{s}"; rd.mkdir(parents=True, exist_ok=True)
                torch.save(head.state_dict(), rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                man = {"arm": arm, "seed": s, "config_sha256": sha256_file(ROOT / ctx.config), "init_a08_sha256": ctx.pin["A08_weights"][s], "intervals_sha256": sha256_file(ctx.out / "data" / f"intervals_s{s}.npz"),
                       "windows_sha256": ctx.pin["A15_windows"][s], "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "trainable_parameters": int(sum(p.numel() for p in head.parameters())),
                       "seconds": time.time() - t0, "phases": {k: {m: float(np.mean([x[m] for x in log[a:b]])) for m in ("loss", "cls", "kl", "grad_norm")} for k, (a, b) in (("first 135", (0, 135)), ("last 270", (UPDATES - 270, UPDATES)))}}
                write_json(rd / "run_manifest.json", man); print(f"[{arm}_s{s}] {len(log)} updates in {man['seconds']:.1f} s; last-270 loss {man['phases']['last 270']['loss']:.4f}; final {man['final_sha256'][:12]}", flush=True); ok = True
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {arm}_s{s} (CPU, one thread)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- check
def head_probs(head, emb: np.ndarray) -> np.ndarray:
    """The head's logits (rr.per_sample turns logits into per-sample probabilities)."""
    with torch.no_grad():
        return head(torch.from_numpy(emb)).numpy()


def check(ctx: Ctx) -> None:
    t0, started, ok, note, res = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config)}
    try:
        torch.set_num_threads(1); import tests.test_interval_recognition as T  # noqa: E402
        res["unit_tests"] = {n: (getattr(T, n)() is None) for n in dir(T) if n.startswith("test_")}
        P = ctx.pool(); seqs = P["seqs"]
        for s in SEEDS:
            D = pool_data(ctx, s); init = ctx.init(s); h0 = ir.copied_head(*init); lg0 = head_probs(h0, D["emb"]); r = {}
            r["copied_head_logits_max_abs"] = float(np.abs(lg0 - D["logits"]).max()); r["copied_head_argmax_agreement"] = float((lg0.argmax(1) == D["logits"].argmax(1)).mean())
            pz = rr.per_sample(P["W"], lg0, seqs); r["zero_update_pool_sequences_equal_base"] = int(sum(np.array_equal(ir.relabel(D["base"][i], pz[q["id"]]), D["base"][i]) for i, q in enumerate(seqs)))
            r["pool_sequences"] = len(seqs)
            I = load_intervals(ctx, s); emb, ql = torch.from_numpy(D["emb"]), torch.from_numpy(D["logits"]); qd = hashlib.sha256(D["logits"].tobytes()).hexdigest()
            head, opt = make(init); steps(head, opt, "I", emb, ql, I, range(6)); straight = {k: v.clone() for k, v in head.state_dict().items()}
            h2, o2 = make(init); steps(h2, o2, "I", emb, ql, I, range(3)); torch.save({"h": h2.state_dict(), "o": o2.state_dict()}, ctx.out / "check_resume.pt")
            st = torch.load(ctx.out / "check_resume.pt", weights_only=False); h3, o3 = make(init); h3.load_state_dict(st["h"]); o3.load_state_dict(st["o"]); steps(h3, o3, "I", emb, ql, I, range(3, 6))
            r["resume_params_max_abs"] = float(max((straight[k] - v).abs().max() for k, v in h3.state_dict().items())); (ctx.out / "check_resume.pt").unlink()
            r["original_q_unchanged"] = hashlib.sha256(D["logits"].tobytes()).hexdigest() == qd and not emb.requires_grad and not ql.requires_grad
            r["trainable_parameters"] = int(sum(p.numel() for p in head.parameters() if p.requires_grad)); r["optimiser_parameters"] = int(sum(p.numel() for g in opt.param_groups for p in g["params"]))
            hw, ow = make(init); a = time.time(); steps(hw, ow, "W", emb, ql, I, range(50)); r["seconds_per_update"] = (time.time() - a) / 50
            res[f"s{s}"] = r
        res["development_zero_update"] = evaluate(ctx, list(SEEDS), zero_only=True)
        res["ok"] = bool(all(res["unit_tests"].values()) and all(res[f"s{s}"]["zero_update_pool_sequences_equal_base"] == res[f"s{s}"]["pool_sequences"] and res[f"s{s}"]["copied_head_logits_max_abs"] <= 1e-4
                                                               and res[f"s{s}"]["resume_params_max_abs"] == 0.0 and res[f"s{s}"]["original_q_unchanged"] and res[f"s{s}"]["trainable_parameters"] == 2565
                                                               and res[f"s{s}"]["optimiser_parameters"] == 2565 for s in SEEDS) and res["development_zero_update"]["ok"])
        res["finished"] = now()
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, json.loads(json.dumps(res, default=str)))
        print(json.dumps(res, indent=1, default=str)[:5000], flush=True)
        if not res["ok"]:
            raise SystemExit("a check failed")
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- freeze
def freeze(ctx: Ctx, seeds: list[int]) -> None:
    t0, started, ok, note, out = time.time(), now(), False, "", {}
    try:
        for s in seeds:
            for arm in ARMS:
                rd = ctx.out / "train" / f"{arm}_s{s}"; man = json.loads((rd / "run_manifest.json").read_text()); dest = ROOT / "models_frozen/video_interval_recognition_v1" / f"{arm}_s{s}"
                if not dest.exists():
                    subprocess.run([sys.executable, "scripts/freeze_model.py", "video_interval_recognition_v1", f"{arm}_s{s}", "--file", str(rd / "final.pt"), "--file", str(rd / "run_manifest.json"),
                                    "--meta", f"run_manifest=runs/video_interval_recognition_v1/train/{arm}_s{s}/run_manifest.json", "--notes", f"A27 {arm}_s{s} recognition head, final update 1350"],
                                   cwd=ROOT, check=True, capture_output=True)
                if sha256_file(dest / "final.pt") != man["final_sha256"]:
                    raise SystemExit(f"{arm}_s{s}: the frozen copy differs")
                out[f"{arm}_s{s}"] = {"final_sha256": man["final_sha256"], "phases": man["phases"], "seconds": man["seconds"], "trainable_parameters": man["trainable_parameters"]}
        rec = {"frozen": out, "config_sha256": sha256_file(ROOT / ctx.config), "time": now()}
        for p in (ctx.out / "training_summary.json", ctx.results / "training_summary.json"):
            write_json(p, rec)
        print(json.dumps(rec, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "freeze (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- evaluate
def frozen_head(s: int, arm: str, want: str):
    p = ROOT / "models_frozen/video_interval_recognition_v1" / f"{arm}_s{s}" / "final.pt"
    if sha256_file(p) != want:
        raise SystemExit(f"{arm}_s{s}: frozen head differs from its record")
    h = torch.nn.Linear(512, 5); h.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)); return h.eval()


def interval_accuracy(base: dict, preds: dict, seqs: list[dict]) -> dict:
    n = good = 0
    for q in seqs:
        y = np.asarray(q["labels"])
        for a, b, _ in ir.runs(base[q["id"]]):
            n += 1; good += int(int(preds[q["id"]][a]) == ir.majority(y[a:b])[0])
    return {"base_intervals": n, "label_accuracy": good / max(1, n)}


def half_systems(ctx: Ctx, a8, h: str, seeds: list[int], heads: dict, zero_only: bool) -> tuple:
    ev, seqs = a8.half(h, rso.DevAccess({h})); refs = sorted({tuple(r) for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text()); pins = rsh.a08_output_pins()
    systems, acc = {}, {}
    for s in seeds:
        z = pinned_npz(ROOT / f"runs/video_frozen_switch_v1/evaluation/{h}/a08_s{s}.npz", ctx.pin["dev_caches"][h][s])
        if not (np.array_equal(z["seq"], W["seq"]) and np.array_equal(z["centre"], W["centre"])):
            raise SystemExit("A13's development cache does not match this half's windows")
        apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
        if sha256_file(apath) != pins[h][s]:
            raise SystemExit("A08 saved outputs differ from their pin")
        lgA, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
        if lgA is None or float(np.abs(lgA - z["logits"]).max()) > 1e-4:
            raise SystemExit("A13's cached logits differ from A08's saved outputs")
        q = rr.per_sample(W, z["logits"].astype(np.float32), seqs); base = decode(ctx.params(s), q); emb = z["emb"].astype(np.float32)
        systems[f"Base s{s}"] = rso.row_of(a8.cfg, base, ev); rso.check_stored(systems[f"Base s{s}"]["row"], a8sum["systems"][f"A + B3 s{s}"][h], f"Base s{s} {h}")
        hz = {"zero-update copy": ir.copied_head(*ctx.init(s))} if zero_only else {arm: heads[(arm, s)] for arm in ARMS}
        for name, hd in hz.items():
            p = rr.per_sample(W, head_probs(hd, emb), seqs); pred = {k: ir.relabel(base[k], v) for k, v in p.items()}   # per_sample applies the softmax
            systems[f"{name} s{s}"] = rso.row_of(a8.cfg, pred, ev); acc[f"{name} s{s}"] = interval_accuracy(base, pred, seqs)
        acc[f"Base s{s}"] = interval_accuracy(base, base, seqs)
    return systems, acc, seqs


def check_systems(ctx: Ctx, a8, seeds: list[int], heads: dict) -> tuple:
    P = ctx.pool(); cidx = bc.members(P, "check"); seqs = [P["seqs"][i] for i in cidx]; ev = bc.evalset(P, cidx, "check"); keep = np.isin(P["W"]["seq"], cidx)
    remap = {g: k for k, g in enumerate(cidx)}; Wc = {"seq": np.asarray([remap[int(x)] for x in P["W"]["seq"][keep]]), "centre": P["W"]["centre"][keep]}
    a19 = json.loads((ROOT / ctx.pin["check_rows"]["path"]).read_text())["systems"]; systems, acc = {}, {}
    for s in seeds:
        D = pool_data(ctx, s); q = rr.per_sample(Wc, D["logits"][keep], seqs); base = decode(ctx.params(s), q)
        m, t = bc.evaluate_pool(a8.cfg, base, ev); systems[f"Base s{s}"] = {"row": bce.check_row(m), "tables": t, "preds": base}
        rso.check_stored(systems[f"Base s{s}"]["row"], a19[f"A08_s{s} old"]["check"], f"Base s{s} check (A19)"); acc[f"Base s{s}"] = interval_accuracy(base, base, seqs)
        for arm in ARMS:
            p = rr.per_sample(Wc, head_probs(heads[(arm, s)], D["emb"][keep]), seqs)
            pred = {k: ir.relabel(base[k], v) for k, v in p.items()}
            m, t = bc.evaluate_pool(a8.cfg, pred, ev); systems[f"{arm} s{s}"] = {"row": bce.check_row(m), "tables": t, "preds": pred}; acc[f"{arm} s{s}"] = interval_accuracy(base, pred, seqs)
    return systems, acc, seqs


def gate(rows: dict, counts: dict) -> dict:
    cells = [(s, h) for s in SEEDS for h in HALVES]; out = {}; F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    for x in ARMS:
        dF = {f"s{s} {h}": F(x, s, h) - F("Base", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in SEEDS])) for h in HALVES}
        dN = float(np.mean([rows[f"{x} s{s}"][h]["native_macro_f1"] - rows[f"Base s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{x} s{s}"][h]["boundary_recall_multi"] - rows[f"Base s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "Base")}
        feas = {f"s{s}": bool(rows[f"{x} s{s}"]["calibration"]["native_macro_f1"] >= 0.980 and rows[f"{x} s{s}"]["calibration"]["boundary_recall_multi"] >= 0.95) for s in SEEDS}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs, "calibration_feasible": feas,
             "selection_native_floor_0.973": {f"s{s}": bool(rows[f"{x} s{s}"]["selection"]["native_macro_f1"] >= 0.973) for s in SEEDS}}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["Base"] + 3); r["feasibility"] = bool(all(feas.values()))
        r["passes"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"] and r["feasibility"]); out[x] = r
    out["train_seed1"] = bool(out["W"]["passes"] or out["I"]["passes"])
    return out


def evaluate(ctx: Ctx, seeds: list[int], zero_only: bool = False) -> dict:
    """zero_only: the zero-update copied head on the development halves (a check; no trained head is read)."""
    a8 = rsh.a08_ctx()
    if zero_only:
        out = {}
        for h in HALVES:
            systems, acc, seqs = half_systems(ctx, a8, h, seeds, {}, True)
            out[h] = {s: {"rows_equal_base": systems[f"zero-update copy s{s}"]["row"] == systems[f"Base s{s}"]["row"],
                          "labels_equal_base": all(np.array_equal(systems[f"zero-update copy s{s}"]["preds"][q["id"]], systems[f"Base s{s}"]["preds"][q["id"]]) for q in seqs)} for s in seeds}
        out["ok"] = all(v["rows_equal_base"] and v["labels_equal_base"] for h in HALVES for v in out[h].values()); return out
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        if (ctx.out / "screen.json").exists():
            raise SystemExit("A27 was evaluated already; evaluation runs once")
        summ = json.loads((ctx.out / "training_summary.json").read_text())["frozen"]; heads = {(arm, s): frozen_head(s, arm, summ[f"{arm}_s{s}"]["final_sha256"]) for s in seeds for arm in ARMS}
        systems, accs, seqs_of = {}, {}, {}
        for h in HALVES:
            S, A, sq = half_systems(ctx, a8, h, seeds, heads, False); seqs_of[h] = sq
            for k, v in S.items():
                systems.setdefault(k, {})[h] = v
            for k, v in A.items():
                accs.setdefault(k, {})[h] = v
            print(f"{h}: rows done", flush=True)
        S, A, sq = check_systems(ctx, a8, seeds, heads); seqs_of["check"] = sq
        for k, v in S.items():
            systems.setdefault(k, {})["check"] = v
        for k, v in A.items():
            accs.setdefault(k, {})["check"] = v
        rows = {k: {t: v[t]["row"] for t in v} for k, v in systems.items()}; counts = {k: {t: rsh.counts_of(seqs_of[t], v[t]) for t in v} for k, v in systems.items()}; cmp_, acc_ = {}, {}
        for t in ("selection", "calibration", "check"):
            for s in seeds:
                rhr.compare(seqs_of[t], systems, t, s, CMPS, cmp_, acc_)
        a14 = json.loads((ROOT / "docs/results/2026-10-21_local_router/screen.json").read_text())["systems"]; a20 = json.loads((ROOT / "docs/results/2026-10-22_hard_replay/screen.json").read_text())["systems"]
        context = {**{f"A14 P s{s}": {h: a14[f"P router s{s}"][h] for h in HALVES} for s in seeds}, **{f"A20 H@A08 s{s}": {h: a20[f"H@A08 s{s}"][h] for h in HALVES} for s in seeds}}
        res = {"stage": "evaluate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "systems": rows, "segment_counts": counts, "interval_label_accuracy": accs, "comparisons": cmp_,
               "accounting": acc_, "context_rows": context, "gate": gate(rows, counts), "heads": {f"{a}_s{s}": summ[f"{a}_s{s}"]["final_sha256"] for s in seeds for a in ARMS},
               "notes": {"deployment": "Base + B3 runs relabelled by the new head's mean probability, then merged", "check": "A15 check pool (recordings trained the encoder; not a holdout)"}, "time": now()}
        ids = {q["id"] for t in seqs_of for q in seqs_of[t] if q["kind"] == "native"} | {r[0] for t in seqs_of for q in seqs_of[t] for r in q["refs"]}
        flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "screen")
        for p in (ctx.out / "screen.json", ctx.results / "screen.json"):
            write_json(p, flat)
        print(json.dumps({k: {x: flat["gate"][k][x] for x in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "calibration_feasible", "passes")} for k in ARMS}, indent=1), flush=True)
        ok = True; return res
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "evaluate once (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("prepare", "check", "train", "freeze", "evaluate"), required=True); ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    {"prepare": lambda: prepare(ctx), "check": lambda: check(ctx), "train": lambda: train(ctx, a.seeds), "freeze": lambda: freeze(ctx, a.seeds), "evaluate": lambda: evaluate(ctx, a.seeds)}[a.stage]()


if __name__ == "__main__":
    main()
