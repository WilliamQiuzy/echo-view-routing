#!/usr/bin/env python
"""A25 (configs/experiments/video_segment_risk_v1.yaml): A10 R's recipe (A08 start, A10's 12 windows, CE + KL to the cached
own A08) plus lambda * L_seg, the exact expected segment-error count of each 4-window group under the model's
per-window categorical distributions: unmatched predicted + unmatched true runs at IoU >= 0.5 (the F1@50 rule), with one
correctly predicted pad position from the true neighbouring request on each side.

  --prepare          CPU: pads and truth tuples of every group (seeds 0, 1, 2)
  --check            cost cases, finite differences, groups / pads, lambda = 0 replay of A10 R_s0 / R_s2, teacher parity,
                     optimiser / encoder, resume, L_seg size and gradient share, timing and budget projection
  --queue K [K ...]  Seg_s0, Seg_s2 (Seg_s1 only after a met rule)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import segment_f1  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

A10_CONFIG = "configs/experiments/video_switch_head_v1.yaml"
WINDOWS, GROUP, K, NONE = 12, 4, 5, -1
now, write_json, ledger_hours, ledger_append, determinism, max_diff = vsh.now, vsh.write_json, vsh.ledger_hours, vsh.ledger_append, vsh.determinism, vsh.max_diff


# ----------------------------------------------------------------------------------------------- the cost
def runs(seq: tuple) -> list[tuple[int, int, int]]:
    """(class, start, end exclusive) of maximal equal-class runs."""
    out, s = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            out.append((int(seq[s]), s, i)); s = i
    return out


def cost(pred: tuple, true: tuple) -> int:
    """Unmatched predicted + unmatched true runs under same-class one-to-one matching at IoU >= 0.5, greedy by IoU."""
    P, T = runs(pred), runs(true); cands = []
    for i, (cp, a, b) in enumerate(P):
        for j, (ct, c, d) in enumerate(T):
            if cp == ct:
                inter = max(0, min(b, d) - max(a, c)); iou = inter / (max(b, d) - min(a, c))
                if iou >= 0.5:
                    cands.append((iou, i, j))
    used_p, used_t, m = set(), set(), 0
    for _, i, j in sorted(cands, reverse=True):
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); m += 1
    return (len(P) - m) + (len(T) - m)


@lru_cache(maxsize=None)
def table(pad_l: int, truth: tuple, pad_r: int) -> torch.Tensor:
    """(5, 5, 5, 5) costs of every output pattern of the group, with the fixed pads around both truth and output."""
    L = (pad_l,) if pad_l != NONE else (); R = (pad_r,) if pad_r != NONE else (); t = L + tuple(int(x) for x in truth) + R; out = torch.empty((K,) * GROUP)
    for pat in itertools.product(range(K), repeat=GROUP):
        out[pat] = cost(L + pat + R, t)
    return out


def expected_cost(p: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """p (4, 5) probabilities; exact E[cost] under independent per-window draws."""
    return torch.einsum("a,b,c,d,abcd->", p[0], p[1], p[2], p[3], C.to(p.device, p.dtype))


def seg_risk(z: torch.Tensor, keys: list) -> torch.Tensor:
    """Mean over the 3 groups of the expected segment-error count."""
    p = F.softmax(z.float(), -1); return torch.stack([expected_cost(p[GROUP * g:GROUP * (g + 1)], table(*keys[g])) for g in range(WINDOWS // GROUP)]).mean()


def cases() -> dict:
    out = {}
    A, B = 0, 1
    out["interior flip costs more than a boundary shift"] = cost((A, A, B, A, A, A), (A, A, A, A, A, A)) > cost((A, A, A, A, A, B), (A, A, A, A, B, B))
    out["shift cost (A A A A A B against A A A A B B)"] = cost((A, A, A, A, A, B), (A, A, A, A, B, B)); out["interior flip cost (A A B A A A against A^6)"] = cost((A, A, B, A, A, A), (A,) * 6)
    out["A-B-A output under A-A-A truth costs > 0"] = cost((A, A, B, A, A), (A, A, A, A, A)) > 0
    out["A-B-A output under A-B-A truth costs 0"] = cost((A, A, B, A, A), (A, A, B, A, A)) == 0
    out["erasing a true short run costs > 0"] = cost((A, A, A, A, A), (A, A, B, A, A)) > 0
    q = torch.tensor([[0.8, 0.2], [0.55, 0.45], [0.8, 0.2]]); s = q + torch.tensor([-0.1, 0.1])
    def e3(p):
        return float(sum(cost(pat, (A, A, A)) * float(p[0, pat[0]] * p[1, pat[1]] * p[2, pat[2]]) for pat in itertools.product(range(2), repeat=3)))
    out["A24 offset example: teacher risk"], out["A24 offset example: student risk"] = e3(q), e3(s); out["A24 offset example: non-zero change"] = abs(e3(s) - e3(q)) > 1e-3
    rng = np.random.default_rng(20261023); mism = 0
    for _ in range(300):
        tr = tuple(int(x) for x in rng.integers(0, 3, 6)); pr = tuple(int(x) for x in rng.integers(0, 3, 6))
        ps, ts = labels_to_segments(np.asarray(pr)), labels_to_segments(np.asarray(tr)); mm = segment_f1(ps, ts, 0.5)["matched"]
        mism += int(cost(pr, tr) != (len(ps) - mm) + (len(ts) - mm))
    out["cost equals echo_routing segment_f1 on 300 random patterns (mismatches)"] = mism; enum = 0
    for key in ((0, (0, 0, 1, 1), 1), (NONE, (2, 2, 2, 2), 2), (3, (3, 4, 3, 3), NONE), (1, (1, 1, 1, 1), 1)):
        L = (key[0],) if key[0] != NONE else (); R = (key[2],) if key[2] != NONE else (); ts = labels_to_segments(np.asarray(L + key[1] + R)); C = table(*key)
        for pat in itertools.product(range(K), repeat=GROUP):
            ps = labels_to_segments(np.asarray(L + pat + R)); mm = segment_f1(ps, ts, 0.5)["matched"]; enum += int(float(C[pat]) != (len(ps) - mm) + (len(ts) - mm))
    out["table equals segment_f1 on all 625 patterns of 4 padded groups (mismatches)"] = enum; mism += enum
    z = torch.randn(WINDOWS, K, dtype=torch.float64, requires_grad=True); keys = [(0, (0, 0, 1, 1), 1), (NONE, (2, 2, 2, 2), 2), (3, (3, 4, 3, 3), NONE)]
    def f(zz):
        p = F.softmax(zz, -1); return torch.stack([expected_cost(p[4 * g:4 * g + 4], table(*keys[g]).double()) for g in range(3)]).mean()
    g = torch.autograd.grad(f(z), z)[0]; eps = 1e-6; fd = torch.zeros_like(z)
    with torch.no_grad():
        for i in range(WINDOWS):
            for k in range(K):
                zp = z.clone(); zp[i, k] += eps; zm = z.clone(); zm[i, k] -= eps; fd[i, k] = (f(zp) - f(zm)) / (2 * eps)
    out["autograd against finite differences (max abs)"] = float((g - fd).abs().max()); out["coupled: one window's gradient depends on its neighbours"] = bool(float(g[1].abs().sum()) > 0)
    out["ok"] = bool(out["interior flip costs more than a boundary shift"] and out["A-B-A output under A-A-A truth costs > 0"] and out["A-B-A output under A-B-A truth costs 0"]
                     and out["erasing a true short run costs > 0"] and out["A24 offset example: non-zero change"] and mism == 0 and out["autograd against finite differences (max abs)"] < 1e-6)
    return out


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vs = cfg, config, cfg["video_segment_risk"]; p = self.vs["pinned"]
        for k in ("A10_config", "A10_manifest_index", "A08_choices", "matching_code", "A19_screen", "A19_logits_index", "A18_screen", "A20_screen", "A15_pool_manifest"):
            if sha256_file(ROOT / p[k]["path"]) != p[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")
        self.a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=A10_CONFIG, set=[])), A10_CONFIG); self.total, self.warmup, self.o = self.a10.total, self.a10.warmup, self.a10.o
        self.lam = float(self.vs["risk"]["coefficient"].split("=")[1].split(",")[0])
        o = self.vs["outputs"]; self.out, self.ckpt_root, self.results, self.ledger = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"], ROOT / self.vs["budget"]["ledger"]

    def start(self, seed: int) -> tuple[dict, str]:
        state, ssha = self.a10.start_state(seed)
        if ssha != self.vs["pinned"]["A08_weights"][seed]:
            raise SystemExit(f"seed {seed}: A08's start weights differ from the pin")
        return state, ssha

    def teacher(self, seed: int) -> dict:
        path = ROOT / "runs/video_ensemble_teacher_v1/cache" / f"teachers_s{seed}.npz"; csha = sha256_file(path); m, msha = self.a10.manifest(seed)
        if csha != self.vs["pinned"]["A18_teacher_cache"][seed]:
            raise SystemExit(f"seed {seed}: A18's teacher cache differs from its pin")
        with np.load(path) as z:
            if str(z["manifest_sha256"]) != msha or not np.array_equal(z["digests"][:self.total], m["digest_ordered"][:self.total]):
                raise SystemExit(f"seed {seed}: A18's cached requests differ from A10's windows")
            return {"logits": z["logits"][:, :self.total].astype(np.float32), "sha256": csha}

    def groups(self, seed: int) -> dict:
        idx = json.loads((self.out / "groups" / "index.json").read_text())[str(seed)]; p = self.out / "groups" / idx["file"]
        if sha256_file(p) != idx["sha256"]:
            raise SystemExit(f"seed {seed}: the group file differs from its index")
        with np.load(p) as z:
            return {k: z[k] for k in z.files}


def keys_of(G: dict, u: int) -> list:
    return [(int(G["pads"][u, g, 0]), tuple(int(x) for x in G["truth"][u, g]), int(G["pads"][u, g, 1])) for g in range(WINDOWS // GROUP)]


# ----------------------------------------------------------------------------------------------- preparation (CPU)
def prepare(ctx: Ctx, seeds: list[int]) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "seeds": {}}
    try:
        seqs = vpr.training_data(ctx.a10.a07p).seqs; gdir = ctx.out / "groups"; gdir.mkdir(parents=True, exist_ok=True); index = {}
        for s in seeds:
            m, msha = ctx.a10.manifest(s); U = ctx.total; pads = np.full((U, 3, 2), NONE, np.int8); truth = np.empty((U, 3, GROUP), np.int8); bad = 0; steps = set()
            for u in range(U):
                for g in range(3):
                    w = slice(GROUP * g, GROUP * (g + 1)); qi = int(m["seq"][u, GROUP * g]); lab = np.asarray(seqs[qi]["labels"]); c = m["c"][u, w]; ci = m["centre_index"][u, w]
                    if not ((m["seq"][u, w] == qi).all() and np.isin(np.diff(ci), (0, 1)).all() and np.array_equal(lab[c], m["target"][u, w])):
                        bad += 1
                    truth[u, g] = m["target"][u, w]; steps |= set(np.diff(c).tolist())
                    if int(c[0]) - 5 >= 0:
                        pads[u, g, 0] = int(lab[int(c[0]) - 5])
                    if int(c[-1]) + 5 <= len(lab) - 1:
                        pads[u, g, 1] = int(lab[int(c[-1]) + 5])
            if bad:
                raise SystemExit(f"seed {s}: {bad} groups are not one sequence with consecutive centres and matching targets")
            p = gdir / f"groups_s{s}.npz"; np.savez(p, pads=pads, truth=truth); index[str(s)] = {"file": p.name, "sha256": sha256_file(p), "a10_manifest_sha256": msha}
            keys = {(int(pads[u, g, 0]), tuple(int(x) for x in truth[u, g]), int(pads[u, g, 1])) for u in range(U) for g in range(3)}
            changing = int(sum(len(set(truth[u, g].tolist())) > 1 for u in range(U) for g in range(3)))
            rec["seeds"][str(s)] = {"groups": int(U * 3), "groups_with_a true change": changing, "left_pad_missing": int((pads[..., 0] == NONE).sum()), "right_pad_missing": int((pads[..., 1] == NONE).sum()),
                                    "sample_steps_within_groups": sorted(int(x) for x in steps), "distinct_cost_tables": len(keys), "zero_cost_floor_check": bool(all(float(table(*k).min()) == 0.0 for k in list(keys)[:200]))}
        write_json(gdir / "index.json", index); rec["finished"] = now()
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, rec)
        print(json.dumps(rec, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare groups and pads (CPU, main process)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- training
def train_step(model, heads, opt, b, zt: torch.Tensor, keys: list, lrs, o: dict, device, lam: float, keep: bool = False) -> dict:
    """A18's own-teacher update (CE + KL on the 12 windows) plus lam * L_seg."""
    u8, slots, target = b[1], b[2], b[3]; t = target.to(device); model.train(); heads.train()
    z = model.classifier(model.embed(tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device)))); zt = zt.to(device)
    ce = F.cross_entropy(z, t); lp = F.log_softmax(z, -1); lpt = F.log_softmax(zt, -1); kl = (lpt.exp() * (lpt - lp)).sum(-1).mean(); seg = seg_risk(z, keys); loss = ce + kl + lam * seg
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward(); params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    enc = float(max((p.grad.abs().max() for p in model.encoder_parameters() if p.grad is not None), default=torch.tensor(0.0))) if keep else None
    opt.step(); pred = z.detach().argmax(1).cpu().numpy()
    rec = {"loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach()), "seg": float(seg.detach()), "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1],
           "correct": int((pred == target.numpy()).sum())}
    if keep:
        rec["encoder_grad_max"] = enc
    return rec


def code_digest() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(tet.__file__), Path(vsh.__file__), Path(tm.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def identity(ctx: Ctx, seed: int, msha: str, ssha: str, csha: str, gsha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "sections": {k: ctx.vs[k] for k in ("arms", "risk", "loss", "training")}, "seed": int(seed), "lambda": ctx.lam, "manifest_sha256": msha,
               "start_sha256": ssha, "teacher_cache_sha256": csha, "groups_sha256": gsha, "environment": prv.environment(), "total_updates": int(ctx.total), "code": code_digest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def train(ctx: Ctx, seed: int, device, workers: int) -> dict:
    m, msha = ctx.a10.manifest(seed); cache = ctx.teacher(seed); G = ctx.groups(seed); gsha = json.loads((ctx.out / "groups" / "index.json").read_text())[str(seed)]["sha256"]
    state, ssha = ctx.start(seed); model, heads = tet.make_model(state, device), vsh.build_heads(seed).to(device); ident = identity(ctx, seed, msha, ssha, cache["sha256"], gsha)
    key = f"Seg_s{seed}"; rd, ck = ctx.out / "train" / key, ctx.ckpt_root / key; rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    if (rd / "run_manifest.json").is_file() and json.loads((rd / "run_manifest.json").read_text()).get("identity") == ident and (ck / "final.pt").is_file():
        return {"key": key, "status": "finished"}
    opt = vsh.make_optimizer(model, heads, ctx.o)
    if not vsh.bound(opt, model, heads):
        raise SystemExit("the optimiser is not bound")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); done, log = 0, []; ul = rd / "updates.jsonl"
    if (ck / "resume.pt").is_file():
        done, log = vsh.load_state(ck / "resume.pt", model, heads, opt, ident); lines = ul.read_text().splitlines()[:done] if ul.is_file() else []; ul.write_text("".join(x + "\n" for x in lines))
        with (rd / "resumes.jsonl").open("a") as fh:
            fh.write(json.dumps({"at_update": done, "time": now()}) + "\n")
    else:
        for p in (ul, rd / "train_log.jsonl", rd / "resumes.jsonl"):
            p.unlink(missing_ok=True)
    print(f"[{key}] identity {ident[:12]}; lambda {ctx.lam}; updates {done} -> {ctx.total}", flush=True); t0 = time.time(); acc = []; torch.cuda.reset_peak_memory_stats(device)
    for b in vsh.loader(m, ctx.a10.frame_paths(), list(range(done, ctx.total)), workers):
        tet.check_batch(m, b); u = int(b[0]); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][seed, u].copy()), keys_of(G, u), lrs, ctx.o, device, ctx.lam); acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"stage": "Seg student training (backward, GPU)", "identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.mean([a[k] for a in acc]))  # noqa: E731
            rec = {"updates": done, **{k: mean(k) for k in ("loss", "ce", "kl", "seg", "grad_norm")}, "accuracy": sum(a["correct"] for a in acc) / (WINDOWS * len(acc)), "lr_encoder_last": acc[-1]["lr_encoder"],
                   "seconds_this_process": time.time() - t0, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); vsh.save_state(ck / "resume.pt", model, heads, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE {rec['ce']:.4f} KL {rec['kl']:.4f} Seg {rec['seg']:.4f} acc {rec['accuracy']:.3f} {rec['seconds_this_process']:.0f}s", flush=True); acc = []
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done}")
    final = ck / "final.pt"; torch.save({"model": model.state_dict(), "heads": heads.state_dict()}, final)
    man = {"identity": ident, "arm": "Seg", "key": key, "seed": seed, "lambda": ctx.lam, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha, "cache_sha256": cache["sha256"],
           "groups_sha256": gsha, "start_sha256": ssha, "environment": prv.environment(), "code": code_digest(), "updates": done, "epochs": log,
           "resumes": (rd / "resumes.jsonl").read_text().splitlines() if (rd / "resumes.jsonl").is_file() else [], "final_checkpoint": str(final), "final_sha256": sha256_file(final),
           "parameters": int(sum(p.numel() for p in model.parameters())), "encoder_forwards_per_window": 1, "seconds_last_process": time.time() - t0}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True); print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"]}


def queue(ctx: Ctx, keys: list[str], device, workers: int) -> None:
    chk = json.loads((ctx.out / "checks.json").read_text())
    if not chk["ok"]:
        raise SystemExit("the checks did not pass")
    spu = float(chk["timing"]["seconds_per_update"]); ev = float(chk["budget"]["evaluation_per_model_hours"])
    for i, key in enumerate(keys):
        spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + len(keys[i:]) * (ctx.total * spu / 3600 + ev)
        print(f"[queue] before {key}: spent {spent:.3f}, projection with evaluation {proj:.3f} of {ctx.vs['budget']['gpu_hours_max']}", flush=True)
        if proj > float(ctx.vs["budget"]["gpu_hours_max"]):
            raise SystemExit("the projection exceeds the cap; nothing new starts")
        t0, started, ok, note = time.time(), now(), False, ""
        try:
            r = train(ctx, int(key.rsplit("_s", 1)[1]), device, workers); ok = True; note = json.dumps(r)
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"train {key}", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- checks
def batch_of(ctx: Ctx, seed: int, u: int):
    m, _ = ctx.a10.manifest(seed); b = vsh.Inputs(m, ctx.a10.frame_paths(), [u])[0]; tet.check_batch(m, b); return b


def run(ctx: Ctx, seed: int, us: list[int], device, lam: float, state_file: Path | None = None) -> dict:
    state, _ = ctx.start(seed); cache = ctx.teacher(seed); G = ctx.groups(seed); model, heads = tet.make_model(state, device), vsh.build_heads(seed).to(device); opt = vsh.make_optimizer(model, heads, ctx.o)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); losses, secs, enc, segs = [], [], [], []
    if state_file is not None and state_file.is_file():
        vsh.load_state(state_file, model, heads, opt, "check")
    for u in us:
        b = batch_of(ctx, seed, u); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o); torch.cuda.synchronize(); a = time.time()
        r = train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][seed, u].copy()), keys_of(G, u), lrs, ctx.o, device, lam, keep=True); torch.cuda.synchronize(); secs.append(time.time() - a)
        losses.append(r["loss"]); enc.append(r["encoder_grad_max"]); segs.append(r["seg"])
    if state_file is not None and not state_file.is_file():
        vsh.save_state(state_file, model, heads, opt, len(us), "check", [])
    return {"model": model, "losses": losses, "seconds": secs, "encoder_grads": enc, "segs": segs}


def gradient_share(ctx: Ctx, seed: int, us: list[int], device) -> list[dict]:
    state, _ = ctx.start(seed); cache = ctx.teacher(seed); G = ctx.groups(seed); out = []
    for u in us:
        model, heads = tet.make_model(state, device), vsh.build_heads(seed).to(device); opt = vsh.make_optimizer(model, heads, ctx.o); vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o); model.train()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); b = batch_of(ctx, seed, u); zt = torch.from_numpy(cache["logits"][seed, u].copy()).to(device); t = b[3].to(device)
        z = model.classifier(model.embed(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device)))); lp = F.log_softmax(z, -1); lpt = F.log_softmax(zt, -1)
        base = F.cross_entropy(z, t) + (lpt.exp() * (lpt - lp)).sum(-1).mean(); seg = ctx.lam * seg_risk(z, keys_of(G, u)); params = [p for p in model.parameters() if p.requires_grad]
        gb = torch.autograd.grad(base, params, retain_graph=True, allow_unused=True); gs = torch.autograd.grad(seg, params, allow_unused=True)
        nb = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gb if g is not None))); ns = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
        dot = float(sum((a.float() * c.float()).sum() for a, c in zip(gb, gs) if a is not None and c is not None))
        out.append({"u": u, "base_loss": float(base), "lambda_seg": float(seg), "base_grad_norm": nb, "seg_grad_norm": ns, "cosine": dot / max(1e-12, nb * ns)}); del model, heads, opt; torch.cuda.empty_cache()
    return out


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note, res = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "started": now(), "lambda": ctx.lam}
    try:
        res["cost_cases"] = cases(); prep = json.loads((ctx.out / "preparation.json").read_text())
        res["groups"] = {**prep["seeds"], "ok": bool(prep["config_sha256"] == res["config_sha256"] and all(v["zero_cost_floor_check"] for v in prep["seeds"].values()))}
        for s, sha in ctx.vs["pinned"]["control_R_weights"].items():
            root = ROOT / ("runs/video_switch_head_v1" if int(s) in (0, 2) else "runs/video_ensemble_teacher_v1"); man = json.loads((root / "train" / f"R_s{s}" / "run_manifest.json").read_text())
            if man["final_sha256"] != sha:
                raise SystemExit(f"control R_s{s}: the run record differs from the pin")
        rep = {}
        for s, us in ((0, list(range(0, ctx.warmup + 2))), (2, list(range(5)))):
            r0 = run(ctx, s, us, device, 0.0); lg = {x["u"]: x for x in (json.loads(l) for l in (ROOT / "runs/video_switch_head_v1/train" / f"R_s{s}" / "updates.jsonl").read_text().splitlines())}
            d = [abs(a - float(lg[u]["loss"])) for u, a in zip(us, r0["losses"])]; rep[f"A10 R_s{s}"] = {"updates": [us[0], us[-1]], "loss_max_abs": max(d), "last_two_abs": d[-2:], "seg_at_start": r0["segs"][:3]}
            del r0; torch.cuda.empty_cache()
        rep["ok"] = bool(all(v["loss_max_abs"] <= 1e-4 for v in rep.values() if isinstance(v, dict))); res["lambda0_replay"] = rep
        state, _ = ctx.start(0); cache = ctx.teacher(0); a08 = tet.make_model(state, device).eval(); d0 = tet.state_digest(a08); par = []
        with torch.no_grad():
            for u in range(4):
                b = batch_of(ctx, 0, u); par.append(float((a08(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device))).float().cpu() - torch.from_numpy(cache["logits"][0, u])).abs().max()))
        res["teacher"] = {"cached_against_online_max_abs": max(par), "windows": 48, "unchanged": tet.state_digest(a08) == d0, "ok": max(par) <= 1e-4 and tet.state_digest(a08) == d0}; del a08
        us = list(range(ctx.warmup - 2, ctx.warmup + 4)); sf = ctx.out / "check_resume.pt"; sf.unlink(missing_ok=True); straight = run(ctx, 0, us, device, ctx.lam)
        repeat = run(ctx, 0, us, device, ctx.lam); run(ctx, 0, us[:3], device, ctx.lam, sf); second = run(ctx, 0, us[3:], device, ctx.lam, sf)
        rp = max_diff(straight["model"], repeat["model"]); dif = max_diff(straight["model"], second["model"]); ld = float(max(abs(a - b2) for a, b2 in zip(straight["losses"][3:], second["losses"])))
        res["resume"] = {"updates": us, "loss_max_abs": ld, "params_max_abs": dif, "repeat_params_max_abs": rp, "ok": bool(ld <= 1e-4 and dif <= max(1e-5, 2 * rp))}
        res["encoder"] = {"grad_max_before_warmup": straight["encoder_grads"][:2], "grad_max_after": straight["encoder_grads"][2:], "ok": bool(all(g == 0.0 for g in straight["encoder_grads"][:2]) and all(g > 0 for g in straight["encoder_grads"][2:]))}
        model, heads = tet.make_model(state, device), vsh.build_heads(0).to(device); opt = vsh.make_optimizer(model, heads, ctx.o); res["optimiser_bound"] = {"ok": bool(vsh.bound(opt, model, heads))}; del model, heads, opt
        sf.unlink(missing_ok=True); spu = float(np.mean(straight["seconds"][2:])); res["seg_values_near_unfreezing"] = straight["segs"]; del straight, repeat, second; torch.cuda.empty_cache()
        res["gradient_share"] = gradient_share(ctx, 0, [0, ctx.warmup, ctx.warmup + 100], device)
        res["timing"] = {"seconds_per_update": spu, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        a08 = tet.make_model(state, device).eval(); x = torch.randint(0, 256, (32, 16, 224, 224, 3), dtype=torch.uint8); mk = tm.marker_planes(torch.zeros(32, dtype=torch.long).to(device))
        with torch.no_grad():
            a08(tm.to_rgb(x.to(device)), mk); torch.cuda.synchronize(); a = time.time()
            for _ in range(3):
                a08(tm.to_rgb(x.to(device)), mk)
            torch.cuda.synchronize(); spw = (time.time() - a) / 96
        ev = (7666 + 7021 + 18175) * spw / 3600; spent = ledger_hours(ctx.ledger, "gpu"); per_run = ctx.total * spu / 3600
        res["budget"] = {"spent": spent, "inference_seconds_per_window": spw, "evaluation_per_model_hours": ev, "per_run_hours": per_run, "projection_two_seeds": spent + 2 * (per_run + ev),
                         "seed1_reserve": per_run + ev + 0.05, "cap": ctx.vs["budget"]["gpu_hours_max"]}
        res["budget"]["fits_with_seed1"] = bool(res["budget"]["projection_two_seeds"] + res["budget"]["seed1_reserve"] <= res["budget"]["cap"])
        res["ok"] = bool(all(res[k]["ok"] for k in ("cost_cases", "groups", "lambda0_replay", "teacher", "resume", "encoder", "optimiser_bound")) and res["budget"]["projection_two_seeds"] <= res["budget"]["cap"])
        res["finished"] = now(); flat = json.loads(json.dumps(res, default=str))
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        print(json.dumps(flat, indent=1)[:6000], flush=True)
        if not res["ok"]:
            raise SystemExit("a check failed")
        ok = True; note = f"{spu:.3f} s/update, projection {res['budget']['projection_two_seeds']:.3f}"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=7); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true"); g.add_argument("--queue", nargs="+"); g.add_argument("--cases", action="store_true")
    a = ap.parse_args(); a.set = []
    if a.cases:
        print(json.dumps(cases(), indent=1)); return
    ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        return prepare(ctx, a.seeds)
    device = torch.device("cuda"); determinism()
    if a.check:
        return check(ctx, device, a.workers)
    queue(ctx, a.queue, device, a.workers)


if __name__ == "__main__":
    main()
