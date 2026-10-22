#!/usr/bin/env python
"""A21 training (configs/experiments/video_spatial_context_v1.yaml): A08 all_s{k} with a residual head g over the shared
encoder's centre-crop and auxiliary-view embeddings, z = classifier(e_crop) + g([e_crop, e_aux]). Full's aux is the whole
frame; Restricted's aux is the same frame with everything outside the old centre crop filled before resizing. A10's
12-window batches, CE + KL to the cached own A08 crop teacher.

  --prepare          CPU: training-manifest geometry summary on a fixed subsample (updates u % 5 == 0): Full/Restricted
                     aux differences, extra-area share, extra-region variation, source sizes
  --check            GPU: pins, crop parity, start logits, cross-arm RNG, synthetic geometry, gradients, sharing, forced
                     Full = Restricted, resume, timing and budget projection
  --queue K [K ...]  runs in order (Full_s0, Restricted_s0, Full_s2, Restricted_s2; seed 1 only after a met rule)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.joint_visual as jv  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

A10_CONFIG = "configs/experiments/video_switch_head_v1.yaml"
WINDOWS, SIDE, FILL, ARMS = 12, 224, (29, 28, 29), ("Full", "Restricted")
now, write_json, ledger_hours, ledger_append, determinism, max_diff = vsh.now, vsh.write_json, vsh.ledger_hours, vsh.ledger_append, vsh.determinism, vsh.max_diff


# ----------------------------------------------------------------------------------------------- pixels
def resized(im: Image.Image) -> Image.Image:
    """crop_uint8's intermediate image R: short side 256 with its max(224, round()) sizes, PIL BILINEAR."""
    w, h = im.size; s = 256 / min(w, h)
    return im.resize((max(SIDE, round(w * s)), max(SIDE, round(h * s))), Image.BILINEAR)


def centre_box(R: Image.Image) -> tuple[int, int, int, int]:
    w, h = R.size; left, top = (w - SIDE) // 2, (h - SIDE) // 2
    return left, top, left + SIDE, top + SIDE


def fit224(R: Image.Image) -> np.ndarray:
    """R scaled so its long side is 224 (other side Python round), BILINEAR, pasted at ((224-w)//2, (224-h)//2) on a
    224x224 FILL canvas: an odd difference leaves the extra pixel right/bottom."""
    w, h = R.size; nw, nh = (SIDE, max(1, round(h * SIDE / w))) if w >= h else (max(1, round(w * SIDE / h)), SIDE)
    canvas = Image.new("RGB", (SIDE, SIDE), FILL); canvas.paste(R.resize((nw, nh), Image.BILINEAR), ((SIDE - nw) // 2, (SIDE - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def views_of(R: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """crop (R's exact centre 224), Full aux (R fitted) and Restricted aux (R outside the centre filled, then fitted)."""
    box = centre_box(R); centre = R.crop(box); masked = Image.new("RGB", R.size, FILL); masked.paste(centre, box[:2])
    return np.asarray(centre, dtype=np.uint8), fit224(R), fit224(masked)


def frame_views(path, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """load_frame_u8's decode (RGB, gamma LUT, resize), then the three views; also the source and R sizes."""
    with Image.open(path) as im:
        src = im.size; im = im.convert("RGB")
        if abs(gamma - 1.0) > 1e-9:
            im = im.point(jv.gamma_lut(gamma))
        R = resized(im)
    return (*views_of(R), src, R.size)


def window_views(m: dict, fp, pairs: list[tuple[int, int]], auxes: tuple[str, ...]):
    """Windows (u, w) in the ordered slot order: crop frames (N, 16, 224, 224, 3), the requested aux views stacked
    (A, N, 16, 224, 224, 3), markers, targets and crop pixel digests (A10's frame_hash / window_digest)."""
    n = len(pairs); crop = np.empty((n, tm.N_SLOTS, SIDE, SIDE, 3), np.uint8); aux = np.empty((len(auxes), *crop.shape), np.uint8)
    digests = np.empty((n, 32), np.uint8); cache = {}; order = m["order_ordered"]
    for k, (u, w) in enumerate(pairs):
        hashes = []
        for slot in range(tm.N_SLOTS):
            j = int(order[u, w, slot]); ref = (int(m["vid"][u, w, j]), int(m["frame"][u, w, j]), round(float(m["gamma"][u, w, j]), 6))
            if ref not in cache:
                paths = fp(str(m["videos"][ref[0]])); c, f, r, _, _ = frame_views(paths[min(ref[1], len(paths) - 1)], ref[2])
                cache[ref] = (c, {"Full": f, "Restricted": r}, tm.frame_hash(c))
            c, av, h = cache[ref]; crop[k, slot] = c; hashes.append(h)
            for a, name in enumerate(auxes):
                aux[a, k, slot] = av[name]
        digests[k] = np.frombuffer(tm.window_digest(hashes), np.uint8)
    us, ws = [p[0] for p in pairs], [p[1] for p in pairs]
    return (torch.from_numpy(crop), torch.from_numpy(aux), torch.from_numpy(m["marker"][us, ws].astype(np.int64)), torch.from_numpy(m["target"][us, ws].astype(np.int64)),
            torch.from_numpy(digests))


class Inputs(Dataset):
    def __init__(self, m: dict, fp, updates: list[int], auxes: tuple[str, ...]) -> None:
        self.m, self.fp, self.updates, self.auxes = m, fp, list(updates), tuple(auxes)

    def __len__(self) -> int:
        return len(self.updates)

    def __getitem__(self, i: int):
        u = self.updates[i]; return (u, *window_views(self.m, self.fp, [(u, w) for w in range(WINDOWS)], self.auxes))


def loader(m: dict, fp, updates: list[int], auxes: tuple[str, ...], workers: int) -> DataLoader:
    return DataLoader(Inputs(m, fp, updates, auxes), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def check_batch(m: dict, b) -> None:
    u = int(b[0])
    if not np.array_equal(b[5].numpy(), m["digest_ordered"][u]):
        raise SystemExit(f"update {u}: the crop pixels differ from A10's prepared digests")
    if not np.array_equal(b[4].numpy(), m["target"][u]) or not np.array_equal(b[3].numpy(), m["marker"][u]):
        raise SystemExit(f"update {u}: targets or markers differ from the manifest")


# ----------------------------------------------------------------------------------------------- model
class SpatialContext(nn.Module):
    """A08's TargetMViT unchanged plus g; both views pass the same encoder and marker stem (one set of parameters)."""

    def __init__(self, base: tm.TargetMViT, seed: int) -> None:
        super().__init__()
        self.base, self.g = base, nn.Sequential(nn.Linear(2 * tm.EMBED_DIM, 32), nn.GELU(), nn.Linear(32, tm.NUM_CLASSES))
        gen = torch.Generator().manual_seed(2100 + int(seed))
        with torch.no_grad():
            self.g[0].weight.copy_(torch.randn(32, 2 * tm.EMBED_DIM, generator=gen) * 0.02); self.g[0].bias.zero_(); self.g[2].weight.zero_(); self.g[2].bias.zero_()

    def logits(self, crop_rgb: torch.Tensor, aux_rgb: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        e_crop = self.base.embed(crop_rgb, marker); e_aux = self.base.embed(aux_rgb, marker)
        return self.base.classifier(e_crop) + self.g(torch.cat([e_crop, e_aux], -1))


def make(state: dict, seed: int, device) -> SpatialContext:
    return SpatialContext(vsh.make_model(state, torch.device("cpu")), seed).to(device)


def make_optimizer(model: SpatialContext, o: dict) -> torch.optim.AdamW:
    return torch.optim.AdamW([{"params": model.base.encoder_parameters(), "lr": 0.0, "name": "encoder"}, {"params": model.base.new_parameters(), "lr": 0.0, "name": "new"},
                              {"params": list(model.g.parameters()), "lr": 0.0, "name": "g"}],
                             betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))


def bound(opt, model: SpatialContext) -> bool:
    ids = lambda ps: [id(p) for p in ps]  # noqa: E731
    reg = [id(p) for g in opt.param_groups for p in g["params"]]
    return (len(opt.param_groups) == 3 and ids(opt.param_groups[0]["params"]) == ids(model.base.encoder_parameters()) and ids(opt.param_groups[1]["params"]) == ids(model.base.new_parameters())
            and ids(opt.param_groups[2]["params"]) == ids(model.g.parameters()) and len(reg) == len(set(reg)) and set(reg) == {id(p) for p in model.parameters()})


def apply_schedule(model: SpatialContext, opt, u: int, warmup: int, total: int, o: dict) -> tuple[float, float, float]:
    f = lr_factor(int(u), int(warmup), int(total)); enc = 0.0 if u < warmup else float(o["lr_encoder"]) * f
    model.base.set_encoder_trainable(u >= warmup); new, g = float(o["lr_new"]) * f, float(o["lr_head"]) * f
    opt.param_groups[0]["lr"], opt.param_groups[1]["lr"], opt.param_groups[2]["lr"] = enc, new, g
    return enc, new, g


def train_step(model: SpatialContext, opt, b, zt: torch.Tensor, lrs, o: dict, device, keep: bool = False, aux_grad: bool = False) -> dict:
    """One update on the 12 windows: crop forward, then aux forward, residual logits, mean CE + mean KL to the cached teacher."""
    t = b[4].to(device); model.train(); mk = tm.marker_planes(b[3].to(device))
    rc = tm.to_rgb(b[1].to(device, non_blocking=True)); ra = tm.to_rgb(b[2][0].to(device, non_blocking=True))
    if aux_grad:
        ra.requires_grad_(True)
    z = model.logits(rc, ra, mk); zt = zt.to(device); ce = F.cross_entropy(z, t); lp = F.log_softmax(z, -1); lpt = F.log_softmax(zt, -1)
    kl = (lpt.exp() * (lpt - lp)).sum(-1).mean(); loss = ce + kl
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward(); params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    grads = {"g_last": float(model.g[2].weight.grad.abs().max()), "g_first": float(model.g[0].weight.grad.abs().max()),
             "encoder": float(max((p.grad.abs().max() for p in model.base.encoder_parameters() if p.grad is not None), default=torch.tensor(0.0))),
             "aux_input": float(ra.grad.abs().max()) if aux_grad else None} if keep else None
    opt.step(); pred = z.detach().argmax(1).cpu().numpy()
    rec = {"loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach()), "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1], "lr_g": lrs[2],
           "correct": int((pred == b[4].numpy()).sum()), "g_last_norm": float(model.g[2].weight.detach().norm())}
    if keep:
        rec["logits"], rec["grads"] = z.detach().cpu(), grads
    return rec


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vc = cfg, config, cfg["video_spatial_context"]; p = self.vc["pinned"]
        for k in ("A10_config", "A10_manifest_index", "A08_choices", "crop_code", "A15_pool_manifest", "A19_screen", "A20_screen"):
            if sha256_file(ROOT / p[k]["path"]) != p[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")
        self.a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=A10_CONFIG, set=[])), A10_CONFIG); self.total, self.warmup, self.o = self.a10.total, self.a10.warmup, self.a10.o
        o = self.vc["outputs"]; self.out, self.ckpt_root, self.results, self.ledger = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"], ROOT / self.vc["budget"]["ledger"]

    def manifest(self, seed: int) -> tuple[dict, str]:
        return self.a10.manifest(seed)

    def start(self, seed: int) -> tuple[dict, str]:
        state, ssha = self.a10.start_state(seed)
        if ssha != self.vc["pinned"]["A08_weights"][seed]:
            raise SystemExit(f"seed {seed}: A08's start weights differ from the pin")
        return state, ssha

    def teacher(self, seed: int) -> tuple[np.ndarray, str]:
        """A18's own-teacher cache: the seed's frozen A08 on the crop view, (1350, 12, 5), identity-checked."""
        path = ROOT / "runs/video_ensemble_teacher_v1/cache" / f"teachers_s{seed}.npz"; csha = sha256_file(path); m, msha = self.manifest(seed)
        if csha != self.vc["pinned"]["A18_teacher_cache"][seed]:
            raise SystemExit(f"seed {seed}: A18's teacher cache differs from its pin")
        with np.load(path) as z:
            if str(z["manifest_sha256"]) != msha or not np.array_equal(z["digests"][:self.total], m["digest_ordered"][:self.total]):
                raise SystemExit(f"seed {seed}: A18's cached requests differ from A10's windows")
            return z["logits"][seed, :self.total].astype(np.float32), csha


def code_digest() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(vsh.__file__), Path(rvc.__file__), Path(tm.__file__), Path(jv.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str, csha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "sections": {k: ctx.vc[k] for k in ("predecessors", "arms", "pixels", "data", "loss", "training")}, "arm": arm, "seed": int(seed),
               "manifest_sha256": msha, "start_sha256": ssha, "teacher_cache_sha256": csha, "environment": prv.environment(), "total_updates": int(ctx.total), "code": code_digest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def save_state(path: Path, model, opt, done: int, ident: str, log: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"identity": ident, "updates": done, "model": model.state_dict(), "opt": opt.state_dict(), "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(), "log": log}, tmp)
    tmp.replace(path)


def load_state(path: Path, model, opt, ident: str) -> tuple[int, list]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["identity"] != ident:
        raise SystemExit(f"{path} belongs to another identity")
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state_all(st["rng_cuda"])
    return int(st["updates"]), list(st["log"])


# ----------------------------------------------------------------------------------------------- training
def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> dict:
    m, msha = ctx.manifest(seed); state, ssha = ctx.start(seed); zt_all, csha = ctx.teacher(seed); model = make(state, seed, device)
    ident = identity(ctx, arm, seed, msha, ssha, csha); key = f"{arm}_s{seed}"; rd, ck = ctx.out / "train" / key, ctx.ckpt_root / key
    rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    if (rd / "run_manifest.json").is_file() and json.loads((rd / "run_manifest.json").read_text()).get("identity") == ident and (ck / "final.pt").is_file():
        print(f"[{key}] already finished", flush=True); return {"key": key, "status": "finished"}
    opt = make_optimizer(model, ctx.o)
    if not bound(opt, model):
        raise SystemExit("the optimiser is not bound to the model")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); done, log = 0, []; ul = rd / "updates.jsonl"
    if (ck / "resume.pt").is_file():
        done, log = load_state(ck / "resume.pt", model, opt, ident); lines = ul.read_text().splitlines()[:done] if ul.is_file() else []
        ul.write_text("".join(x + "\n" for x in lines))
        with (rd / "resumes.jsonl").open("a") as fh:
            fh.write(json.dumps({"at_update": done, "time": now()}) + "\n")
    else:
        for p in (ul, rd / "resumes.jsonl", rd / "train_log.jsonl"):
            p.unlink(missing_ok=True)
    print(f"[{key}] identity {ident[:12]}; updates {done} -> {ctx.total}", flush=True); t0 = time.time(); acc = []
    torch.cuda.reset_peak_memory_stats(device)
    for b in loader(m, ctx.a10.frame_paths(), list(range(done, ctx.total)), (arm,), workers):
        u = int(b[0]); check_batch(m, b); lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = train_step(model, opt, b, torch.from_numpy(zt_all[u]), lrs, ctx.o, device); acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.mean([a[k] for a in acc]))  # noqa: E731
            rec = {"updates": done, "n": len(acc), **{k: mean(k) for k in ("loss", "ce", "kl", "grad_norm")}, "accuracy": sum(a["correct"] for a in acc) / (WINDOWS * len(acc)),
                   "g_last_norm": acc[-1]["g_last_norm"], "lr_encoder_last": acc[-1]["lr_encoder"], "seconds_this_process": time.time() - t0, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); save_state(ck / "resume.pt", model, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE {rec['ce']:.4f} KL {rec['kl']:.4f} acc {rec['accuracy']:.3f} |g_last| {rec['g_last_norm']:.4f} {rec['seconds_this_process']:.0f}s", flush=True)
            acc = []
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done}")
    final = ck / "final.pt"; torch.save({"model": model.state_dict()}, final)
    n_params = int(sum(p.numel() for p in model.parameters())); n_g = int(sum(p.numel() for p in model.g.parameters()))
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha, "start_sha256": ssha, "teacher_cache_sha256": csha,
           "environment": prv.environment(), "code": code_digest(), "updates": done, "epochs": log, "resumes": (rd / "resumes.jsonl").read_text().splitlines() if (rd / "resumes.jsonl").is_file() else [],
           "final_checkpoint": str(final), "final_sha256": sha256_file(final), "parameters": {"total": n_params, "g": n_g, "A08": n_params - n_g}, "encoder_forwards_per_window": 2,
           "seconds_last_process": time.time() - t0}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True)
    print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"]}


def queue(ctx: Ctx, keys: list[str], device, workers: int) -> None:
    chk = json.loads((ctx.out / "checks.json").read_text())
    if not chk["ok"]:
        raise SystemExit("the checks did not pass")
    spu = float(chk["timing"]["seconds_per_update"]); ev = float(chk["budget"]["evaluation_per_model_hours"])
    for i, key in enumerate(keys):
        spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + len(keys[i:]) * (ctx.total * spu / 3600 + ev)
        print(f"[queue] before {key}: spent {spent:.3f}, projection with evaluation {proj:.3f} of {ctx.vc['budget']['gpu_hours_max']}", flush=True)
        if proj > float(ctx.vc["budget"]["gpu_hours_max"]):
            raise SystemExit("the projection exceeds the cap; nothing new starts")
        arm, s = key.rsplit("_s", 1); t0, started, ok, note = time.time(), now(), False, ""
        try:
            r = train(ctx, arm, int(s), device, workers); ok = True; note = json.dumps(r)
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"train {key}", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- preparation (CPU)
class FrameStats(Dataset):
    def __init__(self, items: list[tuple[str, int, float]], fp) -> None:
        self.items, self.fp = items, fp

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        vid, fr, gamma = self.items[i]; paths = self.fp(vid); path = paths[min(fr, len(paths) - 1)]
        with Image.open(path) as im:
            src = im.size; im = im.convert("RGB")
            if abs(gamma - 1.0) > 1e-9:
                im = im.point(jv.gamma_lut(gamma))
            R = resized(im)
        c, f, r = views_of(R); a = np.asarray(R, dtype=np.int16); box = centre_box(R); out = np.ones(a.shape[:2], bool); out[box[1]:box[3], box[0]:box[2]] = False
        px = a[out]; diff = np.any(f != r, axis=-1)
        if not np.array_equal(c, jv.crop_uint8(R)):
            raise SystemExit("the crop differs from crop_uint8 on R")
        return {"src": src, "R": R.size, "extra_share": 1.0 - SIDE * SIDE / (R.size[0] * R.size[1]), "aux_diff_share": float(diff.mean()),
                "extra_mean": float(px.mean()), "extra_std": float(px.std()), "extra_near_black": float((px.max(-1) <= 10).mean())}


def prepare(ctx: Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note, out = time.time(), now(), False, "", {"subsample": "updates u % 5 == 0, all 12 windows, all 16 slots; unique (recording, frame, gamma) requests"}
    try:
        fp = ctx.a10.frame_paths()
        for s in seeds:
            m, msha = ctx.manifest(s); us = list(range(0, ctx.total, 5)); refs, win = {}, []
            for u in us:
                for w in range(WINDOWS):
                    keys = []
                    for slot in range(tm.N_SLOTS):
                        j = int(m["order_ordered"][u, w, slot]); key = (str(m["videos"][int(m["vid"][u, w, j])]), int(m["frame"][u, w, j]), round(float(m["gamma"][u, w, j]), 6))
                        refs.setdefault(key, len(refs)); keys.append(refs[key])
                    win.append(keys)
            items = list(refs); st = [None] * len(items); k = 0
            for x in DataLoader(FrameStats(items, fp), batch_size=None, num_workers=workers, prefetch_factor=4 if workers else None):
                st[k] = x; k += 1
            diff = np.asarray([x["aux_diff_share"] for x in st]); share = np.asarray([x["extra_share"] for x in st]); wdiff = np.asarray([any(diff[i] > 0 for i in ks) for ks in win])
            vids = {}
            for (v, _, _), x in zip(items, st):
                vids.setdefault(v, set()).add(tuple(x["src"]))
            out[f"s{s}"] = {"manifest_sha256": msha, "updates": len(us), "windows": len(win), "requests": len(items), "recordings": len(vids),
                            "recordings_with_one_source_size": int(sum(len(v) == 1 for v in vids.values())),
                            "source_sizes_by_request": {f"{a}x{b}": n for (a, b), n in Counter(tuple(x["src"]) for x in st).most_common()},
                            "R_sizes_by_request": {f"{a}x{b}": n for (a, b), n in Counter(tuple(x["R"]) for x in st).most_common()},
                            "extra_area_share": {"mean": float(share.mean()), "min": float(share.min()), "max": float(share.max())},
                            "aux_pixels_differing_share": {"mean": float(diff.mean()), "min": float(diff.min()), "max": float(diff.max())},
                            "requests_with_identical_aux": int((diff == 0).sum()), "windows_with_identical_aux": int((~wdiff).sum()),
                            "extra_region": {"mean_intensity": float(np.mean([x["extra_mean"] for x in st])), "mean_std": float(np.mean([x["extra_std"] for x in st])),
                                             "near_black_share": float(np.mean([x["extra_near_black"] for x in st]))}}
            print(json.dumps({k2: v for k2, v in out[f"s{s}"].items() if k2 not in ("source_sizes_by_request", "R_sizes_by_request")}), flush=True)
        out["met"] = bool(all(out[f"s{s}"]["windows_with_identical_aux"] < out[f"s{s}"]["windows"] for s in seeds)); out["time"] = now()
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, out)
        if not out["met"]:
            raise SystemExit("Full and Restricted aux are identical everywhere: preparation not met")
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, f"prepare geometry summary (CPU: main process + {workers - 1} more decode workers)", "cpu", started, (time.time() - t0) * max(1, workers), ok, note)


# ----------------------------------------------------------------------------------------------- checks
def batch_of(ctx: Ctx, seed: int, u: int, auxes: tuple[str, ...]):
    m, _ = ctx.manifest(seed); b = Inputs(m, ctx.a10.frame_paths(), [u], auxes)[0]; check_batch(m, b); return b


def synthetic_geometry() -> dict:
    """Synthetic R images: outside the centre 224 perturbed -> crop and Restricted aux unchanged, Full aux changed; fit and
    padding geometry; mixed-size windows processed frame by frame; gamma applied before the resize."""
    rng = np.random.default_rng(20261022); out = {"cases": []}
    for w, h in ((341, 256), (256, 341), (300, 256), (256, 303), (257, 256), (256, 256), (401, 256)):
        a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8); b = a.copy(); box = (( w - SIDE) // 2, (h - SIDE) // 2); mask = np.ones((h, w), bool)
        mask[box[1]:box[1] + SIDE, box[0]:box[0] + SIDE] = False; b[mask] = rng.integers(0, 256, (int(mask.sum()), 3), dtype=np.uint8)
        ca, fa, ra = views_of(Image.fromarray(a)); cb, fb, rb = views_of(Image.fromarray(b))
        nw, nh = (SIDE, round(h * SIDE / w)) if w >= h else (round(w * SIDE / h), SIDE); left, top = (SIDE - nw) // 2, (SIDE - nh) // 2
        pad_ok = bool(np.all(fa[:top] == FILL) and np.all(fa[top + nh:] == FILL) and np.all(fa[:, :left] == FILL) and np.all(fa[:, left + nw:] == FILL))
        out["cases"].append({"R": f"{w}x{h}", "fit": f"{nw}x{nh}", "pad_left_right": [left, SIDE - nw - left], "pad_top_bottom": [top, SIDE - nh - top], "outside_pixels": int(mask.sum()),
                             "crop_unchanged": bool(np.array_equal(ca, cb)), "restricted_unchanged": bool(np.array_equal(ra, rb)), "full_changed": bool(not np.array_equal(fa, fb)) if mask.any() else None,
                             "crop_equals_crop_uint8": bool(np.array_equal(ca, jv.crop_uint8(Image.fromarray(a)))), "padding_is_fill": pad_ok})
    srcs = [Image.fromarray(rng.integers(0, 256, (hh, ww, 3), dtype=np.uint8)) for ww, hh in ((320, 240), (400, 300), (240, 320))]
    per = [views_of(resized(im)) for im in srcs]; mixed = [views_of(resized(srcs[i % 3])) for i in range(6)]
    out["mixed_sizes_frame_by_frame"] = bool(all(all(np.array_equal(x, y) for x, y in zip(mixed[i], per[i % 3])) for i in range(6)))
    im = srcs[0]; g = im.point(jv.gamma_lut(0.8)); out["gamma_before_resize"] = bool(np.array_equal(views_of(resized(g))[0], jv.crop_uint8(g)) and not np.array_equal(views_of(resized(g))[0], views_of(resized(im))[0]))
    out["ok"] = bool(all(c["crop_unchanged"] and c["restricted_unchanged"] and c["full_changed"] is not False and c["crop_equals_crop_uint8"] and c["padding_is_fill"] for c in out["cases"])
                     and out["mixed_sizes_frame_by_frame"] and out["gamma_before_resize"])
    return out


def real_parity(ctx: Ctx, seed: int, updates: list[int]) -> dict:
    """Real requests: frame_views' crop equals load_frame_u8 bit for bit, and the windows' crop digests equal A10's."""
    m, _ = ctx.manifest(seed); fp = ctx.a10.frame_paths(); n = bad = 0; sizes = Counter()
    for u in updates:
        for w in range(WINDOWS):
            for j in range(tm.N_SLOTS):
                vid, fr, gamma = str(m["videos"][int(m["vid"][u, w, j])]), int(m["frame"][u, w, j]), round(float(m["gamma"][u, w, j]), 6); paths = fp(vid); p = paths[min(fr, len(paths) - 1)]
                c, f, r, src, _ = frame_views(p, gamma); n += 1; bad += int(not np.array_equal(c, jv.load_frame_u8(Path(p), gamma))); sizes[f"{src[0]}x{src[1]}"] += 1
        check_batch(m, Inputs(m, fp, [u], ("Full",))[0])
    return {"requests": n, "crop_mismatches": bad, "source_sizes": dict(sizes), "digests_equal": True, "ok": bad == 0}


def run_updates(ctx: Ctx, arm: str, seed: int, us: list[int], device, state_file: Path | None = None, aux_override: str | None = None) -> dict:
    """A fresh run over the given updates (the training step and schedule); optional resume state written at the end."""
    m, _ = ctx.manifest(seed); state, _ = ctx.start(seed); zt_all, _ = ctx.teacher(seed); model = make(state, seed, device); opt = make_optimizer(model, ctx.o)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); losses, t = [], []
    if state_file is not None and state_file.is_file():
        load_state(state_file, model, opt, "check")
    for u in us:
        b = batch_of(ctx, seed, u, (aux_override or arm,)); lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o); torch.cuda.synchronize(); a = time.time()
        r = train_step(model, opt, b, torch.from_numpy(zt_all[u]), lrs, ctx.o, device); torch.cuda.synchronize(); t.append(time.time() - a); losses.append(r["loss"])
    if state_file is not None and not state_file.is_file():
        save_state(state_file, model, opt, len(us), "check", [])
    return {"model": model, "losses": losses, "seconds": t}


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note, res = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "started": now()}
    try:
        for s in (0, 1, 2):
            ctx.teacher(s); ctx.start(s)
        res["pins"] = "A10 config and manifests, A08 weights, A18 teacher caches (sha, manifest sha, request digests), crop code, A15/A19/A20 records"
        res["synthetic_geometry"] = synthetic_geometry(); res["real_parity"] = real_parity(ctx, 0, [0, 1, 2, 3]) | {"seed": 0}
        state, _ = ctx.start(0); zt_all, _ = ctx.teacher(0); b = batch_of(ctx, 0, 0, ("Full", "Restricted")); mk = tm.marker_planes(b[3].to(device))
        rc = tm.to_rgb(b[1].to(device)); a08 = vsh.make_model(state, device).eval(); d0 = tet.state_digest(a08)
        with torch.no_grad():
            za = a08(rc, mk).float().cpu()
            starts = {}
            for arm, ai in (("Full", 0), ("Restricted", 1)):
                mdl = make(state, 0, device).eval(); starts[arm] = float((mdl.logits(rc, tm.to_rgb(b[2][ai].to(device)), mk).float().cpu() - za).abs().max())
        res["start_eval_equals_A08"] = {"max_abs": starts, "cached_teacher_max_abs": float((za - torch.from_numpy(zt_all[0])).abs().max()), "ok": max(starts.values()) == 0.0}
        par = []
        with torch.no_grad():
            for u in range(1, 7):
                bb = batch_of(ctx, 0, u, ("Full",)); par.append(float((a08(tm.to_rgb(bb[1].to(device)), tm.marker_planes(bb[3].to(device))).float().cpu() - torch.from_numpy(zt_all[u])).abs().max()))
        res["teacher"] = {"cached_against_online_max_abs": max(par), "windows": 72, "unchanged": tet.state_digest(a08) == d0, "grads": int(sum(p.grad is not None for p in a08.parameters())),
                          "ok": max(par) <= 1e-4 and tet.state_digest(a08) == d0}
        cross = {}
        for arm, ai in (("Full", 0), ("Restricted", 1)):
            mdl = make(state, 0, device); mdl.train(); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            with torch.no_grad():
                ec = mdl.base.embed(rc, mk); zc = mdl.base.classifier(ec).float().cpu(); mdl.base.embed(tm.to_rgb(b[2][ai].to(device)), mk)
            cross[arm] = (zc, torch.cuda.get_rng_state().clone(), torch.get_rng_state().clone())
        res["cross_arm_train_mode"] = {"crop_logits_equal": bool(torch.equal(cross["Full"][0], cross["Restricted"][0])), "cuda_rng_equal_after_both_forwards": bool(torch.equal(cross["Full"][1], cross["Restricted"][1])),
                                       "cpu_rng_equal": bool(torch.equal(cross["Full"][2], cross["Restricted"][2])), "aux_inputs_differ": bool(not torch.equal(b[2][0], b[2][1]))}
        res["cross_arm_train_mode"]["ok"] = all(res["cross_arm_train_mode"].values())
        mdl = make(state, 0, device); opt = make_optimizer(mdl, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0); g = []
        for u in (0, 1):
            bb = batch_of(ctx, 0, u, ("Full",)); g.append(train_step(mdl, opt, bb, torch.from_numpy(zt_all[u]), apply_schedule(mdl, opt, u, ctx.warmup, ctx.total, ctx.o), ctx.o, device, keep=True, aux_grad=True)["grads"])
        bb = batch_of(ctx, 0, ctx.warmup, ("Full",)); g.append(train_step(mdl, opt, bb, torch.from_numpy(zt_all[ctx.warmup]), apply_schedule(mdl, opt, ctx.warmup, ctx.warmup, ctx.total, ctx.o), ctx.o, device, keep=True, aux_grad=True)["grads"])
        n_all = sum(p.numel() for p in mdl.parameters()); n_g = sum(p.numel() for p in mdl.g.parameters()); n_a08 = sum(p.numel() for p in a08.parameters())
        res["gradients"] = {"u0_encoder_frozen": g[0], "u1": g[1], f"u{ctx.warmup}_unfrozen": g[2],
                            "ok": bool(g[0]["g_last"] > 0 and g[0]["g_first"] == 0.0 and g[0]["encoder"] == 0.0 and g[1]["g_first"] > 0 and g[1]["aux_input"] > 0 and g[1]["encoder"] == 0.0 and g[2]["encoder"] > 0)}
        res["sharing"] = {"parameters_total": int(n_all), "g": int(n_g), "A08": int(n_a08), "unique_parameter_objects": len({id(p) for p in mdl.parameters()}) == len(list(mdl.parameters())),
                          "optimiser_bound_once": bound(opt, mdl), "ok": bool(n_all == n_a08 + n_g and bound(opt, mdl))}
        del mdl, opt; torch.cuda.empty_cache()
        f1 = run_updates(ctx, "Full", 0, [0, 1], device, aux_override="Restricted"); r1 = run_updates(ctx, "Restricted", 0, [0, 1], device)
        res["forced_full_equals_restricted"] = {"loss_max_abs": float(max(abs(a - b2) for a, b2 in zip(f1["losses"], r1["losses"]))), "params_max_abs": max_diff(f1["model"], r1["model"])}
        res["forced_full_equals_restricted"]["ok"] = res["forced_full_equals_restricted"]["loss_max_abs"] <= 1e-5 and res["forced_full_equals_restricted"]["params_max_abs"] <= 1e-5
        del f1, r1; torch.cuda.empty_cache()
        us = list(range(ctx.warmup - 2, ctx.warmup + 4)); sf = ctx.out / "check_resume.pt"; sf.unlink(missing_ok=True); straight = run_updates(ctx, "Full", 0, us, device)
        repeat = run_updates(ctx, "Full", 0, us, device); run_updates(ctx, "Full", 0, us[:3], device, sf); second = run_updates(ctx, "Full", 0, us[3:], device, sf)
        rep = max_diff(straight["model"], repeat["model"]); dif = max_diff(straight["model"], second["model"]); ldif = float(max(abs(a - b2) for a, b2 in zip(straight["losses"][3:], second["losses"])))
        res["resume"] = {"updates": us, "loss_max_abs": ldif, "params_max_abs": dif, "repeat_params_max_abs": rep, "ok": bool(ldif <= 1e-4 and dif <= max(1e-5, 2 * rep))}
        sf.unlink(missing_ok=True); spu = float(np.mean(straight["seconds"][2:])); del straight, repeat, second; torch.cuda.empty_cache()
        res["timing"] = {"seconds_per_update": spu, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        mdl = make(state, 0, device).eval(); x = torch.from_numpy(np.repeat(b[1].numpy()[:1], 32, 0)); xa = torch.from_numpy(np.repeat(b[2][0].numpy()[:1], 32, 0)); mk32 = tm.marker_planes(torch.zeros(32, dtype=torch.long).to(device))
        with torch.no_grad():
            for _ in range(2):
                mdl.logits(tm.to_rgb(x.to(device)), tm.to_rgb(xa.to(device)), mk32)
            torch.cuda.synchronize(); a = time.time()
            for _ in range(3):
                mdl.logits(tm.to_rgb(x.to(device)), tm.to_rgb(xa.to(device)), mk32)
            torch.cuda.synchronize(); spw = (time.time() - a) / 96
        windows = 7666 + 7021 + 18175; ev = windows * spw / 3600; spent = ledger_hours(ctx.ledger, "gpu")
        res["budget"] = {"spent": spent, "inference_seconds_per_window_two_views": spw, "evaluation_per_model_hours": ev, "four_runs": 4 * ctx.total * spu / 3600,
                         "projection_two_seeds": spent + 4 * (ctx.total * spu / 3600 + ev), "seed1_reserve": 2 * (ctx.total * spu / 3600 + ev), "cap": ctx.vc["budget"]["gpu_hours_max"]}
        res["budget"]["fits_with_seed1"] = bool(res["budget"]["projection_two_seeds"] + res["budget"]["seed1_reserve"] <= res["budget"]["cap"])
        res["ok"] = bool(all(res[k]["ok"] for k in ("synthetic_geometry", "real_parity", "start_eval_equals_A08", "teacher", "cross_arm_train_mode", "gradients", "sharing", "forced_full_equals_restricted", "resume"))
                         and res["budget"]["projection_two_seeds"] <= res["budget"]["cap"]); res["finished"] = now()
        flat = json.loads(json.dumps(res, default=str))
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        print(json.dumps({k: (v.get("ok") if isinstance(v, dict) and "ok" in v else v) for k, v in flat.items() if k not in ("synthetic_geometry",)}, indent=1), flush=True)
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
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true"); g.add_argument("--queue", nargs="+")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        return prepare(ctx, a.seeds, a.workers)
    device = torch.device("cuda"); determinism()
    if a.check:
        return check(ctx, device, a.workers)
    queue(ctx, a.queue, device, a.workers)


if __name__ == "__main__":
    main()
