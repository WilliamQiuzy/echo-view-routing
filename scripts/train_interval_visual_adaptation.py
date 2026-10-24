#!/usr/bin/env python
"""A28 (configs/experiments/video_interval_visual_adaptation_v1.yaml): one 16-frame clip per interval
(echo_routing/temporal/interval_clips.py) read by an independent copy of A08; Frozen trains only the copied classifier,
Adapted also the last MultiscaleBlock, final norm and 512-d projection. The original A08 + B3 path is unchanged.

  --stage prepare   rebuild A27's T / P intervals with sample ranges (checked against A27), the clip manifest
  --stage check     clip cases, pixels, split forward, Base rows, trainable sets and optimizer binding, forced equality,
                    3 + 3 resume, a small sanity fit, timing and budget projection
  --stage embed     frozen-copy clip embeddings of every training interval (Init / Frozen inputs), cached by key
  --stage train     Frozen (cached embeddings, CPU) and Adapted (online, GPU) per seed
  --stage freeze    freeze both arms; training summary
  --stage readout   training pool only: interval accuracy, fixed / broken against Base
  --stage evaluate  once: development halves and the A15 check pool
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # before importing A27's CPU-only trainer, which would otherwise hide the GPU

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
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_recognition as t27  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.interval_clips as ic  # noqa: E402
import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.joint_visual import load_frame_u8  # noqa: E402

A27_CONFIG = "configs/experiments/video_interval_recognition_v1.yaml"
SEEDS, ARMS, HALVES = (0, 2), ("Frozen", "Adapted"), ("selection", "calibration")
UPDATES, WARM, PER, BATCH = 1350, 135, 16, 32
LR_HEAD, LR_REP, WD, CLIP = 1e-4, 1e-5, 1e-4, 1.0
RV_MIN = {"source": {"frame_manifest_columns": ["video_id", "frames_dir"]}}
CMPS = {"Adapted - Frozen": ("Adapted", "Frozen"), "Frozen - Base": ("Frozen", "Base"), "Adapted - Base": ("Adapted", "Base"), "Frozen - Init": ("Frozen", "Init"),
        "Adapted - Init": ("Adapted", "Init"), "Init - Base": ("Init", "Base")}
now, write_json, ledger_append = vsh.now, vsh.write_json, vsh.ledger_append


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.va = cfg, config, cfg["video_interval_visual_adaptation"]; self.pin = self.va["pinned"]; o = self.va["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.va["budget"]["ledger"]
        for k in ("A08_choices", "A27_config"):
            if sha256_file(ROOT / self.pin[k]["path"]) != self.pin[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")
        self.a27 = t27.Ctx(load_cfg(SimpleNamespace(config=A27_CONFIG, set=[])), A27_CONFIG)

    def a08(self, s: int, device) -> tm.TargetMViT:
        p = ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt"
        if sha256_file(p) != self.pin["A08_weights"][s]:
            raise SystemExit(f"A08 all_s{s}: weights differ from the pin")
        return vsh.make_model(torch.load(p, map_location="cpu", weights_only=True), device).eval()


def code_digest() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(ic.__file__), Path(tm.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- intervals and clips
def interval_lists(ctx: Ctx, s: int) -> dict:
    """A27's T and P lists, rebuilt in A27's order with (sequence, start, end), checked against A27's file; A27's batches."""
    D = t27.pool_data(ctx.a27, s); P = ctx.a27.pool(); fit = bc.members(P, "large"); T, Pl = [], []
    for i in fit:
        T += [(i, a, b, c) for a, b, c in ir.runs(D["labels"][i])]; Pl += [(i, a, b, ir.majority(D["labels"][i][a:b])[0], c) for a, b, c in ir.runs(D["base"][i])]
    f = ROOT / ctx.pin["A27_intervals"]["path"].split(" ")[0].format(seed=s)        # the pinned path precedes its description
    if sha256_file(f) != ctx.pin["A27_intervals"][s]:
        raise SystemExit(f"seed {s}: A27's interval file differs from its pin")
    with np.load(f) as z:
        same = (np.array_equal(z["T_labels"], [x[3] for x in T]) and np.array_equal(z["T_lengths"], [x[2] - x[1] for x in T]) and np.array_equal(z["P_labels"], [x[3] for x in Pl])
                and np.array_equal(z["P_lengths"], [x[2] - x[1] for x in Pl]))
        bt, bp = z["batch_T"].astype(np.int64), z["batch_P"].astype(np.int64)
    if not same:
        raise SystemExit(f"seed {s}: the rebuilt intervals differ from A27's lists")
    return {"T": T, "P": Pl, "bt": bt, "bp": bp, "seqs": P["seqs"], "D": D}


def clip_of(seq: dict, a: int, b: int) -> tuple[list, int]:
    s = ic.clip_positions(a, b); return [tuple(seq["refs"][int(k)]) for k in s], ic.clip_marker(s)


def frame_paths(seqs) -> FramePaths:
    return FramePaths(prv.frame_dirs(load_paths(), RV_MIN, {r[0] for q in seqs for r in q["refs"]}))


def load_clip(refs: list, fp) -> np.ndarray:
    out, memo = np.empty((16, 224, 224, 3), np.uint8), {}
    for k, (v, f, g) in enumerate(refs):
        if (v, f, g) not in memo:
            p = fp(v); memo[(v, f, g)] = load_frame_u8(p[min(int(f), len(p) - 1)], round(float(g), 6))
        out[k] = memo[(v, f, g)]
    return out


class Clips(Dataset):
    def __init__(self, items: list, fp) -> None:
        self.items, self.fp = items, fp

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        refs, mk = self.items[i]; return torch.from_numpy(load_clip(refs, self.fp)), int(mk)


class Updates(Dataset):
    """Per update: the 32 clips (16 T + 16 P) of A27's batch, their marker slots and labels."""

    def __init__(self, L: dict, us: list[int], fp) -> None:
        self.L, self.us, self.fp = L, list(us), fp

    def __len__(self) -> int:
        return len(self.us)

    def __getitem__(self, i: int):
        u = self.us[i]; rows = [self.L["T"][int(j)] for j in self.L["bt"][u]] + [self.L["P"][int(j)] for j in self.L["bp"][u]]; fr, mk, y = [], [], []
        for r in rows:
            refs, m = clip_of(self.L["seqs"][r[0]], r[1], r[2]); fr.append(load_clip(refs, self.fp)); mk.append(m); y.append(r[3])
        return u, torch.from_numpy(np.stack(fr)), torch.tensor(mk), torch.tensor(y)


def embed_items(model, items: list, fp, device, workers: int, acc: dict, batch: int = BATCH) -> np.ndarray:
    out = []
    with torch.no_grad():
        for u8, mk in DataLoader(Clips(items, fp), batch_size=batch, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True):
            torch.cuda.synchronize(); t0 = time.time(); e = model.embed(tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(mk.to(device))); out.append(e.float().cpu().numpy())
            torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - t0; acc["clips"] += len(mk)
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


# ----------------------------------------------------------------------------------------------- prepare
def prepare(ctx: Ctx) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "seeds": {}}
    try:
        ddir = ctx.out / "data"; ddir.mkdir(parents=True, exist_ok=True)
        for s in SEEDS:
            L = interval_lists(ctx, s); seqs = L["seqs"]; man = {}
            for name in ("T", "P"):
                clips = [clip_of(seqs[r[0]], r[1], r[2]) for r in L[name]]; lengths = np.asarray([r[2] - r[1] for r in L[name]])
                cross = sum(1 for r, (refs, _) in zip(L[name], clips) if len({x[0] for x in refs}) > 1)
                man[name] = {"intervals": len(clips), "length_lt16_repeating": int((lengths < 16).sum()), "length_1": int((lengths == 1).sum()), "length_ge16_even": int((lengths >= 16).sum()),
                             "clips_spanning_two_or_more_recordings": int(cross), "median_length": float(np.median(lengths)), "max_length": int(lengths.max())}
            key = hashlib.sha256(json.dumps({s_: [[r[0], r[1], r[2], r[3]] for r in L[s_]] for s_ in ("T", "P")}, sort_keys=True).encode()).hexdigest()
            rec["seeds"][str(s)] = {**man, "clip_manifest_key": key, "a27_lists_equal": True}; print(json.dumps(rec["seeds"][str(s)]), flush=True)
        rec["rule"] = ctx.va["clip"]["rule"]; rec["finished"] = now()
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- embed (Init / Frozen inputs)
def embed_key(ctx: Ctx, s: int) -> str:
    prep = json.loads((ctx.out / "preparation.json").read_text())
    return hashlib.sha256(json.dumps({"a08": ctx.pin["A08_weights"][s], "clips": prep["seeds"][str(s)]["clip_manifest_key"], "code": code_digest(), "precision": "fp32 eval"}, sort_keys=True).encode()).hexdigest()


def embed(ctx: Ctx, seeds: list[int], device, workers: int) -> None:
    for s in seeds:
        t0, started, ok, note, acc = time.time(), now(), False, "", {"gpu_seconds": 0.0, "clips": 0}
        try:
            L = interval_lists(ctx, s); fp = frame_paths(L["seqs"]); m = ctx.a08(s, device); d0 = tet.state_digest(m); key = embed_key(ctx, s); out = {}
            for name in ("T", "P"):
                out[name] = embed_items(m, [clip_of(L["seqs"][r[0]], r[1], r[2]) for r in L[name]], fp, device, workers, acc)
            if tet.state_digest(m) != d0:
                raise SystemExit("the frozen copy changed during embedding")
            cdir = ctx.out / "cache"; cdir.mkdir(parents=True, exist_ok=True); np.savez(cdir / f"frozen_s{s}.npz", key=key, T=out["T"], P=out["P"])
            print(f"seed {s}: {acc['clips']} clips, GPU {acc['gpu_seconds']:.0f} s", flush=True); ok = True; note = f"{acc['clips']} clips"
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"embed training intervals seed {s} (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def frozen_cache(ctx: Ctx, s: int) -> dict:
    with np.load(ctx.out / "cache" / f"frozen_s{s}.npz") as z:
        if str(z["key"]) != embed_key(ctx, s):
            raise SystemExit(f"seed {s}: the frozen clip cache belongs to another key")
        return {"T": z["T"], "P": z["P"]}


# ----------------------------------------------------------------------------------------------- training
def head_init(ctx: Ctx, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    return ctx.a27.init(s)


def frozen_steps(head, opt, E: dict, L: dict, us, log: list | None = None) -> None:
    for u in us:
        for g in opt.param_groups:
            g["lr"] = LR_HEAD * lr_factor(int(u), WARM, UPDATES)
        x = torch.from_numpy(np.concatenate([E["T"][L["bt"][u]], E["P"][L["bp"][u]]])); y = torch.tensor([L["T"][int(j)][3] for j in L["bt"][u]] + [L["P"][int(j)][3] for j in L["bp"][u]])
        z = head(x); loss = F.cross_entropy(z, y); opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)); opt.step()
        if log is not None:
            log.append({"u": int(u), "loss": float(loss.detach()), "grad_norm": gn, "correct": int((z.detach().argmax(1) == y).sum())})


def adapted_opt(m, rep_lr: float = LR_REP):
    rep = [p for n, p in ic.trainable(m) if not n.startswith("classifier.")]; head = [p for n, p in ic.trainable(m) if n.startswith("classifier.")]
    opt = torch.optim.AdamW([{"params": rep, "lr": 0.0, "base": rep_lr, "name": "representation"}, {"params": head, "lr": 0.0, "base": LR_HEAD, "name": "classifier"}],
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)
    return opt


def adapted_steps(m, opt, L: dict, us, fp, device, workers: int, acc: dict, log: list | None = None, ckpt: Path | None = None, ident: str = "") -> None:
    params = [p for _, p in ic.trainable(m)]
    for u, u8, mk, y in DataLoader(Updates(L, us, fp), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True):
        u = int(u)
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_factor(u, WARM, UPDATES)
        torch.cuda.synchronize(); t0 = time.time(); rgb, mp = tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(mk.to(device)); y = y.to(device)
        with torch.no_grad():
            x, thw = ic.prefix(m, rgb, mp)
        z = m.classifier(ic.suffix(m, x, thw)); loss = F.cross_entropy(z, y)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(params, CLIP)); opt.step(); torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - t0
        if log is not None:
            log.append({"u": u, "loss": float(loss.detach()), "grad_norm": gn, "correct": int((z.detach().argmax(1) == y).sum())})
        if ckpt is not None and (u + 1) % WARM == 0 and u + 1 < UPDATES:
            torch.save({"identity": ident, "updates": u + 1, "trainable": {n: p.detach().cpu() for n, p in ic.trainable(m)}, "opt": opt.state_dict(), "log": log}, ckpt.with_suffix(".tmp")); ckpt.with_suffix(".tmp").replace(ckpt)
        if log is not None and (u + 1) % WARM == 0:
            seg = log[-WARM:]; print(f"  update {u + 1}/{UPDATES}: loss {np.mean([x['loss'] for x in seg]):.4f} acc {sum(x['correct'] for x in seg) / (BATCH * len(seg)):.3f}", flush=True)


def prefix_digest(m) -> str:
    h = hashlib.sha256()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(ctx: Ctx, seeds: list[int], device, workers: int) -> None:
    for s in seeds:
        L = interval_lists(ctx, s); fp = frame_paths(L["seqs"]); E = frozen_cache(ctx, s); init = head_init(ctx, s)
        for arm in ARMS:
            t0, started, ok, note, acc = time.time(), now(), False, "", {"gpu_seconds": 0.0}
            try:
                rd = ctx.out / "train" / f"{arm}_s{s}"; rd.mkdir(parents=True, exist_ok=True); log = []
                ident = hashlib.sha256(json.dumps({"arm": arm, "seed": s, "config": sha256_file(ROOT / ctx.config), "embed_key": embed_key(ctx, s), "code": code_digest()}, sort_keys=True).encode()).hexdigest()
                if arm == "Frozen":
                    torch.set_num_threads(1); head = ir.copied_head(*init); opt = torch.optim.AdamW(head.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)
                    frozen_steps(head, opt, E, L, range(UPDATES), log); state = {f"classifier.{k}": v for k, v in head.state_dict().items()}; d_ok = True; trainable_n = 2565
                else:
                    torch.set_num_threads(8); m = ic.branch(ctx.a08(s, device), True); pd0 = prefix_digest(m); opt = adapted_opt(m); ck = rd / "resume.pt"; done = 0
                    if ck.is_file():
                        st = torch.load(ck, map_location="cpu", weights_only=False)
                        if st["identity"] != ident:
                            raise SystemExit("resume state belongs to another identity")
                        with torch.no_grad():
                            for n, p in ic.trainable(m):
                                p.copy_(st["trainable"][n])
                        opt.load_state_dict(st["opt"]); done, log = int(st["updates"]), list(st["log"])
                    adapted_steps(m, opt, L, range(done, UPDATES), fp, device, workers, acc, log, ck, ident); d_ok = prefix_digest(m) == pd0
                    if not d_ok:
                        raise SystemExit("a frozen prefix parameter changed")
                    state = {n: p.detach().cpu() for n, p in ic.trainable(m)}; trainable_n = int(sum(v.numel() for v in state.values())); ck.unlink(missing_ok=True)
                torch.save(state, rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                ph = lambda a, b: {"loss": float(np.mean([x["loss"] for x in log[a:b]])), "accuracy": sum(x["correct"] for x in log[a:b]) / (BATCH * (b - a)), "grad_norm": float(np.mean([x["grad_norm"] for x in log[a:b]]))}  # noqa: E731
                man = {"arm": arm, "seed": s, "identity": ident, "updates": len(log), "final_sha256": sha256_file(rd / "final.pt"), "trainable_parameters": trainable_n, "frozen_prefix_unchanged": d_ok,
                       "phases": {"first 135": ph(0, WARM), "last 270": ph(UPDATES - 270, UPDATES)}, "seconds": time.time() - t0, "gpu_seconds": acc["gpu_seconds"], "a08_sha256": ctx.pin["A08_weights"][s]}
                write_json(rd / "run_manifest.json", man); print(f"[{arm}_s{s}] {len(log)} updates in {man['seconds']:.0f} s; last-270 loss {man['phases']['last 270']['loss']:.4f}; final {man['final_sha256'][:12]}", flush=True); ok = True
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {arm}_s{s} ({'CPU' if arm == 'Frozen' else 'GPU job wall time'})", "cpu" if arm == "Frozen" else "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- frozen models
def load_branch(ctx: Ctx, s: int, arm: str, device, path: Path | None = None) -> tm.TargetMViT:
    m = ic.branch(ctx.a08(s, device), arm == "Adapted"); st = torch.load(path or ROOT / "models_frozen/video_interval_visual_adaptation_v1" / f"{arm}_s{s}" / "final.pt", map_location="cpu", weights_only=True)
    names = {n for n, _ in ic.trainable(m)}
    if set(st) != names:
        raise SystemExit(f"{arm}_s{s}: stored tensors {sorted(set(st) ^ names)[:4]} do not match the arm's trainable set")
    with torch.no_grad():
        for n, p in ic.trainable(m):
            p.copy_(st[n].to(p.device))
    return m.eval()


def freeze(ctx: Ctx, seeds: list[int]) -> None:
    t0, started, ok, note, out = time.time(), now(), False, "", {}
    try:
        for s in seeds:
            for arm in ARMS:
                rd = ctx.out / "train" / f"{arm}_s{s}"; man = json.loads((rd / "run_manifest.json").read_text()); dest = ROOT / "models_frozen/video_interval_visual_adaptation_v1" / f"{arm}_s{s}"
                if not dest.exists():
                    subprocess.run([sys.executable, "scripts/freeze_model.py", "video_interval_visual_adaptation_v1", f"{arm}_s{s}", "--file", str(rd / "final.pt"), "--file", str(rd / "run_manifest.json"),
                                    "--meta", f"run_manifest=runs/video_interval_visual_adaptation_v1/train/{arm}_s{s}/run_manifest.json", "--meta", f"a08_base={man['a08_sha256']}",
                                    "--notes", f"A28 {arm}_s{s} recognition branch trainable tensors (apply onto A08 all_s{s}); final update 1350"], cwd=ROOT, check=True, capture_output=True)
                if sha256_file(dest / "final.pt") != man["final_sha256"]:
                    raise SystemExit(f"{arm}_s{s}: the frozen copy differs")
                out[f"{arm}_s{s}"] = {k: man[k] for k in ("final_sha256", "trainable_parameters", "frozen_prefix_unchanged", "phases", "seconds", "gpu_seconds", "updates")}
        rec = {"frozen": out, "config_sha256": sha256_file(ROOT / ctx.config), "time": now()}
        for p in (ctx.out / "training_summary.json", ctx.results / "training_summary.json"):
            write_json(p, rec)
        print(json.dumps(rec, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "freeze (CPU)", "cpu", started, time.time() - t0, ok, note)


def frozen_record(ctx: Ctx, s: int, arm: str) -> str:
    summ = json.loads((ctx.out / "training_summary.json").read_text())["frozen"][f"{arm}_s{s}"]["final_sha256"]
    if sha256_file(ROOT / "models_frozen/video_interval_visual_adaptation_v1" / f"{arm}_s{s}" / "final.pt") != summ:
        raise SystemExit(f"{arm}_s{s}: frozen weights differ from the training summary")
    return summ


# ----------------------------------------------------------------------------------------------- systems on fixed Base runs
def classify(ctx: Ctx, s: int, items: list, fp, device, workers: int, acc: dict) -> dict:
    """Init (A08 classifier), Frozen (trained head) on the frozen copy's clip embeddings; Adapted on its own branch."""
    a08 = ctx.a08(s, device); E = embed_items(a08, items, fp, device, workers, acc); out = {}
    with torch.no_grad():
        out["Init"] = a08.classifier(torch.from_numpy(E).to(device)).cpu().numpy()
        fz = load_branch(ctx, s, "Frozen", device); out["Frozen"] = fz.classifier(torch.from_numpy(E).to(device)).cpu().numpy()
    ad = load_branch(ctx, s, "Adapted", device); Ea = embed_items(ad, items, fp, device, workers, acc)
    with torch.no_grad():
        out["Adapted"] = ad.classifier(torch.from_numpy(Ea).to(device)).cpu().numpy()
    return out


def relabelled(base: dict, runs_of: dict, logits: np.ndarray) -> dict:
    out = {}
    for k, b in base.items():
        lab = np.empty(len(b), np.int64)
        for (a, e, _), j in runs_of[k]:
            lab[a:e] = int(logits[j].argmax())
        out[k] = lab
    return out


def interval_table(seqs: list[dict], base: dict, runs_of: dict, preds: dict) -> dict:
    """Base-run five-class accuracy against the majority reference class; fixed / broken against Base with source counts;
    A4C/A5C at interval level."""
    out = {}
    for name, pr in preds.items():
        n = good = fixed = broke = a45 = a54 = 0; fr, fs, br, bs = set(), set(), set(), set()
        for q in seqs:
            y = np.asarray(q["labels"])
            for (a, e, c), _ in runs_of[q["id"]]:
                m = ir.majority(y[a:e])[0]; p = int(pr[q["id"]][a]); n += 1; good += int(p == m); vids = {r[0] for r in q["refs"][a:e]}
                a45 += int(m == 2 and p == 3); a54 += int(m == 3 and p == 2)
                if c != m and p == m:
                    fixed += 1; fr |= vids; fs |= {session_of(v) for v in vids}
                if c == m and p != m:
                    broke += 1; br |= vids; bs |= {session_of(v) for v in vids}
        out[name] = {"base_runs": n, "accuracy": good / max(1, n), "fixed": fixed, "fixed_recordings": len(fr), "fixed_sessions": len(fs), "broken": broke, "broken_recordings": len(br),
                     "broken_sessions": len(bs), "A4C_as_A5C": a45, "A5C_as_A4C": a54}
    return out


def base_runs(base: dict, seqs: list[dict]) -> tuple[dict, list]:
    runs_of, items = {}, []
    for q in seqs:
        rs = []
        for a, e, c in ir.runs(base[q["id"]]):
            rs.append(((a, e, c), len(items))); items.append(clip_of(q, a, e))
        runs_of[q["id"]] = rs
    return runs_of, items


def dev_set(ctx: Ctx, a8, h: str, s: int) -> tuple:
    ev, seqs = a8.half(h, rso.DevAccess({h})); refs = sorted({tuple(r) for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    z = t27.pinned_npz(ROOT / f"runs/video_frozen_switch_v1/evaluation/{h}/a08_s{s}.npz", ctx.a27.pin["dev_caches"][h][s])
    if not (np.array_equal(z["seq"], W["seq"]) and np.array_equal(z["centre"], W["centre"])):
        raise SystemExit("A13's development cache does not match this half's windows")
    base = decode(ctx.a27.params(s), rr.per_sample(W, z["logits"].astype(np.float32), seqs)); a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())
    row = rso.row_of(a8.cfg, base, ev); rso.check_stored(row["row"], a8sum["systems"][f"A + B3 s{s}"][h], f"Base s{s} {h}")
    return ev, seqs, base, row


def check_set(ctx: Ctx, a8, s: int) -> tuple:
    P = ctx.a27.pool(); cidx = bc.members(P, "check"); seqs = [P["seqs"][i] for i in cidx]; ev = bc.evalset(P, cidx, "check"); keep = np.isin(P["W"]["seq"], cidx); remap = {g: k for k, g in enumerate(cidx)}
    Wc = {"seq": np.asarray([remap[int(x)] for x in P["W"]["seq"][keep]]), "centre": P["W"]["centre"][keep]}; D = t27.pool_data(ctx.a27, s)
    base = decode(ctx.a27.params(s), rr.per_sample(Wc, D["logits"][keep], seqs)); m, t = bc.evaluate_pool(a8.cfg, base, ev); row = {"row": bce.check_row(m), "tables": t, "preds": base}
    a19 = json.loads((ROOT / ctx.a27.pin["check_rows"]["path"]).read_text())["systems"]; rso.check_stored(row["row"], a19[f"A08_s{s} old"]["check"], f"Base s{s} check (A19)")
    return ev, seqs, base, row


# ----------------------------------------------------------------------------------------------- check
def sub_batches(L: dict, us, repeat: int | None = None) -> dict:
    """The intervals drawn at updates us as their own lists, with batch rows at the same update indices (or one batch repeated)."""
    us = list(us); T = [L["T"][int(j)] for u in us for j in L["bt"][u]]; P = [L["P"][int(j)] for u in us for j in L["bp"][u]]
    bt = np.zeros((UPDATES, PER), np.int64); bp = np.zeros((UPDATES, PER), np.int64)
    for k, u in enumerate(us):
        bt[u] = np.arange(k * PER, (k + 1) * PER); bp[u] = np.arange(k * PER, (k + 1) * PER)
    return {**L, "T": T, "P": P, "bt": bt, "bp": bp}


def sub_embed(a08, Ls: dict, fp, device, workers: int, acc: dict) -> dict:
    return {k: embed_items(a08, [clip_of(Ls["seqs"][r[0]], r[1], r[2]) for r in Ls[k]], fp, device, workers, acc) for k in ("T", "P")}


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note, res, acc = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config)}, {"gpu_seconds": 0.0, "clips": 0}
    try:
        import tests.test_interval_clips as TC  # noqa: E402
        res["unit_tests"] = {n: (getattr(TC, n)() is None) for n in dir(TC) if n.startswith("test_")}
        s = 0; L = interval_lists(ctx, s); seqs = L["seqs"]; fp = frame_paths(seqs); a08 = ctx.a08(s, device); a08_file = sha256_file(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt")
        inside = all(set(ic.clip_positions(r[1], r[2]).tolist()) <= set(range(r[1], r[2])) for name in ("T", "P") for r in L[name])
        cross = next(r for r in L["P"] if len({x[0] for x in clip_of(seqs[r[0]], r[1], r[2])[0]}) > 1); refs, mk = clip_of(seqs[cross[0]], cross[1], cross[2])
        res["clip"] = {"all_positions_inside_their_interval": inside, "pixels_equal_A08_decode": bool(np.array_equal(load_clip(refs, fp), rr.decode_frames(refs, fp, 0))),
                       "join_crossing_example_recordings": len({x[0] for x in refs}), "join_crossing_example_marker": mk}
        items = [clip_of(seqs[r[0]], r[1], r[2]) for r in L["P"][:8]]; xb = tm.to_rgb(torch.stack([torch.from_numpy(load_clip(r, fp)) for r, _ in items]).to(device)); mp = tm.marker_planes(torch.tensor([m for _, m in items]).to(device))
        ad = ic.branch(a08, True)
        with torch.no_grad():
            full = a08.embed(xb, mp); x, thw = ic.prefix(ad, xb, mp); res["split_forward_max_abs"] = float((full - ic.suffix(ad, x, thw)).abs().max())
        res["base_rows"] = t27.evaluate(ctx.a27, list(SEEDS), zero_only=True)
        tr = ic.trainable(ad); opt = adapted_opt(ad); ids = [id(p) for g in opt.param_groups for p in g["params"]]
        res["trainable"] = {"modules": sorted({".".join(n.split(".")[:3]) if n.startswith("encoder.blocks") else ".".join(n.split(".")[:2]) for n, _ in tr}), "parameters": int(sum(p.numel() for _, p in tr)),
                            "frozen_arm_parameters": int(sum(p.numel() for _, p in ic.trainable(ic.branch(a08, False)))), "optimizer_binds_exactly": sorted(ids) == sorted(id(p) for _, p in tr),
                            "groups": {g["name"]: int(sum(p.numel() for p in g["params"])) for g in opt.param_groups}}
        us = range(WARM, WARM + 10); Lk = sub_batches(L, us); w0 = time.time(); c0 = acc["clips"]; Ek = sub_embed(a08, Lk, fp, device, workers, acc); wall_per_clip = (time.time() - w0) / (acc["clips"] - c0)
        head = ir.copied_head(*head_init(ctx, s)); ho = torch.optim.AdamW(head.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD); frozen_steps(head, ho, Ek, Lk, us)
        fe = ic.branch(a08, True); fo = adapted_opt(fe, rep_lr=0.0); pd0 = prefix_digest(fe); rep0 = {n: p.detach().clone() for n, p in ic.trainable(fe) if not n.startswith("classifier.")}
        adapted_steps(fe, fo, Lk, us, fp, device, workers, acc)
        res["forced_equal"] = {"updates": [us[0], us[-1]], "classifier_max_abs": float(max((fe.classifier.weight.detach().cpu() - head.weight.detach()).abs().max(), (fe.classifier.bias.detach().cpu() - head.bias.detach()).abs().max())),
                               "classifier_moved_max_abs": float((head.weight.detach() - head_init(ctx, s)[0]).abs().max()),
                               "representation_unchanged_at_lr0": all(torch.equal(p.detach(), rep0[n]) for n, p in ic.trainable(fe) if not n.startswith("classifier.")), "prefix_unchanged": prefix_digest(fe) == pd0}

        def run(rng_, m=None, o=None):
            if m is None:
                m = ic.branch(a08, True); o = adapted_opt(m)
            adapted_steps(m, o, L, rng_, fp, device, workers, acc); return m, o
        a, _ = run(range(WARM - 3, WARM + 3)); b, bo = run(range(WARM - 3, WARM)); st = {"t": {n: p.detach().clone() for n, p in ic.trainable(b)}, "o": bo.state_dict()}
        c = ic.branch(a08, True); co = adapted_opt(c)
        with torch.no_grad():
            for n, p in ic.trainable(c):
                p.copy_(st["t"][n])
        co.load_state_dict(st["o"]); run(range(WARM, WARM + 3), c, co); pa = dict(ic.trainable(a))
        res["resume_params_max_abs"] = float(max((pa[n] - p).detach().abs().max() for n, p in ic.trainable(c)))
        fixed = sub_batches(L, [0]); fixed = {**fixed, "bt": np.repeat(fixed["bt"][:1], UPDATES, 0), "bp": np.repeat(fixed["bp"][:1], UPDATES, 0)}; Ef = sub_embed(a08, fixed, fp, device, workers, acc); logs = {"Frozen": [], "Adapted": []}
        hs = ir.copied_head(*head_init(ctx, s)); hso = torch.optim.AdamW(hs.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD); frozen_steps(hs, hso, Ef, fixed, [WARM + 1] * 30, logs["Frozen"])
        sa = ic.branch(a08, True); so = adapted_opt(sa); r0 = {n: p.detach().clone() for n, p in ic.trainable(sa) if not n.startswith("classifier.")}; adapted_steps(sa, so, fixed, [WARM + 1] * 30, fp, device, workers, acc, logs["Adapted"])
        moved = float(sum((p.detach() - r0[n]).norm() for n, p in ic.trainable(sa) if not n.startswith("classifier.")))
        res["sanity_fixed_batch"] = {arm: {"loss_first": logs[arm][0]["loss"], "loss_last": logs[arm][-1]["loss"], "decreases": logs[arm][-1]["loss"] < logs[arm][0]["loss"]} for arm in logs}
        res["sanity_fixed_batch"]["Adapted"]["representation_moved_norm"] = moved
        tm0 = time.time(); run(range(WARM + 10, WARM + 30)); spu = (time.time() - tm0) / 20
        res["a08_file_unchanged"] = sha256_file(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt") == a08_file
        prep = json.loads((ctx.out / "preparation.json").read_text()); n_embed = sum(prep["seeds"][str(x)][k]["intervals"] for x in SEEDS for k in ("T", "P")); clips_eval = len(SEEDS) * 2 * (1011 + 934 + 2807)
        proj = {"embed_training_intervals": n_embed * wall_per_clip / 3600, "adapted_training": len(SEEDS) * UPDATES * spu / 3600, "readout": len(SEEDS) * 7400 * wall_per_clip / 3600, "evaluation": clips_eval * wall_per_clip / 3600}
        spent = vsh.ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600
        res["timing"] = {"adapted_seconds_per_update_wall": spu, "clip_embedding_wall_seconds": wall_per_clip}
        res["budget"] = {"gpu_spent_incl_this_check": spent, "projection_two_seeds": spent + sum(proj.values()), "parts": proj, "seed1_reserve": 0.4 + sum(proj.values()) / len(SEEDS), "cap": 6.0}
        res["budget"]["fits_with_seed1"] = bool(res["budget"]["projection_two_seeds"] + res["budget"]["seed1_reserve"] <= 6.0)
        res["ok"] = bool(all(res["unit_tests"].values()) and inside and res["clip"]["pixels_equal_A08_decode"] and res["split_forward_max_abs"] <= 1e-4 and res["base_rows"]["ok"]
                         and res["trainable"]["parameters"] == 7497989 and res["trainable"]["frozen_arm_parameters"] == 2565 and res["trainable"]["optimizer_binds_exactly"]
                         and res["forced_equal"]["classifier_max_abs"] <= 1e-4 and res["forced_equal"]["representation_unchanged_at_lr0"] and res["forced_equal"]["prefix_unchanged"]
                         and res["resume_params_max_abs"] <= 1e-5 and all(v["decreases"] for v in res["sanity_fixed_batch"].values()) and moved > 0 and res["a08_file_unchanged"] and res["budget"]["fits_with_seed1"])
        res["finished"] = now()
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, json.loads(json.dumps(res, default=str)))
        print(json.dumps(res, indent=1, default=str)[:6000], flush=True)
        if not res["ok"]:
            raise SystemExit("a check failed")
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- readout (training pool)
def readout(ctx: Ctx, seeds: list[int], device, workers: int) -> None:
    t0, started, ok, note, acc, out = time.time(), now(), False, "", {"gpu_seconds": 0.0, "clips": 0}, {}
    try:
        for s in seeds:
            for arm in ARMS:
                frozen_record(ctx, s, arm)
            L = interval_lists(ctx, s); fp = frame_paths(L["seqs"]); E = frozen_cache(ctx, s); a08 = ctx.a08(s, device); fz = load_branch(ctx, s, "Frozen", device); ad = load_branch(ctx, s, "Adapted", device)
            with torch.no_grad():
                lg = {"Init": {k: a08.classifier(torch.from_numpy(E[k]).to(device)).cpu().numpy() for k in ("T", "P")}, "Frozen": {k: fz.classifier(torch.from_numpy(E[k]).to(device)).cpu().numpy() for k in ("T", "P")}}
            Ea = embed_items(ad, [clip_of(L["seqs"][r[0]], r[1], r[2]) for r in L["P"]], fp, device, workers, acc)
            with torch.no_grad():
                lg["Adapted"] = {"P": ad.classifier(torch.from_numpy(Ea).to(device)).cpu().numpy()}
            yP = np.asarray([r[3] for r in L["P"]]); cP = np.asarray([r[4] for r in L["P"]]); yT = np.asarray([r[3] for r in L["T"]])
            for name, d in lg.items():
                pP = d["P"].argmax(1); r = {"P_accuracy": float((pP == yP).mean()), "P_base_accuracy": float((cP == yP).mean()), "P_wrong_base": int((cP != yP).sum()),
                                            "P_fixed": int(((cP != yP) & (pP == yP)).sum()), "P_broken": int(((cP == yP) & (pP != yP)).sum()), "P_changed": int((pP != cP).sum()),
                                            "P_ce": float(F.cross_entropy(torch.from_numpy(d["P"]), torch.from_numpy(yP)))}
                if "T" in d:
                    r["T_accuracy"] = float((d["T"].argmax(1) == yT).mean())
                out[f"{name} s{s}"] = r; print(name, s, r, flush=True)
        rec = {"scope": "training pool (A15 large fit) only; no development, check, test or Pediatric data", "runs": out, "gpu_seconds": acc["gpu_seconds"], "time": now()}
        for p in (ctx.out / "training_readout.json", ctx.results / "training_readout.json"):
            write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "training-pool readout (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- evaluate
def gate(rows: dict, counts: dict) -> dict:
    cells = [(s, h) for s in SEEDS for h in HALVES]; out = {}; Fv = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    for x in ARMS:
        dF = {f"s{s} {h}": Fv(x, s, h) - Fv("Base", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in SEEDS])) for h in HALVES}
        dN = float(np.mean([rows[f"{x} s{s}"][h]["native_macro_f1"] - rows[f"Base s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{x} s{s}"][h]["boundary_recall_multi"] - rows[f"Base s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "Base")}
        feas = {f"s{s}": bool(rows[f"{x} s{s}"]["calibration"]["native_macro_f1"] >= 0.980 and rows[f"{x} s{s}"]["calibration"]["boundary_recall_multi"] >= 0.95) for s in SEEDS}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs, "calibration_feasible": feas,
             "selection_native_floor_0.973": {f"s{s}": bool(rows[f"{x} s{s}"]["selection"]["native_macro_f1"] >= 0.973) for s in SEEDS}}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["Base"] + 3); r["feasibility"] = bool(all(feas.values()))
        r["passes"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"] and r["feasibility"]); out[x] = r
    out["train_seed1"] = bool(out["Frozen"]["passes"] or out["Adapted"]["passes"])
    return out


def evaluate(ctx: Ctx, seeds: list[int], device, workers: int) -> None:
    t0, started, ok, note, acc = time.time(), now(), False, "", {"gpu_seconds": 0.0, "clips": 0}
    try:
        if (ctx.out / "screen.json").exists():
            raise SystemExit("A28 was evaluated already; evaluation runs once")
        heads = {f"{a}_s{s}": frozen_record(ctx, s, a) for s in seeds for a in ARMS}; a8 = rsh.a08_ctx(); systems, itab, seqs_of = {}, {}, {}
        for t in (*HALVES, "check"):
            for s in seeds:
                ev, seqs, base, brow = dev_set(ctx, a8, t, s) if t != "check" else check_set(ctx, a8, s); seqs_of[t] = seqs; runs_of, items = base_runs(base, seqs)
                lg = classify(ctx, s, items, frame_paths(seqs), device, workers, acc); preds = {"Base": base, **{k: relabelled(base, runs_of, v) for k, v in lg.items()}}
                for name, pr in preds.items():
                    if name == "Base":
                        sysd = brow
                    elif t == "check":
                        m, tb = bc.evaluate_pool(a8.cfg, pr, ev); sysd = {"row": bce.check_row(m), "tables": tb, "preds": pr}
                    else:
                        sysd = rso.row_of(a8.cfg, pr, ev)
                    systems.setdefault(f"{name} s{s}", {})[t] = sysd
                for name, v in interval_table(seqs, base, runs_of, preds).items():
                    itab.setdefault(f"{name} s{s}", {})[t] = v
                print(f"{t} s{s}: {len(items)} Base runs classified", flush=True)
        rows = {k: {t: v[t]["row"] for t in v} for k, v in systems.items()}; counts = {k: {t: rsh.counts_of(seqs_of[t], v[t]) for t in v} for k, v in systems.items()}; cmp_, acc_ = {}, {}
        for t in (*HALVES, "check"):
            for s in seeds:
                rhr.compare(seqs_of[t], systems, t, s, CMPS, cmp_, acc_)
        a27 = json.loads((ROOT / "docs/results/2026-10-23_interval_recognition/screen.json").read_text())["systems"]; a14 = json.loads((ROOT / "docs/results/2026-10-21_local_router/screen.json").read_text())["systems"]
        a20 = json.loads((ROOT / "docs/results/2026-10-22_hard_replay/screen.json").read_text())["systems"]
        context = {**{f"A27 {a} s{s}": a27[f"{a} s{s}"] for a in ("W", "I") for s in seeds}, **{f"A14 P s{s}": {h: a14[f"P router s{s}"][h] for h in HALVES} for s in seeds},
                   **{f"A20 H@A08 s{s}": {h: a20[f"H@A08 s{s}"][h] for h in HALVES} for s in seeds}}
        res = {"stage": "evaluate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "systems": rows, "segment_counts": counts, "interval_table": itab, "comparisons": cmp_, "accounting": acc_,
               "context_rows": context, "gate": gate(rows, counts), "heads": heads,
               "cost": {"branch_parameters": 34640549, "trainable": {"Frozen": 2565, "Adapted": 7497989}, "branch_forwards": "one 16-frame clip per Base run, in addition to A08's one forward per request window",
                        "branch_gpu_seconds_per_clip": acc["gpu_seconds"] / max(1, acc["clips"]), "clips": acc["clips"]}, "time": now()}
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("prepare", "check", "embed", "train", "freeze", "readout", "evaluate"), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS)); ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.stage in ("prepare", "freeze"):
        return prepare(ctx) if a.stage == "prepare" else freeze(ctx, a.seeds)
    vsh.determinism(); device = torch.device("cuda")
    {"check": lambda: check(ctx, device, a.workers), "embed": lambda: embed(ctx, a.seeds, device, a.workers), "train": lambda: train(ctx, a.seeds, device, a.workers),
     "readout": lambda: readout(ctx, a.seeds, device, a.workers), "evaluate": lambda: evaluate(ctx, a.seeds, device, a.workers)}[a.stage]()


if __name__ == "__main__":
    main()
