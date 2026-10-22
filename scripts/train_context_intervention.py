#!/usr/bin/env python
"""A23 (configs/experiments/video_context_intervention_v1.yaml): A08 all_s{k} continued with A10's 12 windows per update
plus one copy of each of the same 12 targets whose context, outside a protected local run of real target frames, is a
contiguous 10 Hz run of a donor recording: another view (Other) or the target's own view from another session (Same).

  --prepare          CPU: per-seed Other / Same manifests (24 windows per update), protected runs and donors
  --check            synthetic runs; real copies' pixels; GPU: teacher parity, forced Other = Same, optimiser, encoder
                     states and gradients, resume, timing and budget projection
  --queue K [K ...]  Other_s0, Same_s0, Other_s2, Same_s2 (seed 1 only after a met rule)
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
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.joint_visual import load_frame_u8  # noqa: E402

A10_CONFIG = "configs/experiments/video_switch_head_v1.yaml"
ARMS, BASE, WINDOWS, SLOTS, LEFT, RIGHT = ("Other", "Same"), 12, 24, 16, 4, 3
now, write_json, ledger_hours, ledger_append, determinism, max_diff = vsh.now, vsh.write_json, vsh.ledger_hours, vsh.ledger_append, vsh.determinism, vsh.max_diff


# ----------------------------------------------------------------------------------------------- intervention
def protected_run(s16: np.ndarray, m: int, labels: np.ndarray, joins: set) -> tuple[int, int]:
    """Slots [lo, hi] around the marker inside [m - 4, m + 3]: the target's label and continuous stream positions
    (equal, or consecutive with no fragment join between them)."""
    lab = int(labels[int(s16[m])]); cont = lambda p, q: q == p or (q == p + 1 and q not in joins)  # noqa: E731
    lo = m
    while lo - 1 >= max(0, m - LEFT) and cont(int(s16[lo - 1]), int(s16[lo])) and int(labels[int(s16[lo - 1])]) == lab:
        lo -= 1
    hi = m
    while hi + 1 <= min(SLOTS - 1, m + RIGHT) and cont(int(s16[hi]), int(s16[hi + 1])) and int(labels[int(s16[hi + 1])]) == lab:
        hi += 1
    return lo, hi


def draws(seed: int, u: int, w: int) -> tuple[float, float]:
    rng = np.random.default_rng([20261022, 23, int(seed), int(u), int(w)]); x, y = rng.random(2); return float(x), float(y)


def copy_slots(orig: list, lo: int, hi: int, donor_refs: list, offset: int) -> list:
    """The copy: protected slots keep the original refs; the other slots take the donor run in slot order."""
    out, k = [], 0
    for j in range(SLOTS):
        if lo <= j <= hi:
            out.append(orig[j])
        else:
            out.append(donor_refs[offset + k]); k += 1
    return out


class Donors:
    """Protocol-train native recordings by label (sorted ids), their sessions, sample counts and references."""

    def __init__(self, data) -> None:
        self.refs = {q["id"]: q["refs"] for q in data.seqs if q["kind"] == "native"}; self.label = {v: int(data.label[v]) for v in self.refs}
        self.n = {v: len(r) for v, r in self.refs.items()}; self.ids = sorted(self.refs); self._pools = {}

    def _base(self, arm: str, label: int, n_ctx: int) -> list:
        key = (arm, label, n_ctx)
        if key not in self._pools:
            self._pools[key] = [v for v in self.ids if (self.label[v] != label if arm == "Other" else self.label[v] == label) and self.n[v] >= n_ctx]
        return self._pools[key]

    def pick(self, arm: str, label: int, session: str, n_ctx: int, x: float, y: float) -> tuple[str, int]:
        pool = self._base(arm, label, n_ctx)
        if arm == "Same":
            pool = [v for v in pool if session_of(v) != session]
        if not pool:
            raise SystemExit(f"no {arm} donor for label {label} with {n_ctx} samples")
        v = pool[min(int(x * len(pool)), len(pool) - 1)]; span = self.n[v] - n_ctx + 1
        return v, min(int(y * span), span - 1)


def synthetic_runs() -> dict:
    """protected_run on synthetic windows: joins, label changes, repeats, window edges and the 8-slot cap."""
    lab = np.array([0] * 10 + [1] * 30 + [0] * 10); cases = {}
    s = np.arange(12, 28); cases["inside one run (cap 8)"] = protected_run(s, 8, lab, set()) == (4, 11)
    s = np.arange(2, 18); cases["label change on the left"] = protected_run(s, 8, lab, set()) == (8, 11)
    cases["fragment join on the right"] = protected_run(np.arange(12, 28), 8, lab, {22}) == (4, 9)
    s = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]); lab2 = np.zeros(8, int); cases["repeated positions"] = protected_run(s, 8, lab2, set()) == (4, 11)
    cases["window start"] = protected_run(np.arange(10, 26), 1, lab, set()) == (0, 4)
    cases["window end"] = protected_run(np.arange(24, 40), 15, lab, set()) == (11, 15)
    orig = [("T", i, 1.0) for i in range(16)]; donor = [("D", 100 + i, 1.0) for i in range(30)]; cp = copy_slots(orig, 4, 11, donor, 3)
    cases["copy keeps the run and fills the rest in order"] = cp[4:12] == orig[4:12] and cp[:4] == donor[3:7] and cp[12:] == donor[7:11]
    return {"cases": cases, "ok": bool(all(cases.values()))}


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vc = cfg, config, cfg["video_context_intervention"]; p = self.vc["pinned"]
        for k in ("A10_config", "A10_manifest_index", "A08_choices", "A15_pool_manifest", "A19_screen", "A19_logits_index", "A18_screen", "A20_screen"):
            if sha256_file(ROOT / p[k]["path"]) != p[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")
        self.a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=A10_CONFIG, set=[])), A10_CONFIG); self.total, self.warmup, self.o = self.a10.total, self.a10.warmup, self.a10.o
        o = self.vc["outputs"]; self.out, self.ckpt_root, self.results, self.ledger = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"], ROOT / self.vc["budget"]["ledger"]
        self._m, self._fp = {}, None

    def manifest(self, arm: str, seed: int) -> tuple[dict, str]:
        key = f"{arm}_s{seed}"
        if key not in self._m:
            idx = json.loads((self.out / "manifests" / "index.json").read_text())[key]; p = self.out / "manifests" / idx["file"]
            if (got := sha256_file(p)) != idx["sha256"]:
                raise SystemExit(f"{key}: the manifest differs from its index")
            with np.load(p) as z:
                self._m[key] = ({k: z[k] for k in z.files}, got)
        return self._m[key]

    def frame_paths(self, m: dict):
        """Frame directories of every protocol training clip (one table for all arms and seeds in a process)."""
        if self._fp is None:
            from scripts.train_joint_visual_temporal import FramePaths
            self._fp = FramePaths(prv.frame_dirs(self.a10.paths, self.a10.rv, set(prv.allowlist(self.a10.rv, ROOT))))
        if {str(v) for v in m["videos"].tolist()} - set(self._fp.dirs):
            raise SystemExit("a manifest recording is not a protocol training clip")
        return self._fp

    def start(self, seed: int) -> tuple[dict, str]:
        state, ssha = self.a10.start_state(seed)
        if ssha != self.vc["pinned"]["A08_weights"][seed]:
            raise SystemExit(f"seed {seed}: A08's start weights differ from the pin")
        return state, ssha

    def teacher(self, seed: int) -> dict:
        path = ROOT / "runs/video_ensemble_teacher_v1/cache" / f"teachers_s{seed}.npz"; csha = sha256_file(path); m, msha = self.a10.manifest(seed)
        if csha != self.vc["pinned"]["A18_teacher_cache"][seed]:
            raise SystemExit(f"seed {seed}: A18's teacher cache differs from its pin")
        with np.load(path) as z:
            if str(z["manifest_sha256"]) != msha or not np.array_equal(z["digests"][:self.total], m["digest_ordered"][:self.total]):
                raise SystemExit(f"seed {seed}: A18's cached requests differ from A10's windows")
            return {"logits": z["logits"][:, :self.total].astype(np.float32), "sha256": csha}


# ----------------------------------------------------------------------------------------------- preparation (CPU)
def prepare(ctx: Ctx, seeds: list[int]) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "started": now(), "seeds": {}}
    try:
        data = vpr.training_data(ctx.a10.a07p); seqs = data.seqs; donors = Donors(data); index = {}; mdir = ctx.out / "manifests"; mdir.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            m10, msha = ctx.a10.manifest(s); U = ctx.total; arr = {}; runs = np.empty((U, BASE, 2), np.int8); offs = np.empty((U, BASE), np.int32); nctx = np.empty((U, BASE), np.int8)
            copies = {a: [[None] * BASE for _ in range(U)] for a in ARMS}; donor_of = {a: np.empty((U, BASE), object) for a in ARMS}; bad = Counter()
            for u in range(U):
                for w in range(BASE):
                    qi, c, m = int(m10["seq"][u, w]), int(m10["c"][u, w]), int(m10["marker"][u, w]); q = seqs[qi]; lab = np.asarray(q["labels"]); s16 = m10["s"][u, w]
                    orig = [(str(m10["videos"][int(m10["vid"][u, w, j])]), int(m10["frame"][u, w, j]), round(float(m10["gamma"][u, w, j]), 6)) for j in range(SLOTS)]
                    if [(r[0], int(r[1]), round(float(r[2]), 6)) for r in (q["refs"][int(x)] for x in s16)] != orig or int(s16[m]) != c or int(lab[c]) != int(m10["target"][u, w]):
                        bad["original window differs from its sequence"] += 1
                    lo, hi = protected_run(s16, m, lab, set(int(j) for j in np.asarray(q["joins"]).tolist())); n_ctx = SLOTS - (hi - lo + 1); x, y = draws(s, u, w)
                    runs[u, w], nctx[u, w] = (lo, hi), n_ctx; sess = session_of(orig[m][0]); o_arm = {}
                    for a in ARMS:
                        v, o = donors.pick(a, int(lab[c]), sess, n_ctx, x, y); o_arm[a] = o; donor_of[a][u, w] = v
                        copies[a][u][w] = copy_slots(orig, lo, hi, [(r[0], int(r[1]), 1.0) for r in donors.refs[v]], o)
                    offs[u, w] = o_arm["Other"]
            if bad:
                raise SystemExit(f"seed {s}: {dict(bad)}")
            for a in ARMS:
                vids = sorted({str(v) for v in m10["videos"].tolist()} | {r[0] for row in copies[a] for cp in row for r in cp}); vix = {v: i for i, v in enumerate(vids)}
                remap = np.asarray([vix[str(v)] for v in m10["videos"]], np.int32)
                m = {"vid": np.empty((U, WINDOWS, SLOTS), np.int32), "frame": np.empty((U, WINDOWS, SLOTS), np.int32), "gamma": np.empty((U, WINDOWS, SLOTS), np.float32)}
                m["vid"][:, :BASE] = remap[m10["vid"][:U]]; m["frame"][:, :BASE] = m10["frame"][:U]; m["gamma"][:, :BASE] = m10["gamma"][:U]
                for u in range(U):
                    for w in range(BASE):
                        cp = copies[a][u][w]; m["vid"][u, BASE + w] = [vix[r[0]] for r in cp]; m["frame"][u, BASE + w] = [int(r[1]) for r in cp]; m["gamma"][u, BASE + w] = [float(r[2]) for r in cp]
                mk = np.concatenate([m10["marker"][:U], m10["marker"][:U]], 1).astype(np.int8); tg = np.concatenate([m10["target"][:U], m10["target"][:U]], 1).astype(np.int8)
                dig = np.zeros((U, WINDOWS, 32), np.uint8); dig[:, :BASE] = m10["digest_ordered"][:U]
                m.update({"marker": mk, "target": tg, "order_ordered": np.broadcast_to(np.arange(SLOTS, dtype=np.int8), (U, WINDOWS, SLOTS)).copy(), "videos": np.asarray(vids), "digest_ordered": dig,
                          "runs": runs, "n_context": nctx, "seq": m10["seq"][:U], "c": m10["c"][:U]})
                p = mdir / f"{a}_s{s}.npz"
                if p.is_file():
                    raise SystemExit(f"{p.name} exists; manifests are never overwritten")
                np.savez(p, **m); index[f"{a}_s{s}"] = {"file": p.name, "sha256": sha256_file(p), "a10_manifest_sha256": msha}
            plen = (runs[..., 1] - runs[..., 0] + 1).ravel(); dlab = {a: np.asarray([donors.label[v] for v in donor_of[a].ravel()]) for a in ARMS}; tlab = m10["target"][:ctx.total].ravel()
            rec["seeds"][str(s)] = {"a10_manifest_sha256": msha, "copies": int(plen.size), "protected_slots": {"mean": float(plen.mean()), "distribution": {str(k): int((plen == k).sum()) for k in range(1, 9)}},
                                    "replaced_slots_mean": float(SLOTS - plen.mean()), "donor_label_differs": {a: float((dlab[a] != tlab).mean()) for a in ARMS},
                                    "donor_recordings": {a: int(len(set(donor_of[a].ravel().tolist()))) for a in ARMS},
                                    "donor_label_distribution": {a: {str(k): int((dlab[a] == k).sum()) for k in range(5)} for a in ARMS},
                                    "teacher_error_rate": float((ctx.teacher(s)["logits"][s].argmax(-1) != m10["target"][:ctx.total]).mean())}
        rec["donor_pool"] = {"natives": len(donors.ids), "by_label": {str(k): int(sum(donors.label[v] == k for v in donors.ids)) for k in range(5)}}
        rec["finished"] = now(); write_json(mdir / "index.json", index)
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, rec)
        print(json.dumps(rec, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare manifests (CPU, main process)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- training
class Inputs(Dataset):
    def __init__(self, m: dict, fp, updates: list[int]) -> None:
        self.m, self.fp, self.updates = m, fp, list(updates)

    def __len__(self) -> int:
        return len(self.updates)

    def __getitem__(self, i: int):
        u = self.updates[i]; return (u, *rvc.window_inputs(self.m, self.m["order_ordered"], self.fp, [(u, w) for w in range(WINDOWS)]))


def loader(m: dict, fp, updates: list[int], workers: int) -> DataLoader:
    return DataLoader(Inputs(m, fp, updates), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def check_batch(m: dict, b) -> None:
    u = int(b[0])
    if not np.array_equal(b[4].numpy()[:BASE], m["digest_ordered"][u, :BASE]):
        raise SystemExit(f"update {u}: the original windows' pixels differ from A10's digests")
    if not np.array_equal(b[3].numpy(), m["target"][u]) or not np.array_equal(b[2].numpy(), m["marker"][u]):
        raise SystemExit(f"update {u}: targets or markers differ from the manifest")


def train_step(model, heads, opt, b, zt: torch.Tensor, lrs, o: dict, device, keep: bool = False) -> dict:
    """A18's own-teacher update on the 12 originals (CE + KL) plus CE on the 12 copies, one forward over 24 windows."""
    u8, slots, target = b[1], b[2], b[3]; t = target.to(device); model.train(); heads.train()
    z = model.classifier(model.embed(tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device)))); zt = zt.to(device)
    lp = F.log_softmax(z[:BASE], -1); lpt = F.log_softmax(zt, -1); ce_o = F.cross_entropy(z[:BASE], t[:BASE]); kl = (lpt.exp() * (lpt - lp)).sum(-1).mean()
    ce_c = F.cross_entropy(z[BASE:], t[BASE:]); loss = ce_o + kl + ce_c
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward(); params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    enc = float(max((p.grad.abs().max() for p in model.encoder_parameters() if p.grad is not None), default=torch.tensor(0.0))) if keep else None
    opt.step(); pred = z.detach().argmax(1).cpu().numpy(); tn = target.numpy()
    rec = {"loss": float(loss.detach()), "ce_original": float(ce_o.detach()), "kl": float(kl.detach()), "ce_copy": float(ce_c.detach()), "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1],
           "correct_original": int((pred[:BASE] == tn[:BASE]).sum()), "correct_copy": int((pred[BASE:] == tn[BASE:]).sum())}
    if keep:
        rec["encoder_grad_max"] = enc
    return rec


def code_digest() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(tet.__file__), Path(vsh.__file__), Path(rvc.__file__), Path(tm.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str, csha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "sections": {k: ctx.vc[k] for k in ("predecessors", "arms", "intervention", "loss", "training")}, "arm": arm, "seed": int(seed),
               "manifest_sha256": msha, "start_sha256": ssha, "teacher_cache_sha256": csha, "environment": prv.environment(), "total_updates": int(ctx.total), "code": code_digest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def student(ctx: Ctx, seed: int, device):
    state, ssha = ctx.start(seed); return tet.make_model(state, device), vsh.build_heads(seed).to(device), ssha


def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> dict:
    m, msha = ctx.manifest(arm, seed); cache = ctx.teacher(seed); model, heads, ssha = student(ctx, seed, device); ident = identity(ctx, arm, seed, msha, ssha, cache["sha256"])
    key = f"{arm}_s{seed}"; rd, ck = ctx.out / "train" / key, ctx.ckpt_root / key; rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
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
    print(f"[{key}] identity {ident[:12]}; updates {done} -> {ctx.total}", flush=True); t0 = time.time(); acc = []; torch.cuda.reset_peak_memory_stats(device)
    for b in loader(m, ctx.frame_paths(m), list(range(done, ctx.total)), workers):
        check_batch(m, b); u = int(b[0]); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][seed, u].copy()), lrs, ctx.o, device); acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"stage": f"{arm} student training (backward, GPU)", "identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.mean([a[k] for a in acc]))  # noqa: E731
            rec = {"updates": done, **{k: mean(k) for k in ("loss", "ce_original", "kl", "ce_copy", "grad_norm")}, "accuracy_original": sum(a["correct_original"] for a in acc) / (BASE * len(acc)),
                   "accuracy_copy": sum(a["correct_copy"] for a in acc) / (BASE * len(acc)), "lr_encoder_last": acc[-1]["lr_encoder"], "seconds_this_process": time.time() - t0,
                   "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); vsh.save_state(ck / "resume.pt", model, heads, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE orig {rec['ce_original']:.4f} KL {rec['kl']:.4f} CE copy {rec['ce_copy']:.4f} acc orig {rec['accuracy_original']:.3f} copy {rec['accuracy_copy']:.3f} "
                  f"{rec['seconds_this_process']:.0f}s", flush=True); acc = []
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done}")
    final = ck / "final.pt"; torch.save({"model": model.state_dict(), "heads": heads.state_dict()}, final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha, "cache_sha256": cache["sha256"], "start_sha256": ssha,
           "environment": prv.environment(), "code": code_digest(), "updates": done, "epochs": log, "resumes": (rd / "resumes.jsonl").read_text().splitlines() if (rd / "resumes.jsonl").is_file() else [],
           "final_checkpoint": str(final), "final_sha256": sha256_file(final), "parameters": int(sum(p.numel() for p in model.parameters())), "encoder_forwards_per_window": 1,
           "seconds_last_process": time.time() - t0}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True); print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
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


# ----------------------------------------------------------------------------------------------- checks
def batch_of(ctx: Ctx, arm: str, seed: int, u: int):
    m, _ = ctx.manifest(arm, seed); b = Inputs(m, ctx.frame_paths(m), [u])[0]; check_batch(m, b); return b


def legality(ctx: Ctx, seeds: list[int]) -> dict:
    """Every copy: protected run legal and equal to the original, context slots one contiguous donor run with the arm's
    label/session relation; both arms share runs, offsets and slot layout."""
    data = vpr.training_data(ctx.a10.a07p); donors = Donors(data); out, ok = {}, True
    for s in seeds:
        mO, _ = ctx.manifest("Other", s); mS, _ = ctx.manifest("Same", s); m10, _ = ctx.a10.manifest(s); bad = Counter(); n = 0
        if not (np.array_equal(mO["runs"], mS["runs"]) and np.array_equal(mO["n_context"], mS["n_context"])):
            bad["runs differ across arms"] += 1
        for u in range(ctx.total):
            for w in range(BASE):
                n += 1; lo, hi = (int(x) for x in mO["runs"][u, w]); mk = int(m10["marker"][u, w]); q = data.seqs[int(m10["seq"][u, w])]; lab = np.asarray(q["labels"]); s16 = m10["s"][u, w]
                if not (lo <= mk <= hi and hi - lo + 1 <= LEFT + RIGHT + 1 and lo >= mk - LEFT and hi <= mk + RIGHT and all(int(lab[int(s16[j])]) == int(lab[int(s16[mk])]) for j in range(lo, hi + 1))):
                    bad["protected run illegal"] += 1
                offs = {}
                for a, m in (("Other", mO), ("Same", mS)):
                    cp = [(str(m["videos"][m["vid"][u, BASE + w, j]]), int(m["frame"][u, BASE + w, j]), round(float(m["gamma"][u, BASE + w, j]), 6)) for j in range(SLOTS)]
                    orig = [(str(m["videos"][m["vid"][u, w, j]]), int(m["frame"][u, w, j]), round(float(m["gamma"][u, w, j]), 6)) for j in range(SLOTS)]
                    if cp[lo:hi + 1] != orig[lo:hi + 1]:
                        bad[f"{a}: protected slots differ from the original"] += 1
                    ctxs = [cp[j] for j in range(SLOTS) if not lo <= j <= hi]; v = ctxs[0][0]
                    if any(x[0] != v or x[2] != 1.0 for x in ctxs) or v not in donors.refs:
                        bad[f"{a}: context not one native donor at gamma 1"] += 1; continue
                    plan = [int(r[1]) for r in donors.refs[v]]; o = plan.index(ctxs[0][1]) if ctxs[0][1] in plan else -1
                    if o < 0 or [x[1] for x in ctxs] != plan[o:o + len(ctxs)]:
                        bad[f"{a}: context not one contiguous donor run"] += 1
                    offs[a] = o; tl = int(m["target"][u, BASE + w]); dl = donors.label[v]
                    if (a == "Other" and dl == tl) or (a == "Same" and (dl != tl or session_of(v) == session_of(orig[mk][0]))):
                        bad[f"{a}: donor label/session relation"] += 1
                    if int(m["target"][u, BASE + w]) != int(m10["target"][u, w]) or int(m["marker"][u, BASE + w]) != mk:
                        bad[f"{a}: copy target or marker"] += 1
        out[f"s{s}"] = {"copies": n, "violations": dict(bad)}; ok = ok and not bad
    out["ok"] = bool(ok)
    return out


def pixel_check(ctx: Ctx, seed: int, us: list[int]) -> dict:
    """Real copies through the training loader: protected slots bit-equal to the original window's, context slots equal
    to independently decoded donor frames."""
    out, ok = {}, True
    for a in ARMS:
        m, _ = ctx.manifest(a, seed); fp = ctx.frame_paths(m); prot = ctxok = 0
        for u in us:
            b = batch_of(ctx, a, seed, u); fr = b[1].numpy()
            for w in range(BASE):
                lo, hi = (int(x) for x in m["runs"][u, w]); prot += int(np.array_equal(fr[BASE + w, lo:hi + 1], fr[w, lo:hi + 1]))
                good = True
                for j in range(SLOTS):
                    if lo <= j <= hi:
                        continue
                    v = str(m["videos"][m["vid"][u, BASE + w, j]]); p = fp(v); img = load_frame_u8(p[min(int(m["frame"][u, BASE + w, j]), len(p) - 1)], 1.0); good = good and np.array_equal(fr[BASE + w, j], img)
                ctxok += int(good)
        n = len(us) * BASE; out[a] = {"copies": n, "protected_equal": prot, "context_equal": ctxok}; ok = ok and prot == n and ctxok == n
    out["ok"] = bool(ok)
    return out


def run(ctx: Ctx, arm: str, seed: int, us: list[int], device, state_file: Path | None = None, inputs_of: str | None = None) -> dict:
    cache = ctx.teacher(seed); model, heads, _ = student(ctx, seed, device); opt = vsh.make_optimizer(model, heads, ctx.o)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); losses, secs, enc = [], [], []
    if state_file is not None and state_file.is_file():
        vsh.load_state(state_file, model, heads, opt, "check")
    for u in us:
        b = batch_of(ctx, inputs_of or arm, seed, u); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o); torch.cuda.synchronize(); a = time.time()
        r = train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][seed, u].copy()), lrs, ctx.o, device, keep=True); torch.cuda.synchronize(); secs.append(time.time() - a)
        losses.append(r["loss"]); enc.append(r["encoder_grad_max"])
    if state_file is not None and not state_file.is_file():
        vsh.save_state(state_file, model, heads, opt, len(us), "check", [])
    return {"model": model, "losses": losses, "seconds": secs, "encoder_grads": enc}


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note, res = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "started": now()}
    try:
        res["synthetic_runs"] = synthetic_runs(); res["legality"] = legality(ctx, [0, 2]); res["pixels"] = pixel_check(ctx, 0, [0, 1])
        state, _ = ctx.start(0); cache = ctx.teacher(0); a08 = tet.make_model(state, device).eval(); d0 = tet.state_digest(a08); par = []
        with torch.no_grad():
            for u in range(6):
                b = batch_of(ctx, "Other", 0, u)
                par.append(float((a08(tm.to_rgb(b[1][:BASE].to(device)), tm.marker_planes(b[2][:BASE].to(device))).float().cpu() - torch.from_numpy(cache["logits"][0, u])).abs().max()))
        res["teacher"] = {"cached_against_online_max_abs": max(par), "windows": 72, "unchanged": tet.state_digest(a08) == d0, "ok": max(par) <= 1e-4 and tet.state_digest(a08) == d0}; del a08
        f1 = run(ctx, "Other", 0, [0, 1], device, inputs_of="Same"); r1 = run(ctx, "Same", 0, [0, 1], device)
        res["forced_other_equals_same"] = {"loss_max_abs": float(max(abs(a - b2) for a, b2 in zip(f1["losses"], r1["losses"]))), "params_max_abs": max_diff(f1["model"], r1["model"])}
        res["forced_other_equals_same"]["ok"] = res["forced_other_equals_same"]["loss_max_abs"] <= 1e-5 and res["forced_other_equals_same"]["params_max_abs"] <= 1e-5
        del f1, r1; torch.cuda.empty_cache()
        us = list(range(ctx.warmup - 2, ctx.warmup + 4)); sf = ctx.out / "check_resume.pt"; sf.unlink(missing_ok=True); straight = run(ctx, "Other", 0, us, device)
        repeat = run(ctx, "Other", 0, us, device); run(ctx, "Other", 0, us[:3], device, sf); second = run(ctx, "Other", 0, us[3:], device, sf)
        rp = max_diff(straight["model"], repeat["model"]); dif = max_diff(straight["model"], second["model"]); ld = float(max(abs(a - b2) for a, b2 in zip(straight["losses"][3:], second["losses"])))
        res["resume"] = {"updates": us, "loss_max_abs": ld, "params_max_abs": dif, "repeat_params_max_abs": rp, "ok": bool(ld <= 1e-4 and dif <= max(1e-5, 2 * rp))}
        res["encoder"] = {"grad_max_before_warmup": straight["encoder_grads"][:2], "grad_max_after": straight["encoder_grads"][2:], "ok": bool(all(g == 0.0 for g in straight["encoder_grads"][:2]) and all(g > 0 for g in straight["encoder_grads"][2:]))}
        model, heads, _ = student(ctx, 0, device); opt = vsh.make_optimizer(model, heads, ctx.o); res["optimiser_bound"] = {"ok": bool(vsh.bound(opt, model, heads))}; del model, heads, opt
        sf.unlink(missing_ok=True); spu = float(np.mean(straight["seconds"][2:])); del straight, repeat, second; torch.cuda.empty_cache()
        res["timing"] = {"seconds_per_update": spu, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        a08 = tet.make_model(state, device).eval(); x = torch.randint(0, 256, (32, 16, 224, 224, 3), dtype=torch.uint8); mk = tm.marker_planes(torch.zeros(32, dtype=torch.long).to(device))
        with torch.no_grad():
            a08(tm.to_rgb(x.to(device)), mk); torch.cuda.synchronize(); a = time.time()
            for _ in range(3):
                a08(tm.to_rgb(x.to(device)), mk)
            torch.cuda.synchronize(); spw = (time.time() - a) / 96
        ev = (7666 + 7021 + 18175) * spw / 3600; spent = ledger_hours(ctx.ledger, "gpu"); per_run = ctx.total * spu / 3600
        res["budget"] = {"spent": spent, "inference_seconds_per_window": spw, "evaluation_per_model_hours": ev, "per_run_hours": per_run, "projection_two_seeds": spent + 4 * (per_run + ev),
                         "seed1_reserve": 2 * (per_run + ev), "cap": ctx.vc["budget"]["gpu_hours_max"]}
        res["budget"]["fits_with_seed1"] = bool(res["budget"]["projection_two_seeds"] + res["budget"]["seed1_reserve"] <= res["budget"]["cap"])
        res["ok"] = bool(all(res[k]["ok"] for k in ("synthetic_runs", "legality", "pixels", "teacher", "forced_other_equals_same", "resume", "encoder", "optimiser_bound"))
                         and res["budget"]["projection_two_seeds"] <= res["budget"]["cap"]); res["finished"] = now()
        flat = json.loads(json.dumps(res, default=str))
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
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=7); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2])
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true"); g.add_argument("--queue", nargs="+")
    g.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(); a.set = []
    if a.synthetic:
        print(json.dumps(synthetic_runs(), indent=1)); return
    ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        return prepare(ctx, a.seeds)
    device = torch.device("cuda"); determinism()
    if a.check:
        return check(ctx, device, a.workers)
    queue(ctx, a.queue, device, a.workers)


if __name__ == "__main__":
    main()
