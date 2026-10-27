#!/usr/bin/env python
"""A41 (configs/experiments/video_echojepa_recognition_v1.yaml): EchoJEPA (E) against generic V-JEPA 2 (G) ViT-L, each with
the same r8 QKV LoRA and target / global head, trained 900 updates of 8 fit windows; Base cuts unchanged.

  --stage extract  strict target_encoder extraction of both downloaded checkpoints -> FP32 encoders + sources.json (CPU)
  --stage check    source / token / parity / formula / gradient / mechanics checks on fit frames, Base reproduction,
                   throughput / peak memory and the launch projection (GPU)
  --stage train    --seed s: G_f0, E_f0, G_f1, E_f1 (fit fold f: G then E), resumable (GPU)
  --stage freeze   --seed s --commit c --manifest-sha h: record that weights_s{s}.json (sha h) was committed in c (CPU)
  --stage readout  --seed s: after the weights are committed, one opposite-fold online readout (+ the source-2 rule) (GPU)
The frozen A08 + B3 path supplies Base and its runs; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # A27's CPU-only trainer import would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.diagnose_a5c_evidence as d32  # noqa: E402
import scripts.prepare_video_sequence_objective as a07  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_masked_feature as t40  # noqa: E402
import scripts.train_native_spatial_detail as t36  # noqa: E402
import scripts.train_patch_evidence as t39  # noqa: E402
import scripts.train_predicted_run_context as t38  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
import scripts.train_wider_temporal_evidence as t37  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.temporal.echojepa_lora as jl  # noqa: E402
import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

A31_CONFIG, A32_CONFIG = t36.A31_CONFIG, t36.A32_CONFIG
FOLDS, ARMS, U, WARM, PER, LOG_EVERY, SAVE_EVERY = (0, 1), ("G", "E"), 900, 90, 8, 100, 100
LR_LORA, LR_HEAD, CLIP, READ_BATCH = 1e-4, 1e-3, 1.0, 32
SCH = {"updates": U, "warmup": WARM}
LIMITS = {"loss": 1e-3, "params": 1e-4}
TOL = {"parity_bf16": 1e-3, "parity_fp32": 1e-4, "formula_rel": 1e-5, "microbatch_fp64_rel": 1e-8, "checkpointing_rel": 1e-5}   # FP64 microbatch gate since check attempt 2
DRAW_BASE, DRAW_ROUND = 20261027, 41
FILES = {"G": "vitl.pt", "E": "vitl-vmix22m-pt220-c55.pt"}
A4C, A5C = d32.A4C, d32.A5C
now, write_json, ledger_append, tables, a5c_extras, ceil_pct = t31.now, t31.write_json, t31.ledger_append, t36.tables, t39.a5c_extras, t39.ceil_pct


def model_seed(s: int, f: int) -> int:
    return jl.INIT_BASE + 10 * int(s) + int(f)


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["echojepa_recognition"]; o = self.v["outputs"]
        self.out, self.results, self.ledger, self.src = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"], ROOT / o["sources_dir"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG); self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t36.staged(ctx, job, kind, fn)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    t36.publish(ctx, name, rec)


# ----------------------------------------------------------------------------------------------- upstream and sources
def upstream_git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(jl.UPSTREAM), *a], check=True, capture_output=True, text=True).stdout.strip()


def verify_upstream() -> dict:
    """Clone at the pinned commit and clean; every loaded upstream file equals the pinned commit's blob."""
    jl.upstream(); files = jl.upstream_files(); blobs = {}
    for name, rel in sorted(files.items()):
        pinned = upstream_git("ls-tree", jl.COMMIT, rel).split(); blobs[rel] = {"module": name, "equal_pinned_blob": bool(pinned) and pinned[2] == upstream_git("hash-object", rel)}
    return {"commit": upstream_git("rev-parse", "HEAD"), "pinned": jl.COMMIT, "clean": upstream_git("status", "--porcelain") == "", "files": blobs, "pydeps": jl.pydeps(),
            "all_equal": bool(blobs) and all(b["equal_pinned_blob"] for b in blobs.values()), "src_path": [str(Path(p).resolve().relative_to(ROOT)) for p in sys.modules["src"].__path__]}


def sources(ctx: Ctx) -> dict:
    p = ctx.results / "sources.json"; rec = json.loads(p.read_text())
    if sha256_file(ctx.out / "sources.json") != sha256_file(p):
        raise SystemExit("server and results sources.json differ")
    return rec


_BASES: dict = {}


def base_state(ctx: Ctx, arm: str) -> tuple[dict, str]:
    """The arm's FP32 encoder state, sha-bound to sources.json (verified once per process, then kept in memory read-only)."""
    if arm not in _BASES:
        rec = sources(ctx)["encoders"][arm]; p = ctx.src / f"encoder_{arm}.pt"
        if sha256_file(p) != rec["sha256"]:
            raise SystemExit(f"encoder_{arm}.pt differs from sources.json")
        _BASES[arm] = (torch.load(p, map_location="cpu", weights_only=True), rec["sha256"])
    return _BASES[arm]


def trainable_digest(st: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(st):
        h.update(k.encode()); h.update(st[k].contiguous().numpy().tobytes())
    return h.hexdigest()


def load_checkpoint(p: Path) -> dict:
    """weights_only (tensors and plain types only); memory-mapped when the file is in the zip format."""
    try:
        return torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:
        return torch.load(p, map_location="cpu", weights_only=True)


def extract(ctx: Ctx) -> str:
    dl = json.loads((ctx.out / "downloads.json").read_text()); rec = {"config_sha256": sha256_file(ROOT / ctx.config), "upstream": verify_upstream(), "encoders": {}, "downloads": {}}
    if not (rec["upstream"]["clean"] and rec["upstream"]["commit"] == jl.COMMIT and rec["upstream"]["all_equal"] and rec["upstream"]["pydeps"]["in_round_dir"]):
        raise SystemExit("upstream clone not at the pinned commit / clean: engineering stop")
    ref = jl.build_encoder(use_activation_checkpointing=False); arch = {"blocks": len(ref.blocks), "dim": ref.embed_dim, "heads": ref.num_heads, "pos_embed_is_none": ref.pos_embed is None,
                                                                        "qkv": list(ref.blocks[0].attn.qkv.weight.shape), "norm_eps": ref.norm.eps, "use_rope": ref.use_rope, "uniform_power": ref.uniform_power}
    states = {}
    for arm in ARMS:
        d = dl["files"][arm]; p = ctx.src / FILES[arm]; sha = sha256_file(p)
        if sha != d["sha256"] or p.stat().st_size != d["bytes"]:
            raise SystemExit(f"{FILES[arm]} differs from downloads.json")
        rec["downloads"][arm] = {k: d.get(k) for k in ("file", "url", "bytes", "sha256", "downloaded_at", "drive_file_id", "drive_folder", "head_http_status", "head_content_length", "get_status_final_url_bytes")}
        ck = load_checkpoint(p); enc = jl.build_encoder(use_activation_checkpointing=False)
        st, rep = jl.strict_encoder_state(ck, enc); jl.load_base(enc, st); rep["epoch"] = ck.get("epoch") if isinstance(ck.get("epoch"), (int, float)) else None
        out = ctx.src / f"encoder_{arm}.pt"; torch.save(st, out); back = torch.load(out, map_location="cpu", weights_only=True)
        rep["reload_equal"] = sorted(back) == sorted(st) and all(torch.equal(back[k], st[k]) for k in st); rep["sha256"] = sha256_file(out); rep["bytes"] = out.stat().st_size
        rec["encoders"][arm] = rep; states[arm] = st; del ck, enc, back
    num = sum(float((states["E"][k].double() - states["G"][k].double()).pow(2).sum()) for k in states["G"]); den = sum(float(states["G"][k].double().pow(2).sum()) for k in states["G"])
    rec["E_vs_G"] = {"relative_l2": (num / den) ** 0.5, "identical_tensors": int(sum(torch.equal(states["E"][k], states["G"][k]) for k in states["G"])), "tensors": len(states["G"])}
    rec["architecture"] = arch; rec["contract"] = jl.CONTRACT; rec["written"] = now(); publish(ctx, "sources.json", rec)
    return f"E {rec['encoders']['E']['sha256'][:12]}, G {rec['encoders']['G']['sha256'][:12]}; E vs G relative L2 {rec['E_vs_G']['relative_l2']:.4f}"


# ----------------------------------------------------------------------------------------------- groups and batches
class Groups(t40.Groups):
    def draw(self, s: int, f: int, u: int) -> np.ndarray:
        """The 8 pool_W rows of update u: native group, then constructed group (A07's draw order), A41's stream."""
        rng = np.random.default_rng([DRAW_BASE, DRAW_ROUND, int(s), int(f), int(u), 1])
        i = int(rng.integers(len(self.nat))); j = int(rng.integers(len(self.nat[i][1]))); k = int(rng.integers(len(self.con))); l_ = int(rng.integers(len(self.con[k][1])))
        (ni, ns), (ci, cs) = self.nat[i], self.con[k]; rn, rc = self.rows[ni], self.rows[ci]
        return np.concatenate([rn[a07.group_indices(len(rn), int(ns[j]))], rc[a07.group_indices(len(rc), int(cs[l_]))]])


class Fold:
    def __init__(self, ctx: Ctx, f: int) -> None:
        self.f, self.seqs = f, t31.load_pool(ctx.a31, f); self.W = t31.pool_W(self.seqs); self.G = Groups(self.seqs, self.W); self.fp = t28.frame_paths(self.seqs)


def batches(F_: Fold, s: int, us):
    return t40.Batches(F_, s, us)


def rows_sha(F_: Fold, s: int, us) -> str:
    h = hashlib.sha256()
    for u in us:
        h.update(F_.G.draw(s, F_.f, u).astype(np.int64).tobytes())
    return h.hexdigest()


def on_device(b, device) -> tuple:
    u, rows, u8, mk, y = b
    return int(u), rows.numpy(), u8.to(device, non_blocking=True), mk.to(device), y.to(device)


# ----------------------------------------------------------------------------------------------- model, update
def build(ctx: Ctx, arm: str, s: int, f: int, device, base: str | None = None, bf16: bool = True) -> jl.Recogniser:
    """The arm's strict base encoder (or `base`'s, for the forced E = G check), frozen; LoRA + head from the fixed generator."""
    st, _ = base_state(ctx, base or arm)
    with torch.device(device):
        enc = jl.build_encoder()                       # upstream random init on the device, then fully replaced by the strict state
    m = jl.Recogniser(jl.load_base(enc, st), bf16=bf16).to(device); jl.init_trainable(m, s, f)
    return m.train()


def trainable_params(m: jl.Recogniser) -> list:
    return [p for _, p in m.trainable()]


def make_opt(m: jl.Recogniser):
    return t31.opt_of([("lora", [p for k, p in m.trainable() if k.startswith("lora.")], LR_LORA), ("head", [m.head.weight, m.head.bias], LR_HEAD)])


def base_digest(m: jl.Recogniser) -> str:
    h = hashlib.sha256()
    for k, v in m.base_named():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def group_norms(m: jl.Recogniser) -> dict:
    sq = lambda ps: float(sum(0.0 if p.grad is None else float(p.grad.double().pow(2).sum()) for p in ps)) ** 0.5  # noqa: E731
    return {"g_A": sq([x.A for x in m.loras]), "g_B": sq([x.B for x in m.loras]), "g_head": sq([m.head.weight, m.head.bias])}


def update(m: jl.Recogniser, opt, b, device) -> dict:
    """One real update: 8 windows in draw order, microbatch 1, CE / 8 accumulated; one clip and one AdamW step."""
    u, _, u8, mk, y = on_device(b, device); t31.sched(opt, u, SCH); opt.zero_grad(set_to_none=True); ces = []
    for w in range(PER):
        ce = F.cross_entropy(m(u8[w:w + 1], mk[w:w + 1]), y[w:w + 1]); (ce / PER).backward(); ces.append(ce.detach())
    rec = {"u": u, "ce": float(torch.stack(ces).mean()), **group_norms(m)}; rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(trainable_params(m), CLIP)); opt.step(); return rec


def run_updates(m, opt, F_: Fold, s: int, us, device, workers: int, log: list, hook=None) -> float:
    gsec = 0.0
    for b in t40.loader(F_, s, us, workers):
        torch.cuda.synchronize(); a = time.time(); rec = update(m, opt, b, device); torch.cuda.synchronize(); gsec += time.time() - a; log.append(rec)
        if hook is not None:
            hook(rec["u"], rec)
    return gsec


def state_of(m: jl.Recogniser, opt) -> dict:
    return {"trainable": m.trainable_state(), "opt": opt.state_dict(), "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}


def load_state(m: jl.Recogniser, opt, st: dict) -> None:
    m.load_trainable(st["trainable"]); opt.load_state_dict(st["opt"]); torch.set_rng_state(st["cpu_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"])


def params_max_diff(a: dict, b: dict) -> float:
    return float(max((a[k].float() - b[k].float()).abs().max() for k in a))


@torch.no_grad()
def fit_monitor(m: jl.Recogniser, F_: Fold, s: int, device) -> dict:
    """Training inputs only: CE and accuracy of updates 0-3's batches (32 windows) in eval mode."""
    was = m.training; m.eval(); ce, acc = [], []
    for b in (batches(F_, s, range(4))[i] for i in range(4)):
        u, _, u8, mk, y = on_device(b, device); z = m(u8, mk); ce.append(float(F.cross_entropy(z, y))); acc.append(float((z.argmax(1) == y).float().mean()))
    m.train(was); return {"ce": float(np.mean(ce)), "accuracy": float(np.mean(acc)), "batches": "updates 0-3 (eval)"}


def lora_stats(m: jl.Recogniser) -> dict:
    d = jl.lora_delta_norms(m)
    return {"delta_W_fro_by_block": d, "delta_W_fro_total": float(np.sqrt(np.sum(np.square(d)))), "A_norm": float(sum(float(x.A.detach().double().pow(2).sum()) for x in m.loras) ** 0.5),
            "B_norm": float(sum(float(x.B.detach().double().pow(2).sum()) for x in m.loras) ** 0.5), "head_weight_norm": float(m.head.weight.detach().norm())}


# ----------------------------------------------------------------------------------------------- train
def identity(ctx: Ctx, arm: str, s: int, f: int) -> str:
    h = hashlib.sha256()
    for p in (Path(jl.__file__), Path(__file__)):
        h.update(p.read_bytes())
    sp = json.loads((ctx.a31.out / "split.json").read_text())
    h.update(json.dumps({"arm": arm, "seed": s, "fold": f, "base_sha256": sources(ctx)["encoders"][arm]["sha256"], "upstream": jl.COMMIT, "pools": sp["pool_sha256"], "config": sha256_file(ROOT / ctx.config)}, sort_keys=True).encode())
    return h.hexdigest()


def train_one(ctx: Ctx, arm: str, s: int, F_: Fold, ident: str, device, workers: int) -> dict:
    key = f"{arm}_s{s}_f{F_.f}"; rd = ctx.out / "train" / key; rd.mkdir(parents=True, exist_ok=True); rp = rd / "resume.pt"
    m = build(ctx, arm, s, F_.f, device); opt = make_opt(m); bd0 = base_digest(m); init_sha = trainable_digest(m.trainable_state())
    before = fit_monitor(m, F_, s, device); torch.manual_seed(model_seed(s, F_.f)); start, log, resumes = 0, [], []
    if rp.exists():
        st = torch.load(rp, map_location="cpu", weights_only=False)
        if st["identity"] != ident:
            raise SystemExit(f"{key}: the resume file belongs to another identity")
        load_state(m, opt, st); start, log, before = st["update"], st["log"], st["fit_before"]; resumes.append({"from_update": start, "at": now()})
    prog = ctx.out / f"progress_s{s}.json"; t0 = time.time()

    def hook(u: int, rec: dict) -> None:
        if (u + 1) % LOG_EVERY == 0:
            last = log[-LOG_EVERY:]; print(f"[{key}] update {u + 1}/{U}: CE {np.mean([x['ce'] for x in last]):.4f} grad {np.mean([x['grad_norm'] for x in last]):.3f} gA {np.mean([x['g_A'] for x in last]):.3g} gB {np.mean([x['g_B'] for x in last]):.3g}  {time.time() - t0:.0f} s", flush=True)
            write_json(prog, {"seed": s, "model": key, "update": u + 1, "total": U, "updated_at": now(), "pid": os.getpid(), "device": torch.cuda.get_device_name(0)})
        if (u + 1) % SAVE_EVERY == 0 and u + 1 < U:
            tmp = rp.with_suffix(".tmp"); torch.save({**state_of(m, opt), "update": u + 1, "log": log, "identity": ident, "fit_before": before}, tmp); tmp.replace(rp)

    gsec = run_updates(m, opt, F_, s, range(start, U), device, workers, log, hook)
    steps = [int(st_["step"]) for st_ in opt.state_dict()["state"].values()]
    if len(log) != U or [x["u"] for x in log] != list(range(U)) or not steps or set(steps) != {U}:
        raise SystemExit(f"{key}: {len(log)} logged updates, optimiser steps {sorted(set(steps))}")
    bd1 = base_digest(m)
    if bd1 != bd0:
        raise SystemExit(f"{key}: the base encoder changed")
    after = fit_monitor(m, F_, s, device); torch.save(m.trainable_state(), rd / "final.pt"); (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    man = {"arm": arm, "seed": s, "fit_fold": F_.f, "updates": len(log), "optimizer_steps": U, "identity": ident, "final_sha256": sha256_file(rd / "final.pt"), "base_sha256": sources(ctx)["encoders"][arm]["sha256"],
           "base_digest_unchanged": bd0 == bd1, "init_digest": init_sha, "updates_sha256": sha256_file(rd / "updates.jsonl"), "rows_sha256": rows_sha(F_, s, range(U)), "trainable_parameters": int(sum(p.numel() for p in trainable_params(m))),
           "fit_before": before, "fit_after": after, "lora": lora_stats(m), "resumes": resumes, "gpu_seconds_this_process": gsec, "seconds_this_process": time.time() - t0,
           "loss": {k: {"first_90": float(np.mean([x[k] for x in log[:WARM]])), "last_90": float(np.mean([x[k] for x in log[-WARM:]]))} for k in ("ce", "grad_norm", "g_A", "g_B", "g_head")},
           "first_updates": [{k: x[k] for k in ("u", "ce", "g_A", "g_B", "g_head", "grad_norm")} for x in log[:3]], "every_100": [{k: x[k] for k in ("u", "ce", "g_A", "g_B", "g_head", "grad_norm")} for x in log if x["u"] % 100 == 99],
           "label_exposure": dict(sorted(Counter(int(v) for u in range(U) for v in F_.W["label"][F_.G.draw(s, F_.f, u)]).items())), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    write_json(rd / "run_manifest.json", man); return man


def train(ctx: Ctx, s: int, device, workers: int) -> None:
    for f in FOLDS:
        F_ = Fold(ctx, f)
        for arm in ARMS:
            key = f"{arm}_s{s}_f{f}"; rd = ctx.out / "train" / key; ident = identity(ctx, arm, s, f); mp = rd / "run_manifest.json"
            if mp.exists() and json.loads(mp.read_text())["identity"] == ident and sha256_file(rd / "final.pt") == json.loads(mp.read_text())["final_sha256"]:
                print(f"[{key}] finished earlier", flush=True); continue
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                man = train_one(ctx, arm, s, F_, ident, device, workers); torch.cuda.empty_cache(); ok = True; note = f"final {man['final_sha256'][:12]}"
                if sha256_file(rd / "final.pt") == man["final_sha256"] and (rd / "resume.pt").exists():
                    (rd / "resume.pt").unlink()                                  # regenerable; the final weights are verified
                print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "final_sha256", "fit_before", "fit_after")}), flush=True)
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {key} (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)
    man = {f"{arm}_s{s}_f{f}": json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for f in FOLDS for arm in ARMS}
    publish(ctx, f"weights_s{s}.json", {"seed": s, "models": man, "sources_sha256": sha256_file(ctx.results / "sources.json"), "config_sha256": sha256_file(ROOT / ctx.config),
                                        "upstream": verify_upstream(), "loaded_code_sha256": t38.loaded_code(), "written": now()})


def freeze(ctx: Ctx, s: int, commit: str, manifest_sha: str) -> str:
    p = ctx.results / f"weights_s{s}.json"
    if sha256_file(p) != manifest_sha or sha256_file(ctx.out / f"weights_s{s}.json") != manifest_sha:
        raise SystemExit("the committed weight manifest differs from the server's")
    models = json.loads(p.read_text())["models"]; com = ctx.out / "weights_committed.json"; cur = json.loads(com.read_text()) if com.exists() else {}
    for k, man in models.items():
        if sha256_file(ctx.out / "train" / k / "final.pt") != man["final_sha256"] or sha256_file(ctx.src / f"encoder_{man['arm']}.pt") != man["base_sha256"]:
            raise SystemExit(f"{k}: weights or base differ from the manifest")
        cur[k] = {"final_sha256": man["final_sha256"], "base_sha256": man["base_sha256"]}
    cur[f"commit_s{s}"] = commit; write_json(com, cur); return f"{len(models)} models pinned at {commit}"


# ----------------------------------------------------------------------------------------------- check
def group_checks(ctx: Ctx, s: int) -> dict:
    out = {}
    for f in FOLDS:
        F_ = Fold(ctx, f); grid_ok = True
        for i, q in enumerate(F_.seqs):
            w = vsf.windows(len(q["refs"])); r = F_.G.rows[i]
            grid_ok &= bool(np.array_equal(w["centres"], F_.W["centre"][r]) and np.array_equal(w["samples"], F_.W["s"][r]) and np.array_equal(w["markers"], F_.W["marker"][r]))
        valid, lab, nat_seq, con_seq = True, Counter(), set(), set()
        for u in range(U):
            rows = F_.G.draw(s, f, u); sq = F_.W["seq"][rows]; c = F_.W["centre"][rows]; y = F_.W["label"][rows]; lab.update(y.tolist()); nat_seq.add(int(sq[0])); con_seq.add(int(sq[4]))
            nat, con = F_.seqs[int(sq[0])], F_.seqs[int(sq[4])]
            valid &= bool(len(set(sq[:4].tolist())) == 1 and len(set(sq[4:].tolist())) == 1 and nat["kind"] == "native" and con["kind"] == "multi" and len(set(y[4:].tolist())) > 1
                          and np.all(np.diff(c[4:]) > 0) and np.all(np.diff(c[:4]) >= 0) and np.array_equal(y, np.asarray([F_.seqs[int(a)]["labels"][int(b)] for a, b in zip(sq, c)])))
        out[str(f)] = {"grid_equals_A07_windows": grid_ok, "native_sequences": len(F_.G.nat), "eligible_streams": len(F_.G.con), "eligible_starts": int(sum(len(x[1]) for x in F_.G.con)),
                       "short_natives_repeating": int(sum(1 for _, st in F_.G.nat if len(F_.G.rows[_]) < 4)), "draws_valid_900": valid, "label_counts_900x8": {str(k): int(v) for k, v in sorted(lab.items())},
                       "distinct_native_sequences_drawn": len(nat_seq), "distinct_streams_drawn": len(con_seq), "draw_repeat_equal": rows_sha(F_, s, range(U)) == rows_sha(F_, s, range(U)),
                       "rows_sha256": rows_sha(F_, s, range(U)), "differs_from_A40_draws": rows_sha(F_, s, range(U)) != t40.rows_sha(t40.Fold(ctx_a40(ctx), f), s, range(U))}
    return out


def ctx_a40(ctx: Ctx):
    return SimpleNamespace(a31=ctx.a31)


def token_checks(m: jl.Recogniser, u8: torch.Tensor, mk: torch.Tensor) -> dict:
    enc = m.encoder; x = jl.jepa_input(u8[:1])
    with torch.no_grad():
        conv = enc.patch_embed.proj(x); pe = enc.patch_embed(x); t_, h_, w_ = conv.shape[2:]
        loop = max(float((pe[:, t * h_ * w_ + y * w_ + xx] - conv[:, :, t, y, xx]).abs().max()) for t in range(t_) for y in range(0, h_, 3) for xx in range(0, w_, 3))
        ids = torch.arange(jl.N_TOKENS); d, h, w = enc.blocks[0].attn.separate_positions(ids, 14, 14); ref = torch.stack([ids // 196, (ids % 196) // 14, ids % 14])
        tok = torch.randn(3, jl.N_TOKENS, 8); mm = torch.tensor([0, 7, 15]); feat = jl.readout_features(tok, mm)
        exp = torch.stack([torch.cat([tok[b, (int(mm[b]) // 2) * 196:(int(mm[b]) // 2 + 1) * 196].mean(0), tok[b].mean(0)]) for b in range(3)])
        nx = jl.jepa_input(u8[:1]); manual = ((u8[:1].float() / 255.0 - torch.tensor(jl.MEAN, device=u8.device)) / torch.tensor(jl.STD, device=u8.device)).permute(0, 4, 1, 2, 3)
        err = False
        try:
            jl.readout_features(tok, torch.tensor([0, 7, 16]))
        except ValueError:
            err = True
    return {"conv_thw": [int(t_), int(h_), int(w_)], "patch_tokens_equal_conv_time_major_loop": loop, "rope_positions_time_major": bool(torch.equal(torch.stack([d, h, w]), ref)),
            "readout_vs_loop_max_abs": float((feat - exp).abs().max()), "marker_16_rejected": err, "input_normalisation_max_abs": float((nx - manual).abs().max()), "input_shape": list(nx.shape),
            "fit_marker_slots": sorted({int(v) for v in mk.tolist()}), "readout_inputs": "tokens and marker slot only (no label, no interval boundary)"}


def parity_checks(ctx: Ctx, u8: torch.Tensor, mk: torch.Tensor, device) -> dict:
    """Zero-LoRA wrapped encoder against a separately built unwrapped official encoder on the same state (both arms)."""
    out = {}
    for arm in ARMS:
        m = build(ctx, arm, 0, 0, device).eval(); st, _ = base_state(ctx, arm); r = {}
        with torch.device(device):
            off = jl.load_base(jl.build_encoder(), st).eval()      # separately built, unwrapped official encoder
        with torch.no_grad():
            for mode, n in (("bf16", 4), ("fp32", 2)):
                m.bf16 = mode == "bf16"; x = jl.jepa_input(u8[:n])
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=m.bf16):
                    to = off(x)
                tw = m.tokens(u8[:n]); za, zb = m.head(jl.readout_features(tw, mk[:n])), m.head(jl.readout_features(to, mk[:n]))
                r[mode] = {"tokens_max_abs": float((tw.float() - to.float()).abs().max()), "token_shape": list(tw.shape), "head_argmax_equal": bool(torch.equal(za.argmax(1), zb.argmax(1))), "windows": n}
            m.bf16 = True; m.encoder.use_activation_checkpointing = False; ta = m.tokens(u8[:2]); m.encoder.use_activation_checkpointing = True; tb = m.tokens(u8[:2])
            r["eval_checkpointing_on_off_tokens_max_abs"] = float((ta.float() - tb.float()).abs().max())
        out[arm] = r; del m, off; torch.cuda.empty_cache()
    return out


def formula_check(device) -> dict:
    g = torch.Generator().manual_seed(7); base = torch.nn.Linear(jl.DIM, 3 * jl.DIM).to(device); mod = jl.LoRAQKV(base).to(device)
    with torch.no_grad():
        mod.A.copy_(torch.randn(mod.A.shape, generator=g)); mod.B.copy_(torch.randn(mod.B.shape, generator=g) * 0.1); x = torch.randn(2, 64, jl.DIM, generator=g).to(device)
        y = mod(x).double(); xd = x.double(); ref = xd @ base.weight.double().T + base.bias.double() + 2.0 * (xd @ mod.A.double().T) @ mod.B.double().T
        zero = jl.LoRAQKV(base).to(device); same = bool(torch.equal(zero(x), base(x)))
    return {"max_abs": float((y - ref).abs().max()), "out_max": float(ref.abs().max()), "scale": mod.scale, "zero_B_equals_base_exactly": same, "base_frozen": not any(p.requires_grad for p in base.parameters())}


def init_checks(ctx: Ctx, device) -> dict:
    a, b = build(ctx, "G", 0, 0, device), build(ctx, "E", 0, 0, device); sa, sb = a.trainable_state(), b.trainable_state(); c = build(ctx, "G", 0, 1, device).trainable_state()
    cnt = int(sum(p.numel() for p in a.parameters() if p.requires_grad)); base_rg = any(p.requires_grad for _, p in a.base_named())
    rec = {"trainable_identical_E_G": sorted(sa) == sorted(sb) and all(torch.equal(sa[k], sb[k]) for k in sa), "fold_inits_differ": any(not torch.equal(sa[k], c[k]) for k in sa if k.endswith(".A") or k == "head.weight"),
           "B_zero_and_head_bias_zero": all(float(v.abs().sum()) == 0 for k, v in sa.items() if k.endswith(".B") or k == "head.bias"), "A_std": float(torch.cat([v.flatten() for k, v in sa.items() if k.endswith(".A")]).std()),
           "trainable_count_requires_grad": cnt, "trainable_count_list": int(sum(v.numel() for v in sa.values())), "expected": jl.TRAINABLE, "base_requires_grad_any": base_rg,
           "bases_differ": base_digest(a) != base_digest(b), "lora_wrappers": len(a.loras), "global_rng_untouched_by_init": None}
    cpu0 = torch.get_rng_state(); jl.init_trainable(a, 0, 0); rec["global_rng_untouched_by_init"] = bool(torch.equal(cpu0, torch.get_rng_state())); del a, b; torch.cuda.empty_cache(); return rec


def grads(m: jl.Recogniser) -> dict:
    return {k: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for k, p in m.trainable()}


def rel(ga: dict, gb: dict) -> dict:
    mx = max(float(v.abs().max()) for v in gb.values()); d = max(float((ga[k] - gb[k]).abs().max()) for k in gb)
    va, vb = torch.cat([ga[k].flatten().double() for k in gb]), torch.cat([gb[k].flatten().double() for k in gb])
    return {"max_abs": d, "grad_max": mx, "cosine": float(F.cosine_similarity(va, vb, dim=0))}


def fp64_microbatch(ctx: Ctx, u8: torch.Tensor, mk: torch.Tensor, y: torch.Tensor, device) -> dict:
    """Batch coupling test: the whole recogniser in FP64 (math SDPA), check-only nonzero B, 2 windows: batch CE gradient against
    microbatch-1 accumulation of CE / 2 (FP32 differences are round-off amplified through the encoder; see check attempt 1)."""
    m = build(ctx, "E", 0, 0, device, bf16=False); g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for x in m.loras:
            x.B.copy_(torch.randn(x.B.shape, generator=g) * 0.02)
    m.double(); x64 = jl.jepa_input(u8[:2]).double()

    def logits(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        gr = jl.token_grid(m.encoder(x)); t = torch.div(k.long(), 2, rounding_mode="floor")
        return m.head(torch.cat([gr[torch.arange(gr.shape[0], device=gr.device), t].mean(1), gr.mean((1, 2))], 1))

    F.cross_entropy(logits(x64, mk[:2]), y[:2]).backward(); gb = grads(m); m.zero_grad(set_to_none=True)
    for w in range(2):
        (F.cross_entropy(logits(x64[w:w + 1], mk[w:w + 1]), y[w:w + 1]) / 2).backward()
    return {**rel(grads(m), gb), "windows": 2, "dtype": "float64"}


def gradient_checks(ctx: Ctx, F_: Fold, device) -> dict:
    """Check-only nonzero B; microbatch-1 accumulation = batch gradient (FP64 gate; FP32 / BF16 recorded); checkpointing on / off;
    then the real zero-B start."""
    m = build(ctx, "E", 0, F_.f, device); g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for x in m.loras:
            x.B.copy_(torch.randn(x.B.shape, generator=g) * 0.02)
    b = batches(F_, 0, [3])[0]; u, _, u8, mk, y = on_device(b, device); out = {}
    for mode in ("fp32", "bf16"):
        m.bf16 = mode == "bf16"; m.zero_grad(set_to_none=True); F.cross_entropy(m(u8, mk), y).backward(); gb = grads(m); m.zero_grad(set_to_none=True)
        for w in range(PER):
            (F.cross_entropy(m(u8[w:w + 1], mk[w:w + 1]), y[w:w + 1]) / PER).backward()
        out[f"microbatch_vs_batch_{mode}"] = rel(grads(m), gb); m.zero_grad(set_to_none=True)
        m.encoder.use_activation_checkpointing = False; F.cross_entropy(m(u8[:2], mk[:2]), y[:2]).backward(); g_off = grads(m); m.zero_grad(set_to_none=True)
        m.encoder.use_activation_checkpointing = True; F.cross_entropy(m(u8[:2], mk[:2]), y[:2]).backward(); out[f"checkpointing_on_vs_off_{mode}"] = rel(grads(m), g_off); m.zero_grad(set_to_none=True)
    del m; torch.cuda.empty_cache(); out["microbatch_vs_batch_fp64"] = fp64_microbatch(ctx, u8, mk, y, device); torch.cuda.empty_cache()
    m = build(ctx, "E", 0, F_.f, device); F.cross_entropy(m(u8[:1], mk[:1]), y[:1]).backward(); g0 = group_norms(m); d0 = grads(m)
    out["real_zero_B_start"] = {**g0, "A_grad_tensors_zero": int(sum(float(v.abs().sum()) == 0 for k, v in d0.items() if k.endswith(".A"))), "B_grad_tensors_nonzero": int(sum(float(v.abs().sum()) > 0 for k, v in d0.items() if k.endswith(".B"))),
                                "head_grad_nonzero": float(d0["head.weight"].abs().sum()) > 0, "all_finite": all(bool(torch.isfinite(v).all()) for v in d0.values())}
    del m; torch.cuda.empty_cache(); return out


def mech_run(ctx: Ctx, arm: str, F_: Fold, us, device, workers: int, state: dict | None = None, base: str | None = None):
    m = build(ctx, arm, 0, F_.f, device, base=base); opt = make_opt(m); torch.manual_seed(model_seed(0, F_.f))
    if state is not None:
        load_state(m, opt, state)
    log = []; gs = run_updates(m, opt, F_, 0, us, device, workers, log); return m, opt, log, gs


def training_checks(ctx: Ctx, F_: Fold, device, workers: int) -> dict:
    us = range(85, 91); res = {}; m = build(ctx, "E", 0, F_.f, device); opt = make_opt(m); bd0 = base_digest(m); init = m.trainable_state()
    ids = sorted(id(p) for g in opt.param_groups for p in g["params"])
    res["binding"] = {"binds_exactly_trainable": ids == sorted(id(p) for p in trainable_params(m)) and len(ids) == len(set(ids)), "bound_parameters": int(sum(p.numel() for g in opt.param_groups for p in g["params"])),
                      "lr": {g["name"]: g["base"] for g in opt.param_groups}, "weight_decay": [g["weight_decay"] for g in opt.param_groups], "betas_eps": [list(opt.defaults["betas"]), opt.defaults["eps"]]}
    del m, opt; torch.cuda.empty_cache()
    L = lambda a, b: max(abs(x["ce"] - y["ce"]) for x, y in zip(a[2], b[2]))  # noqa: E731
    P = lambda a, b: params_max_diff(a[0].trainable_state(), b[0].trainable_state())  # noqa: E731
    ra = mech_run(ctx, "E", F_, us, device, workers); bd1 = base_digest(ra[0]); fin = ra[0].trainable_state()
    res["real_update"] = {"A_moved_max_abs": max(float((fin[k] - init[k]).abs().max()) for k in fin if k.endswith(".A")), "B_moved_max_abs": max(float((fin[k] - init[k]).abs().max()) for k in fin if k.endswith(".B")),
                          "head_moved_max_abs": float((fin["head.weight"] - init["head.weight"]).abs().max()), "optimizer_steps": sorted({int(v["step"]) for v in ra[1].state_dict()["state"].values()}),
                          "base_digest_unchanged": bd0 == bd1, "g_A_first": ra[2][0]["g_A"], "g_B_first": ra[2][0]["g_B"], "ce": [x["ce"] for x in ra[2]]}
    rb = mech_run(ctx, "E", F_, us, device, workers); res["repeat_6_E"] = {"loss_max_abs": L(ra, rb), "params_max_abs": P(ra, rb)}; del rb; torch.cuda.empty_cache()
    gg = mech_run(ctx, "G", F_, range(85, 88), device, workers); ge = mech_run(ctx, "E", F_, range(85, 88), device, workers, base="G")
    res["forced_E_equals_G_3"] = {"loss_max_abs": L(gg, ge), "params_max_abs": P(gg, ge)}; del gg, ge; torch.cuda.empty_cache()
    h3 = mech_run(ctx, "E", F_, range(85, 88), device, workers); p = ctx.out / "check_resume.pt"; torch.save(state_of(h3[0], h3[1]), p); log3 = h3[2]; del h3
    st = torch.load(p, map_location="cpu", weights_only=False); hr = mech_run(ctx, "E", F_, range(88, 91), device, workers, st); p.unlink()
    res["resume_3_plus_3_E"] = {"loss_max_abs": max(abs(x["ce"] - y["ce"]) for x, y in zip(ra[2], log3 + hr[2])), "params_max_abs": P(ra, hr)}
    res["timing"] = {"seconds_per_update_gpu_6": ra[3] / len(us)}; del ra, hr; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); m, opt, log, gsec = mech_run(ctx, "E", F_, range(200, 230), device, workers); wall = time.time() - t0
    res["timing"].update({"seconds_per_update_gpu_30": gsec / 30, "seconds_per_update_wall_30_incl_build_and_loader": wall / 30, "peak_gpu_gb_training": torch.cuda.max_memory_allocated() / 1e9}); del m, opt
    torch.cuda.empty_cache(); return res


def readout_timing(ctx: Ctx, device, workers: int) -> dict:
    F_ = Fold(ctx, 0); rows = np.flatnonzero(F_.W["seq"] < 20); t0 = time.time(); fr, idx = t37.chunk_frames(F_.seqs, {"seq": F_.W["seq"][rows]}, F_.W["s"][rows], F_.fp, workers); td = time.time() - t0
    m = build(ctx, "E", 0, 0, device).eval(); g = 0.0; torch.cuda.reset_peak_memory_stats()
    for b in range(0, len(rows), READ_BATCH):
        ks = np.arange(b, min(b + READ_BATCH, len(rows))); torch.cuda.synchronize(); t1 = time.time(); forward(m, fr, idx[ks], F_.W["marker"][rows[ks]], device); torch.cuda.synchronize(); g += time.time() - t1
    del m; torch.cuda.empty_cache()
    return {"timing_windows": int(len(rows)), "gpu_ms_per_window_one_model": 1e3 * g / len(rows), "decode_ms_per_window": 1e3 * td / len(rows), "peak_gpu_gb_readout": torch.cuda.max_memory_allocated() / 1e9}


def forward(m: jl.Recogniser, frames: np.ndarray, idx: np.ndarray, marker: np.ndarray, device) -> torch.Tensor:
    with torch.no_grad():
        return m(torch.from_numpy(frames[idx]).to(device), torch.from_numpy(np.asarray(marker)).long().to(device)).float()


def check(ctx: Ctx, device, workers: int) -> str:
    s = 0; t_start = time.time(); src = sources(ctx)
    res = {"config_sha256": sha256_file(ROOT / ctx.config), "sources_sha256": sha256_file(ctx.results / "sources.json"), "tolerances": TOL, "limits": LIMITS, "contract": jl.CONTRACT,
           "numerics": {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}}
    res["upstream"] = verify_upstream(); res["encoder_files"] = {a: sha256_file(ctx.src / f"encoder_{a}.pt") == src["encoders"][a]["sha256"] for a in ARMS}
    sp = json.loads((ctx.a31.out / "split.json").read_text()); res["isolation"] = sp["isolation"]; res["pool_sha256"] = sp["pool_sha256"]; res["groups"] = group_checks(ctx, s)
    F0 = Fold(ctx, 0); b = batches(F0, s, [0])[0]; u, rows, u8c, mk_c, y = b; rows = rows.numpy()
    fr, idx = t37.chunk_frames(F0.seqs, {"seq": F0.W["seq"][rows]}, F0.W["s"][rows], F0.fp, 0); res["pixels_equal_A31_path"] = bool(np.array_equal(fr[idx], u8c.numpy()))
    res["identical_arm_pixels"] = bool(torch.equal(batches(F0, s, [0])[0][2], u8c)); u8, mk = u8c.to(device), mk_c.to(device)
    m = build(ctx, "E", 0, 0, device).eval(); res["tokens"] = token_checks(m, u8, mk); del m; torch.cuda.empty_cache()
    res["parity"] = parity_checks(ctx, u8, mk, device); res["formula"] = formula_check(device); res["init"] = init_checks(ctx, device)
    res["base_reproduced"] = t40.init_base(ctx, s); res["gradients"] = gradient_checks(ctx, F0, device); res["training"] = training_checks(ctx, F0, device, workers)
    spu = max(res["training"]["timing"]["seconds_per_update_gpu_30"], res["training"]["timing"]["seconds_per_update_wall_30_incl_build_and_loader"]); tm_ = readout_timing(ctx, device, workers)
    nw = sum(len(t31.pool_W(t31.load_pool(ctx.a31, f))["seq"]) for f in FOLDS); spent = t31.vsh.ledger_hours(ctx.ledger, "gpu") + (time.time() - t_start) / 3600
    train_h = 4 * U * spu * 1.15 / 3600; read_h = nw * (2 * tm_["gpu_ms_per_window_one_model"] + tm_["decode_ms_per_window"]) / 1e3 * 1.15 / 3600; proj = spent + train_h + read_h + 0.75
    res["projection"] = {"spent_gpu_h_incl_this_check": spent, "train_4x900_h": train_h, "readout_h": read_h, "reserve_h": 0.75, "source0_total_h": proj, "source2_if_triggered_h": train_h + read_h,
                         "cap_h": ctx.v["budget"]["gpu_hours_max"], "windows": nw, "seconds_per_update_used": spu, **tm_}
    res["disk"] = {"free_gb": shutil.disk_usage(ctx.out).free / 1e9, "sources_dir_gb": sum(p.stat().st_size for p in ctx.src.iterdir()) / 1e9, "per_model_final_mb": 4 * jl.TRAINABLE / 1e6}
    res["loaded_code_sha256"] = t38.loaded_code(); res["ok"] = check_ok(res); res["finished"] = now(); publish(ctx, "checks.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("loaded_code_sha256", "groups", "upstream")}, indent=1, default=str)[:12000], flush=True)
    if not res["ok"]:
        raise SystemExit("a check failed")
    return f"{spu:.2f} s per update; source-0 projection {proj:.2f} GPU h"


def check_ok(r: dict) -> bool:
    T, Pa, Fo, I, Gr, Tr, Pr = r["tokens"], r["parity"], r["formula"], r["init"], r["gradients"], r["training"], r["projection"]; lim = lambda d: d["loss_max_abs"] <= LIMITS["loss"] and d["params_max_abs"] <= LIMITS["params"]  # noqa: E731
    up = r["upstream"]
    return bool(up["clean"] and up["commit"] == jl.COMMIT and up["all_equal"] and up["src_path"] == ["third_party/EchoJEPA/src"] and up["pydeps"]["in_round_dir"] and all(r["encoder_files"].values())
                and not any(r["isolation"].values()) and all(g["grid_equals_A07_windows"] and g["draws_valid_900"] and g["draw_repeat_equal"] and g["eligible_streams"] > 0 for g in r["groups"].values())
                and not r["numerics"]["tf32_matmul"] and not r["numerics"]["tf32_cudnn"] and r["numerics"]["cudnn_deterministic"] and not r["numerics"]["cudnn_benchmark"] and r["numerics"]["deterministic_algorithms"] and not r["numerics"]["deterministic_warn_only"]
                and r["pixels_equal_A31_path"] and r["identical_arm_pixels"]
                and T["conv_thw"] == [8, 14, 14] and T["patch_tokens_equal_conv_time_major_loop"] == 0 and T["rope_positions_time_major"] and T["readout_vs_loop_max_abs"] <= 1e-6 and T["marker_16_rejected"]
                and T["input_normalisation_max_abs"] <= 1e-6 and T["input_shape"] == [1, 3, 16, 224, 224]
                and all(Pa[a]["bf16"]["tokens_max_abs"] <= TOL["parity_bf16"] and Pa[a]["bf16"]["head_argmax_equal"] and Pa[a]["fp32"]["tokens_max_abs"] <= TOL["parity_fp32"] and Pa[a]["fp32"]["head_argmax_equal"]
                        and Pa[a]["bf16"]["token_shape"][1:] == [jl.N_TOKENS, jl.DIM] for a in ARMS)
                and Fo["max_abs"] <= TOL["formula_rel"] * max(1.0, Fo["out_max"]) and Fo["zero_B_equals_base_exactly"] and Fo["base_frozen"] and Fo["scale"] == 2.0
                and I["trainable_identical_E_G"] and I["fold_inits_differ"] and I["B_zero_and_head_bias_zero"] and I["trainable_count_requires_grad"] == jl.TRAINABLE == I["trainable_count_list"] == 796677
                and not I["base_requires_grad_any"] and I["bases_differ"] and I["lora_wrappers"] == 24 and I["global_rng_untouched_by_init"]
                and all(all(v[k] for k in ("runs_equal_base", "sequences_equal_base", "rows_equal_A32_base")) for v in r["base_reproduced"].values())
                and Gr["microbatch_vs_batch_fp64"]["max_abs"] <= TOL["microbatch_fp64_rel"] * Gr["microbatch_vs_batch_fp64"]["grad_max"]
                and Gr["checkpointing_on_vs_off_fp32"]["max_abs"] <= TOL["checkpointing_rel"] * Gr["checkpointing_on_vs_off_fp32"]["grad_max"]
                and Gr["real_zero_B_start"]["B_grad_tensors_nonzero"] == 24 and Gr["real_zero_B_start"]["head_grad_nonzero"] and Gr["real_zero_B_start"]["all_finite"]
                and Tr["binding"]["binds_exactly_trainable"] and Tr["binding"]["bound_parameters"] == jl.TRAINABLE
                and Tr["real_update"]["A_moved_max_abs"] > 0 and Tr["real_update"]["B_moved_max_abs"] > 0 and Tr["real_update"]["optimizer_steps"] == [6] and Tr["real_update"]["base_digest_unchanged"]
                and lim(Tr["repeat_6_E"]) and lim(Tr["forced_E_equals_G_3"]) and lim(Tr["resume_3_plus_3_E"])
                and Pr["source0_total_h"] <= Pr["cap_h"] and r["disk"]["free_gb"] > 15)


# ----------------------------------------------------------------------------------------------- readout
def load_model(ctx: Ctx, arm: str, s: int, f: int, device) -> jl.Recogniser:
    rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text()).get(f"{arm}_s{s}_f{f}", {})
    if sha256_file(rd / "final.pt") != man["final_sha256"] or com.get("final_sha256") != man["final_sha256"] or com.get("base_sha256") != man["base_sha256"]:
        raise SystemExit(f"{arm}_s{s}_f{f}: weights not committed or changed")
    m = build(ctx, arm, s, f, device)
    if sources(ctx)["encoders"][arm]["sha256"] != man["base_sha256"]:
        raise SystemExit(f"{arm}: base sha differs from the manifest")
    m.load_trainable(torch.load(rd / "final.pt", map_location="cpu", weights_only=True)); return m.eval()


def heldout_logits(ctx: Ctx, s: int, h: int, models: dict, device, workers: int, acc: dict) -> dict:
    """Online forward of the fit-(1 - h) models on every original request window of held-out fold h (decoded once)."""
    seqs = t31.load_pool(ctx.a31, h); W = t31.pool_W(seqs); fp = t28.frame_paths(seqs); n = len(W["seq"]); Z = {a: np.zeros((n, 5), np.float32) for a in models}
    for c0 in range(0, len(seqs), 150):
        rows = np.flatnonzero((W["seq"] >= c0) & (W["seq"] < c0 + 150)); a = time.time(); fr, idx = t37.chunk_frames(seqs, {"seq": W["seq"][rows]}, W["s"][rows], fp, workers); acc["decode_seconds"] += time.time() - a
        for b in range(0, len(rows), READ_BATCH):
            ks = np.arange(b, min(b + READ_BATCH, len(rows))); r = rows[ks]
            for arm, m in models.items():
                torch.cuda.synchronize(); a = time.time(); Z[arm][r] = forward(m, fr, idx[ks], W["marker"][r], device).cpu().numpy(); torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a
        del fr; print(f"seed {s} held-out fold {h}: {int(rows.max()) + 1}/{n} windows; GPU {acc['gpu_seconds']:.0f} s", flush=True)
    if not all(np.isfinite(z).all() for z in Z.values()):
        raise SystemExit("a held-out logit is not finite")
    return Z


def readout(ctx: Ctx, s: int, device, workers: int) -> str:
    name = f"readout_s{s}.json"
    if (ctx.out / name).exists():
        raise SystemExit("this seed's readout exists; it runs once")
    X = t40.full_q(ctx, s); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"seed": s, "cells": {}}; acc = {m_: [] for m_ in ("Base", *ARMS)}; ys, bs = [], []
    cov = {a: [] for a in ARMS}; s1recs = {a: [] for a in ARMS}; old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]; tim = {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    a40 = json.loads((ROOT / "docs/results/2026-10-27_masked_feature_recognition/readout_s0.json").read_text())["cells"] if s == 0 else None; cache = {}
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); off = X["offset"][h]; P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        Z = heldout_logits(ctx, s, h, {a: load_model(ctx, a, s, f, device) for a in ARMS}, device, workers, tim); torch.cuda.empty_cache(); cell = {}
        for arm in ARMS:
            z = X["q"].clone(); z[off:off + len(Z[arm])] = torch.from_numpy(Z[arm]); t = tables(z, D, off, a8.cfg, P, ev); p = t.pop("pred"); acc[arm].append(p); cell[arm] = t
            cell[arm]["coverage"], sets = t37.coverage(D, D["pk"]["labels"], D["pk"]["base"], p); cov[arm].append(sets); cell[arm]["boundaries"] = t38.boundaries(z, D, off)
            ex, recs = a5c_extras(D, p); cell[arm].update(ex); s1recs[arm].append(recs); cache[f"logits_{arm}_heldout_f{h}"] = Z[arm]; cache[f"runs_{arm}_heldout_f{h}"] = p.astype(np.int8)
            cell[arm]["window_argmax_counts"] = {str(k): int(v) for k, v in sorted(Counter(Z[arm].argmax(1).tolist()).items())}
        t = tables(X["q"], D, off, a8.cfg, P, ev); p = t.pop("pred"); acc["Base"].append(p); cell["Base"] = t; cell["Base"]["boundaries"] = t38.boundaries(X["q"], D, off); cell["Base"].update(a5c_extras(D, p)[0])
        s1 = np.flatnonzero(D["M"]["S1"]); cell["S1_denominator"] = {"runs": int(len(s1)), **d32.concentration([(D, int(i)) for i in s1], "true_src", None)}
        ob = old[f"s{s}_heldout_f{h}"]["A08 Base"]; cell["Base_reproduced"] = {"A32_row": cell["Base"]["sequence_row"] == ob["sequence_row"] and cell["Base"]["counts"] == ob["counts"]}
        if a40 is not None:
            js = lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731
            cell["Base_reproduced"]["A40_Base_cell"] = all(js(cell["Base"][k]) == a40[f"heldout_f{h}"]["Base"][k] for k in a40[f"heldout_f{h}"]["Base"] if k in cell["Base"])
        row = lambda m_: cell[m_]["sequence_row"]  # noqa: E731
        cell["comparisons"] = {f"{a} - {bb}": {"net_fixed": cell[a]["runs"]["all"]["net_fixed"] - (cell[bb]["runs"]["all"]["net_fixed"] if bb != "Base" else 0), "S1_fixed": cell[a]["S1_fixed"]["fixed"] - cell[bb]["S1_fixed"]["fixed"],
                                               "correct_A5C_kept": cell[a]["correct_A5C_kept"]["kept"] - cell[bb]["correct_A5C_kept"]["kept"], "correct_A4C_kept": cell[a]["correct_A4C_kept"]["kept"] - cell[bb]["correct_A4C_kept"]["kept"],
                                               **{k: row(a)[k] - row(bb)[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")},
                                               **{k: cell[a]["counts"][k] - cell[bb]["counts"][k] for k in ("fp", "fp_short", "fn", "fn_short")}} for a, bb in (("E", "G"), ("E", "Base"), ("G", "Base"))}
        cell["E_vs_G_runs_differ"] = int((acc["E"][-1] != acc["G"][-1]).sum())
        mans = {arm: json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for arm in ARMS}
        cell["fit_only"] = {arm: {k: mans[arm][k] for k in ("fit_before", "fit_after", "lora", "loss", "final_sha256", "base_sha256")} for arm in ARMS}
        rec["cells"][f"heldout_f{h}"] = cell; ys.append(D["pk"]["labels"]); bs.append(D["pk"]["base"])
    y, b = np.concatenate(ys), np.concatenate(bs); wr = b != y; rec["oof"] = {}
    for m_ in ARMS:
        p = np.concatenate(acc[m_]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); un = {k: set().union(*[c[k] for c in cov[m_]]) for k in ("fixed", "broken")}
        rec["oof"][m_] = {"runs": int(len(y)), "fixed": fx, "broken": br, "net_fixed": fx - br, **{f"{k}_recordings": len(v) for k, v in un.items()}, **{f"{k}_sessions": len({session_of(x) for x in v}) for k, v in un.items()},
                          "S1_fixed_true_class_recordings": len(set().union(*s1recs[m_]))}
    rec["oof"]["E_vs_G_runs_differ"] = int(sum(c["E_vs_G_runs_differ"] for c in rec["cells"].values()))
    rec["seed2_rule"], rec["values"], rec["summary"] = rule_of(rec, s1recs)
    cp = ctx.out / f"readout_cache_s{s}.npz"; np.savez_compressed(cp, **cache); rec["private_cache"] = {"path": str(cp.relative_to(ROOT)), "sha256": sha256_file(cp), "contents": "per-window held-out logits and per-run labels of E / G (server only; no pixels / tokens / ids)"}
    nw = sum(len(t31.pool_W(t31.load_pool(ctx.a31, f_))["seq"]) for f_ in FOLDS); pr = json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text())
    rec["deployment_cost"] = {"forwards": "Base's one A08 224 forward per window plus one forward of the new recognition branch (ViT-L, 16 x 224, BF16 autocast)", "parameters": f"A08 34.6M (Base) + ViT-L {sources(ctx)['encoders']['E']['parameters'] / 1e6:.1f}M frozen + LoRA / head {jl.TRAINABLE / 1e6:.3f}M",
                              "gpu_ms_per_window_new_branch": 1e3 * tim["gpu_seconds"] / (2 * nw), "cpu_decode_ms_per_window_shared": 1e3 * tim["decode_seconds"] / nw, "gpu_ms_per_window_A31_base_pass": 1e3 * pr["gpu_seconds"] / sum(pr["rows"].values()),
                              "readout_gpu_seconds": tim["gpu_seconds"], "readout_decode_seconds": tim["decode_seconds"], "note": "two models (G, E) per held-out window share one decode; batch 32"}
    rec["written"] = now(); publish(ctx, name, rec)
    print(json.dumps({"summary": rec["summary"], "oof": rec["oof"], "rule": rec["seed2_rule"], "values": rec["values"], "Base_reproduced": {k: v["Base_reproduced"] for k, v in rec["cells"].items()}}, indent=1)); return f"rule met: {rec['seed2_rule']['met']}"


def rule_of(rec: dict, s1recs: dict) -> tuple[dict, dict, dict]:
    C = lambda h: rec["cells"][f"heldout_f{h}"]  # noqa: E731
    tot = lambda m_, k, sub: sum(C(h)[m_][k][sub] for h in FOLDS)  # noqa: E731
    d_ = lambda a, bb, h, k: C(h)[a]["sequence_row"][k] - C(h)[bb]["sequence_row"][k]  # noqa: E731
    net = lambda a, h: C(h)[a]["runs"]["all"]["net_fixed"]  # noqa: E731
    n1, n5, n4 = tot("E", "S1_fixed", "runs"), tot("E", "correct_A5C_kept", "runs"), tot("E", "correct_A4C_kept", "runs"); s1E, s1G = tot("E", "S1_fixed", "fixed"), tot("G", "S1_fixed", "fixed")
    k5E, k5G, k4E, k4G = tot("E", "correct_A5C_kept", "kept"), tot("G", "correct_A5C_kept", "kept"), tot("E", "correct_A4C_kept", "kept"), tot("G", "correct_A4C_kept", "kept")
    recE = [len(r) for r in s1recs["E"]]; recE_all = len(set().union(*s1recs["E"]))
    rule = {"target_each_fold_E_minus_G_S1_ge_1": all(C(h)["E"]["S1_fixed"]["fixed"] - C(h)["G"]["S1_fixed"]["fixed"] >= 1 for h in FOLDS), "target_sum_E_minus_G_S1_ge_5": s1E - s1G >= 5,
            "S1_E": f"{s1E}/{n1}", "S1_G": f"{s1G}/{n1}", "S1_needed": ceil_pct(10, n1), "signal_S1_runs_ok": s1E >= ceil_pct(10, n1),
            "S1_E_true_class_recordings_by_fold": recE, "S1_E_true_class_recordings_total": recE_all, "signal_recordings_ok": recE_all >= 3 and all(r >= 1 for r in recE),
            "correct_A5C_E": f"{k5E}/{n5}", "correct_A5C_G": f"{k5G}/{n5}", "A5C_needed": ceil_pct(98, n5), "retain_A5C_ok": k5E >= ceil_pct(98, n5) and k5E >= k5G - 3,
            "correct_A4C_E": f"{k4E}/{n4}", "correct_A4C_G": f"{k4G}/{n4}", "A4C_needed": ceil_pct(98, n4), "retain_A4C_ok": k4E >= ceil_pct(98, n4) and k4E >= k4G - 3,
            "net_E_minus_Base_each_fold_ge_0": all(net("E", h) >= 0 for h in FOLDS), "net_E_minus_G_each_fold_ge_0": all(net("E", h) - net("G", h) >= 0 for h in FOLDS),
            "dF_E_minus_Base_each_fold_ge_0": all(d_("E", "Base", h, "segment_f1_50_multi") >= 0 for h in FOLDS), "dF_E_minus_G_each_fold_ge_0": all(d_("E", "G", h, "segment_f1_50_multi") >= 0 for h in FOLDS),
            "native_E_minus_Base_each_fold_ge_-0.003": all(d_("E", "Base", h, "native_macro_f1") >= -0.003 for h in FOLDS), "boundary_recall_E_minus_Base_each_fold_ge_-0.005": all(d_("E", "Base", h, "boundary_recall_multi") >= -0.005 for h in FOLDS),
            "missed_short_E_le_Base_plus_3": tot("E", "counts", "fn_short") <= tot("Base", "counts", "fn_short") + 3}
    rule["met"] = all(v for v in rule.values() if isinstance(v, bool))
    values = {"E_minus_G_S1": [C(h)["E"]["S1_fixed"]["fixed"] - C(h)["G"]["S1_fixed"]["fixed"] for h in FOLDS], "E_minus_G_net": [net("E", h) - net("G", h) for h in FOLDS], "E_minus_Base_net": [net("E", h) for h in FOLDS],
              "G_minus_Base_net": [net("G", h) for h in FOLDS], "E_minus_G_dF": [d_("E", "G", h, "segment_f1_50_multi") for h in FOLDS], "E_minus_Base_dF": [d_("E", "Base", h, "segment_f1_50_multi") for h in FOLDS],
              "G_minus_Base_dF": [d_("G", "Base", h, "segment_f1_50_multi") for h in FOLDS], "E_minus_Base_native": [d_("E", "Base", h, "native_macro_f1") for h in FOLDS],
              "E_minus_Base_boundary_recall": [d_("E", "Base", h, "boundary_recall_multi") for h in FOLDS], "missed_short": {m_: tot(m_, "counts", "fn_short") for m_ in ("Base", *ARMS)}}
    summary = {f"heldout_f{h}": {m_: {"net": net(m_, h) if m_ != "Base" else 0, "F1": C(h)[m_]["sequence_row"]["segment_f1_50_multi"], "S1": C(h)[m_]["S1_fixed"]["fixed"], "A5C_kept": C(h)[m_]["correct_A5C_kept"]["kept"],
                                       "A4C_kept": C(h)[m_]["correct_A4C_kept"]["kept"]} for m_ in ("Base", *ARMS)} for h in FOLDS}
    return rule, values, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("extract", "check", "train", "freeze", "readout"), required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--commit", default=""); ap.add_argument("--manifest-sha", default="")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "extract":
        return staged(ctx, "extract strict target_encoder E / G (CPU)", "cpu", lambda: extract(ctx))
    if a.stage == "freeze":
        return staged(ctx, f"freeze s{a.seed} (CPU)", "cpu", lambda: freeze(ctx, a.seed, a.commit, a.manifest_sha))
    t31.vsh.determinism(); torch.use_deterministic_algorithms(True, warn_only=False); device = torch.device("cuda")   # deterministic SDPA backward (check attempt 1)
    if a.stage == "train":
        return train(ctx, a.seed, device, a.workers)
    fn = {"check": lambda: check(ctx, device, a.workers), "readout": lambda: readout(ctx, a.seed, device, a.workers)}[a.stage]
    staged(ctx, f"{a.stage}{'' if a.stage == 'check' else ' s' + str(a.seed)} (GPU job wall time)", "gpu", fn)


if __name__ == "__main__":
    main()
