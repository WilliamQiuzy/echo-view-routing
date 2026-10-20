#!/usr/bin/env python
"""A13 (configs/experiments/video_frozen_switch_v1.yaml): frozen A08 features on A11's sealed manifests, then head-only
training of A11's sustained-switch head (U and H). The whole TargetMViT stays in eval mode without gradients; its
parameter and buffer digests are checked before and after. The head trains on CPU (exactly deterministic).

  --extract S [S ...]   GPU: A08 eval logits and embeddings of every manifest window of the seeds
  --check               CPU: frozen-path digests, head-only gradients, optimiser binding, U/H pairing, masses, resume
  --train K [K ...]     head runs (U_s0, H_s0, ...)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_sustained_switch as tss  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402

ARMS = ("U", "H")
now, write_json, ledger_append, ledger_hours = vsh.now, vsh.write_json, vsh.ledger_append, vsh.ledger_hours


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vf = cfg, config, cfg["video_frozen_switch"]; src = self.vf["source"]
        for p in (src["a11_config"], src["a11_code"]["train"], src["a11_code"]["report"], src["a11_manifests"]["index"]):
            vsh.pinned(p)
        self.a11 = tss.Ctx(load_cfg(SimpleNamespace(config=src["a11_config"]["path"], set=[])), src["a11_config"]["path"])
        self.total, self.warmup = self.a11.total, self.a11.warmup; o = self.vf["outputs"]; self.out, self.results = ROOT / o["root"], ROOT / o["results_dir"]
        self.ledger = ROOT / self.vf["budget"]["ledger"]; self.o = self.a11.o

    def manifest(self, seed: int) -> dict:
        m, sha = self.a11.manifest(seed)
        if sha != self.vf["source"]["a11_manifests"][seed]:
            raise SystemExit(f"seed {seed}: A11's manifest differs from this round's pin")
        return m

    def features(self, seed: int) -> dict:
        idx = json.loads((self.out / "features" / "index.json").read_text())[str(seed)]; p = self.out / "features" / idx["file"]
        if sha256_file(p) != idx["sha256"]:
            raise SystemExit(f"seed {seed}: the feature cache differs from its index")
        with np.load(p) as z:
            return {k: z[k] for k in z.files}


def frozen(ctx: Ctx, seed: int, device):
    state, ssha = ctx.a11.a10.start_state(seed); model = vsh.make_model(state, device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ssha


def path_digest(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():                  # parameters and buffers
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def extract(ctx: Ctx, seeds: list[int], device, workers: int) -> None:
    for s in seeds:
        t0, started, ok, note = time.time(), now(), False, ""
        try:
            m = ctx.manifest(s); model, ssha = frozen(ctx, s, device); d0 = path_digest(model); U = ctx.total
            logits, emb = np.zeros((U, 12, 5), np.float32), np.zeros((U, 12, tm.EMBED_DIM), np.float32)
            with torch.no_grad():
                for b in vsh.loader(m, ctx.a11.frame_paths(), list(range(U)), workers):
                    u = int(b[0]); rvc.check_digests(m, vsh.vso.GROUP_NAME, u, b[4])
                    e = model.embed(tm.to_rgb(b[1].to(device, non_blocking=True)), tm.marker_planes(b[2].to(device))); z = model.classifier(e)
                    logits[u], emb[u] = z.float().cpu().numpy(), e.float().cpu().numpy()
            d1 = path_digest(model)
            if d0 != d1 or model.training or any(p.requires_grad for p in model.parameters()):
                raise SystemExit("the frozen path changed or left eval mode")
            fdir = ctx.out / "features"; fdir.mkdir(parents=True, exist_ok=True); path = fdir / f"train_s{s}.npz"
            np.savez(path, logits=logits, emb=emb, start_sha256=ssha, manifest_sha256=ctx.vf["source"]["a11_manifests"][s], path_digest=d0)
            ip = fdir / "index.json"; idx = json.loads(ip.read_text()) if ip.is_file() else {}
            idx[str(s)] = {"file": path.name, "sha256": sha256_file(path), "start_sha256": ssha, "path_digest_before": d0, "path_digest_after": d1, "eval": True, "time": now()}
            write_json(ip, idx); ok = True; note = f"seed {s}: {U * 12} windows"; print(f"seed {s}: features cached ({time.time() - t0:.0f} s)", flush=True); del model; torch.cuda.empty_cache()
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"extract features s{s} (frozen A08, eval)", "gpu", started, time.time() - t0, ok, note)


def make_opt(head, o: dict) -> torch.optim.AdamW:
    return torch.optim.AdamW(list(head.parameters()), lr=0.0, betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))


def step(head, opt, f: dict, m: dict, arm: str, u: int, o: dict, total: int, warmup: int) -> dict:
    lr = float(o["lr_head"]) * lr_factor(u, warmup, total); opt.param_groups[0]["lr"] = lr; head.train()
    z, e = torch.from_numpy(f["logits"][u]), torch.from_numpy(f["emb"][u]); s = head(tss.central_features(z, e, tss.TRAIN_IDX))
    y = torch.as_tensor(m["c_edge"][u]).float(); w = torch.as_tensor(m[f"w_{arm}"][u]).float(); km = y >= 0
    bce = (w[km] * F.binary_cross_entropy_with_logits(s[km], y[km], reduction="none")).mean() if bool(km.any()) else s.sum() * 0.0
    opt.zero_grad(set_to_none=True); bce.backward(); gn = float(torch.nn.utils.clip_grad_norm_(list(head.parameters()), float(o["grad_clip"]))); opt.step()
    return {"bce": float(bce.detach()), "grad_norm": gn, "lr": lr, "edges": int(km.sum()), "edge_acc": float(((s.detach()[km] > 0).float() == y[km]).float().mean()) if bool(km.any()) else float("nan")}


def identity(ctx: Ctx, arm: str, seed: int, fsha: str) -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(tss.__file__)):
        h.update(Path(p).read_bytes())
    payload = {"sections": {k: ctx.vf[k] for k in ("predecessors", "source", "arms", "model", "data", "training")}, "arm": arm, "seed": int(seed), "features_sha256": fsha,
               "manifest_sha256": ctx.vf["source"]["a11_manifests"][seed], "code": h.hexdigest(), "torch": torch.__version__}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def run_head(ctx: Ctx, arm: str, seed: int, us: range, state: dict | None = None):
    torch.manual_seed(seed); head = tss.build_head(seed); opt = make_opt(head, ctx.o)
    if state is not None:
        head.load_state_dict(state["head"]); opt.load_state_dict(state["opt"])
    f, m = ctx.features(seed), ctx.manifest(seed); recs = [step(head, opt, f, m, arm, u, ctx.o, ctx.total, ctx.warmup) for u in us]
    return head, opt, recs


def train(ctx: Ctx, keys: list[str]) -> None:
    for key in keys:
        arm, s = key.rsplit("_s", 1); s = int(s); t0, started, ok, note = time.time(), now(), False, ""
        try:
            fsha = json.loads((ctx.out / "features" / "index.json").read_text())[str(s)]["sha256"]; ident = identity(ctx, arm, s, fsha)
            head, opt, recs = run_head(ctx, arm, s, range(ctx.total)); rd = ctx.out / "train" / key; rd.mkdir(parents=True, exist_ok=True)
            final = rd / "final.pt"; torch.save(head.state_dict(), final)
            log = [{"updates": k + ctx.warmup, **{x: float(np.nanmean([r[x] for r in recs[k:k + ctx.warmup]])) for x in ("bce", "grad_norm", "edge_acc")}} for k in range(0, ctx.total, ctx.warmup)]
            write_json(rd / "run_manifest.json", {"identity": ident, "arm": arm, "key": key, "seed": s, "features_sha256": fsha, "updates": len(recs), "epochs": log,
                                                  "final_checkpoint": str(final), "final_sha256": sha256_file(final), "head_parameters": int(sum(p.numel() for p in head.parameters()))})
            ok = True; note = f"final bce {log[-1]['bce']:.4f} edge_acc {log[-1]['edge_acc']:.3f}"; print(f"[{key}] {note}", flush=True)
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"train head {key} (CPU)", "cpu", started, time.time() - t0, ok, note)


def check(ctx: Ctx) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"config_sha256": sha256_file(ROOT / ctx.config), "started": started}
    try:
        idx = json.loads((ctx.out / "features" / "index.json").read_text())
        out["frozen_path"] = {s: {"unchanged": v["path_digest_before"] == v["path_digest_after"], "eval": v["eval"]} for s, v in idx.items()}
        a, b = tss.build_head(0), tss.build_head(0); out["init_identical"] = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
        head = tss.build_head(0); opt = make_opt(head, ctx.o); out["optimiser_only_head"] = [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in head.parameters()]
        masses = {}
        for s in idx:
            m = ctx.manifest(int(s)); neg, pos = m["c_edge"] == 0, m["c_edge"] == 1
            masses[s] = {"neg_equal": bool(abs(m["w_U"][neg].sum() - m["w_H"][neg].sum()) <= 1e-6 * m["w_U"][neg].sum()), "pos_equal": bool(np.array_equal(m["w_U"][pos], m["w_H"][pos]))}
        out["masses"] = masses
        model, _ = frozen(ctx, 0, torch.device("cpu")); f = ctx.features(0); m = ctx.manifest(0)
        z, e = torch.from_numpy(f["logits"][0]), torch.from_numpy(f["emb"][0]); h = tss.build_head(0); s_ = h(tss.central_features(z, e, tss.TRAIN_IDX)); s_.sum().backward()
        out["gradients"] = {"head": all(p.grad is not None for p in h.parameters()), "frozen_without_grad": all(p.grad is None and not p.requires_grad for p in model.parameters())}; del model
        full = run_head(ctx, "H", 0, range(ctx.warmup, ctx.warmup + 6)); a3 = run_head(ctx, "H", 0, range(ctx.warmup, ctx.warmup + 3))
        b3 = run_head(ctx, "H", 0, range(ctx.warmup + 3, ctx.warmup + 6), state={"head": a3[0].state_dict(), "opt": a3[1].state_dict()})
        out["resume"] = {"loss_max_abs": max(abs(x["bce"] - y["bce"]) for x, y in zip(full[2][3:], b3[2])), "params_max_abs": vsh.max_diff(full[0], b3[0])}
        out["resume"]["ok"] = out["resume"]["loss_max_abs"] == 0.0 and out["resume"]["params_max_abs"] == 0.0
        out["budget"] = {"gpu_spent": ledger_hours(ctx.ledger, "gpu"), "cap": float(ctx.vf["budget"]["gpu_hours_max"])}
        out["ok"] = bool(all(v["unchanged"] and v["eval"] for v in out["frozen_path"].values()) and out["init_identical"] and out["optimiser_only_head"]
                         and all(v["neg_equal"] and v["pos_equal"] for v in masses.values()) and out["gradients"]["head"] and out["gradients"]["frozen_without_grad"] and out["resume"]["ok"])
        ok = out["ok"]; out["finished"] = now()
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, out)
        print(json.dumps(out, indent=1, default=str), flush=True)
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--extract", nargs="+", type=int); g.add_argument("--check", action="store_true"); g.add_argument("--train", nargs="+")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); vsh.determinism()
    if a.extract:
        return extract(ctx, a.extract, torch.device("cuda"), a.workers)
    if a.check:
        return check(ctx)
    train(ctx, a.train)


if __name__ == "__main__":
    main()
