#!/usr/bin/env python
"""Train one arm (A: E1 features; B: L1_s{seed} features) and seed of the L1 + ASRF round (plan Task 2;
configs/experiments/l1_asrf_integration_v1.yaml).

The official ASRF (pinned commit; model, losses and post-processing imported unchanged through ASRFHead) on frozen
512-d features. Recipe: each update reads 4 constructed training streams and 4 native windows (TrainData.epoch_batches;
1,500 streams and 1,500 windows of at most 256 samples per epoch, 375 updates); 50 epochs, 18,750 updates; AdamW
lr 3e-4, weight decay 1e-4, clip 1.0; 2 warm-up epochs then cosine to 0; fp32, TF32 off; no early stopping. Class
weights and pos_weight follow the official rules on the seed's epoch-0 batch labels, which are identical for A and B.
One saved initial state per seed is loaded by both arms; the batch identities of every update are checked against the
seed's saved plan.

After every epoch the model decodes the calibration half only (official refinement_with_boundary, boundary_th 0.5). The
selection half is never loaded here. The checkpoint rule: among epochs with calibration native macro-F1 >= 0.980 and
multi boundary recall >= 0.95, the highest segment F1@50, ties to fewer native internal cuts, then the earlier epoch.
Without a feasible epoch the final epoch is kept as a diagnostic. Resume state after every epoch.

--smoke: same-state reproducibility, resume (3 + 3 updates against 6), padding labels without effect on the loss,
finite non-zero gradients, the A/B plan and initial state, and timing / memory for the budget.

  python scripts/train_l1_asrf.py --config configs/experiments/l1_asrf_integration_v1.yaml --arm B --seed 0 [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_to_dict  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore, TrainData, collate, eval_set, stream_seq  # noqa: E402
from echo_routing.temporal.dev_heads import ASRFHead, asrf_weights  # noqa: E402
from echo_routing.temporal.dev_run import check_index, digest, load_json  # noqa: E402
from echo_routing.temporal.dev_train import LOGGED_METRICS, lr_factor, seed_everything  # noqa: E402

NS = "l1_asrf_integration_v1"
ARMS = ("A", "B")
CODE = ("scripts/train_l1_asrf.py", "echo_routing/temporal/dev_heads.py", "echo_routing/temporal/dev_data.py",
        "echo_routing/temporal/dev_train.py", "echo_routing/evaluate/dev_metrics.py", "echo_routing/compose/stream_assembler.py")


# ----------------------------------------------------------------------------------------------- data
def feature_manifest(la: dict, arm: str, seed: int, root: Path) -> Path:
    if arm == "A":
        return root / la["inputs"]["e1_feature_manifest"]["path"]
    if arm == "B":
        return root / "runs" / NS / "features" / f"feature_manifest_L1_s{seed}.json"
    raise ValueError(arm)


def selection_forbidden(*_a, **_k):
    raise RuntimeError("the selection half is not available to the training entry of this round")


class Data:
    """Protocol, feature store, the 1,500-stream training bank (digest checked) and the calibration half only."""

    def __init__(self, cfg: dict, la: dict, paths, fman_path: Path) -> None:
        root = paths.repo_root; self.proto, self.proto_sha = load_json(root / la["inputs"]["protocol"]["path"])
        if self.proto_sha != la["inputs"]["protocol"]["sha256"]:
            raise SystemExit("protocol sha256 differs from the config")
        self.fman, self.fman_sha = load_json(fman_path); key = self.fman["encoder"]["cache_key"]
        if int(self.fman["feature_dim"]) != int(la["asrf"]["model"]["in_channel"]):
            raise SystemExit("feature dimension differs from the ASRF input")
        self.store = FeatureStore(paths.cache_root, key); idx = read_index(variant_dir(paths.cache_root, key, "orig")); check_index(idx, self.proto)
        self.index_sha = {v: sha256_file(variant_dir(paths.cache_root, key, v) / "index.csv") for v in ("orig", "gamma090")}
        self.labels = dict(zip(idx["video_id"], idx["label_index"].astype(int)))
        bank = multi_bank(idx, "train", cfg); bank_digest = digest([recipe_to_dict(r) for r in bank.recipes])
        if bank_digest != self.proto["train"]["multi_bank"]["digest"]:
            raise SystemExit(f"training bank digest {bank_digest[:16]} != protocol")
        self.bank_digest = bank_digest; self.cache_key = key
        self.train = TrainData(tuple(stream_seq(render(r, self.store.get)) for r in bank.recipes),
                               tuple((v, self.labels[v]) for v in self.proto["train"]["clips"]), self.store)
        self.calibration = eval_set(self.proto, "calibration", self.labels, self.store)
        self.selection = selection_forbidden


def batches(data: Data, tr: dict, seed: int, epoch: int) -> list:
    return data.train.epoch_batches(seed, epoch, int(tr["native_per_epoch"]), int(tr["native_window"]), int(tr["streams_per_batch"]),
                                    int(tr["natives_per_batch"]))


def batch_digest(seqs: list) -> str:
    """Identity of one update: sequence ids, lengths, labels and joins (never the features)."""
    h = hashlib.sha256()
    for s in seqs:
        h.update(f"{s.seq_id}|{s.labels.size}|".encode()); h.update(np.asarray(s.labels, np.int64).tobytes()); h.update(np.asarray(s.joins, np.int64).tobytes()); h.update(b";")
    return h.hexdigest()[:16]


def make_plan(data: Data, tr: dict, seed: int) -> dict:
    epochs = []
    for e in range(int(tr["epochs"])):
        b = batches(data, tr, seed, e)
        if len(b) != int(tr["updates_per_epoch"]):
            raise SystemExit(f"epoch {e}: {len(b)} updates, expected {tr['updates_per_epoch']}")
        epochs.append([batch_digest(x) for x in b])
    first = [s for x in batches(data, tr, seed, 0) for s in x]
    cw, pw = asrf_weights(first, num_classes=5)
    return {"seed": seed, "epochs": epochs, "class_weight": cw.tolist(), "pos_weight": float(pw)}


def save_or_check_plan(plan: dict, root: Path) -> tuple[Path, str]:
    path = root / "runs" / NS / "plans" / f"plan_s{plan['seed']}.json"
    if path.is_file():
        old = json.loads(path.read_text())
        if old != json.loads(json.dumps(plan)):
            raise SystemExit(f"{path}: this arm's batches or weights differ from the saved plan of the seed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(plan))
    return path, sha256_file(path)


# ----------------------------------------------------------------------------------------------- model
ASRF_BUILD = {"in_channel": 512, "n_classes": 5, "n_features": 64, "n_layers": 10, "n_stages": 4, "n_stages_asb": 4, "n_stages_brb": 4}   # ASRFHead.build


def make_head(la: dict, paths, plan: dict, device) -> ASRFHead:
    a = la["asrf"]; root = paths.repo_root
    if {k: int(v) for k, v in a["model"].items()} != ASRF_BUILD:
        raise SystemExit(f"the config's ASRF architecture {a['model']} differs from what ASRFHead.build constructs {ASRF_BUILD}")
    head = ASRFHead(root / a["repo"], int(a["model"]["in_channel"]), torch.tensor(plan["class_weight"], dtype=torch.float32),
                    torch.tensor(plan["pos_weight"]), num_classes=int(a["model"]["n_classes"]), lambda_b=float(a["lambda_b"]),
                    boundary_th=float(a["boundary_th"]), device=device)
    if head.commit != a["commit"]:
        raise SystemExit(f"ASRF commit {head.commit} != {a['commit']}")
    return head


def save_or_check_init(head: ASRFHead, seed: int, root: Path) -> tuple[Path, str]:
    path = root / "runs" / NS / "inits" / f"asrf_init_s{seed}.pt"
    seed_everything(seed); state = {k: v.detach().cpu().clone() for k, v in head.build().state_dict().items()}
    if path.is_file():
        old = torch.load(path, map_location="cpu", weights_only=True)
        if set(old) != set(state) or any(not torch.equal(old[k], state[k]) for k in state):
            raise SystemExit(f"{path} exists with other content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True); torch.save(state, path)
    return path, sha256_file(path)


def new_model(head: ASRFHead, init_path: Path, device):
    model = head.build().to(device); model.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True)); return model


def make_optimiser(model, tr: dict):
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    total = int(tr["epochs"]) * int(tr["updates_per_epoch"]); warm = int(tr["warmup_epochs"]) * int(tr["updates_per_epoch"])
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warm, total))


def update(head: ASRFHead, model, seqs: list, opt, sched, clip: float, device) -> dict:
    model.train(); batch = collate(seqs, device); loss, parts = head.loss(model, batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {parts}")
    opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    opt.step(); sched.step()
    return {"total": float(loss.detach()), **parts, "grad_norm": gn}


def evaluate_calibration(head: ASRFHead, model, data: Data, cfg: dict, device) -> tuple[dict, dict]:
    decoded = head.decode(model, data.calibration.sequences(), device)
    m, _ = evaluate_predictions({k: v["labels"] for k, v in decoded.items()}, data.calibration, float(cfg["protocol"]["hz"]),
                                float(cfg["selection"]["boundary_tolerance_s"]))
    return m, decoded


def choose(records: list[dict], rule: dict) -> dict:
    """The round's checkpoint rule on the calibration records."""
    ok = [r for r in records if r["native_macro_f1"] >= float(rule["native_min"]) and (r["boundary_recall_multi"] or 0) >= float(rule["recall_min"])]
    if not ok:
        last = records[-1]
        return {"epoch": last["epoch"], "status": "infeasible (final epoch shown as diagnostic)", "n_feasible": 0}
    best = min(ok, key=lambda r: (-r["segment_f1_50_multi"], r["native_internal_cuts_per_clip"], r["epoch"]))
    return {"epoch": best["epoch"], "status": "feasible", "n_feasible": len(ok)}


def identity(cfg: dict, arm: str, seed: int, data: Data, plan_sha: str, init_sha: str, root: Path) -> str:
    payload = {"experiment": cfg.get("experiment"), "arm": arm, "seed": int(seed), "config": cfg["l1_asrf"], "protocol_sha256": data.proto_sha,
               "feature_manifest_sha256": data.fman_sha, "cache_key": data.cache_key, "index_sha256": data.index_sha, "bank_digest": data.bank_digest,
               "plan_sha256": plan_sha, "init_sha256": init_sha, "code": {c: sha256_file(root / c) for c in CODE}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def save_resume(path: Path, model, opt, sched, epoch: int, ident: str, records: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"identity": ident, "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(), "records": records}, tmp)
    tmp.replace(path)


def load_resume(path: Path, model, opt, sched, ident: str) -> tuple[int, list]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["identity"] != ident:
        raise SystemExit(f"{path} belongs to another run identity")
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state_all(st["rng_cuda"])
    return int(st["epoch"]) + 1, list(st["records"])


def finished_run(manifest: Path, ident: str):
    """True: a finished run of this identity with intact checkpoints; False: a finished run of another identity or with
    changed checkpoints; None: no finished run."""
    if not manifest.is_file():
        return None
    man = json.loads(manifest.read_text())
    ok = man.get("identity") == ident and all(Path(man[f"{k}_checkpoint"]).is_file() and sha256_file(Path(man[f"{k}_checkpoint"])) == man[f"{k}_sha256"]
                                              for k in ("selected", "final"))
    return bool(ok)


# ----------------------------------------------------------------------------------------------- smoke
def smoke(cfg: dict, la: dict, head: ASRFHead, data: Data, plan: dict, init_path: Path, device) -> dict:
    tr = la["training"]; clip = float(tr["grad_clip"]); b0 = batches(data, tr, plan["seed"], 0)[:6]; rep = {}
    if [batch_digest(x) for x in b0] != plan["epochs"][0][:6]:
        raise SystemExit("smoke batches differ from the plan")

    def run(k, state=None):   # ASRF has dropout: the RNG is seeded before a fresh run and restored on resume, as in training
        model = new_model(head, init_path, device); opt, sched = make_optimiser(model, tr)
        if state:
            model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
            torch.set_rng_state(state["rng_cpu"]); torch.cuda.set_rng_state_all(state["rng_cuda"])
        else:
            seed_everything(plan["seed"])
        losses = [update(head, model, b, opt, sched, clip, device)["total"] for b in k]
        return model, opt, sched, losses
    a = run(b0); b = run(b0)
    pa, pb = a[0].state_dict(), b[0].state_dict()
    rep["reproducibility"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(a[3], b[3])), "params_max_abs": max(float((pa[k].float() - pb[k].float()).abs().max()) for k in pa)}
    c = run(b0[:3]); st = {"model": c[0].state_dict(), "opt": c[1].state_dict(), "sched": c[2].state_dict(),
                           "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}
    d = run(b0[3:], state=st); pd_ = d[0].state_dict()
    rep["resume_params_max_abs"] = max(float((pa[k].float() - pd_[k].float()).abs().max()) for k in pa)
    model = new_model(head, init_path, device); model.train(); batch = collate(b0[0], device)   # same dropout masks for both losses
    torch.manual_seed(1); l1, _ = head.loss(model, batch); alt = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = -7; alt["labels"] = lab
    torch.manual_seed(1); l2, _ = head.loss(model, alt)
    rep["padding_label_loss_diff"] = abs(float(l1) - float(l2)); rep["padded_positions"] = int((batch["valid"] == 0).sum())
    model.zero_grad(); l1.backward()
    grads = {n: (float(p.grad.abs().max()) if p.grad is not None else None) for n, p in model.named_parameters()}
    rep["zero_or_missing_gradients"] = [n for n, g in grads.items() if g is None or g == 0.0 or not np.isfinite(g)]
    rep["parameters"] = len(grads); rep["loss_first"] = float(l1)
    try:
        data.selection(); rep["selection_guard"] = False
    except RuntimeError:
        rep["selection_guard"] = True
    torch.cuda.reset_peak_memory_stats(device); t0 = time.time(); model2 = new_model(head, init_path, device); opt, sched = make_optimiser(model2, tr)
    for x in batches(data, tr, plan["seed"], 0)[:50]:
        update(head, model2, x, opt, sched, clip, device)
    rep["seconds_per_update"] = (time.time() - t0) / 50; t1 = time.time(); evaluate_calibration(head, model2, data, cfg, device)
    rep["seconds_per_calibration_eval"] = time.time() - t1; rep["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    rep["estimated_minutes_per_run"] = (rep["seconds_per_update"] * int(tr["epochs"]) * int(tr["updates_per_epoch"]) + rep["seconds_per_calibration_eval"] * int(tr["epochs"])) / 60
    rep["ok"] = bool(rep["reproducibility"]["loss_max_abs"] <= 1e-6 and rep["reproducibility"]["params_max_abs"] <= 1e-6 and rep["resume_params_max_abs"] <= 1e-6
                     and rep["padding_label_loss_diff"] == 0.0 and rep["padded_positions"] > 0 and not rep["zero_or_missing_gradients"] and rep["selection_guard"])
    return rep


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--arm", choices=ARMS, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); la = cfg["l1_asrf"]; paths = load_paths(); root = paths.repo_root; tr = la["training"]
    if a.seed not in [int(s) for s in la["seeds"]]:
        raise SystemExit(f"seed {a.seed} is not one of the round's seeds")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    data = Data(cfg, la, paths, feature_manifest(la, a.arm, a.seed, root))
    plan = make_plan(data, tr, a.seed); plan_path, plan_sha = save_or_check_plan(plan, root)
    head = make_head(la, paths, plan, device); init_path, init_sha = save_or_check_init(head, a.seed, root)
    ident = identity(cfg, a.arm, a.seed, data, plan_sha, init_sha, root); key = f"{a.arm}_s{a.seed}"
    if a.smoke:
        rep = smoke(cfg, la, head, data, plan, init_path, device)
        out = root / "runs" / NS / "smoke" / f"{key}.json"; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**rep, "identity": ident, "plan_sha256": plan_sha, "init_sha256": init_sha, "cache_key": data.cache_key}, indent=1))
        print(json.dumps(rep, indent=1))
        if not rep["ok"]:
            raise SystemExit("smoke checks failed")
        return
    run_dir = root / "runs" / NS / "train" / key; ckpt = root / "checkpoints" / NS / key
    done = finished_run(run_dir / "run_manifest.json", ident)
    if done is True:
        print(f"{key}: finished with this identity, nothing to do", flush=True); return
    if done is False:
        raise SystemExit(f"{key}: a finished run of another identity exists in {run_dir}; move it before training again")
    run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    model = new_model(head, init_path, device); opt, sched = make_optimiser(model, tr); start, records = 0, []
    seed_everything(a.seed)   # ASRF's dropout draws from this state; A and B of a seed start from the same one
    if (ckpt / "resume.pt").is_file():
        start, records = load_resume(ckpt / "resume.pt", model, opt, sched, ident); print(f"{key}: resumed after epoch {start - 1}", flush=True)
    clip = float(tr["grad_clip"]); log = run_dir / "train_log.jsonl"
    if start == 0:
        log.write_text("")
    for epoch in range(start, int(tr["epochs"])):
        torch.cuda.reset_peak_memory_stats(device); t0 = time.time(); sums = defaultdict(float)
        bs = batches(data, tr, a.seed, epoch)
        if [batch_digest(x) for x in bs] != plan["epochs"][epoch]:
            raise SystemExit(f"epoch {epoch}: batches differ from the saved plan")
        for x in bs:
            for k, v in update(head, model, x, opt, sched, clip, device).items():
                sums[k] += v
        t_train = time.time() - t0; m, _ = evaluate_calibration(head, model, data, cfg, device)
        rec = {"epoch": epoch, "updates": (epoch + 1) * len(bs), "lr": float(sched.get_last_lr()[0]), "loss": {k: v / len(bs) for k, v in sums.items()},
               "seconds_train": t_train, "seconds_eval": time.time() - t0 - t_train, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
               **{k: m[k] for k in LOGGED_METRICS}}
        records.append(rec); torch.save(model.state_dict(), ckpt / f"epoch_{epoch:03d}.pt")
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{key} epoch {epoch} loss {rec['loss']['total']:.4f} | cal native {rec['native_macro_f1']:.4f} segF1 {rec['segment_f1_50_multi']:.4f} "
              f"rec {rec['boundary_recall_multi']:.3f} cuts {rec['native_internal_cuts_per_clip']:.3f} ({t_train:.0f}s+{rec['seconds_eval']:.0f}s)", flush=True)
        save_resume(ckpt / "resume.pt", model, opt, sched, epoch, ident, records)
    if len(records) != int(tr["epochs"]):
        raise SystemExit(f"{key}: {len(records)} epochs recorded")
    ch = choose(records, la["checkpoint_rule"]); last = records[-1]["epoch"]
    shutil.copyfile(ckpt / f"epoch_{ch['epoch']:03d}.pt", ckpt / "selected.pt"); shutil.copyfile(ckpt / f"epoch_{last:03d}.pt", ckpt / "final.pt")
    for p in ckpt.glob("epoch_*.pt"):
        p.unlink()
    (ckpt / "resume.pt").unlink(missing_ok=True)
    man = {"identity": ident, "arm": a.arm, "seed": a.seed, "epochs": len(records), "updates": records[-1]["updates"], "chosen": ch,
           "selected_checkpoint": str(ckpt / "selected.pt"), "selected_sha256": sha256_file(ckpt / "selected.pt"),
           "final_checkpoint": str(ckpt / "final.pt"), "final_sha256": sha256_file(ckpt / "final.pt"), "plan": str(plan_path), "plan_sha256": plan_sha,
           "init": str(init_path), "init_sha256": init_sha, "class_weight": plan["class_weight"], "pos_weight": plan["pos_weight"],
           "feature_manifest_sha256": data.fman_sha, "cache_key": data.cache_key, "index_sha256": data.index_sha, "bank_digest": data.bank_digest,
           "asrf_commit": head.commit, "records": records}
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"run": key, "chosen": ch, "minutes": round(sum(r["seconds_train"] + r["seconds_eval"] for r in records) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
