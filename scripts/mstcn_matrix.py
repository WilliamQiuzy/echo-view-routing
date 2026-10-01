#!/usr/bin/env python
"""MS-TCN training/test permutation matrix on the fixed, correctly ordered image set (tasks 0 and 3 of the
2026-10-01 round, docs/research_plan.md).

Inputs: the ladder banks (recipe seed from the config) over B0 79f4a41af6e3 features in cache_v2 (numeric frame
order). Training condition `true` renders the multi-fragment train bank in acquisition order; `rand` permutes every
training fragment once with TRAIN_PERMUTATION_SEED (fixed per fragment, never reshuffled between epochs). Model seeds
0, 1, 2 control Python, NumPy, PyTorch and the official batch order through the seeded copy
third_party/ms-tcn_seeded (third_party/patches/ms-tcn-seed.patch). The two conditions of one model seed therefore
share initial weights, batch order and budget (the authors' 50 epochs, batch 1, lr 5e-4); only the order inside
training fragments differs.

Isolated names: data  data/mstcn_matrix/<enc>-r<recipe digest>-tr<condition>-ps<permutation seed>-<frame order>
                model third_party/ms-tcn_seeded/models/<data name>-ms<model seed>/split_1 (DONE.json: seeds, digests)
                preds cache_v2/mstcn_matrix/<data name>-ms<model seed>/<test condition>/<stream id>.npz
Test conditions: true order and the ten task-1 random seeds, built with the task-1 functions, so B2/B3 rows computed
here are paired with MS-TCN on identical inputs. Decoding is fixed (argmax of the MS-TCN softmax, as b4); routing
thresholds come from true-order validation of each model.

Subcommands: verify (tiny run: seeds, digests, isolation) | prepare | train | predict | evaluate | queue (all of it
after a given PID exits).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.compare_ladder_runs import METRICS, get  # noqa: E402
from scripts.permutation_mechanism import (SEEDS as TEST_SEEDS, Frozen, assemble, frozen_thresholds, native_rows,  # noqa: E402
                                           natives, same_view_split_rate)
from scripts.run_mstcn_official import FEATURES_DIM, PINNED, export_layout  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.audit.label_map import load_ontology  # noqa: E402
from echo_routing.compose.banks import multi_bank, native_bank, pair_bank  # noqa: E402
from echo_routing.config import FRAME_ORDER_VERSION, load_paths  # noqa: E402
from echo_routing.evaluate.ladder import evaluate_method  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.realvideo.pediatric import recipes_digest  # noqa: E402
from echo_routing.temporal.baselines.registry import AUDITED_B3  # noqa: E402

ENC = "79f4a41af6e3"
MODEL_SEEDS = (0, 1, 2)
TRAIN_PERMUTATION_SEED = 500          # fixed before training; distinct from the test seeds 100-109
CONDITIONS = ("true", "rand")
EPOCHS = 50
TEST_CONDITIONS = [("true", 0)] + [("random", s) for s in TEST_SEEDS]
PAIRED_METHODS = ("b2", "b3", "b3a")  # B0-probability decoders recomputed on the same test inputs
PATCH = Path("third_party/patches/ms-tcn-seed.patch")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tag(kind: str, seed: int) -> str:
    return "true" if kind == "true" else f"{kind}_s{seed}"


class Ctx:
    def __init__(self, cfg_path: str) -> None:
        ns = argparse.Namespace(config=cfg_path, set=[])
        self.cfg = load_cfg(ns); self.cfg["decode"].setdefault("b3a", dict(AUDITED_B3))
        self.paths = load_paths(); self.root = self.paths.repo_root
        self.seeded = self.root / "third_party" / "ms-tcn_seeded"
        self.data = self.paths.data_root / "mstcn_matrix"; self.pred = self.paths.cache_root / "mstcn_matrix"
        self.idx = read_index(variant_dir(self.paths.cache_root, ENC, "orig"))
        self.banks = {"train": multi_bank(self.idx, "train", self.cfg)}
        for s in ("validation", "test"):
            self.banks[f"pairs_{s}"] = pair_bank(self.idx, s, self.cfg); self.banks[f"multi_{s}"] = multi_bank(self.idx, s, self.cfg)
            self.banks[f"native_{s}"] = native_bank(self.idx, s)
        self.rec_hash = recipes_digest([r for b in self.banks.values() for r in b.recipes])[:10]
        self.classes = list(load_ontology().classes(self.cfg["task"]))
        self.fz = Frozen(self.paths, ENC)

    def data_name(self, cond: str, suffix: str = "") -> str:
        ps = TRAIN_PERMUTATION_SEED if cond == "rand" else "na"
        return f"{ENC}-r{self.rec_hash}-tr{cond}-ps{ps}-{FRAME_ORDER_VERSION}{suffix}"


def features_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((root / "features").iterdir()):
        h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
    for p in sorted((root / "groundTruth").iterdir()):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def prepare(ctx: Ctx, limit: int | None = None, suffix: str = "") -> dict[str, str]:
    """Export each training condition once; returns condition -> data name. Existing exports are digest-checked."""
    recipes = list(ctx.banks["train"].recipes)[:limit] if limit else list(ctx.banks["train"].recipes)
    out = {}
    for cond in CONDITIONS:
        name = ctx.data_name(cond, suffix); root = ctx.data / name; done = root / "DONE.json"
        if done.exists():
            if json.loads(done.read_text())["features_digest"] != features_digest(root):
                raise SystemExit(f"{root} changed after export")
            out[cond] = name; continue
        kind = "random" if cond == "rand" else "true"
        streams = {r.recipe_id: assemble(ctx.fz, r, kind, TRAIN_PERMUTATION_SEED, None, "b0")[0] for r in recipes}
        export_layout(root, ctx.classes, list(streams), [], streams)
        done.write_text(json.dumps({"data_name": name, "condition": cond, "permutation_seed": TRAIN_PERMUTATION_SEED if cond == "rand" else None,
                                    "recipe_digest": ctx.rec_hash, "encoder": ENC, "frame_order": FRAME_ORDER_VERSION,
                                    "train_streams": len(streams), "features_digest": features_digest(root)}, indent=1))
        out[cond] = name
    return out


def train(ctx: Ctx, data_name: str, model_seed: int, epochs: int = EPOCHS, env: dict | None = None, run_suffix: str = "",
          wait: bool = True):
    """Train through the seeded official main.py. Returns the run name (or the Popen when wait=False)."""
    run = f"{data_name}-ms{model_seed}{run_suffix}"
    link = ctx.data / run
    if not link.exists():
        link.symlink_to(ctx.data / data_name, target_is_directory=True)
    if not (ctx.seeded / "data").exists():
        (ctx.seeded / "data").symlink_to(ctx.data, target_is_directory=True)
    model_dir = ctx.seeded / "models" / run / "split_1"
    if (model_dir / "DONE.json").exists():
        return run
    log = (ctx.paths.logs_root / f"mstcn_matrix_{run}.log").open("w")
    cmd = [sys.executable, "main.py", "--action=train", f"--dataset={run}", "--split=1", f"--seed={model_seed}", f"--num_epochs={epochs}"]
    proc = subprocess.Popen(cmd, cwd=ctx.seeded, env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}, stdout=log, stderr=subprocess.STDOUT)
    proc.meta = {"run": run, "data_name": data_name, "model_seed": model_seed, "epochs": epochs, "t0": time.time(),  # type: ignore[attr-defined]
                 "device": "cpu" if (env or {}).get("CUDA_VISIBLE_DEVICES") == "" else "cuda"}
    return finish(ctx, proc) if wait else proc


def finish(ctx: Ctx, proc) -> str:
    if proc.wait() != 0:
        raise SystemExit(f"training {proc.meta['run']} failed (see logs/mstcn_matrix_{proc.meta['run']}.log)")
    m = proc.meta; model_dir = ctx.seeded / "models" / m["run"] / "split_1"
    init = dict(line.split() for line in (model_dir / "init_digest.txt").read_text().splitlines())
    if int(init["seed"]) != m["model_seed"]:
        raise SystemExit(f"{m['run']}: entry point used seed {init['seed']}, expected {m['model_seed']}")
    data_done = json.loads((ctx.data / m["data_name"] / "DONE.json").read_text())
    head = subprocess.run(["git", "-C", str(ctx.root / "third_party" / "ms-tcn"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = {"run": m["run"], "entry": "third_party/ms-tcn_seeded/main.py", "clone_commit": head, "pinned": PINNED,
           "seed_patch_sha256": sha256_file(ctx.root / PATCH), "model_seed": int(init["seed"]), "init_sha256": init["init_sha256"],
           "epochs": m["epochs"], "final_model": f"epoch-{m['epochs']}.model",
           "final_sha256": sha256_file(model_dir / f"epoch-{m['epochs']}.model"), "device": m["device"],
           "seconds": round(time.time() - m["t0"], 1), "data": data_done}
    (model_dir / "DONE.json").write_text(json.dumps(rec, indent=1))
    return m["run"]


def load_model(ctx: Ctx, run: str, epochs: int, device):
    import torch
    sys.path.insert(0, str(ctx.seeded))
    from model import MultiStageModel  # official class
    model = MultiStageModel(4, 10, 64, FEATURES_DIM, len(ctx.classes)).to(device)
    model.load_state_dict(torch.load(ctx.seeded / "models" / run / "split_1" / f"epoch-{epochs}.model", map_location=device))
    return model.eval()


def predict(ctx: Ctx, run: str, epochs: int = EPOCHS) -> dict:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = load_model(ctx, run, epochs, device)
    ncls = len(ctx.classes)

    @torch.no_grad()
    def write(items, out_dir: Path) -> float:
        out_dir.mkdir(parents=True, exist_ok=True); spent = 0.0
        for sid, feat in items:
            x = np.zeros((FEATURES_DIM, feat.shape[0]), dtype=np.float32); x[: feat.shape[1]] = feat.T
            t0 = time.perf_counter(); xt = torch.from_numpy(x).unsqueeze(0).to(device)
            p = torch.softmax(model(xt, torch.ones(1, ncls, xt.shape[2], device=device))[-1], dim=1)[0].T.cpu().numpy()
            spent += time.perf_counter() - t0
            np.savez_compressed(out_dir / f"{sid}.npz", prob=p.astype(np.float32))
        return spent

    base = ctx.pred / run
    val = [(s.recipe.recipe_id, s.feat) for b in ("pairs_validation", "multi_validation")
           for s in (assemble(ctx.fz, r, "true", 0, None, "b0")[0] for r in ctx.banks[b].recipes)]
    val += [(c.video_id, c.feat) for c in natives(ctx.fz, ctx.idx, "validation", "true", 0, None, "b0")[0]]
    seconds = {"validation": write(val, base / "_validation")}
    for kind, seed in TEST_CONDITIONS:
        out = base / tag(kind, seed)
        items = [(s.recipe.recipe_id, s.feat) for b in ("pairs_test", "multi_test")
                 for s in (assemble(ctx.fz, r, kind, seed, None, "b0")[0] for r in ctx.banks[b].recipes)]
        items += [(c.video_id, c.feat) for c in natives(ctx.fz, ctx.idx, "test", kind, seed, None, "b0")[0]]
        seconds[tag(kind, seed)] = write(items, out)
        for p in (base / "_validation").iterdir():  # validation stays in true order: hard links, identical bytes
            if not (out / p.name).exists():
                os.link(p, out / p.name)
    (base / "PREDICTED.json").write_text(json.dumps({"run": run, "epochs": epochs, "device": str(device), "forward_seconds": seconds}, indent=1))
    return seconds


def evaluate(ctx: Ctx, runs: list[str], out: Path) -> None:
    import copy
    cfg = ctx.cfg; ncls = len(ctx.classes)
    val_pairs = [assemble(ctx.fz, r, "true", 0, None, "b0")[0] for r in ctx.banks["pairs_validation"].recipes]
    val_multi = [assemble(ctx.fz, r, "true", 0, None, "b0")[0] for r in ctx.banks["multi_validation"].recipes]
    val_nat = natives(ctx.fz, ctx.idx, "validation", "true", 0, None, "b0")[0]
    rows, nat_rows, frag_rows = [], [], []
    for kind, seed in TEST_CONDITIONS:
        tp = [assemble(ctx.fz, r, kind, seed, None, "b0")[0] for r in ctx.banks["pairs_test"].recipes]
        tm = [assemble(ctx.fz, r, kind, seed, None, "b0")[0] for r in ctx.banks["multi_test"].recipes]
        tn = natives(ctx.fz, ctx.idx, "test", kind, seed, None, "b0")[0]
        jobs = [(m, None, cfg) for m in PAIRED_METHODS]
        for run in runs:
            c = copy.deepcopy(cfg); c["mstcn_official"] = {"pred_dir": str(ctx.pred / run / tag(kind, seed))}
            jobs.append(("b4", run, c))
        for m, run, c in jobs:
            res = evaluate_method(m, val_nat, tn, val_pairs, tp, ncls, c, test_multi=tm, val_multi=val_multi)
            meta = {"model": run or "-", "method": m, "test_condition": kind, "test_seed": seed}
            row = {**meta, **{k: get(res, p) for k, p in METRICS}, "same_view_split_pairs": same_view_split_rate(res, "constructed"),
                   "same_view_split_multi": same_view_split_rate(res, "constructed_multi"), "thresholds": frozen_thresholds(res)}
            rows.append(row)
            for bank, fr in res["_fragments"].items():
                frag_rows.extend({**meta, "bank": bank, **{k: v for k, v in f.items() if not k.startswith(("acc|", "wrong|"))}} for f in fr)
            for r in native_rows(tn, c, (m,), "b0", kind, seed):
                nat_rows.append({**r, "model": run or "-"})
        print(f"evaluated {tag(kind, seed)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False); pd.DataFrame(nat_rows).to_csv(out / "natives.csv.gz", index=False)
    pd.DataFrame(frag_rows).to_csv(out / "fragments.csv.gz", index=False)


def verify(ctx: Ctx, out: Path, device: str) -> dict:
    """Tiny run (12 training streams, 2 epochs): seed control, shared initialisation of a condition pair, isolation."""
    env = {"CUDA_VISIBLE_DEVICES": ""} if device == "cpu" else {}
    suffix = f"-verify12-{device}"
    names = prepare(ctx, limit=12, suffix=suffix)
    runs = {"true_s0": train(ctx, names["true"], 0, 2, env), "true_s0_rep": train(ctx, names["true"], 0, 2, env, run_suffix="-rep"),
            "true_s1": train(ctx, names["true"], 1, 2, env), "rand_s0": train(ctx, names["rand"], 0, 2, env)}
    done = {k: json.loads((ctx.seeded / "models" / r / "split_1" / "DONE.json").read_text()) for k, r in runs.items()}
    ref = ctx.data.parent / "mstcn" / f"ev9v_{ENC}" / "features"   # the v2 export used for the frozen MS-TCN v2
    same_as_v2 = [bool(np.array_equal(np.load(ctx.data / names["true"] / "features" / f"{sid}.npy"), np.load(ref / f"{sid}.npy")))
                  for sid in [r.recipe_id for r in list(ctx.banks["train"].recipes)[:12]] if (ref / f"{sid}.npy").exists()]
    checks = {
        "same_seed_same_init": done["true_s0"]["init_sha256"] == done["true_s0_rep"]["init_sha256"],
        "same_seed_same_final_weights": done["true_s0"]["final_sha256"] == done["true_s0_rep"]["final_sha256"],
        "different_seed_different_init": done["true_s0"]["init_sha256"] != done["true_s1"]["init_sha256"],
        "condition_pair_same_init": done["true_s0"]["init_sha256"] == done["rand_s0"]["init_sha256"],
        "condition_pair_different_final": done["true_s0"]["final_sha256"] != done["rand_s0"]["final_sha256"],
        "recorded_seed_matches": all(d["model_seed"] == int(k.split("_s")[1][0]) for k, d in done.items()),
        "distinct_model_dirs": len({r for r in runs.values()}) == len(runs),
        "true_export_equals_v2_export": f"{sum(same_as_v2)}/{len(same_as_v2)}",
        "true_and_rand_data_differ": json.loads((ctx.data / names["true"] / "DONE.json").read_text())["features_digest"]
                                     != json.loads((ctx.data / names["rand"] / "DONE.json").read_text())["features_digest"]}
    rec = {"device": device, "runs": runs, "checks": checks, "done": done}
    out.mkdir(parents=True, exist_ok=True); (out / f"verify_{device}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(checks, indent=1), flush=True)
    return rec


def queue(ctx: Ctx, after_pid: int | None, out: Path) -> None:
    while after_pid and Path(f"/proc/{after_pid}").exists():
        time.sleep(60)
    print(f"start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    verify(ctx, out, "cuda")
    names = prepare(ctx)
    runs = []
    for seed in MODEL_SEEDS:  # the two conditions of one seed train side by side, same budget
        procs = [train(ctx, names[c], seed, EPOCHS, wait=False) for c in CONDITIONS]
        runs.extend(finish(ctx, p) if not isinstance(p, str) else p for p in procs)
        print(f"trained seed {seed} {time.strftime('%H:%M:%S')}", flush=True)
    for run in runs:
        predict(ctx, run)
    evaluate(ctx, runs, out)
    print(f"done {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="MS-TCN permutation matrix")
    ap.add_argument("action", choices=("verify", "prepare", "queue", "predict", "evaluate"))
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="predict: which saved epoch to load")
    ap.add_argument("-c", "--config", default="runs/20260928-001145-baselines-1b027c50-s0/config.resolved.yaml")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda")); ap.add_argument("--after-pid", type=int, default=None)
    ap.add_argument("--out", default="runs/_demo/mstcn_matrix"); ap.add_argument("--runs", default=None)
    args = ap.parse_args(); ctx = Ctx(args.config); out = Path(args.out)
    print(json.dumps({"recipe_digest": ctx.rec_hash, "data_names": {c: ctx.data_name(c) for c in CONDITIONS}}), flush=True)
    if args.action == "verify":
        verify(ctx, out, args.device)
    elif args.action == "prepare":
        print(prepare(ctx))
    elif args.action == "queue":
        queue(ctx, args.after_pid, out)
    elif args.action == "predict":
        for run in args.runs.split(","):
            print(run, predict(ctx, run, args.epochs), flush=True)
    else:
        evaluate(ctx, args.runs.split(","), out)


if __name__ == "__main__":
    main()
