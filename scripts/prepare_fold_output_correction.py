#!/usr/bin/env python
"""Freeze the data and identity of the fold-output correction round (plan Task 1;
configs/experiments/fold_output_correction_v1.yaml). No training and no development-set score.

  --extract  T-in caches: model (k + 1) % 3 on the clips of fold k (orig and gamma090, 10 Hz, 224 letterbox, numeric
             frame order) into new caches <cache_key>-seen-fold<k>. The T-out caches are never written. (GPU)
  --check    Input identities (sha256); the training list of each fold model and the seen / unseen rule per fold; cache
             coverage, class order and frame alignment; the 1,500 library streams and the native windows; the paired
             batch plan of each seed with its class weights and pos_weight; real paired batches of the three arms; and
             the input description (in-sample against held-out error, clip confusion and confidence per fold). Writes
             the plans and details (server) and preparation.json (counts and digests only).

  python scripts/prepare_fold_output_correction.py --config configs/experiments/fold_output_correction_v1.yaml \
      --out docs/results/2026-10-08_fold_output_correction --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.compose.recipe_builder import recipe_from_dict  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.temporal.dev_data import Seq  # noqa: E402

NS = "fold_output_correction_v1"
ARMS = ("T-in", "T-out", "P-out")
N_FOLDS = 3
VARIANTS = ("orig", "gamma090")
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")


# ----------------------------------------------------------------------------------------------- generators and pairing
def generator(arm: str, fold: int) -> int:
    """The fold model whose probabilities an arm reads for the clips of `fold`: (k + 1) % 3 for T-in (it trained on
    fold k), k for T-out and P-out (it never trained on fold k)."""
    if arm not in ARMS:
        raise ValueError(arm)
    return (fold + 1) % N_FOLDS if arm == "T-in" else fold


def seen_key(cache_key: str, fold: int) -> str:
    return f"{cache_key}-seen-fold{fold}"


def arm_cache_key(arm: str, fold: int, keys: dict[int, str]) -> str:
    j = generator(arm, fold)
    return seen_key(keys[j], fold) if arm == "T-in" else keys[j]


def check_generator(arm: str, session: str, fold: int, train_sessions: dict[int, set]) -> None:
    """T-in must read a model that trained on the session; T-out and P-out one that did not."""
    seen = session in train_sessions[generator(arm, fold)]
    if seen != (arm == "T-in"):
        raise SystemExit(f"{arm}: fold {fold}'s generator {'trained' if seen else 'did not train'} on a session it must {'not ' if seen else ''}have")


def check_training_clips(clips, protocol_train: set) -> None:
    """Training sequences may hold protocol training clips only (never validation or test)."""
    bad = sorted(set(clips) - protocol_train)
    if bad:
        raise SystemExit(f"{len(bad)} clips outside the protocol's training split would enter training")


@dataclass(frozen=True)
class Item:
    """One training sequence, independent of the encoder: the arms differ only in the probabilities read for it."""
    kind: str             # "stream" or "native"
    seq_id: str
    fold: int
    sources: tuple        # ((video, variant, start, end), ...) in order
    labels: np.ndarray
    joins: np.ndarray


def stream_item(recipe: dict, fold: int, fold_of_clip: dict[str, int]) -> Item:
    r = recipe_from_dict(recipe); frs = r.fragments
    if any(fold_of_clip[f.video_id] != fold for f in frs):
        raise SystemExit(f"stream {r.recipe_id} uses recordings of another fold; one stream must read one encoder")
    labels = np.concatenate([np.full(f.end - f.start, int(f.label), np.int64) for f in frs])
    joins = np.cumsum([f.end - f.start for f in frs])[:-1].astype(int)
    return Item("stream", r.recipe_id, fold, tuple((f.video_id, f.variant, int(f.start), int(f.end)) for f in frs), labels, joins)


def epoch_items(seed: int, epoch: int, streams: list[Item], clips: list[tuple[str, int]], n_samples: dict[str, int], variant: dict[str, str],
                fold_of_clip: dict[str, int], n_native: int, window: int, spb: int, npb: int) -> list[list[Item]]:
    """TrainData.epoch_batches' order and window rule (same generator calls in the same order), with each native window
    read in its clip's fixed contrast variant instead of the hard-coded orig."""
    rng = np.random.default_rng([int(seed), int(epoch)])
    order = rng.permutation(len(streams))
    picks = rng.choice(len(clips), size=n_native, replace=n_native > len(clips))
    natives = []
    for i in picks:
        vid, lab = clips[int(i)]; n = int(n_samples[vid]); length = min(window, n); start = int(rng.integers(0, n - length + 1))
        natives.append(Item("native", f"{vid}@{start}", fold_of_clip[vid], ((vid, variant[vid], start, start + length),),
                            np.full(length, int(lab), np.int64), np.zeros(0, int)))
    n_batches = max(math.ceil(len(order) / spb), math.ceil(n_native / npb))
    return [[streams[int(k)] for k in order[b * spb:(b + 1) * spb]] + natives[b * npb:(b + 1) * npb] for b in range(n_batches)]


def item_digest(it: Item) -> str:
    h = hashlib.sha256(f"{it.kind}|{it.seq_id}|{it.fold}|{json.dumps(it.sources)}|".encode())
    h.update(np.asarray(it.labels, np.int64).tobytes()); h.update(b"|"); h.update(np.asarray(it.joins, np.int64).tobytes())
    return h.hexdigest()


def batch_digest(items: list[Item]) -> str:
    """Identity of one update: every sequence's sources, labels and joins (never the probabilities)."""
    return hashlib.sha256("".join(item_digest(it) for it in items).encode()).hexdigest()[:16]


class ProbStore:
    """Memoised per-sample probabilities of one cache key; only the prob array is read."""

    def __init__(self, cache_root: Path, key: str) -> None:
        self.cache_root, self.key = Path(cache_root), key
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def get(self, video: str, variant: str) -> np.ndarray:
        k = (video, variant)
        if k not in self._cache:
            with np.load(variant_dir(self.cache_root, self.key, variant) / f"{video}.npz") as z:
                self._cache[k] = np.asarray(z["prob"], np.float32)
        return self._cache[k]


def materialize(it: Item, prob_of) -> Seq:
    """The Seq an arm trains on: its probabilities of every source span, concatenated; labels and joins unchanged."""
    feat = np.concatenate([prob_of(v, var)[a:b] for v, var, a, b in it.sources]).astype(np.float32)
    if feat.shape != (it.labels.size, len(CLASSES)):
        raise SystemExit(f"{it.seq_id}: probabilities {feat.shape} do not match {it.labels.size} labels x {len(CLASSES)} classes")
    return Seq(it.seq_id, it.kind == "native", feat, it.labels, it.joins)


def assert_no_ids(obj, ids: set[str], where: str = "") -> None:
    """Refuse a repo-bound record that carries a recording or session id anywhere (keys or string values)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in ids:
                raise SystemExit(f"an id appears as a key at {where or 'top'}")
            assert_no_ids(v, ids, f"{where}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_ids(v, ids, f"{where}[{i}]")
    elif isinstance(obj, str) and (obj in ids or any(part in ids for part in obj.replace("|", " ").replace("@", " ").replace(":", " ").split())):
        raise SystemExit(f"an id appears as a value at {where}")


def cache_digest(cache_root: Path, key: str, clips: list[str]) -> str:
    """Content identity of one cache over the given clips and both variants (every file's bytes and the index)."""
    h = hashlib.sha256()
    for variant in VARIANTS:
        d = variant_dir(cache_root, key, variant); h.update(f"{variant}|{sha256_file(d / 'index.csv')}|".encode())
        for v in sorted(clips):
            h.update(f"{v}|{sha256_file(d / f'{v}.npz')}|".encode())
    return h.hexdigest()


class ArmInputs:
    """Per arm and fold, the probability store; every read checks the seen / unseen rule of its session."""

    def __init__(self, cache_root: Path, keys: dict[int, str], train_sessions: dict[int, set]) -> None:
        self.stores = {(arm, k): ProbStore(cache_root, arm_cache_key(arm, k, keys)) for arm in ARMS for k in range(N_FOLDS)}
        self.train_sessions = train_sessions

    def seq(self, arm: str, it: Item) -> Seq:
        for v, *_ in it.sources:
            check_generator(arm, session_of(v), it.fold, self.train_sessions)
        return materialize(it, self.stores[(arm, it.fold)].get)


# ----------------------------------------------------------------------------------------------- context
def check_inputs(fo: dict, root: Path) -> dict:
    specs = {k: fo["inputs"][k] for k in ("protocol", "manifest", "session_folds", "library_config", "library_models")}
    specs.update({f"banks_f{k}": v for k, v in fo["inputs"]["banks"].items()})
    for k, m in fo["inputs"]["fold_models"].items():
        specs[f"fold_model_{k}"] = {"path": m["checkpoint"], "sha256": m["sha256"]}; specs[f"fold_run_manifest_{k}"] = m["run_manifest"]
    specs["b0_v3"] = fo["inputs"]["b0_v3"]["checkpoint"]
    for name, spec in specs.items():
        if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
            raise SystemExit(f"input {name}: sha256 {got[:12]} != {spec['sha256'][:12]}")
    return {name: spec["sha256"] for name, spec in specs.items()}


@dataclass
class Context:
    proto: dict
    labels: dict[str, int]
    fold_of_session: dict[str, int]
    fold_of_clip: dict[str, int]
    train_sessions: dict[int, set]
    banks: dict[int, dict]
    keys: dict[int, str]
    clips: list[tuple[str, int]]
    variant: dict[str, str]
    streams: list[Item]


def load_context(cfg: dict, fo: dict, root: Path) -> Context:
    import pandas as pd
    proto = json.loads((root / fo["inputs"]["protocol"]["path"]).read_text()); train = list(proto["train"]["clips"])
    man = pd.read_csv(root / fo["inputs"]["manifest"]["path"]).set_index("video_id")
    if man.loc[train, "split"].ne("train").any():
        raise SystemExit("a protocol training clip is not in the manifest's training split")
    labels = {v: int(man.loc[v, "family5_index"]) for v in train}
    fold_of_session = {s: int(f) for s, f in json.loads((root / fo["inputs"]["session_folds"]["path"]).read_text()).items()}
    fold_of_clip = {v: fold_of_session[session_of(v)] for v in train}
    train_sessions = {k: {s for s, f in fold_of_session.items() if f != k} for k in range(N_FOLDS)}
    banks = {int(k): json.loads((root / v["path"]).read_text()) for k, v in fo["inputs"]["banks"].items()}
    variant = {v: var for k in range(N_FOLDS) for v, var in banks[k]["natives"].items()}
    if set(variant) != set(train) or any(fold_of_clip[v] != k for k in range(N_FOLDS) for v in banks[k]["natives"]):
        raise SystemExit("the library's native contrast choices do not cover the training clips fold by fold")
    streams = [stream_item(r, k, fold_of_clip) for k in range(N_FOLDS) for r in banks[k]["recipes"]]
    check_training_clips({s[0] for it in streams for s in it.sources} | set(variant), set(train))
    keys = {int(k): m["cache_key"] for k, m in fo["inputs"]["fold_models"].items()}
    return Context(proto, labels, fold_of_session, fold_of_clip, train_sessions, banks, keys, [(v, labels[v]) for v in train], variant, streams)


def n_samples_of(ctx: Context, cache_root: Path) -> dict[str, int]:
    out = {}
    for k in range(N_FOLDS):
        idx = read_index(variant_dir(cache_root, ctx.keys[k], "orig")).set_index("video_id")
        out.update({v: int(idx.loc[v, "n_samples"]) for v, _ in ctx.clips if ctx.fold_of_clip[v] == k})
    return out


def plan_of(ctx: Context, tr: dict, n_samples: dict[str, int], seed: int, epoch: int) -> list[list[Item]]:
    return epoch_items(seed, epoch, ctx.streams, ctx.clips, n_samples, ctx.variant, ctx.fold_of_clip, int(tr["native_windows_per_epoch"]),
                       int(tr["native_window"]), int(tr["streams_per_batch"]), int(tr["natives_per_batch"]))


def make_plan(ctx: Context, tr: dict, n_samples: dict[str, int], seed: int) -> dict:
    from echo_routing.temporal.dev_heads import asrf_weights
    epochs = []
    for e in range(int(tr["epochs"])):
        b = plan_of(ctx, tr, n_samples, seed, e)
        if len(b) != int(tr["updates_per_epoch"]) or any(len(x) != int(tr["streams_per_batch"]) + int(tr["natives_per_batch"]) for x in b):
            raise SystemExit(f"epoch {e}: {len(b)} updates or a short batch, expected {tr['updates_per_epoch']} of 4 + 4")
        epochs.append([batch_digest(x) for x in b])
    cw, pw = asrf_weights([it for x in plan_of(ctx, tr, n_samples, seed, 0) for it in x], num_classes=len(CLASSES))
    return {"seed": int(seed), "epochs": epochs, "class_weight": cw.tolist(), "pos_weight": float(pw)}


def save_or_check_plan(plan: dict, root: Path) -> tuple[Path, str]:
    path = root / "runs" / NS / "plans" / f"plan_s{plan['seed']}.json"
    if path.is_file():
        if json.loads(path.read_text()) != json.loads(json.dumps(plan)):
            raise SystemExit(f"{path}: the batches or weights differ from the saved plan of the seed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(plan))
    return path, sha256_file(path)


# ----------------------------------------------------------------------------------------------- steps
def run_extract(cfg: dict, fo: dict, paths) -> None:
    import pandas as pd
    import torch

    from echo_routing.config import gpu_memory_fraction
    from echo_routing.features.encoder import load_checkpoint
    from echo_routing.features.extract import extract_split
    root = paths.repo_root; check_inputs(fo, root); ctx = load_context(cfg, fo, root)
    man = pd.read_csv(root / fo["inputs"]["manifest"]["path"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    for k in range(N_FOLDS):
        j = generator("T-in", k); key = arm_cache_key("T-in", k, ctx.keys)
        if variant_dir(paths.cache_root, key, "orig") in {variant_dir(paths.cache_root, ctx.keys[i], "orig") for i in range(N_FOLDS)}:
            raise SystemExit("the T-in cache would overwrite a T-out cache")
        clips = sorted(v for v, _ in ctx.clips if ctx.fold_of_clip[v] == k)
        for v in clips:
            check_generator("T-in", session_of(v), k, ctx.train_sessions)
        model, meta = load_checkpoint(root / fo["inputs"]["fold_models"][j]["checkpoint"])
        if tuple(meta["classes"]) != CLASSES:
            raise SystemExit(f"fold model {j}: class order {meta['classes']}")
        rows = man[man.video_id.isin(clips)]
        for variant in VARIANTS:
            extract_split(model, device, rows, "family5_index", paths.cache_root, key, variant, cfg["sampling"]["target_hz"], cfg["preprocess"]["image_size"],
                          cfg["features"]["batch_size"], cfg["features"]["num_workers"],
                          meta={"ckpt": fo["inputs"]["fold_models"][j]["checkpoint"], "task": cfg["task"], "classes": list(CLASSES), "model": j, "fold": k, "use": "T-in (seen fold)"})
        print(json.dumps({"model": j, "fold": k, "cache": key, "clips": len(clips)}), flush=True)


def training_lists(ctx: Context, fo: dict, root: Path) -> dict:
    """Each fold model's training clips as its trainer formed them (protocol training clips outside its fold), checked
    against its run manifest and training log."""
    out = {}
    for k in range(N_FOLDS):
        clips = sorted(v for v, _ in ctx.clips if ctx.fold_of_clip[v] != k); sess = {session_of(v) for v in clips}
        rm = json.loads((root / fo["inputs"]["fold_models"][k]["run_manifest"]["path"]).read_text())
        log = [json.loads(x) for x in (root / Path(fo["inputs"]["fold_models"][k]["checkpoint"]).parent / "train_log.jsonl").read_text().splitlines()]
        if rm["fold"] != k or rm["n_train_clips"] != len(clips) or rm["n_train_sessions"] != len(sess) or {r["n_train_records"] for r in log} != {len(clips)}:
            raise SystemExit(f"fold model {k}: its recorded training list differs from the two other folds")
        if sess & {s for s, f in ctx.fold_of_session.items() if f == k}:
            raise SystemExit(f"fold model {k} trained on a session of its own fold")
        out[str(k)] = {"clips": len(clips), "sessions": len(sess), "digest": hashlib.sha256("\n".join(clips).encode()).hexdigest(),
                       "epochs_logged": len(log), "final_epoch": log[-1]["epoch"]}
    return out


def caches(ctx: Context, fo: dict, cache_root: Path, protocol_train: set) -> dict:
    """Coverage of every arm's caches (fold clips only, both variants) and equal frame indices for T-in and T-out."""
    out = {}
    for k in range(N_FOLDS):
        fold_clips = {v for v, _ in ctx.clips if ctx.fold_of_clip[v] == k}; rec = {}
        for arm in ("T-in", "T-out"):
            key = arm_cache_key(arm, k, ctx.keys)
            for variant in VARIANTS:
                idx = read_index(variant_dir(cache_root, key, variant))
                if set(idx.video_id) != fold_clips or set(idx.split) != {"train"}:
                    raise SystemExit(f"{arm} fold {k} {variant}: the cache does not hold exactly the fold's training clips")
                meta = json.loads((variant_dir(cache_root, key, variant) / "meta.json").read_text())
                if tuple(meta.get("classes") or ()) != CLASSES:
                    raise SystemExit(f"{arm} fold {k} {variant}: class order {meta.get('classes')}")
            rec[arm] = key
        bad = 0
        for v in sorted(fold_clips):
            for variant in VARIANTS:
                with np.load(variant_dir(cache_root, rec["T-in"], variant) / f"{v}.npz") as a, np.load(variant_dir(cache_root, rec["T-out"], variant) / f"{v}.npz") as b:
                    bad += int(not (np.array_equal(a["frame_idx"], b["frame_idx"]) and np.array_equal(a["t_sec"], b["t_sec"])))
        if bad:
            raise SystemExit(f"fold {k}: {bad} clip-variants differ in frame indices or times between T-in and T-out")
        out[str(k)] = {"clips": len(fold_clips), "T-in cache": rec["T-in"], "T-out cache": rec["T-out"], "frame alignment": "identical"}
    check_training_clips({v for v, _ in ctx.clips}, protocol_train)
    return out


def description(ctx: Context, arms: ArmInputs) -> dict:
    """Input description per fold (orig variant, every native clip): per-sample error, clip confusion (majority vote)
    and mean confidence, in-sample (T-in generator) against held-out (T-out generator). Never used for a choice."""
    out = {}
    for k in range(N_FOLDS):
        out[str(k)] = {}
        for arm in ("T-in", "T-out"):
            n = wrong = 0; conf = np.zeros((len(CLASSES), len(CLASSES)), int); top = []; true_p = []
            for v, lab in ctx.clips:
                if ctx.fold_of_clip[v] != k:
                    continue
                p = arms.stores[(arm, k)].get(v, "orig"); pred = p.argmax(1)
                n += pred.size; wrong += int((pred != lab).sum()); conf[lab, int(np.bincount(pred, minlength=len(CLASSES)).argmax())] += 1
                top.append(float(p.max(1).mean())); true_p.append(float(p[:, lab].mean()))
            out[str(k)][arm] = {"generator model": generator(arm, k), "sample error": wrong / n, "samples": n, "clip confusion": conf.tolist(),
                                "clips correct": int(np.trace(conf)), "mean confidence": float(np.mean(top)), "mean true-class probability": float(np.mean(true_p))}
    return out


def paired_batches(ctx: Context, arms: ArmInputs, tr: dict, n_samples: dict, seeds: list[int], n_check: int) -> dict:
    """Real paired batches: the first n_check updates of epoch 0 per seed, materialised for the three arms."""
    out = {}
    for seed in seeds:
        b = plan_of(ctx, tr, n_samples, seed, 0)[:n_check]; diff = same_out = 0; rows_ok = True
        for items in b:
            for it in items:
                s = {arm: arms.seq(arm, it) for arm in ARMS}
                if not all(np.array_equal(s[a].labels, s["T-out"].labels) and np.array_equal(s[a].joins, s["T-out"].joins) for a in ARMS):
                    raise SystemExit(f"seed {seed}: the arms differ in labels or joins")
                if not np.array_equal(s["P-out"].feat, s["T-out"].feat):
                    raise SystemExit(f"seed {seed}: P-out does not read the T-out inputs")
                rows_ok &= bool(all(np.isfinite(s[a].feat).all() and np.abs(s[a].feat.sum(1) - 1).max() < 1e-3 for a in ARMS))
                diff += int(not np.array_equal(s["T-in"].feat, s["T-out"].feat)); same_out += 1
        out[str(seed)] = {"updates": len(b), "sequences": same_out, "T-in differs from T-out": diff, "probability rows finite and summing to 1": rows_ok}
    return out


def run_check(cfg: dict, fo: dict, paths, out_dir: Path) -> None:
    from echo_routing.audit.label_map import load_ontology
    root = paths.repo_root; t0 = time.time(); shas = check_inputs(fo, root); ctx = load_context(cfg, fo, root); tr = fo["training"]
    if tuple(load_ontology().classes(cfg["task"])) != CLASSES:
        raise SystemExit("the frozen ontology's class order differs")
    import torch
    for name, ck in [*[(f"fold model {k}", m["checkpoint"]) for k, m in fo["inputs"]["fold_models"].items()], ("B0 v3", fo["inputs"]["b0_v3"]["checkpoint"]["path"])]:
        meta = torch.load(root / ck, map_location="cpu", weights_only=False)
        if tuple(meta["classes"]) != CLASSES:
            raise SystemExit(f"{name}: class order {meta['classes']}")
    if int(torch.load(root / fo["inputs"]["b0_v3"]["checkpoint"]["path"], map_location="cpu", weights_only=False)["epoch"]) != int(fo["inputs"]["b0_v3"]["epoch"]):
        raise SystemExit("B0 v3 checkpoint epoch differs from the config")
    lists = training_lists(ctx, fo, root); cache = caches(ctx, fo, paths.cache_root, set(ctx.proto["train"]["clips"])); n_samples = n_samples_of(ctx, paths.cache_root)
    arms = ArmInputs(paths.cache_root, ctx.keys, ctx.train_sessions)
    for v, _ in ctx.clips:                                   # the seen / unseen rule on every training recording and arm
        for arm in ARMS:
            check_generator(arm, session_of(v), ctx.fold_of_clip[v], ctx.train_sessions)
    plans = {}
    for seed in [int(s) for s in tr["seeds"]]:
        plan = make_plan(ctx, tr, n_samples, seed); _, sha = save_or_check_plan(plan, root)
        plans[str(seed)] = {"sha256": sha, "class_weight": plan["class_weight"], "pos_weight": plan["pos_weight"], "updates": sum(len(e) for e in plan["epochs"])}
    paired = paired_batches(ctx, arms, tr, n_samples, [int(s) for s in tr["seeds"]], 25); desc = description(ctx, arms)
    res = {"inputs_sha256": shas, "class_order": list(CLASSES), "training_lists": lists, "caches": cache,
           "streams": {"per fold": {str(k): len(ctx.banks[k]["recipes"]) for k in range(N_FOLDS)}, "total": len(ctx.streams),
                       "fragments": sum(len(it.sources) for it in ctx.streams), "fragment variants": dict(Counter(s[1] for it in ctx.streams for s in it.sources))},
           "natives": {"clips": len(ctx.clips), "variants": dict(Counter(ctx.variant.values())), "sessions": len({session_of(v) for v, _ in ctx.clips})},
           "seen_unseen": "checked on every training recording for every arm", "plans": plans, "paired_batches": paired, "input_description": desc,
           "seconds": time.time() - t0}
    ids = {v for v, _ in ctx.clips} | {session_of(v) for v, _ in ctx.clips}
    details = {**res, "_fold_of_clip": ctx.fold_of_clip, "_native_variant": ctx.variant}          # server only: carries ids
    (root / "runs" / NS).mkdir(parents=True, exist_ok=True); (root / "runs" / NS / "preparation_details.json").write_text(json.dumps(details, indent=1))
    assert_no_ids(res, ids, "preparation")
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "preparation.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def check_out(out: str, fo: dict) -> None:
    if Path(out) != Path(fo["results_dir"]):
        raise SystemExit(f"--out must be {fo['results_dir']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    step = ap.add_mutually_exclusive_group(required=True); step.add_argument("--extract", action="store_true"); step.add_argument("--check", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); fo = cfg["fold_output_correction"]; check_out(a.out, fo); paths = load_paths()
    if a.extract:
        run_extract(cfg, fo, paths)
    else:
        run_check(cfg, fo, paths, paths.repo_root / a.out)


if __name__ == "__main__":
    main()
