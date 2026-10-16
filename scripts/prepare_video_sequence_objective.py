#!/usr/bin/env python
"""Prepare the A07 sequence-objective round (docs/superpowers/plans/2026-10-15-end-to-end-sequence-objective.md, Task 1;
configs/experiments/video_sequence_objective_v1.yaml). Training split only; nothing of the development halves is read.

1. Sources: every pin of the configuration (task sheet, source configurations and summaries, the loss module, the three
   V01 Ordered final states, the protocol, the archived summaries), the reused helper files and the environment.
2. Training sequences through A01's allowlisted path (prepare_video_feature_temporal.TrainingData): the protocol's 3,364
   training clips (labels parsed for training rows only), the rebuilt 1,500-stream bank (digest checked) and every
   sequence's 10 Hz references, streams assembled from their recipes first. No feature is extracted.
3. Groups: on every sequence's whole stride-5 grid (video_sequence_features.windows), native starts (four consecutive
   centres, the last repeated below four) and eligible constructed starts (a change among the four sampled labels).
4. Per seed, 3,750 updates drawn with default_rng([20261015, seed, update]) in the order native clip, native start,
   eligible stream, its start; eight windows per update (the native group, then the constructed group) in V01's manifest
   layout (videos, vid, frame, gamma per slot; marker, target, kind, seq, c per window; identity slot order) plus the group
   fields; every distinct (recording, frame, gamma) decoded once with V01's load_frame_u8 for the pixel digests.
5. Everything is built a second time from scratch (plans, manifests and pixel digests) and must be byte-identical.
Writes runs/video_sequence_objective_v1/manifests/VSO_s{seed}.npz and index.json (ids on the server only), the id-free
preparation.json in the results directory and a CPU ledger line.

  python scripts/prepare_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.prepare_video_feature_temporal as pvft  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402

import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.losses import segment_objective as so  # noqa: E402

WINDOWS, GROUP, SLOTS = 8, 4, tm.N_SLOTS
RNG_BASE = 20261015
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
PIN_KEYS = ("raw_video_config", "dense_config", "a01_config", "raw_video_summary", "a05_summary", "loss_module", "a03_config")


# ----------------------------------------------------------------------------------------------- groups
def native_starts(k: int) -> np.ndarray:
    """Starts of the four-centre groups of a native clip with k centres; below four centres the single start 0."""
    return np.arange(k - 3, dtype=np.int64) if k >= GROUP else np.zeros(1, np.int64)


def group_indices(k: int, start: int) -> np.ndarray:
    """The four centre indices of a group; a clip with fewer than four centres repeats its last centre."""
    return np.minimum(np.arange(int(start), int(start) + GROUP, dtype=np.int64), int(k) - 1)


def repeated_targets(idx: np.ndarray) -> int:
    return int(GROUP - len(set(np.asarray(idx).tolist())))


def eligible_starts(centre_labels: np.ndarray) -> np.ndarray:
    """Starts j of a constructed stream whose four sampled labels (centres j..j+3) contain at least one change."""
    lab = np.asarray(centre_labels)
    return np.array([j for j in range(lab.size - GROUP + 1) if len(set(lab[j:j + GROUP].tolist())) > 1], dtype=np.int64)


def skipped_runs(labels: np.ndarray, centres: np.ndarray) -> int:
    """Reference runs inside the group's span [first centre, last centre] that hold none of the sampled centres."""
    r = so.reference_runs(np.asarray(labels)); c = np.asarray(centres); lo, hi = int(c.min()), int(c.max())
    return int(sum(1 for s, e in zip(r.starts, r.ends) if s <= hi and e - 1 >= lo and not ((c >= s) & (c < e)).any()))


def draw(seed: int, update: int, native_counts, eligible_counts) -> tuple[int, int, int, int]:
    """The frozen draw of one update: native clip, its start, eligible stream, its start (in this order)."""
    rng = np.random.default_rng([RNG_BASE, int(seed), int(update)])
    i = int(rng.integers(len(native_counts))); j = int(rng.integers(int(native_counts[i])))
    k = int(rng.integers(len(eligible_counts))); l_ = int(rng.integers(int(eligible_counts[k])))
    return i, j, k, l_


def sequence_info(seqs: list[dict]) -> list[dict]:
    """Per sequence: its kind, length, whole-sequence windows and its group starts (eligible starts for streams)."""
    out = []
    for q in seqs:
        n = len(q["refs"]); w = vsf.windows(n); k = len(w["centres"])
        if np.asarray(q["labels"]).size != n:
            raise SystemExit("a sequence has another number of labels than references")
        starts = native_starts(k) if q["kind"] == "native" else eligible_starts(np.asarray(q["labels"])[w["centres"]])
        out.append({"kind": q["kind"], "n": n, "windows": w, "starts": starts})
    return out


def make_plan(info: list[dict], seed: int, total: int) -> dict:
    natives = [i for i, q in enumerate(info) if q["kind"] == "native"]; streams = [i for i, q in enumerate(info) if q["kind"] != "native" and len(q["starts"])]
    if not natives or not streams:
        raise SystemExit("no native clip or no eligible constructed stream")
    nc, ec = [len(info[i]["starts"]) for i in natives], [len(info[i]["starts"]) for i in streams]
    plan = {k: np.empty(int(total), np.int64) for k in ("native_seq", "native_start", "stream_seq", "stream_start")}
    for u in range(int(total)):
        a, b, c, d = draw(seed, u, nc, ec)
        plan["native_seq"][u], plan["native_start"][u] = natives[a], info[natives[a]]["starts"][b]
        plan["stream_seq"][u], plan["stream_start"][u] = streams[c], info[streams[c]]["starts"][d]
    return plan


def plan_stats(info: list[dict], plan: dict) -> dict:
    nat = [q for q in info if q["kind"] == "native"]; bank = [q for q in info if q["kind"] != "native"]
    return {"native_clips": len(nat), "native_starts": int(sum(len(q["starts"]) for q in nat)),
            "native_clips_with_repeats": int(sum(len(q["windows"]["centres"]) < GROUP for q in nat)),
            "streams": {"eligible": int(sum(len(q["starts"]) > 0 for q in bank)), "ineligible": int(sum(len(q["starts"]) == 0 for q in bank))},
            "eligible_starts": int(sum(len(q["starts"]) for q in bank)),
            "ineligible_starts": int(sum(max(0, len(q["windows"]["centres"]) - GROUP + 1) - len(q["starts"]) for q in bank)),
            "updates": int(len(plan["native_seq"]))}


# ----------------------------------------------------------------------------------------------- manifests
def build_manifest(seqs: list[dict], info: list[dict], plan: dict) -> dict:
    """V01's manifest layout for the plan: eight windows per update, each cut from its sequence's whole grid."""
    U = int(len(plan["native_seq"])); groups = []
    for u in range(U):
        g = []
        for qi, start in ((int(plan["native_seq"][u]), int(plan["native_start"][u])), (int(plan["stream_seq"][u]), int(plan["stream_start"][u]))):
            g.append((qi, group_indices(len(info[qi]["windows"]["centres"]), start)))
        groups.append(g)
    videos = sorted({seqs[qi]["refs"][int(x)][0] for g in groups for qi, idx in g for ci in idx for x in info[qi]["windows"]["samples"][int(ci)]})
    vix = {v: i for i, v in enumerate(videos)}
    m = {"vid": np.empty((U, WINDOWS, SLOTS), np.int32), "frame": np.empty((U, WINDOWS, SLOTS), np.int32), "gamma": np.empty((U, WINDOWS, SLOTS), np.float32),
         "s": np.empty((U, WINDOWS, SLOTS), np.int32), "c": np.empty((U, WINDOWS), np.int32), "target": np.empty((U, WINDOWS), np.int8),
         "kind": np.empty((U, WINDOWS), np.int8), "seq": np.empty((U, WINDOWS), np.int32), "centre_index": np.empty((U, WINDOWS), np.int32),
         "marker": np.empty((U, WINDOWS), np.int8), "repeats": np.zeros((U, WINDOWS), bool), "group_seq": np.empty((U, 2), np.int32),
         "group_start": np.empty((U, 2), np.int32)}
    for u, g in enumerate(groups):
        for gi, (qi, idx) in enumerate(g):
            q, W = seqs[qi], info[qi]["windows"]; labels = np.asarray(q["labels"])
            m["group_seq"][u, gi], m["group_start"][u, gi] = qi, int(idx[0])
            for t, ci in enumerate(idx):
                w = gi * GROUP + t; s = W["samples"][int(ci)]; c = int(W["centres"][int(ci)])
                refs = [q["refs"][int(x)] for x in s]
                m["vid"][u, w] = [vix[r[0]] for r in refs]; m["frame"][u, w] = [int(r[1]) for r in refs]; m["gamma"][u, w] = [float(r[2]) for r in refs]
                m["s"][u, w], m["c"][u, w], m["target"][u, w], m["kind"][u, w] = s, c, int(labels[c]), int(gi)
                m["seq"][u, w], m["centre_index"][u, w], m["marker"][u, w] = qi, int(ci), int(W["markers"][int(ci)])
                m["repeats"][u, w] = bool(t > 0 and int(ci) == int(idx[t - 1]))
    m["order_ordered"] = np.broadcast_to(np.arange(SLOTS, dtype=np.int8), (U, WINDOWS, SLOTS)).copy()
    m["videos"] = np.asarray(videos)
    return m


def arrays_digest(m: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(m):
        a = np.ascontiguousarray(m[k]); h.update(f"{k}|{a.dtype.str}|{a.shape};".encode()); h.update(a.tobytes() if a.dtype.kind != "U" else "|".join(a.tolist()).encode())
    return h.hexdigest()


def slot_refs(m: dict) -> set[tuple[str, int, float]]:
    return {(str(m["videos"][v]), int(f), round(float(g), 6)) for v, f, g in zip(m["vid"].ravel(), m["frame"].ravel(), m["gamma"].ravel())}


def window_digests(m: dict, hashes: dict) -> np.ndarray:
    """Per window, SHA256 over its 16 slot pixel digests in input order (the identity order of Ordered context)."""
    videos = m["videos"]; U = m["vid"].shape[0]; out = np.empty((U, WINDOWS, 32), np.uint8)
    for u in range(U):
        for w in range(WINDOWS):
            slot = [hashes[(str(videos[m["vid"][u, w, i]]), int(m["frame"][u, w, i]), round(float(m["gamma"][u, w, i]), 6))] for i in range(SLOTS)]
            out[u, w] = np.frombuffer(tm.window_digest([slot[j] for j in m["order_ordered"][u, w]]), np.uint8)
    return out


def epoch_digests(m: dict, upe: int) -> list[str]:
    out = []
    for e in range(m["vid"].shape[0] // int(upe)):
        h = hashlib.sha256(); h.update("|".join(map(str, m["videos"].tolist())).encode()); sl = slice(e * int(upe), (e + 1) * int(upe))
        for k in ("vid", "frame", "gamma", "s", "c", "marker", "target", "kind", "seq", "centre_index", "order_ordered", "digest_ordered"):
            h.update(np.ascontiguousarray(m[k][sl]).tobytes())
        out.append(h.hexdigest())
    return out


def _stats(x) -> dict:
    x = np.asarray(x, float)
    return {"min": float(x.min()), "mean": float(x.mean()), "max": float(x.max())} if x.size else {}


def manifest_counts(m: dict, info: list[dict], plan: dict, seqs: list[dict] | None = None) -> dict:
    """Class, run, transcript-length, repeat, skip and source-exposure counts of one manifest (descriptive only); the
    skipped reference runs need the sequences' labels."""
    U = m["vid"].shape[0]; nat_runs, con_runs, tl = [], [], Counter(); skipped = skipped_total(seqs, info, plan) if seqs is not None else None
    for u in range(U):
        nat_runs.append(len(so.reference_runs(m["target"][u, :GROUP].astype(np.int64)).transcript))
        r = so.reference_runs(m["target"][u, GROUP:].astype(np.int64)); con_runs.append(len(r.transcript)); tl[len(r.transcript)] += 1
    return {"windows": int(U * WINDOWS), "target_classes": {CLASSES[k]: int(v) for k, v in sorted(Counter(m["target"].ravel().tolist()).items())},
            "target_classes_constructed": {CLASSES[k]: int(v) for k, v in sorted(Counter(m["target"][:, GROUP:].ravel().tolist()).items())},
            "native_runs": _stats(nat_runs), "constructed_runs": _stats(con_runs), "transcript_lengths": {str(k): int(v) for k, v in sorted(tl.items())},
            "repeated_native_targets": int(m["repeats"][:, :GROUP].sum()), "skipped_reference_runs": skipped,
            "sources": {"native_clips_drawn": int(len(set(plan["native_seq"].tolist()))), "streams_drawn": int(len(set(plan["stream_seq"].tolist()))),
                        "max_draws_per_native_clip": int(max(Counter(plan["native_seq"].tolist()).values())),
                        "max_draws_per_stream": int(max(Counter(plan["stream_seq"].tolist()).values())), "recordings_in_windows": int(len(m["videos"]))}}


def skipped_total(seqs: list[dict], info: list[dict], plan: dict) -> int:
    return int(sum(skipped_runs(seqs[int(q)]["labels"], info[int(q)]["windows"]["centres"][group_indices(len(info[int(q)]["windows"]["centres"]), int(s))])
                   for q, s in zip(plan["stream_seq"], plan["stream_start"])))


# ----------------------------------------------------------------------------------------------- context and sources
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vso = cfg, config, cfg["video_sequence_objective"]
        self.paths = load_paths(); self.root = self.paths.repo_root; self.src = self.vso["source"]
        self.out = self.root / self.vso["outputs"]["root"]; self.results = self.root / self.vso["outputs"]["results_dir"]
        self.ledger = self.root / self.vso["budget"]["ledger"]; tr = self.vso["training"]
        self.seeds, self.total, self.upe = [int(s) for s in tr["seeds"]], int(tr["total_updates"]), int(tr["updates_per_epoch"])
        if self.upe * 10 != self.total:
            raise SystemExit("the epochs do not divide the total updates")
        self.rv = load_config(pinned(self.root, self.src["raw_video_config"]))["raw_video_context"]

    def a01(self):
        """A01's preparation context (its own pins checked); only its training-data path is used here."""
        path = self.src["a01_config"]["path"]; pinned(self.root, self.src["a01_config"])
        return pvft.Ctx(load_cfg(SimpleNamespace(config=path, set=[])), path)


def pinned(root: Path, pin: dict) -> Path:
    return pvft.pinned(root, pin)


def environment() -> dict:
    import pandas
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__, **prv.environment()}


def verify_sources(ctx: Ctx) -> dict:
    src = ctx.src; out = {"task_sheet": sha256_file(pinned(ctx.root, ctx.vso["task_sheet"]))}
    for k in PIN_KEYS:
        out[k] = sha256_file(pinned(ctx.root, src[k]))
    out["weights"] = {str(s): sha256_file(pinned(ctx.root, p)) for s, p in src["weights"].items()}
    out["protocol"] = sha256_file(pinned(ctx.root, src["recipes"]["protocol"]))
    out["archived_rows"] = {k: sha256_file(pinned(ctx.root, p)) for k, p in src["archived_rows"].items()}
    bad = [p for p, h in src["helpers"].items() if sha256_file(ctx.root / p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} reused helper file(s) differ from their pins")
    out["helpers"] = len(src["helpers"])
    env = environment(); want = src["environment"]
    if any(str(env[k]) != str(v) for k, v in want.items()):
        raise SystemExit(f"environment {env} differs from the pinned {want}")
    out["environment"] = env
    rv_src = ctx.rv["source"]
    if rv_src["protocol"] != src["recipes"]["protocol"]:
        raise SystemExit("the protocol pin differs from V01's")
    return out


def training_data(ctx: Ctx):
    """A01's allowlisted training sequences: protocol clips, labels of training rows only, the rebuilt bank."""
    data = pvft.TrainingData(ctx.a01()); rec = ctx.src["recipes"]
    if len(data.clips) != int(rec["training_clips"]) or len(data.recipes) != int(rec["bank_streams"]):
        raise SystemExit("the training clips or bank streams differ from the configured counts")
    if data.bank_digest != data.proto["train"]["multi_bank"]["digest"]:
        raise SystemExit("the rebuilt bank differs from the protocol's")
    return data


def build_all(ctx: Ctx, seqs: list[dict], info: list[dict], workers: int, dirs: dict) -> tuple[dict, dict, dict]:
    """Plans, manifests and pixel digests of every seed, from scratch."""
    plans = {s: make_plan(info, s, ctx.total) for s in ctx.seeds}; ms = {s: build_manifest(seqs, info, plans[s]) for s in ctx.seeds}
    refs = set().union(*(slot_refs(m) for m in ms.values())); t0 = time.time()
    hashes = prv.pixel_hashes(refs, dirs, workers); print(f"{len(hashes)} distinct frames digested ({time.time() - t0:.0f} s)", flush=True)
    for s, m in ms.items():
        m["digest_ordered"] = window_digests(m, hashes)
    return plans, ms, {"distinct_frames": len(hashes), "hash_table_digest": hashlib.sha256(b"".join(k[0].encode() + str(k[1:]).encode() + v
                                                                                                     for k, v in sorted(hashes.items()))).hexdigest()}


def ledger_append(path: Path, job: str, kind: str, started: str, seconds: float, ok: bool, note: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as fh:
        fh.write(json.dumps({"job": job, "kind": kind, "started": started, "finished": pvft.now(), "hours": float(seconds) / 3600, "ok": bool(ok), "note": note}) + "\n")


def run(ctx: Ctx, workers: int) -> None:
    t0, started, ok, note = time.time(), pvft.now(), False, ""
    try:
        rec = {"config": ctx.config, "config_sha256": sha256_file(ctx.root / ctx.config), "started": started, "sources": verify_sources(ctx)}
        data = training_data(ctx); seqs = data.seqs; ids = data.ids() | {session_of(v) for v in data.clips}
        info = sequence_info(seqs); rec["sequences"] = {"natives": len(data.clips), "bank_streams": len(data.recipes), "bank_digest": data.bank_digest}
        dirs = prv.frame_dirs(ctx.paths, ctx.rv, set(data.clips))
        if set(dirs) - set(prv.allowlist(ctx.rv, ctx.root)):
            raise SystemExit("a frame directory lies outside the allowlist")
        plans, ms, pix = build_all(ctx, seqs, info, workers, dirs); print("first pass built", flush=True)
        plans2, ms2, pix2 = build_all(ctx, seqs, info, workers, dirs); print("second pass built", flush=True)
        twice = {str(s): {"plan": all(np.array_equal(plans[s][k], plans2[s][k]) for k in plans[s]), "arrays": arrays_digest(ms[s]) == arrays_digest(ms2[s])} for s in ctx.seeds}
        if pix != pix2 or not all(v["plan"] and v["arrays"] for v in twice.values()):
            raise SystemExit(f"the second build differs from the first ({twice}, pixel tables equal {pix == pix2})")
        mdir = ctx.out / "manifests"; mdir.mkdir(parents=True, exist_ok=True); index = {"config": ctx.config, "seeds": {}, "pixels": pix}
        rec["plan_stats"] = plan_stats(info, plans[ctx.seeds[0]]); rec["seeds"] = {}
        for s in ctx.seeds:
            m = ms[s]; path = mdir / f"VSO_s{s}.npz"
            if path.is_file() and arrays_digest(dict(np.load(path))) != arrays_digest(m):
                raise SystemExit(f"{path.name} exists with other content")
            if not path.is_file():
                tmp = path.with_suffix(".tmp.npz"); np.savez(tmp, **m); tmp.replace(path)
            counts = manifest_counts(m, info, plans[s], seqs)
            e = {"file": path.name, "sha256": sha256_file(path), "arrays_digest": arrays_digest(m), "epoch_digests": epoch_digests(m, ctx.upe), "counts": counts}
            index["seeds"][str(s)] = e; rec["seeds"][str(s)] = {"sha256": e["sha256"], "arrays_digest": e["arrays_digest"], "epoch_digests": e["epoch_digests"], "counts": counts}
            print(f"seed {s}: {json.dumps(counts['sources'])}", flush=True)
        rec.update({"second_build_identical": twice, "pixels": pix, "finished": pvft.now(), "seconds": time.time() - t0})
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation"); assert_no_ids(json.loads(json.dumps(index, default=str)), ids, "index")
        (mdir / "index.json").write_text(json.dumps(index, indent=1, default=str))
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            pvft.write_atomic_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.set = []; run(Ctx(load_cfg(a), a.config), a.workers)


if __name__ == "__main__":
    main()
