#!/usr/bin/env python
"""Dense target-time inference of the frozen Ordered context models and a remaining-error audit
(docs/superpowers/plans/2026-10-12-dense-target-inference.md; configs/experiments/dense_target_inference_v1.yaml).
An implementation check of the same trained models, not a new model: no weight, readout, optimiser or update is added.

Every 10 Hz sample gets one model output three ways:
- Sparse: the source round's saved stride-5 window logits, each sample taking its nearest centre (ties to the earlier);
- Interpolated: the same window probabilities, linear between centres, the end values held outside them;
- Dense: a 16-frame window requested at every sample; the saved logits are reused at the old centres and only the other
  centres are inferred, so the three conditions share every old observation.
All three are decoded with each seed's inherited B3 numbers; scripts/dense_target_inference_tables.py does the report.

  --check  sources, Sparse reproduction, alignment, source parity, timing and the budget projection (a check record)
  --run    every new centre of three seeds and two halves, then the report; it needs a passed check with the same run
           identity and creates no training process

  python scripts/dense_target_inference.py --config configs/experiments/dense_target_inference_v1.yaml --check
  python scripts/dense_target_inference.py --config configs/experiments/dense_target_inference_v1.yaml --run
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.dense_target_inference_tables as dtt  # noqa: E402
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.train_joint_visual_temporal as tjv  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.evaluate.window_target as wt  # noqa: E402
import echo_routing.ingest.frames as ingest_frames  # noqa: E402
import echo_routing.temporal.joint_visual as jv  # noqa: E402
import echo_routing.temporal.windowed as windowed  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.features.extract import VARIANT_GAMMA  # noqa: E402

STRIDE, SIZE, FRAMES, BATCH = 5, 16, 16, 32
HALVES = ("selection", "calibration")
CONDITIONS = ("Sparse", "Interpolated", "Dense")
INFERENCE_KEYS = ("task_sheet", "source", "conditions", "windows", "inference", "determinism")
INFERENCE_FILES = (Path(__file__), Path(tm.__file__), Path(wt.__file__), Path(windowed.__file__), Path(jv.__file__), Path(ingest_frames.__file__),
                   Path(rr.__file__), Path(rcs.__file__), Path(prv.__file__), Path(tjv.__file__))


# ----------------------------------------------------------------------------------------------- centres, windows, mappings
def old_centres(n: int) -> np.ndarray:
    """The source round's centres: window_centres(n, 5)."""
    return windowed.window_centres(int(n), STRIDE)


def new_centres(n: int) -> np.ndarray:
    """Every sample that is not an old centre, in order."""
    return np.setdiff1d(np.arange(int(n)), old_centres(n))


def window_spec(n: int, c: int) -> tuple[np.ndarray, int]:
    """The 16 sample indices of the window requested at c and its marker slot (from s and c only)."""
    s = wt.window_frames(int(c), int(n), SIZE, FRAMES)
    return s, tm.marker_slot(s, int(c))


def _windows(seqs_h: list[dict], at: dict, centres_of) -> dict:
    seq, cen, ref, mk = [], [], [], []
    for qi, q in enumerate(seqs_h):
        n = len(q["refs"])
        for c in centres_of(n):
            s, m = window_spec(n, int(c))
            seq.append(qi); cen.append(int(c)); ref.append([at[q["refs"][i]] for i in s]); mk.append(m)
    return {"seq": np.asarray(seq, dtype=np.int64), "centre": np.asarray(cen, dtype=np.int64),
            "ref": np.asarray(ref, dtype=np.int64).reshape(-1, FRAMES), "marker": np.asarray(mk, dtype=np.int64)}


def sparse_windows(seqs_h: list[dict], at: dict) -> dict:
    """The source round's stride-5 windows of a half (sequence order, then centre order); reads references only."""
    return _windows(seqs_h, at, old_centres)


def dense_windows(seqs_h: list[dict], at: dict) -> dict:
    """A window at every sample of every sequence, with a flag for the old centres; reads references only."""
    W = _windows(seqs_h, at, np.arange)
    W["old"] = np.concatenate([np.isin(np.arange(len(q["refs"])), old_centres(len(q["refs"]))) for q in seqs_h]) if seqs_h else np.zeros(0, bool)
    return W


def softmax32(logits: np.ndarray) -> np.ndarray:
    """The source round's per-window softmax: torch, float32, over the last axis."""
    return torch.softmax(torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32)), -1).numpy()


def sparse_probs(n: int, centres: np.ndarray, wprob: np.ndarray) -> np.ndarray:
    """Every sample takes its nearest centre's probabilities (argmin, ties to the earlier centre)."""
    return np.asarray(wprob)[wt.nearest_centre(int(n), np.asarray(centres))]


def interp_probs(n: int, centres: np.ndarray, wprob: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Per class numpy.interp over the sample axis, the end values held outside the centres, cast to float32 once.
    Refuses negative values, row sums off 1 by more than tol, and any change at an old centre."""
    c = np.asarray(centres, dtype=np.int64); w = np.asarray(wprob, dtype=np.float32)
    if c.ndim != 1 or c.size == 0 or c.size != len(w) or np.any(np.diff(c) <= 0):
        raise ValueError("centres must be a non-empty increasing vector with one probability row each")
    x = np.arange(int(n), dtype=float)
    out = np.stack([np.interp(x, c.astype(float), w[:, k].astype(float)) for k in range(w.shape[1])], 1).astype(np.float32)
    if (out < 0).any() or np.abs(out.astype(float).sum(1) - 1.0).max() > tol or not np.array_equal(out[c], w):
        raise ValueError("interpolated rows are not probabilities, or an old centre changed")
    return out


def assemble(n: int, old_c: np.ndarray, old_v: np.ndarray, new_c: np.ndarray, new_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows by sample: the old centres' values unchanged, the new centres' values; exactly one row per sample."""
    old_c, new_c = np.asarray(old_c, dtype=np.int64), np.asarray(new_c, dtype=np.int64)
    if len(old_c) != len(old_v) or len(new_c) != len(new_v):
        raise ValueError("one value row per centre")
    both = np.concatenate([old_c, new_c])
    if both.size != int(n) or not np.array_equal(np.sort(both), np.arange(int(n))):
        raise ValueError("every sample needs exactly one output")
    out = np.empty((int(n), np.asarray(old_v).shape[1]), np.float32); out[old_c] = old_v; out[new_c] = new_v
    is_old = np.zeros(int(n), bool); is_old[old_c] = True
    return out, is_old


def bounds(seq: np.ndarray, n_seqs: int) -> np.ndarray:
    """Row boundaries of each sequence in a window table sorted by sequence."""
    return np.searchsorted(np.asarray(seq), np.arange(n_seqs + 1))


def condition_probs(seqs_h: list[dict], O: dict, W: dict, old_logits: np.ndarray, dense_logits: np.ndarray) -> dict:
    """{condition: {sequence id: (n, 5) float32}}. Dense takes the old softmax rows at the old centres; all three
    conditions must agree there bit for bit."""
    wprob = softmax32(old_logits); dprob = np.empty_like(dense_logits)
    dprob[W["old"]] = wprob; dprob[~W["old"]] = softmax32(dense_logits[~W["old"]])
    ob, wb = bounds(O["seq"], len(seqs_h)), bounds(W["seq"], len(seqs_h)); out = {c: {} for c in CONDITIONS}
    for qi, q in enumerate(seqs_h):
        n = len(q["refs"]); o, w = slice(ob[qi], ob[qi + 1]), slice(wb[qi], wb[qi + 1]); cen = O["centre"][o]
        if not (np.array_equal(cen, old_centres(n)) and np.array_equal(W["centre"][w], np.arange(n))):
            raise SystemExit("a sequence's windows are not in sequence and centre order")
        out["Sparse"][q["id"]] = sparse_probs(n, cen, wprob[o]); out["Interpolated"][q["id"]] = interp_probs(n, cen, wprob[o])
        out["Dense"][q["id"]], _ = assemble(n, cen, wprob[o], new_centres(n), dprob[w][~W["old"][w]])
        if not all(np.array_equal(out[c][q["id"]][cen], wprob[o]) for c in CONDITIONS):
            raise SystemExit("the three conditions differ at an old centre")
    return out


def pinned(root: Path, pin: dict) -> Path:
    """The pinned file, refused when it is missing or its sha256 differs."""
    path = Path(root) / pin["path"]
    if not path.is_file():
        raise SystemExit(f"{pin['path']}: missing")
    if (got := sha256_file(path)) != pin["sha256"]:
        raise SystemExit(f"{pin['path']}: sha256 {got[:12]} differs from the pin {pin['sha256'][:12]}")
    return path


# ----------------------------------------------------------------------------------------------- context and identity
class Ctx:
    """This round's configuration and the source round's (read only); outputs go to this round's root only."""

    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.dt = cfg, config, cfg["dense_target_inference"]
        self.paths = load_paths(); self.root = self.paths.repo_root; src = self.dt["source"]
        pinned(self.root, self.dt["task_sheet"])
        self.rv = load_config(pinned(self.root, src["config"]))["raw_video_context"]
        if (src["protocol"] != self.rv["source"]["protocol"] or src["echoprime_checkpoint"] != self.rv["source"]["echoprime"]["checkpoint"]
                or src["e1_feature_manifest"] != self.rv["evaluation"]["reference_inputs"]["e1_feature_manifest"]):
            raise SystemExit("a source pin differs from the source configuration's")
        self.seeds = [int(s) for s in src["models"]]
        self.out = self.root / self.dt["outputs"]["root"]; self.results = self.root / self.dt["outputs"]["results_dir"]
        self.b3 = {s: {"half_window": int(self.dt["b3_fixed"]["half_window"]), "threshold": float(self.dt["b3_fixed"]["thresholds"][s]),
                       "nms_radius": int(self.dt["b3_fixed"]["nms_radius"])} for s in self.seeds}
        if not bool(self.dt["b3_fixed"]["require_label_change"]):
            raise SystemExit("B3 keeps the persistent label change of the source round")

    def verify_sources(self) -> dict:
        """Every pin, the frozen models' identities and final digests, the inherited B3 numbers and the environment."""
        src = self.dt["source"]
        for pin in (*src["report"].values(), *(src[k] for k in ("protocol", "e1_feature_manifest", "window_plans", "echoprime_checkpoint"))):
            pinned(self.root, pin)
        stored = json.loads((self.root / src["report"]["summary"]["path"]).read_text())["systems"]
        for s in self.seeds:
            m = src["models"][s]
            for k in ("final", "init", "run_manifest"):
                pinned(self.root, m[k])
            man = json.loads((self.root / m["run_manifest"]["path"]).read_text())
            if man["identity"] != m["identity"] or man["final_sha256"] != m["final"]["sha256"] or int(man["updates"]) != int(self.rv["training"]["total_updates"]):
                raise SystemExit(f"ordered_s{s}: the frozen run manifest does not carry the pinned identity, weights and updates")
            for h in HALVES:
                pinned(self.root, src["outputs"][h][s])
            p = stored[f"ordered_s{s}"]["b3_choice"]
            if p["status"] != "feasible" or {k: p["params"][k] for k in ("half_window", "threshold", "nms_radius")} != self.b3[s]:
                raise SystemExit(f"ordered_s{s}: the inherited B3 numbers differ from the source round's chosen setting")
        env = prv.environment(); want = src["environment"]
        if any(str(env[k]) != str(want[k]) for k in want):
            raise SystemExit(f"environment {env} differs from the pinned {want}")
        return env


def file_digest(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).name.encode()); h.update(Path(p).read_bytes())
    return h.hexdigest()


def inference_identity(ctx: Ctx, seed: int) -> str:
    """Keys the saved outputs of one model: sources, windows, inference, determinism, environment and the full files of
    the entry script and the modules that build its windows, frames and model; not the report code."""
    payload = {"experiment": ctx.cfg.get("experiment"), "dense_target_inference": {k: ctx.dt[k] for k in INFERENCE_KEYS}, "seed": int(seed),
               "environment": prv.environment(), "code": file_digest(INFERENCE_FILES)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run_identity(ctx: Ctx, pixel_digest: str) -> str:
    """The inference identities, the whole configuration, the environment, the decoded pixels and the report code."""
    payload = {"inference": {s: inference_identity(ctx, s) for s in ctx.seeds}, "config": ctx.dt, "environment": prv.environment(),
               "pixels": pixel_digest, "report_code": file_digest(dtt.report_files())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- data, frames, sources
def load_data(ctx: Ctx) -> tuple:
    """The two halves with every sequence's frame references (B7's window plans must agree) and the frame index."""
    src = ctx.dt["source"]
    inputs, seqs = rcs.validation(ctx.cfg, {"inputs": {"protocol": src["protocol"], "feature_manifest": src["e1_feature_manifest"]}}, ctx.paths)
    refs = sorted({r for h in HALVES for q in seqs[h] for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}
    return inputs, seqs, refs, at


def decode_all(ctx: Ctx, refs: list[tuple], workers: int) -> tuple[np.ndarray, float, str]:
    """Every referenced frame with the source round's decoder, its seconds and a digest of all pixels in reference order."""
    fp = tjv.FramePaths(prv.frame_dirs(ctx.paths, ctx.rv, {r[0] for r in refs}))
    t0 = time.time(); frames = rr.decode_frames(refs, fp, workers); secs = time.time() - t0
    h = hashlib.sha256()
    for f in frames:
        h.update(tm.frame_hash(f))
    return frames, secs, h.hexdigest()


def old_outputs(ctx: Ctx, seed: int, half: str, seqs_h: list[dict], O: dict) -> dict:
    """The saved stride-5 outputs of one model and half; identity, sequences, centres, markers and order must equal the
    source run's and this round's sparse windows."""
    path = pinned(ctx.root, ctx.dt["source"]["outputs"][half][seed])
    with np.load(path) as z:
        same = (str(z["identity"]) == ctx.dt["source"]["models"][seed]["identity"] and z["sequence_ids"].tolist() == [q["id"] for q in seqs_h]
                and np.array_equal(z["seq"], O["seq"]) and np.array_equal(z["centre"], O["centre"]) and np.array_equal(z["marker"], O["marker"])
                and np.array_equal(z["order"], np.tile(np.arange(FRAMES, dtype=np.int8), (len(O["seq"]), 1))))
        if not same:
            raise SystemExit(f"ordered_s{seed} {half}: the saved windows differ from the source rule")
        return {"logits": z["logits"].astype(np.float32), "offsets": z["sample_offsets"].copy(), "probs": z["sample_probs"].copy()}


def id_set(ctx: Ctx, seqs: dict) -> set:
    """Recordings (training and development), their sessions and the sequence ids themselves, for the id guards."""
    train = set(prv.allowlist(ctx.rv, ctx.root)); dev = {r[0] for h in HALVES for q in seqs[h] for r in q["refs"]}
    return train | dev | {session_of(v) for v in train | dev} | {q["id"] for h in HALVES for q in seqs[h]}


# ----------------------------------------------------------------------------------------------- model and inference
def setup_gpu() -> torch.device:
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest"); torch.use_deterministic_algorithms(True, warn_only=True)
    return device


def build_model(ctx: Ctx, seed: int, device) -> tm.TargetMViT:
    """The frozen Ordered model as the source evaluation built it, in eval mode with no trainable parameter."""
    m = ctx.dt["source"]["models"][seed]
    init = torch.load(pinned(ctx.root, m["init"]), map_location="cpu", weights_only=True)
    model = tm.build_model(pinned(ctx.root, ctx.dt["source"]["echoprime_checkpoint"]), init)
    model.load_state_dict(torch.load(pinned(ctx.root, m["final"]), map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def state_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def logits_of(model, ref: np.ndarray, marker: np.ndarray, frames: np.ndarray, device, batch: int = BATCH) -> tuple[np.ndarray, float]:
    """Logits of windows given by frame indices (time order) and marker slots, in the given order and batch size."""
    out = []; torch.cuda.synchronize(); t0 = time.time()
    for b in range(0, len(ref), batch):
        u8 = torch.from_numpy(frames[ref[b:b + batch]]).to(device, non_blocking=True)
        out.append(model(tm.to_rgb(u8), tm.marker_planes(torch.from_numpy(marker[b:b + batch]).long().to(device))).float().cpu().numpy())
    torch.cuda.synchronize()
    return (np.concatenate(out) if out else np.zeros((0, tm.NUM_CLASSES), np.float32)), time.time() - t0


def cache_path(ctx: Ctx, seed: int, half: str) -> Path:
    return ctx.out / "inference" / half / f"ordered_s{seed}.npz"


def save_cache(path: Path, ident: str, pixel: str, W: dict, logits: np.ndarray, probs: dict, seqs_h: list[dict], seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); ids = [q["id"] for q in seqs_h]; tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, identity=ident, pixel_digest=pixel, sequence_ids=np.asarray(ids), seq=W["seq"], centre=W["centre"], marker=W["marker"],
                 ref=W["ref"], old=W["old"], provenance=np.where(W["old"], "reused", "new"), logits=logits.astype(np.float32),
                 sample_offsets=np.cumsum([0, *[len(q["refs"]) for q in seqs_h]]), seconds=seconds,
                 **{c.lower(): np.concatenate([probs[c][i] for i in ids]).astype(np.float32) for c in CONDITIONS})
    tmp.replace(path)


def load_cache(path: Path, ident: str, pixel: str, W: dict, seqs_h: list[dict]) -> tuple[np.ndarray, float] | None:
    """Saved dense logits, reused only when the inference identity, pixels, sequences and every window array match."""
    if not path.is_file():
        return None
    with np.load(path) as z:
        same = (str(z["identity"]) == ident and str(z["pixel_digest"]) == pixel and z["sequence_ids"].tolist() == [q["id"] for q in seqs_h]
                and all(np.array_equal(z[k], W[k]) for k in ("seq", "centre", "marker", "ref", "old")))
        return (z["logits"].astype(np.float32), float(z["seconds"])) if same else None


# ----------------------------------------------------------------------------------------------- checks
def alignment(inputs, seqs: dict, W: dict, O: dict, refs: list[tuple]) -> dict:
    """Every dense window: 16 slots inside the sequence, ordered, the target slots contain c, the marker is their middle,
    the marked frame is the sequence's 10 Hz reference of c; old centres equal the sparse windows; native frames equal
    E1's cache; stream samples carry their recipe fragment's recording and gamma. Raises on any failure."""
    out = {}
    for h in HALVES:
        Wh, ev = W[h], inputs.halves[h]; wb = bounds(Wh["seq"], len(seqs[h])); bad = []
        frag = {st.recipe.recipe_id: st.recipe.fragments for st in (*ev.pairs, *ev.multi)}
        e1 = {s.seq_id: inputs.store.get(s.seq_id, "orig").frame_idx for s in ev.natives}
        short = clamped = crossing = 0
        for qi, q in enumerate(seqs[h]):
            n = len(q["refs"]); rows = range(wb[qi], wb[qi + 1])
            if list(Wh["centre"][wb[qi]:wb[qi + 1]]) != list(range(n)):
                bad.append("centres")
            short += int(n < SIZE)
            if q["kind"] == "native":
                if len(e1[q["id"]]) != n or any(int(e1[q["id"]][c]) != int(q["refs"][c][1]) or q["refs"][c][0] != q["id"] for c in range(n)):
                    bad.append("native frames")
            else:
                want = [(fr.video_id, round(float(VARIANT_GAMMA[fr.variant]), 6)) for fr in frag[q["id"]] for _ in range(fr.start, fr.end)]
                if want != [(r[0], round(float(r[2]), 6)) for r in q["refs"]]:
                    bad.append("stream recording or gamma")
            for i in rows:
                c = int(Wh["centre"][i]); r = [refs[j] for j in Wh["ref"][i]]; m = int(Wh["marker"][i])
                start = max(0, min(c - SIZE // 2, n - SIZE))
                s = wt.window_frames(c, n, SIZE, FRAMES)
                slots = np.flatnonzero(s == c)
                ok = (len(r) == FRAMES and slots.size > 0 and m == int(slots[slots.size // 2]) and r[m] == q["refs"][c]
                      and all(r[k] == q["refs"][int(s[k])] for k in range(FRAMES)) and np.all(np.diff(s) >= 0)
                      and int(s[0]) == start and int(s[-1]) == min(n, start + SIZE) - 1
                      and (n < SIZE or np.array_equal(s, np.arange(start, start + SIZE)))
                      and (n >= SIZE or set(s.tolist()) == set(range(n))))
                if not ok:
                    bad.append(f"window {c}")
                clamped += int(m != SIZE // 2); crossing += int(len({x[0] for x in r}) > 1 or len({round(float(x[2]), 6) for x in r}) > 1)
        if not all(np.array_equal(O[h][k], Wh[k][Wh["old"]]) for k in ("seq", "centre", "ref", "marker")):
            bad.append("old windows")
        if bad:
            raise SystemExit(f"{h}: alignment failed ({len(bad)} problems, first {bad[:3]})")
        out[h] = {"sequences": len(seqs[h]), "windows": int(len(Wh["seq"])), "old_windows": int(Wh["old"].sum()), "new_windows": int((~Wh["old"]).sum()),
                  "short_sequences": short, "windows_with_marker_off_slot_8": clamped, "windows_crossing_a_join_or_gamma": crossing}
    return out


def sparse_reproduction(ctx: Ctx, inputs, seqs: dict, O: dict) -> dict:
    """Every seed and half: Sparse per-sample probabilities bit-equal to the saved ones, and every stored argmax and B3
    row equal field by field. Also times B3 decoding for the budget."""
    stored = json.loads((ctx.root / ctx.dt["source"]["report"]["summary"]["path"]).read_text())["systems"]; out = {}; t_b3 = 0.0
    for s in ctx.seeds:
        for h in HALVES:
            old = old_outputs(ctx, s, h, seqs[h], O[h]); wprob = softmax32(old["logits"]); ob = bounds(O[h]["seq"], len(seqs[h]))
            P = {q["id"]: sparse_probs(len(q["refs"]), O[h]["centre"][ob[i]:ob[i + 1]], wprob[ob[i]:ob[i + 1]]) for i, q in enumerate(seqs[h])}
            saved = np.concatenate([P[q["id"]] for q in seqs[h]])
            if saved.shape != old["probs"].shape or not np.array_equal(saved, old["probs"]):
                raise SystemExit(f"ordered_s{s} {h}: Sparse probabilities differ from the saved ones")
            t0 = time.time(); b3 = decode(ctx.b3[s], P); t_b3 += time.time() - t0; fields = 0
            for row, preds in (("argmax", {k: v.argmax(1) for k, v in P.items()}), ("b3", b3)):
                m, _ = rcs.evaluate(ctx.cfg, preds, inputs.halves[h]); got = dtt.system_row(m); want = stored[f"ordered_s{s}"][row][h]
                diff = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
                if diff:
                    raise SystemExit(f"ordered_s{s} {row} {h}: the stored row is not reproduced ({sorted(diff)})")
                fields += len(want)
            out[f"s{s}|{h}"] = {"samples": int(saved.shape[0]), "probabilities": "bit-equal", "row_fields_equal": fields}
    return {"rows": out, "b3_seconds_per_decode": t_b3 / (len(ctx.seeds) * len(HALVES))}


def parity_rows(seqs_h: list[dict], O: dict) -> np.ndarray:
    """The first 8 old windows of native clips and the first 8 of multi streams, in sequence order."""
    kind = np.array([seqs_h[i]["kind"] for i in O["seq"]])
    return np.r_[np.flatnonzero(kind == "native")[:8], np.flatnonzero(kind == "multi")[:8]]


def source_parity(ctx: Ctx, seqs: dict, O: dict, frames: np.ndarray, device) -> dict:
    """Re-inferred old windows against the saved logits, every seed (max |difference| and argmax), with an unchanged state."""
    rows, tol, out = parity_rows(seqs["selection"], O["selection"]), float(ctx.dt["checks"]["parity_max_abs"]), {}
    for s in ctx.seeds:
        old = old_outputs(ctx, s, "selection", seqs["selection"], O["selection"]); model = build_model(ctx, s, device); before = state_digest(model)
        lg, _ = logits_of(model, O["selection"]["ref"][rows], O["selection"]["marker"][rows], frames, device)
        d = float(np.abs(lg - old["logits"][rows]).max()); same = bool(np.array_equal(lg.argmax(1), old["logits"][rows].argmax(1)))
        out[f"s{s}"] = {"windows": int(rows.size), "max_abs": d, "argmax_equal": same, "state_unchanged": state_digest(model) == before}
        del model; torch.cuda.empty_cache()
        if d > tol or not same or not out[f"s{s}"]["state_unchanged"]:
            raise SystemExit(f"source parity failed for seed {s}: {out[f's{s}']}")
    return out


def time_new_windows(ctx: Ctx, Wsel: dict, frames: np.ndarray, device) -> dict:
    """Seed 0, selection: one warm-up batch, then ten timed batches of new windows in the fixed order."""
    t = ctx.dt["checks"]["timing"]; new = np.flatnonzero(~Wsel["old"])[:BATCH * (int(t["warmup_batches"]) + int(t["timed_batches"]))]
    warm = BATCH * int(t["warmup_batches"]); model = build_model(ctx, ctx.seeds[0], device)
    logits_of(model, Wsel["ref"][new[:warm]], Wsel["marker"][new[:warm]], frames, device)
    _, sec = logits_of(model, Wsel["ref"][new[warm:]], Wsel["marker"][new[warm:]], frames, device)
    del model; torch.cuda.empty_cache()
    return {"timed_windows": int(new.size - warm), "seconds": sec, "s_per_window": sec / max(1, new.size - warm)}


# ----------------------------------------------------------------------------------------------- ledger and budget
def ledger_path(ctx: Ctx) -> Path:
    return ctx.out / "ledger.jsonl"


def ledger_hours(ctx: Ctx) -> float:
    p = ledger_path(ctx)
    return sum(float(json.loads(x)["hours"]) for x in p.read_text().splitlines() if x.strip()) if p.is_file() else 0.0


def ledger(ctx: Ctx, job: str, started: str, seconds: float, ok: bool, identity: str | None, note: str = "") -> None:
    ledger_path(ctx).parent.mkdir(parents=True, exist_ok=True)
    line = {"job": job, "started": started, "finished": now(), "hours": seconds / 3600, "ok": bool(ok), "run_identity": identity, "note": note}
    with ledger_path(ctx).open("a") as f:
        f.write(json.dumps(line) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def projection(ctx: Ctx, s_per_window: float, new_windows: int, decode_s: float, b3_s: float, elapsed_s: float) -> dict:
    """Ledger hours + this job so far + all remaining new windows + frame and B3 decoding + the reserve, against the cap."""
    b = ctx.dt["budget"]; n_decodes = len(CONDITIONS) * len(ctx.seeds) * len(HALVES) * 2
    parts = {"spent": ledger_hours(ctx) + elapsed_s / 3600, "inference": new_windows * s_per_window / 3600, "frame_decoding": decode_s / 3600,
             "b3_decoding": n_decodes * b3_s / 3600, "reserve": float(b["reserve_hours"])}
    total = sum(parts.values())
    return {"hours": parts, "total": total, "cap": float(b["gpu_hours_max"]), "within_cap": total <= float(b["gpu_hours_max"])}


# ----------------------------------------------------------------------------------------------- jobs
def run_check(ctx: Ctx, workers: int) -> None:
    t0, started, rec = time.time(), now(), {"passed": False}
    try:
        device = setup_gpu(); env = ctx.verify_sources(); inputs, seqs, refs, at = load_data(ctx)
        frames, dec_s, pixel = decode_all(ctx, refs, workers); rec["run_identity"] = run_identity(ctx, pixel)
        O = {h: sparse_windows(seqs[h], at) for h in HALVES}; W = {h: dense_windows(seqs[h], at) for h in HALVES}
        align = alignment(inputs, seqs, W, O, refs); repro = sparse_reproduction(ctx, inputs, seqs, O)
        parity = source_parity(ctx, seqs, O, frames, device); timing = time_new_windows(ctx, W["selection"], frames, device)
        new = sum(align[h]["new_windows"] for h in HALVES) * len(ctx.seeds)
        proj = projection(ctx, timing["s_per_window"], new, dec_s, repro["b3_seconds_per_decode"], time.time() - t0)
        rec.update({"passed": bool(proj["within_cap"]), "started": started, "finished": now(), "inference_identity": {s: inference_identity(ctx, s) for s in ctx.seeds},
                    "pixel_digest": pixel, "frames": len(refs), "frame_decode_seconds": dec_s, "environment": env, "alignment": align,
                    "sparse_reproduction": repro, "parity": parity, "timing": timing, "new_windows_all_seeds": new, "projection": proj})
        write_record(ctx, rec, id_set(ctx, seqs))
        print(json.dumps({k: rec[k] for k in ("passed", "alignment", "parity", "timing", "projection")}, indent=1, default=str), flush=True)
        if not rec["passed"]:
            raise SystemExit("the projected GPU hours exceed the cap: nothing runs")
    finally:
        ledger(ctx, "check", started, time.time() - t0, rec["passed"], rec.get("run_identity"))


def write_record(ctx: Ctx, rec: dict, ids: set) -> None:
    from scripts.prepare_fold_output_correction import assert_no_ids
    flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "check")
    for d in (ctx.out / "checks", ctx.results):
        d.mkdir(parents=True, exist_ok=True); tmp = d / "checks.json.tmp"; tmp.write_text(json.dumps(flat, indent=1)); tmp.replace(d / "checks.json")


def run_full(ctx: Ctx, workers: int) -> None:
    t0, started, ok, rid = time.time(), now(), False, None
    try:
        rec_path = ctx.out / "checks" / "checks.json"
        check = json.loads(rec_path.read_text()) if rec_path.is_file() else {}
        if not check.get("passed"):
            raise SystemExit("no passed check: run --check first")
        device = setup_gpu(); ctx.verify_sources(); inputs, seqs, refs, at = load_data(ctx)
        frames, dec_s, pixel = decode_all(ctx, refs, workers); rid = run_identity(ctx, pixel)
        if rid != check["run_identity"]:
            raise SystemExit("the run identity differs from the passed check's: check again")
        O = {h: sparse_windows(seqs[h], at) for h in HALVES}; W = {h: dense_windows(seqs[h], at) for h in HALVES}
        new_total = sum(int((~W[h]["old"]).sum()) for h in HALVES) * len(ctx.seeds); done = 0
        probs, seconds = {}, {}
        for s in ctx.seeds:
            ident, model, finals = inference_identity(ctx, s), None, sha256_file(ctx.root / ctx.dt["source"]["models"][s]["final"]["path"])
            probs[s] = {}
            for h in HALVES:
                proj = projection(ctx, check["timing"]["s_per_window"], new_total - done, 0.0, 0.0, time.time() - t0)
                if not proj["within_cap"]:
                    raise SystemExit(f"stopping before ordered_s{s} {h}: projected {proj['total']:.2f} GPU hours")
                old = old_outputs(ctx, s, h, seqs[h], O[h]); path = cache_path(ctx, s, h); cached = load_cache(path, ident, pixel, W[h], seqs[h])
                if cached is None:
                    model = model if model is not None else build_model(ctx, s, device); before = state_digest(model); rows = np.flatnonzero(~W[h]["old"])
                    new_lg, sec = logits_of(model, W[h]["ref"][rows], W[h]["marker"][rows], frames, device)
                    if state_digest(model) != before:
                        raise SystemExit(f"ordered_s{s}: the model state changed during inference")
                    lg = np.empty((len(W[h]["seq"]), tm.NUM_CLASSES), np.float32); lg[W[h]["old"]] = old["logits"]; lg[rows] = new_lg
                    probs[s][h] = condition_probs(seqs[h], O[h], W[h], old["logits"], lg); save_cache(path, ident, pixel, W[h], lg, probs[s][h], seqs[h], sec)
                else:
                    lg, sec = cached
                    if not np.array_equal(lg[W[h]["old"]], old["logits"]):
                        raise SystemExit(f"ordered_s{s} {h}: the cached logits differ from the saved ones at an old centre")
                    probs[s][h] = condition_probs(seqs[h], O[h], W[h], old["logits"], lg)
                seconds[f"s{s}|{h}"] = sec; done += int((~W[h]["old"]).sum())
                print(f"ordered_s{s} {h}: {int((~W[h]['old']).sum())} new windows ({sec:.0f} s)", flush=True)
            if sha256_file(ctx.root / ctx.dt["source"]["models"][s]["final"]["path"]) != finals:
                raise SystemExit(f"ordered_s{s}: the frozen weights changed")
            del model; torch.cuda.empty_cache()
        del frames
        ids = id_set(ctx, seqs); a = check["alignment"]
        counts = {h: {"sequences": {k: int(sum(q["kind"] == k for q in seqs[h])) for k in ("native", "pairs", "multi")},
                      "samples": int(sum(len(q["refs"]) for q in seqs[h])), **{k: a[h][k] for k in ("old_windows", "new_windows", "short_sequences",
                                                                                                  "windows_with_marker_off_slot_8", "windows_crossing_a_join_or_gamma")}}
                  for h in HALVES}
        cost = {"new_windows": new_total, "inference_seconds": seconds, "ms_per_new_window": 1000 * sum(seconds.values()) / max(1, new_total),
                "frame_decode_seconds": dec_s, "ledger_hours_before_run": ledger_hours(ctx)}
        stored = json.loads((ctx.root / ctx.dt["source"]["report"]["summary"]["path"]).read_text())["systems"]
        dtt.run_report(ctx.cfg, ctx.dt, ctx.b3, inputs, seqs, probs, counts, cost, ctx.out, ctx.results, ids, t0, stored)
        ok = True
    finally:
        ledger(ctx, "run", started, time.time() - t0, ok, rid)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=6)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--check", action="store_true"); g.add_argument("--run", action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    run_check(ctx, a.workers) if a.check else run_full(ctx, a.workers)


if __name__ == "__main__":
    main()
