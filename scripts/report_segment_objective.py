#!/usr/bin/env python
"""Evaluate the segment objective round (docs/superpowers/plans/2026-10-14-segment-objective-training.md, Task 4;
configs/experiments/segment_objective_v1.yaml). Runs only after the nine trainings are finished and frozen.

Development data come from A03's metadata-only helpers (half_index, meta_stream, half_sets: one half's protocol
entries, the clip index rows of that half's clips and the pinned window plans) and from a read-only adapter of A01's
development feature caches. A01's cache holds both halves; the adapter checks that every member is stored without
compression, parses the NPY header and copies only the open half's rows by bounded seeks. No video encoder is built and
no eager development loader is called (a guard makes them raise). Labels and transcripts never enter a head: the heads
see the features alone, and MS-TCN and ASRF see the features with placeholder labels of the valid length.

  --calibrate   the calibration half only: parity of the source classifier against V01's saved logits, A01's stored
                frozen Ordered, Linear, MS-TCN and ASRF rows reproduced, and one B3 choice per new head (40 settings)
  --dry-run     a code check of the selection pipeline before the choices are committed; calibration stands in for
                both halves and nothing reaches the results
  --proof       (local) the git-object proof that a commit holds exactly the choice file; copied to the server
  --selection   with --choices-sha256 and --choices-commit: the barrier (committed file, proof, provenance), then the
                selection half opened once, the systems, comparisons, accounting, goal and readings

  python scripts/report_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --calibrate
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import base64  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from collections import namedtuple  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.dense_target_inference_tables as dtt  # noqa: E402
import scripts.prepare_video_feature_temporal as pv  # noqa: E402
import scripts.raw_video_context_tables as rt  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_segment_objective as ts  # noqa: E402
import scripts.train_video_feature_temporal as tv  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import write_atomic  # noqa: E402
from scripts.trained_token_readout_tables import check_choice_record  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from echo_routing.evaluate.complementarity import select_setting, sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_data import Seq  # noqa: E402

NS = "segment_objective_v1"
HALVES = st.HALVES
ARMS = ts.ARMS
ARM_NAME = {"frame": "F", "segment": "B", "order": "O"}
DIAG = {"evaluation": {"b3_infeasible_diagnostic": {"half_window": 9, "quantile": 0.95}}}
CMP = {"b3": {"O - B": ("O + B3", "B + B3"), "B - F": ("B + B3", "F + B3"), "O - F": ("O + B3", "F + B3"),
              "O - frozen Ordered": ("O + B3", "frozen Ordered + B3"), "B - frozen Ordered": ("B + B3", "frozen Ordered + B3"),
              "F - A01 Linear": ("F + B3", "A01 Linear + B3"), "F - frozen Ordered": ("F + B3", "frozen Ordered + B3")},
       "argmax": {"O - B": ("O argmax", "B argmax"), "B - F": ("B argmax", "F argmax"), "O - F": ("O argmax", "F argmax")},
       "ordered_setting": {"O - B": ("O at Ordered setting", "B at Ordered setting"), "B - F": ("B at Ordered setting", "F at Ordered setting"),
                           "O - F": ("O at Ordered setting", "F at Ordered setting")}}
PAIRS = {"O vs B": ("O + B3", "B + B3"), "O vs F": ("O + B3", "F + B3"), "O vs frozen Ordered": ("O + B3", "frozen Ordered + B3"),
         "B vs F": ("B + B3", "F + B3"), "B vs frozen Ordered": ("B + B3", "frozen Ordered + B3"), "F vs frozen Ordered": ("F + B3", "frozen Ordered + B3"),
         "F vs A01 Linear": ("F + B3", "A01 Linear + B3")}
REFERENCE_ROWS = {"frozen Ordered + B3": "frozen Ordered + B3", "frozen Ordered argmax": "frozen Ordered argmax", "A01 Linear + B3": "Linear + B3",
                  "A01 Linear argmax": "Linear argmax", "A01 MS-TCN": "MS-TCN", "A01 ASRF": "ASRF"}


def _repo_modules() -> list[str]:
    """The repository's own modules (scripts/, echo_routing/) loaded together with this one."""
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        q = Path(f).resolve()
        if q.is_relative_to(ROOT) and q.relative_to(ROOT).parts[0] in ("scripts", "echo_routing"):
            out.add(q.relative_to(ROOT).as_posix())
    return sorted(out)


IMPORTED = _repo_modules()            # fixed when this module is imported
HELPER_FILES = ("scripts/freeze_model.py", "third_party/ms-tcn/model.py")


def code_digest() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    for f in sorted(set(IMPORTED) | set(HELPER_FILES)):
        q = ROOT / f
        h.update(f.encode() + b"\0" + (q.read_bytes() if q.is_file() else b"missing"))
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- the bounded cache reader
Layout = namedtuple("Layout", "offset dtype shape")


def _read_exact(f, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = f.read(n - len(out))
        if not chunk:
            raise SystemExit("a bounded read ended before the requested bytes")
        out += chunk
    return bytes(out)


def member_layout(path, member: str, opener=open) -> Layout:
    """The data offset, dtype and shape of one stored NPY member; compressed, encrypted or other layouts stop here."""
    name = Path(path).name
    with opener(path, "rb") as f:
        try:
            with zipfile.ZipFile(f) as zf:
                info = zf.getinfo(f"{member}.npy")
        except (zipfile.BadZipFile, KeyError) as e:
            raise SystemExit(f"{name}: no readable member {member} ({type(e).__name__})") from None
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1 or info.compress_size != info.file_size:
            raise SystemExit(f"{name}: member {member} is compressed or encrypted; there is no eager fallback")
        f.seek(info.header_offset); sig, _, _, comp, _, _, _, _, _, nlen, elen = struct.unpack("<IHHHHHIIIHH", _read_exact(f, 30))
        if sig != 0x04034B50 or comp != 0:
            raise SystemExit(f"{name}: member {member} has an unexpected local header")
        start = info.header_offset + 30 + nlen + elen; f.seek(start); version = np.lib.format.read_magic(f)
        if version not in ((1, 0), (2, 0)):
            raise SystemExit(f"{name}: member {member} has NPY version {version}")
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        offset = f.tell()
    if fortran or dtype.hasobject:
        raise SystemExit(f"{name}: member {member} is Fortran-ordered or holds objects")
    if offset - start + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize != info.file_size:
        raise SystemExit(f"{name}: member {member}'s size disagrees with its header")
    return Layout(int(offset), dtype, tuple(int(x) for x in shape))


def read_rows(path, lay: Layout, spans, opener=open) -> np.ndarray:
    """Copies of the given [a, b) row spans of a member, by bounded seeks; nothing else of the member is read."""
    row = int(np.prod(lay.shape[1:], dtype=np.int64)) * lay.dtype.itemsize; parts = []
    with opener(path, "rb") as f:
        for a, b in spans:
            if not 0 <= int(a) <= int(b) <= lay.shape[0]:
                raise SystemExit("a requested row span lies outside the member")
            f.seek(lay.offset + int(a) * row); buf = _read_exact(f, (int(b) - int(a)) * row) if b > a else b""
            parts.append(np.frombuffer(buf, dtype=lay.dtype).reshape((int(b) - int(a), *lay.shape[1:])).copy())
    return np.concatenate(parts) if parts else np.zeros((0, *lay.shape[1:]), lay.dtype)


def _read_all(path, lay: Layout, opener=open) -> np.ndarray:
    if not lay.shape:
        with opener(path, "rb") as f:
            f.seek(lay.offset)
            return np.frombuffer(_read_exact(f, lay.dtype.itemsize), dtype=lay.dtype).reshape(()).copy()
    return read_rows(path, lay, [(0, lay.shape[0])], opener)


def read_members(path, names) -> dict:
    """Whole members of a one-half archive (V01's saved outputs), read through their zip streams."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for n in names:
            with zf.open(f"{n}.npy") as fh:
                out[n] = np.lib.format.read_array(fh, allow_pickle=False)
    return out


def half_block(all_ids, half_ids) -> tuple[int, int]:
    """The half's sequences as the leading or the trailing block of the cache, in the half's order."""
    all_ids, half_ids = [str(x) for x in all_ids], [str(x) for x in half_ids]; pos = {v: i for i, v in enumerate(all_ids)}
    if not half_ids or len(pos) != len(all_ids) or any(v not in pos for v in half_ids):
        raise SystemExit("the half's sequences are not all listed once in the cache")
    i0 = pos[half_ids[0]]; i1 = i0 + len(half_ids)
    if all_ids[i0:i1] != half_ids or not (i0 == 0 or i1 == len(all_ids)):
        raise SystemExit("the half's sequences are not one leading or trailing block of the cache, in order")
    return i0, i1


class DevAccess:
    """Which development halves a stage may open; selection only after the barrier."""

    def __init__(self, allowed) -> None:
        if not set(allowed) <= set(HALVES):
            raise SystemExit("only the two development halves can be opened")
        self.allowed = frozenset(allowed)

    def require(self, half: str) -> None:
        if half not in self.allowed:
            raise SystemExit(f"the {half} half is not open at this stage")


def dev_half(path, half: str, half_ids, access: DevAccess, opener=open) -> dict:
    """One half of an A01 development cache: its sample features per sequence and its window features, centres and
    markers; metadata (ids, lengths, offsets) are read whole, feature rows only within the half's block."""
    access.require(half)
    lay = {m: member_layout(path, m, opener) for m in ("identity", "ids", "n", "sample_offsets", "window_offsets", "centres", "markers", "window", "samples")}
    meta = {m: _read_all(path, lay[m], opener) for m in ("identity", "ids", "n", "sample_offsets", "window_offsets")}
    i0, i1 = half_block(meta["ids"].tolist(), half_ids); s_off, w_off = meta["sample_offsets"], meta["window_offsets"]
    if lay["samples"].dtype != np.float32 or lay["window"].dtype != np.float32 or lay["samples"].shape[1:] != (512,) or lay["window"].shape[1:] != (512,):
        raise SystemExit(f"{Path(path).name}: the features are not float32 (rows, 512)")
    if int(s_off[-1]) != lay["samples"].shape[0] or int(w_off[-1]) != lay["window"].shape[0] or not np.array_equal(np.diff(s_off), meta["n"]) \
            or lay["centres"].shape[0] != lay["window"].shape[0]:
        raise SystemExit(f"{Path(path).name}: the offsets disagree with the members")
    S = read_rows(path, lay["samples"], [(int(s_off[i0]), int(s_off[i1]))], opener); span = [(int(w_off[i0]), int(w_off[i1]))]
    W, C, M = (read_rows(path, lay[m], span, opener) for m in ("window", "centres", "markers"))
    loc, wloc = s_off[i0:i1 + 1] - s_off[i0], w_off[i0:i1 + 1] - w_off[i0]
    return {"identity": str(meta["identity"]), "ids": list(half_ids), "block": (i0, i1), "samples": {sid: S[loc[j]:loc[j + 1]] for j, sid in enumerate(half_ids)},
            "window": W, "window_offsets": wloc, "centres": C, "markers": M}


# ----------------------------------------------------------------------------------------------- the git-object proof
def _git(repo, *args, binary: bool = False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args[:2])} failed: {r.stderr.decode()[:200]}")
    return r.stdout if binary else r.stdout.decode().strip()


def _obj_sha(kind: str, raw: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(raw)}".encode() + b"\0" + raw).hexdigest()


def _tree_entries(raw: bytes) -> dict:
    out, i = {}, 0
    while i < len(raw):
        sp = raw.index(b" ", i); nul = raw.index(b"\0", sp)
        out[raw[sp + 1:nul].decode()] = (raw[i:sp].decode(), raw[nul + 1:nul + 21].hex()); i = nul + 21
    return out


def _walk(get, commit: str, relpath: str) -> str:
    """The blob id at relpath in the commit, through objects returned by get(kind, sha) -> raw."""
    first = get("commit", commit).split(b"\n", 1)[0].decode()
    if not first.startswith("tree "):
        raise SystemExit("the commit object names no tree")
    sha, parts = first[5:], relpath.split("/")
    for i, name in enumerate(parts):
        entries = _tree_entries(get("tree", sha))
        if name not in entries:
            raise SystemExit(f"{relpath} is not in the commit")
        mode, sha = entries[name]
        if (i < len(parts) - 1) != (mode == "40000"):
            raise SystemExit(f"{relpath}: an entry on the path has an unexpected mode")
    return sha


def git_proof(repo, commit: str, relpath: str) -> dict:
    """Raw commit and tree objects from the local history showing that the commit holds exactly the working copy of relpath."""
    repo = Path(repo); commit = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"]).returncode != 0:
        raise SystemExit("the commit is not in the local history")
    objects = {}

    def get(kind: str, sha: str) -> bytes:
        raw = _git(repo, "cat-file", kind, sha, binary=True)
        if _obj_sha(kind, raw) != sha:
            raise SystemExit("a git object does not hash to its name")
        objects[sha] = {"kind": kind, "raw": base64.b64encode(raw).decode()}
        return raw
    blob = _walk(get, commit, relpath); path = repo / relpath
    data = path.read_bytes() if path.is_file() else b""
    if _obj_sha("blob", data) != blob or _git(repo, "status", "--porcelain", "--", relpath):
        raise SystemExit("the working copy of the file differs from the committed blob")
    return {"commit": commit, "path": relpath, "blob": blob, "file_sha256": hashlib.sha256(data).hexdigest(), "objects": objects, "made": ts.now()}


def verify_proof(proof: dict, data: bytes, relpath: str, commit: str) -> str:
    """Without git: every object hashes to its name, the commit's trees lead to relpath, and its blob is these bytes."""
    if not commit or len(str(commit)) != 40 or proof.get("commit") != commit:
        raise SystemExit("the proof belongs to another commit")
    if proof.get("path") != relpath:
        raise SystemExit("the proof covers another path")
    objs = {}
    for sha, o in proof.get("objects", {}).items():
        raw = base64.b64decode(o["raw"])
        if o["kind"] not in ("commit", "tree") or _obj_sha(o["kind"], raw) != sha:
            raise SystemExit("a proof object does not hash to its name")
        objs[sha] = (o["kind"], raw)

    def get(kind: str, sha: str) -> bytes:
        if objs.get(sha, (None,))[0] != kind:
            raise SystemExit(f"the proof misses a {kind} object on the path")
        return objs[sha][1]
    blob = _walk(get, commit, relpath)
    if _obj_sha("blob", data) != blob or proof.get("blob") != blob or hashlib.sha256(data).hexdigest() != proof.get("file_sha256"):
        raise SystemExit("the choice file differs from the committed blob")
    return blob


def barrier(results: Path, twin: Path, proof_path: Path, relpath: str, expected_sha: str | None, commit: str | None, provenance: dict) -> dict:
    """The launch barrier of the selection stage: the committed choice file (results copy equal to the calibration
    stage's copy, the given sha256, the git-object proof of the given commit) made under the current provenance."""
    if not expected_sha or not commit:
        raise SystemExit("--selection needs --choices-sha256 and --choices-commit of the committed choice file")
    for p in (results, twin, proof_path):
        if not Path(p).is_file():
            raise SystemExit(f"{Path(p).name} is missing")
    data = Path(results).read_bytes()
    if Path(twin).read_bytes() != data:
        raise SystemExit("the results copy of the choices differs from the calibration stage's copy")
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise SystemExit("choices.json is not the committed record (its sha256 differs)")
    verify_proof(json.loads(Path(proof_path).read_text()), data, relpath, commit)
    choices = json.loads(data)
    if choices.get("provenance") != json.loads(json.dumps(provenance, default=str)):
        raise SystemExit("the choices belong to other code, configuration, runs or sources")
    return choices


# ----------------------------------------------------------------------------------------------- guards and choices
def forbid_encoders():
    """Encoder construction or forward and A01's eager development loaders raise from here on; returns the undo."""
    import echo_routing.temporal.dev_run as dev_run
    import echo_routing.vit.target_mvit as tm
    import scripts.report_video_feature_temporal as rvt

    def blocked(name):
        def f(*a, **k):
            raise RuntimeError(f"{name} is forbidden in this round (no encoder, no eager development loader)")
        return f
    targets = [(tm, "build_model"), (tm.TargetMViT, "forward"), (tm.TargetMViT, "embed"), (pv, "build_source"), (pv, "extract"), (pv, "encode"),
               (rcs, "validation"), (rvt, "dev_features"), (dev_run.Inputs, "__init__")]
    saved = [(o, n, o.__dict__[n]) for o, n in targets]
    for o, n in targets:
        setattr(o, n, blocked(n))

    def undo() -> None:
        for o, n, v in saved:
            setattr(o, n, v)
    return undo


def placeholder_seqs(feats: dict) -> list[Seq]:
    """Decoder inputs from features alone: zero labels of the valid length, no join, no native flag."""
    return [Seq(k, False, np.ascontiguousarray(f, np.float32), np.zeros(len(f), np.int64), np.zeros(0, np.int64)) for k, f in feats.items()]


def rederive_choice(rec: dict, goal: dict) -> None:
    """The calibration rule applied again to the record's own grid must give its setting, status and feasible count."""
    sel = select_setting(rec["grid"], float(goal["native_macro_f1_min"]["calibration"]), float(goal["boundary_recall_min"]))
    if sel["chosen"] is None:
        d = DIAG["evaluation"]["b3_infeasible_diagnostic"]
        diag = next(r for r in rec["grid"] if r["params"]["half_window"] == int(d["half_window"]) and abs(r["params"]["threshold_quantile"] - float(d["quantile"])) < 1e-9)
        want = ("infeasible (diagnostic setting shown)", diag["params"], 0)
    else:
        want = ("feasible", sel["chosen"]["params"], sel["n_feasible"])
    if (rec["status"], rec["params"], rec["n_feasible"]) != want:
        raise SystemExit("a choice record does not follow the calibration rule on its own grid")


# ----------------------------------------------------------------------------------------------- context and provenance
class Ctx(ts.Ctx):
    def __init__(self, cfg: dict, config: str) -> None:
        super().__init__(cfg, config); self._cache = {}; self.gpu_seconds = 0.0
        self.relpath = self.so["outputs"]["choices"]; self.twin = self.out / "calibration" / "choices.json"; self.proof = self.root / self.so["outputs"]["choices_proof"]

    def _pinned(self, key: str, pin: dict, kind: str = "json"):
        if key not in self._cache:
            text = pinned(self.root, pin).read_text(); self._cache[key] = json.loads(text) if kind == "json" else yaml.safe_load(text)
        return self._cache[key]

    def a03src(self) -> dict:
        return self._pinned("a03", self.src["a03_config"], "yaml")["trained_token_readout"]["source"]

    def proto(self) -> dict:
        p = self._pinned("protocol", self.a03src()["protocol"]); ttr.check_class_order(rcs.CLASSES, p["mapping_version"])
        return p

    def plans(self) -> dict:
        return self._pinned("plans", self.a03src()["window_plans"])

    def half(self, h: str) -> tuple:
        return ttr.half_sets(self.proto(), h, pinned(self.root, self.a03src()["clip_index"]), self.plans())

    def a01_summary(self) -> dict:
        return self._pinned("a01_summary", self.src["a01_summary"])

    def v01_summary(self) -> dict:
        return self._pinned("v01_summary", self.a01().vt["source"]["report"]["summary"])


def pinned(root: Path, pin: dict) -> Path:
    return pv.pinned(root, pin)


def verify_runs(ctx: Ctx) -> dict:
    prep = ts.read_record(ctx, "preparation.json"); idents = ts.identities(ctx, prep); out = {}
    for key, ident in idents.items():
        man_path = ctx.out / "train" / key / "run_manifest.json"
        if not tv.finished(man_path, ident):
            raise SystemExit(f"{key}: not a finished run of this identity")
        man = json.loads(man_path.read_text()); frozen = ctx.root / "models_frozen" / NS / key / "final.pt"
        if not frozen.is_file() or sha256_file(frozen) != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen final weights are missing or differ")
        out[key] = {"identity": ident, "updates": man["updates"], "final_sha256": man["final_sha256"], "interrupted": man["interrupted"],
                    "parameters": man["parameters"], "frozen": str(frozen.parent.relative_to(ctx.root)), "records": man["records"]}
    return out


def provenance(ctx: Ctx, runs: dict) -> dict:
    a3 = ctx.a03src(); a01 = ctx.a01(); prep = ts.read_record(ctx, "preparation.json")
    if a01.vt["source"]["protocol"] != a3["protocol"] or a01.vt["source"]["window_plans"] != a3["window_plans"]:
        raise SystemExit("A03's metadata pins differ from A01's protocol or window plans")
    return {"report_code": code_digest(), "train_code": ts.code_digest(), "config_sha256": ctx.config_sha(), "preparation_code": prep.get("code"),
            "runs": {k: {"identity": v["identity"], "final_sha256": v["final_sha256"]} for k, v in runs.items()},
            "development_caches": {s: c["development"]["sha256"] for s, c in prep["caches"].items()},
            "a01_summary": ctx.src["a01_summary"]["sha256"], "metadata": {k: a3[k]["sha256"] for k in ("protocol", "clip_index", "window_plans")},
            "v01_outputs": {f"{h}_s{s}": a01.vt["source"]["outputs"][h][s]["sha256"] for h in HALVES for s in ctx.seeds}}


def ids_of(ctx: Ctx, seqs: dict) -> set:
    proto = ctx.proto(); ids = {c for h in HALVES for c in proto["validation"][h]["clips"]}
    ids |= {r[0] for q in (x for h in seqs for x in seqs[h]) for r in q["refs"]} | {q["id"] for h in seqs for q in seqs[h]}
    return ids | {session_of(v) for v in list(ids)}


# ----------------------------------------------------------------------------------------------- systems of one half
def row_of(cfg: dict, preds: dict, ev) -> dict:
    m, tables = rcs.evaluate(cfg, preds, ev)
    return {"row": json.loads(json.dumps(dtt.system_row(m), default=str)), "tables": tables, "preds": preds}


def check_stored(got: dict, want: dict, what: str) -> None:
    bad = sorted(k for k in set(want) | set(got) if got.get(k) != want.get(k))
    if bad:
        raise SystemExit(f"{what}: the stored row is not reproduced ({bad[:6]})")


def frozen_linear(path: Path, sha: str, device) -> torch.nn.Module:
    if sha256_file(path) != sha:
        raise SystemExit(f"{path}: the frozen weights differ from their record")
    m = ts.new_model(torch.load(path, map_location="cpu", weights_only=True), device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def a01_head(ctx: Ctx, head: str, seed: int, device):
    a01 = ctx.a01(); plan, _ = a01.plan(seed); h = tv.make_head(a01, head, plan, device)
    f = ctx.root / "models_frozen" / "video_feature_temporal_v1" / f"{head}_s{seed}" / "final.pt"
    if sha256_file(f) != ctx.a01_summary()["runs"][f"{head}_s{seed}"]["final_sha256"]:
        raise SystemExit(f"A01 {head}_s{seed}: the frozen weights differ from A01's summary")
    model = tv.new_model(h, torch.load(f, map_location="cpu", weights_only=True), device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return h, model


def half_systems(ctx: Ctx, h: str, ev, seqs: list[dict], access: DevAccess, device, runs: dict, choices: dict | None) -> dict:
    """One half: per seed the bounded cache read, the parity, A01's stored reference rows reproduced and the new heads'
    probabilities; with choices also the new heads' B3, argmax and Ordered-setting systems."""
    a01 = ctx.a01(); stored = ctx.a01_summary()["systems"]; v01 = ctx.v01_summary()["systems"]; prep = ts.read_record(ctx, "preparation.json")
    ids = [q["id"] for q in seqs]; out = {"systems": {}, "probs": {}, "parity": {}, "reproduced": [], "seqs": seqs, "ev": ev}
    for s in ctx.seeds:
        cache = ctx.root / prep["caches"][str(s)]["development"]["path"]
        if sha256_file(cache) != prep["caches"][str(s)]["development"]["sha256"]:
            raise SystemExit(f"seed {s}: the development cache differs from its preparation record")
        d = dev_half(cache, h, ids, access)
        v = read_members(pinned(ctx.root, a01.vt["source"]["outputs"][h][s]), ("sequence_ids", "centre", "logits", "sample_offsets", "sample_probs"))
        if v["sequence_ids"].tolist() != ids or not np.array_equal(v["centre"], d["centres"]):
            raise SystemExit(f"seed {s} {h}: V01's saved outputs list other sequences or windows than the cache")
        init, _ = a01.init_state("linear", s); mine = d["window"].astype(np.float64) @ init["weight"].double().numpy().T + init["bias"].double().numpy()
        saved = v["logits"].astype(np.float64)
        par = {"windows": int(len(saved)), "max_abs": float(np.abs(mine - saved).max()), "argmax_equal": bool(np.array_equal(mine.argmax(1), saved.argmax(1)))}
        if par["max_abs"] > 1e-4 or not par["argmax_equal"]:
            raise SystemExit(f"seed {s} {h}: the source classifier's parity failed ({par})")
        out["parity"][f"s{s}"] = par; feats = d["samples"]; off = v["sample_offsets"]
        P = {sid: v["sample_probs"][off[i]:off[i + 1]] for i, sid in enumerate(ids)}; oparams = v01[f"ordered_s{s}"]["b3_choice"]["params"]
        sysd = {"frozen Ordered + B3": row_of(ctx.cfg, decode(oparams, P), ev), "frozen Ordered argmax": row_of(ctx.cfg, {k: p.argmax(1) for k, p in P.items()}, ev)}
        lin = frozen_linear(ctx.root / "models_frozen" / "video_feature_temporal_v1" / f"linear_s{s}" / "final.pt",
                            ctx.a01_summary()["runs"][f"linear_s{s}"]["final_sha256"], device)
        lp = ts.predict(lin, feats, device); out["probs"][("a01_linear", s)] = lp
        sysd["A01 Linear + B3"] = row_of(ctx.cfg, decode(ctx.a01_summary()["linear_b3"][str(s)]["params"], lp), ev)
        sysd["A01 Linear argmax"] = row_of(ctx.cfg, {k: p.argmax(1) for k, p in lp.items()}, ev)
        for head, name in (("mstcn", "A01 MS-TCN"), ("asrf", "A01 ASRF")):
            hd, model = a01_head(ctx, head, s, device); o = hd.decode(model, placeholder_seqs(feats), device)
            sysd[name] = row_of(ctx.cfg, {k: np.asarray(x["labels"], int) for k, x in o.items()}, ev); del model
        for name, key in REFERENCE_ROWS.items():
            check_stored(sysd[name]["row"], stored[f"{key} s{s}"][h], f"{name} s{s} {h}"); out["reproduced"].append(f"{name} s{s} {h}")
        for arm in ARMS:
            key = ts.run_key(arm, s); m = frozen_linear(ctx.root / runs[key]["frozen"] / "final.pt", runs[key]["final_sha256"], device)
            probs = ts.predict(m, feats, device); out["probs"][(arm, s)] = probs; n = ARM_NAME[arm]
            if choices is not None:
                sysd[f"{n} + B3"] = row_of(ctx.cfg, decode(choices["choices"][key]["params"], probs), ev)
                sysd[f"{n} argmax"] = row_of(ctx.cfg, {k: p.argmax(1) for k, p in probs.items()}, ev)
                sysd[f"{n} at Ordered setting"] = row_of(ctx.cfg, decode(oparams, probs), ev)
        for name, x in sysd.items():
            out["systems"][f"{name} s{s}"] = x
        torch.cuda.empty_cache(); print(f"{h} seed {s}: parity, references and heads done", flush=True)
    return out


# ----------------------------------------------------------------------------------------------- stages
def run_stage(ctx: Ctx, name: str, fn) -> None:
    """Every stage, finished or failed, appends its GPU part (the inference phase) and its CPU remainder to the ledger."""
    b = ctx.so["budget"]
    if ts.ledger_hours(ctx.ledger, "cpu") >= float(b["cpu_hours_max"]) or ts.ledger_hours(ctx.ledger, "gpu") >= float(b["gpu_hours_max"]):
        raise SystemExit("the CPU or GPU budget is spent")
    t0, started, ok, note = time.time(), ts.now(), False, ""; ctx.gpu_seconds = 0.0
    try:
        env = ts.environment()
        if any(str(env[k]) != str(v) for k, v in ctx.src["environment"].items()):
            raise SystemExit(f"environment {env} differs from the pinned {ctx.src['environment']}")
        fn(ctx); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(float(ctx.gpu_seconds), total)
        if g > 0:
            ts.ledger_append(ctx.ledger, f"{name} (GPU inference)", "gpu", started, g, ok, note)
        ts.ledger_append(ctx.ledger, f"{name} (CPU)", "cpu", started, total - g, ok, note)


def read_halves(ctx: Ctx, data: dict, access: DevAccess, runs: dict, choices: dict | None) -> dict:
    """data: reported half -> the half whose files are read; a half read once serves every reported half naming it."""
    device = pv.setup_gpu(); t0 = time.time(); per = {}
    try:
        for h in ("calibration", "selection"):
            if h not in data:
                continue
            if data[h] != h and data[h] in per:
                per[h] = per[data[h]]; continue
            ev, seqs = ctx.half(data[h]); per[h] = half_systems(ctx, data[h], ev, seqs, access, device, runs, choices)
    finally:
        torch.cuda.empty_cache(); ctx.gpu_seconds += time.time() - t0
    return per


def stage_calibrate(ctx: Ctx) -> None:
    undo = forbid_encoders()
    try:
        runs = verify_runs(ctx); prov = provenance(ctx, runs)
        per = read_halves(ctx, {"calibration": "calibration"}, DevAccess({"calibration"}), runs, None)["calibration"]
        rec = {"provenance": prov, "half": "calibration", "parity": per["parity"], "reproduced": per["reproduced"], "systems": {}}
        for s in ctx.seeds:
            for arm in ARMS:
                ch = rcs.b3_choose(ctx.cfg, DIAG, per["probs"][(arm, s)], per["ev"]); key = ts.run_key(arm, s)
                rec["systems"][key] = {"status": ch["status"], "params": ch["params"], "n_feasible": ch["n_feasible"], "grid": ch["grid"]}
                print(f"{key}: {ch['status']} {ch['params']}", flush=True)
        rec["choices"] = {k: {x: v[x] for x in ("status", "params", "n_feasible")} for k, v in rec["systems"].items()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids_of(ctx, {"calibration": per["seqs"]}), "choices")
        for p in (ctx.twin, ctx.root / ctx.relpath):
            if p.is_file() and json.loads(p.read_text()) != flat:
                raise SystemExit("an earlier choice record differs from this one; it is never overwritten")
        for p in (ctx.twin, ctx.root / ctx.relpath):
            p.parent.mkdir(parents=True, exist_ok=True); write_atomic(p, json.dumps(flat, indent=1))
    finally:
        undo()


def load_choices(ctx: Ctx, choices: dict) -> dict:
    keys = {ts.run_key(a, s) for a in ARMS for s in ctx.seeds}
    if set(choices["systems"]) != keys or set(choices["choices"]) != keys:
        raise SystemExit("the choice record does not hold exactly one record per head")
    for k, rec in choices["systems"].items():
        check_choice_record(rec, ctx.so["evaluation"]["b3"]["grid"]); rederive_choice(rec, ctx.so["goal"])
        if choices["choices"][k] != {x: rec[x] for x in ("status", "params", "n_feasible")}:
            raise SystemExit("a summary choice differs from its full record")
    return choices


def check_calibration_rows(per: dict, choices: dict) -> None:
    """The committed grid rows must be reproduced by the chosen settings on calibration."""
    for key, rec in choices["systems"].items():
        arm, s = key.rsplit("_s", 1); g = next(r for r in rec["grid"] if r["params"] == rec["params"])
        row = per["systems"][f"{ARM_NAME[arm]} + B3 s{s}"]["row"]
        if any(row[f] != g[f] for f in rcs.FLAT):
            raise SystemExit(f"{key}: the calibration row differs from the committed choice record")


def summarise(ctx: Ctx, per: dict, runs: dict, choices: dict) -> tuple[dict, dict]:
    res = {"systems": {}, "kind_counts": {}, "transcripts": {}, "switches": {}, "join_cuts": {}, "comparisons": {}, "accounting": {}, "native_changes": {}}
    private = {}
    for h in HALVES:
        x = per[h]; seqs = x["seqs"]
        for name, sysd in x["systems"].items():
            res["systems"].setdefault(name, {})[h] = sysd["row"]
            res["kind_counts"].setdefault(name, {})[h] = st.kind_counts(seqs, sysd["preds"], sysd["tables"])
            res["transcripts"].setdefault(name, {})[h] = st.transcript_summary(seqs, sysd["preds"])
            res["switches"].setdefault(name, {})[h] = st.switch_summary(seqs, sysd["preds"])
            mf = sysd["tables"]["multi_frag"]; res["join_cuts"].setdefault(name, {})[h] = {"cuts": int(mf["false_split"].sum()), "same_view_joins": int(mf["same_join"].sum())}
        for s in ctx.seeds:
            draws = {}
            for dec, cmps in CMP.items():
                for cname, (a, b) in cmps.items():
                    for k in (a, b):
                        if k not in draws:
                            draws[k] = st.system_draws(x["systems"][f"{k} s{s}"]["tables"])
                    res["comparisons"][f"s{s}|{dec}|{cname}|{h}"] = st.paired(draws[a], draws[b])
            for pname, (a, b) in PAIRS.items():
                A, B = x["systems"][f"{a} s{s}"], x["systems"][f"{b} s{s}"]; seg, priv = st.segment_accounting(seqs, A["preds"], B["preds"])
                nc = rcs.native_changes(A["tables"]["native"], B["tables"]["native"])
                res["accounting"][f"s{s}|{pname}|{h}"] = {"segments": seg}
                res["native_changes"][f"s{s}|{pname}|{h}"] = {k: ({"fixed": len(v["fixed"]), "new": len(v["new"]), "fixed_sessions": v["fixed_sessions"],
                                                                  "new_sessions": v["new_sessions"]} if "fixed_sessions" in v else v) for k, v in nc.items()}
                private[f"s{s}|{pname}|{h}"] = {"segments": priv, "native": {k: v for k, v in nc.items() if "fixed_sessions" in v}}
            print(f"{h} seed {s}: comparisons and accounting done", flush=True)
    res["verdicts"] = {f"{dec}|{c}|{m}": {h: st.consistency([res["comparisons"][f"s{s}|{dec}|{c}|{h}"][m] for s in ctx.seeds]) for h in HALVES}
                       for dec, cmps in CMP.items() for c in cmps for m in st.BOOT_METRICS}
    res["accounting_pooled"] = pooled(ctx, res["accounting"])
    res["goal"] = goals(ctx, res["systems"], choices)
    res["decision"] = decide(ctx, res)
    return res, private


def pooled(ctx: Ctx, acc: dict) -> dict:
    out = {}
    for pname in PAIRS:
        for h in HALVES:
            for kind in st.KINDS:
                xs = [acc[f"s{s}|{pname}|{h}"]["segments"][kind] for s in ctx.seeds]
                out[f"{pname}|{h}|{kind}"] = {n: int(sum(x[n]["segments"] for x in xs)) for n in ("recovered", "lost", "new_unmatched_pred", "removed_unmatched_pred")} | \
                    {"recovered_xyx": int(sum(x["recovered"]["xyx_a4c_a5c"] for x in xs)), "lost_xyx": int(sum(x["lost"]["xyx_a4c_a5c"] for x in xs)),
                     "recovered_short": int(sum(x["recovered"]["short"] for x in xs)), "lost_short": int(sum(x["lost"]["short"] for x in xs))}
    return out


def goals(ctx: Ctx, systems: dict, choices: dict) -> dict:
    out = {}
    for n in ("F", "B", "O", "frozen Ordered"):
        name = f"{n} + B3"; rows = [{h: systems[f"{name} s{s}"][h] for h in HALVES} for s in ctx.seeds]
        feas = [choices["choices"][ts.run_key({v: k for k, v in ARM_NAME.items()}[n], s)]["status"] == "feasible" if n in ("F", "B", "O") else True for s in ctx.seeds]
        out[n] = rt.goal(rows, feas, ctx.so["goal"])
    return out


def decide(ctx: Ctx, res: dict) -> dict:
    """The fixed readings with their operational inputs (fixed in code before any selection output): a transcript or
    recovery gain is an O - B or B - F pair whose B3 multi edit distance is lower in every seed and half, or whose pooled
    multi recoveries exceed its losses in both halves; a trade-off is a pair among O - B, B - F, O - F and B/O - frozen
    Ordered with pooled multi recoveries in a half together with consistently more multi FP segments, a consistently
    lower native macro-F1 or a recall-floor failure of the candidate in that half."""
    v = res["verdicts"]; ver = {c: v[f"b3|{c}|multi segment F1@50"] for c in ("O - B", "B - F", "O - F", "B - frozen Ordered", "O - frozen Ordered")}
    gain = {}
    for pair, (a, b) in (("O vs B", ("O", "B")), ("B vs F", ("B", "F"))):
        ed = all(res["transcripts"][f"{a} + B3 s{s}"][h]["multi"]["mean"] < res["transcripts"][f"{b} + B3 s{s}"][h]["multi"]["mean"] for s in ctx.seeds for h in HALVES)
        rec = all(res["accounting_pooled"][f"{pair}|{h}|multi"]["recovered"] > res["accounting_pooled"][f"{pair}|{h}|multi"]["lost"] for h in HALVES)
        gain[pair] = {"edit_distance_lower_everywhere": bool(ed), "recoveries_exceed_losses_both_halves": bool(rec)}
    trade = {}
    for c, pair in (("O - B", "O vs B"), ("B - F", "B vs F"), ("O - F", "O vs F"), ("B - frozen Ordered", "B vs frozen Ordered"), ("O - frozen Ordered", "O vs frozen Ordered")):
        cand = c.split(" - ")[0]
        for h in HALVES:
            fails = any(f["half"] == h and f["boundary_recall"] < float(ctx.so["goal"]["boundary_recall_min"]) for f in res["goal"][cand]["constraint_failures"])
            cost = v[f"b3|{c}|multi FP segments"][h] == st.POS or v[f"b3|{c}|native macro-F1"][h] == st.NEG or fails
            trade[f"{c}|{h}"] = bool(res["accounting_pooled"][f"{pair}|{h}|multi"]["recovered"] > 0 and cost)
    extra = {"transcript_or_recovery_gain": any(any(g.values()) for g in gain.values()), "tradeoff": any(trade.values()), "gain_inputs": gain, "tradeoff_inputs": trade}
    return st.readings(ver, {"B": res["goal"]["B"], "O": res["goal"]["O"]}, extra)


def figures(ctx: Ctx, runs: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); col = {"frame": "#8172b2", "segment": "#dd8452", "order": "#4c72b0"}
    for key, r in runs.items():
        arm, s = key.rsplit("_s", 1); recs = r["records"]; x = [e["updates"] for e in recs]
        kw = {"color": col[arm], "ls": ("-", "--", ":")[int(s)], "lw": 1.1, "label": f"{ARM_NAME[arm]} s{s}"}
        ax[0].plot(x, [e["monitor"]["order"] for e in recs], **kw); ax[1].plot(x, [e["monitor"]["multi_accuracy"] for e in recs], **kw)
        ax[2].plot(x, [e["monitor"]["edit_distance_multi"] for e in recs], **kw)
    titles = ("transcript term L_order on the 16 check sequences", "argmax accuracy of the 8 check streams", "normalised transcript edit distance (check streams)")
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=9); a.set_xlabel("updates"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=3); fig.suptitle("Training data only; the final weights are kept and nothing is chosen from these curves", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=130); plt.close(fig)


def save_private(out: Path, per: dict, private: dict) -> None:
    """Per-sequence probabilities and decodings of the new heads, and the case lists; server only."""
    for h in HALVES:
        x = per[h]; ids = [q["id"] for q in x["seqs"]]; d = out / h; d.mkdir(parents=True, exist_ok=True)
        for (arm, s), probs in x["probs"].items():
            name = f"{arm}_s{s}" if arm != "a01_linear" else f"a01_linear_s{s}"; extra = {}
            if arm in ARM_NAME:
                extra = {f"{dec}": np.concatenate([np.asarray(x["systems"][f"{ARM_NAME[arm]} {lab} s{s}"]["preds"][i], np.int8) for i in ids])
                         for dec, lab in (("b3", "+ B3"), ("argmax", "argmax"), ("ordered_setting", "at Ordered setting"))}
            np.savez(d / f"{name}.npz", sequence_ids=np.asarray(ids), sample_offsets=np.cumsum([0, *[len(probs[i]) for i in ids]]),
                     sample_probs=np.concatenate([probs[i] for i in ids]).astype(np.float32), **extra)
    write_atomic(out / "private_accounting.json", json.dumps(private, default=str))


def evaluate_all(ctx: Ctx, choices: dict, data: dict, access: DevAccess, runs: dict) -> tuple[dict, dict, dict, set]:
    per = read_halves(ctx, data, access, runs, choices)
    check_calibration_rows(per["calibration"], choices)
    res, private = summarise(ctx, per, runs, choices)
    res["parity"] = {h: per[h]["parity"] for h in HALVES}; res["reproduced"] = {h: per[h]["reproduced"] for h in HALVES}
    res["choices"] = choices["choices"]; res["runs"] = {k: {x: v for x, v in r.items() if x != "records"} for k, r in runs.items()}
    res["training_last_epoch"] = {k: r["records"][-1] for k, r in runs.items()}
    ids = ids_of(ctx, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "summary")
    if any(x in st.tables_md(flat) for x in ids):
        raise SystemExit("an id appears in the report tables")
    return flat, per, private, ids


def stage_dry_run(ctx: Ctx) -> None:
    undo = forbid_encoders()
    try:
        runs = verify_runs(ctx); prov = provenance(ctx, runs)
        if not ctx.twin.is_file():
            raise SystemExit("run --calibrate first")
        choices = load_choices(ctx, json.loads(ctx.twin.read_text()))
        if choices["provenance"] != json.loads(json.dumps(prov, default=str)):
            raise SystemExit("the choices belong to other code, configuration, runs or sources")
        flat, per, private, _ = evaluate_all(ctx, choices, {"calibration": "calibration", "selection": "calibration"}, DevAccess({"calibration"}), runs)
        out = ctx.out / "dry_run"; out.mkdir(parents=True, exist_ok=True)
        flat["dry_run"] = "the calibration half stands in for both halves; a code check, not a result"
        write_atomic(out / "summary.json", json.dumps(flat, indent=1)); write_atomic(out / "report_tables.md", st.tables_md(flat))
        print("dry run ok (a code check, not a result)", flush=True)
    finally:
        undo()


def stage_selection(ctx: Ctx) -> None:
    undo = forbid_encoders()
    try:
        runs = verify_runs(ctx); prov = provenance(ctx, runs)
        choices = load_choices(ctx, barrier(ctx.root / ctx.relpath, ctx.twin, ctx.proof, ctx.relpath, ctx.choices_sha256, ctx.choices_commit, prov))
        out = ctx.out / "selection"; done = out / "done.json"
        if done.is_file():
            raise SystemExit("the selection half was already evaluated once; it is never evaluated again")
        flat, per, private, ids = evaluate_all(ctx, choices, {"calibration": "calibration", "selection": "selection"}, DevAccess({"calibration", "selection"}), runs)
        led = [json.loads(x) for x in ctx.ledger.read_text().splitlines() if x.strip()]
        flat["identity"] = {**prov, "choices_sha256": ctx.choices_sha256, "choices_commit": ctx.choices_commit,
                            "proof_made": json.loads(ctx.proof.read_text()).get("made")}
        flat["cost"] = {"gpu_hours_before_this_stage": ts.ledger_hours(ctx.ledger, "gpu"), "cpu_hours_before_this_stage": ts.ledger_hours(ctx.ledger, "cpu"),
                        "earlier_stages": [{k: x[k] for k in ("job", "kind", "started", "finished", "hours", "ok")} for x in led]}
        assert_no_ids(flat, ids, "results summary"); out.mkdir(parents=True, exist_ok=True); save_private(out, per, private)
        write_atomic(ctx.results / "summary.json", json.dumps(flat, indent=1)); write_atomic(ctx.results / "report_tables.md", st.tables_md(flat))
        figures(ctx, runs, ctx.results)
        write_atomic(done, json.dumps({"choices_sha256": ctx.choices_sha256, "choices_commit": ctx.choices_commit, "finished": ts.now()}))
        print(json.dumps(flat["decision"]["applied"]), flush=True)
    finally:
        undo()


def stage_proof(ctx: Ctx) -> None:
    if not ctx.choices_commit:
        raise SystemExit("--proof needs --choices-commit")
    proof = git_proof(ctx.root, ctx.choices_commit, ctx.relpath); verify_proof(proof, (ctx.root / ctx.relpath).read_bytes(), ctx.relpath, proof["commit"])
    ctx.proof.parent.mkdir(parents=True, exist_ok=True); write_atomic(ctx.proof, json.dumps(proof, indent=1))
    print(json.dumps({"commit": proof["commit"], "blob": proof["blob"], "file_sha256": proof["file_sha256"], "objects": len(proof["objects"])}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--choices-sha256"); ap.add_argument("--choices-commit")
    g = ap.add_mutually_exclusive_group(required=True)
    for s in ("calibrate", "dry_run", "proof", "selection"):
        g.add_argument(f"--{s.replace('_', '-')}", dest=s, action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.choices_sha256, ctx.choices_commit = a.choices_sha256, a.choices_commit
    if a.proof:
        stage_proof(ctx); return
    stage = next(s for s in ("calibrate", "dry_run", "selection") if getattr(a, s))
    run_stage(ctx, stage, {"calibrate": stage_calibrate, "dry_run": stage_dry_run, "selection": stage_selection}[stage])


if __name__ == "__main__":
    main()
