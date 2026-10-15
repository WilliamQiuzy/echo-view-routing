"""Data isolation, replication, the choices barrier and the report tables of the segment objective round
(configs/experiments/segment_objective_v1.yaml). Everything here runs on the CPU with synthetic data: A01-like training
caches, A01-like development caches with canary selection rows, a throwaway git repository and synthetic evaluator
tables. No real recording, label or feature is read.
"""
from __future__ import annotations

import base64
import inspect
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.report_segment_objective as rs
import scripts.segment_objective_tables as st
import scripts.train_segment_objective as ts
import scripts.train_video_feature_temporal as tv
from echo_routing.temporal.losses import segment_objective as so
from scripts.report_model_development import paired_bootstrap
from scripts.train_l1_asrf import batch_digest

ALPHA = np.array([0.7340, 0.9877, 1.0, 1.2529, 1.4164])
TRAINING = {"optimiser": {"name": "AdamW", "lr": 3.0e-4, "weight_decay": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "grad_clip": 1.0},
            "schedule": {"name": "warmup_cosine", "warmup_updates": 4}}
DATA = {"epochs": 2, "updates_per_epoch": 5, "total_updates": 10, "streams_per_batch": 2, "natives_per_batch": 2, "native_window": 16,
        "native_per_epoch": 10}


# ----------------------------------------------------------------------------------------------- a synthetic A01 training cache
def library(rng):
    seqs = []
    for i in range(10):
        n = int(rng.integers(18, 40)); seqs.append({"id": f"nat{i:02d}", "kind": "native", "labels": np.full(n, i % 5, np.int64), "joins": np.zeros(0, np.int64)})
    pattern = [(2, 3, 2), (0, 0, 1), (4, 1), (2, 3), (1, 2, 1, 0), (3, 3), (0, 4, 0), (2, 2, 3), (1, 0), (4, 3, 4)]
    for i, p in enumerate(pattern):
        lens = rng.integers(5, 20, len(p)); labels = np.concatenate([np.full(k, c, np.int64) for k, c in zip(lens, p)])
        seqs.append({"id": f"bank{i:02d}", "kind": "bank", "labels": labels, "joins": np.cumsum(lens)[:-1].astype(np.int64)})
    return seqs


def write_training_cache(root: Path, seqs, seed: int = 0, identity: str = "feat-id") -> SimpleNamespace:
    d = root / "features" / "train"; d.mkdir(parents=True, exist_ok=True); n = np.array([q["labels"].size for q in seqs])
    joins = [q["joins"] for q in seqs]
    np.savez(d / "sequences.npz", ids=np.asarray([q["id"] for q in seqs]), kinds=np.asarray([q["kind"] for q in seqs]), n=n,
             labels=np.concatenate([q["labels"] for q in seqs]), join_offsets=np.concatenate([[0], np.cumsum([j.size for j in joins])]),
             joins=np.concatenate(joins), refs_digest=np.asarray(["x"] * len(seqs)))
    rng = np.random.default_rng(seed)
    np.savez(d / f"ordered_s{seed}.npz", identity=identity, ids=np.asarray([q["id"] for q in seqs]), n=n,
             samples=rng.standard_normal((int(n.sum()), 512)).astype(np.float32))
    from echo_routing.evaluate.complementarity import sha256_file
    return SimpleNamespace(out=root, vt={"data": DATA}, prep={"features": {str(seed): {"identity": identity}},
                                                                "sequence_file": {"sha256": sha256_file(d / "sequences.npz")}})


@pytest.fixture()
def trainset(tmp_path):
    ctx = write_training_cache(tmp_path, library(np.random.default_rng(0)))
    data = tv.TrainSet(ctx, 0)
    plan = {"epochs": [[batch_digest(x) for x in data.epoch(ctx, e)] for e in range(DATA["epochs"])], "class_weight": ALPHA.tolist()}
    return ctx, data, plan


def init_state(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"weight": torch.randn(5, 512, generator=g) * 0.05, "bias": torch.randn(5, generator=g) * 0.1}


# ----------------------------------------------------------------------------------------------- initial states and batches
def test_arm_inits_are_bit_identical_copies_and_never_overwritten(tmp_path):
    state = init_state()
    rec = ts.make_arm_inits(state, tmp_path / "inits", 0, tmp_path)
    assert set(rec) == {f"{a}_s0" for a in so.ARMS}
    for a in so.ARMS:
        got = torch.load(tmp_path / rec[f"{a}_s0"]["path"], map_location="cpu", weights_only=True)
        assert set(got) == {"weight", "bias"} and all(torch.equal(got[k], state[k]) for k in state)
        assert rec[f"{a}_s0"]["parameters"] == 2565
    assert ts.make_arm_inits(state, tmp_path / "inits", 0, tmp_path) == rec           # an equal copy is accepted again
    torch.save({"weight": state["weight"] + 1, "bias": state["bias"]}, tmp_path / rec["order_s0"]["path"])
    with pytest.raises(SystemExit):
        ts.make_arm_inits(state, tmp_path / "inits", 0, tmp_path)


def test_three_arms_read_identical_batches_that_match_the_plan(trainset):
    ctx, data, plan = trainset
    rep = ts.check_pairing(data, plan, ctx)
    assert rep["epochs_checked"] == DATA["epochs"] and rep["updates_checked"] > 0
    bad = json.loads(json.dumps(plan)); bad["epochs"][1][2] = "0" * 16
    with pytest.raises(SystemExit):
        ts.check_pairing(data, bad, ctx)


def test_initial_predictions_are_identical_across_arms(tmp_path, trainset):
    _, data, _ = trainset; rec = ts.make_arm_inits(init_state(), tmp_path / "inits", 0, tmp_path)
    feats = {s.seq_id: s.feat for s in data.check}; outs = []
    for a in so.ARMS:
        m = torch.nn.Linear(512, 5); m.load_state_dict(torch.load(tmp_path / rec[f"{a}_s0"]["path"], weights_only=True)); outs.append(ts.predict(m, feats, "cpu"))
    assert all(np.array_equal(outs[0][k], o[k]) for o in outs[1:] for k in feats)


# ----------------------------------------------------------------------------------------------- updates, resume, padding
@pytest.mark.parametrize("arm", so.ARMS)
def test_six_updates_reproduce_and_three_plus_three_resume(arm, trainset):
    ctx, data, _ = trainset; b = data.epoch(ctx, 0) + data.epoch(ctx, 1); state = init_state()
    a = ts.run_updates(arm, state, b[:6], ALPHA, TRAINING, 10, "cpu", 0); c = ts.run_updates(arm, state, b[:6], ALPHA, TRAINING, 10, "cpu", 0)
    assert a[3] == c[3] and all(torch.equal(a[0].state_dict()[k], c[0].state_dict()[k]) for k in state)
    h = ts.run_updates(arm, state, b[:3], ALPHA, TRAINING, 10, "cpu", 0)
    resume = {"model": {k: v.clone() for k, v in h[0].state_dict().items()}, "opt": h[1].state_dict(), "sched": h[2].state_dict(), "rng": tv.rng_state()}
    d = ts.run_updates(arm, state, b[3:6], ALPHA, TRAINING, 10, "cpu", 0, resume=resume)
    assert max(abs(x - y) for x, y in zip(a[3][3:], d[3])) <= 1e-6
    assert max(float((a[0].state_dict()[k] - d[0].state_dict()[k]).abs().max()) for k in state) <= 1e-6
    assert any(not torch.equal(a[0].state_dict()[k], state[k]) for k in state)


def test_optimiser_binds_only_the_head_with_the_a01_settings():
    m = torch.nn.Linear(512, 5); opt, sched, clip = ts.make_optimiser(TRAINING, 10, m)
    assert isinstance(opt, torch.optim.AdamW) and len(opt.param_groups) == 1
    assert {id(p) for p in opt.param_groups[0]["params"]} == {id(p) for p in m.parameters()}
    g = opt.param_groups[0]
    assert (g["weight_decay"], tuple(g["betas"]), g["eps"], clip) == (1e-4, (0.9, 0.999), 1e-8, 1.0)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sched.step()
    assert abs(lrs[0] - 3e-4 / 4) < 1e-12 and abs(lrs[3] - 3e-4) < 1e-12 and lrs[-1] < lrs[4]


@pytest.mark.parametrize("arm", so.ARMS)
def test_padding_changes_leave_the_batch_loss_and_gradient_unchanged(arm, trainset):
    ctx, data, _ = trainset; seqs = data.epoch(ctx, 0)[0]; runs = ts.batch_runs(seqs)
    m = torch.nn.Linear(512, 5); m.load_state_dict(init_state())
    from echo_routing.temporal import dev_data as dd
    raw = dd.collate(seqs, "cpu"); alt = dict(raw); pad = raw["valid"] == 0
    assert pad.any()
    x = raw["x"].clone(); x[pad[:, None, :].expand_as(x)] = 99.0; lab = raw["labels"].clone(); lab[pad] = 3
    alt["x"], alt["labels"] = x, lab
    l1, _ = ts.loss_from_batch(arm, m, tv.normalise(raw), runs, ALPHA); g1 = torch.autograd.grad(l1, list(m.parameters()))
    l2, _ = ts.loss_from_batch(arm, m, tv.normalise(alt), runs, ALPHA); g2 = torch.autograd.grad(l2, list(m.parameters()))
    assert float(l1) == float(l2) and all(torch.equal(a, b) for a, b in zip(g1, g2))


def test_resume_needs_the_same_identity(tmp_path):
    m = torch.nn.Linear(512, 5); opt, sched, _ = ts.make_optimiser(TRAINING, 10, m)
    tv.save_resume(tmp_path / "resume.pt", m, opt, sched, 3, "identity-a", [])
    with pytest.raises(SystemExit):
        tv.load_resume(tmp_path / "resume.pt", m, opt, sched, "identity-b")


def test_inference_reads_features_and_length_only(trainset):
    _, data, _ = trainset; m = torch.nn.Linear(512, 5); m.load_state_dict(init_state())
    assert list(inspect.signature(ts.predict).parameters) == ["model", "feats", "device"]
    feats = {s.seq_id: s.feat for s in data.check}; renamed = {f"other{i}": s.feat for i, s in enumerate(data.check)}
    a, b = ts.predict(m, feats, "cpu"), ts.predict(m, renamed, "cpu")
    assert all(np.array_equal(a[s.seq_id], b[f"other{i}"]) for i, s in enumerate(data.check))
    lab = {s.seq_id: s.labels for s in data.check}
    seqs = rs.placeholder_seqs(feats)
    assert all(np.array_equal(q.feat, feats[q.seq_id]) and not q.labels.any() and q.labels.size == lab[q.seq_id].size and q.joins.size == 0 for q in seqs)


def test_monitor_reports_every_term_and_training_fit(trainset):
    _, data, _ = trainset; m = torch.nn.Linear(512, 5); m.load_state_dict(init_state())
    mon = ts.monitor(m, data.check, ALPHA, "cpu")
    for k in ("frame", "segment", "order", "objective_order", "segment_mean_ce", "native_accuracy", "multi_accuracy",
              "edit_distance_native", "edit_distance_multi"):
        assert k in mon and np.isfinite(mon[k])


# ----------------------------------------------------------------------------------------------- ledger, projection, counts
def test_ledger_keeps_kinds_and_failed_attempts(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ts.ledger_append(p, "check", "gpu", "t0", 360.0, True); ts.ledger_append(p, "prepare", "cpu", "t1", 720.0, False, "boom")
    assert abs(ts.ledger_hours(p, "gpu") - 0.1) < 1e-12 and abs(ts.ledger_hours(p, "cpu") - 0.2) < 1e-12
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert rows[1]["ok"] is False and rows[1]["note"] == "boom"


def test_projection_rule():
    per = {"frame": 0.1, "segment": 0.1, "order": 0.3}
    p = ts.projection(1.0, per, ["frame_s0", "order_s1", "order_s2"], 0.75, 4.0)
    assert abs(p["total"] - (1.0 + 0.1 + 0.3 + 0.3 + 0.75)) < 1e-12 and p["within_cap"]
    assert not ts.projection(3.0, per, ["order_s0", "order_s1"], 0.75, 4.0)["within_cap"]


def test_training_counts_describe_runs_joins_and_loss_mass(trainset):
    ctx, data, _ = trainset; lib = library(np.random.default_rng(0))
    c = ts.training_counts(lib, [data.epoch(ctx, e) for e in range(DATA["epochs"])], ALPHA)
    assert c["library"]["natives"] == 10 and c["library"]["streams"] == 10
    assert c["library"]["same_view_joins"] == 3                       # (0, 0, 1), (3, 3) and (2, 2, 3)
    assert c["library"]["xyx_a4c_a5c_middles"] == 1                   # (2, 3, 2) only
    for w in ("frame", "segment"):
        shares = c["batches"]["mass"][w]
        assert abs(shares["native"] + shares["stream"] - 1) < 1e-9 and abs(sum(shares["by_class"].values()) - 1) < 1e-9


# ----------------------------------------------------------------------------------------------- the development cache adapter
def write_dev_cache(path: Path, first: list[str], second: list[str], canary_first: bool, compressed: bool = False) -> dict:
    """A01's member layout; the first block's rows hold the canary value."""
    rng = np.random.default_rng(1); ids = first + second; n = rng.integers(7, 15, len(ids)); s_off = np.concatenate([[0], np.cumsum(n)])
    nwin = (n + 4) // 5; w_off = np.concatenate([[0], np.cumsum(nwin)])
    samples = rng.standard_normal((int(s_off[-1]), 512)).astype(np.float32); window = rng.standard_normal((int(w_off[-1]), 512)).astype(np.float32)
    k = len(first)
    if canary_first:
        samples[:s_off[k]] = 1e30; window[:w_off[k]] = 1e30
    save = np.savez_compressed if compressed else np.savez
    with open(path, "wb") as f:
        save(f, identity="dev-id", ids=np.asarray(ids), n=n, sample_offsets=s_off, window_offsets=w_off, centres=np.arange(int(w_off[-1])),
             markers=np.zeros(int(w_off[-1]), np.int64), window=window, samples=samples)
    return {"samples": samples, "window": window, "s_off": s_off, "w_off": w_off, "ids": ids}


class Recorder:
    """An opener that records every byte range read, for canary checks."""

    def __init__(self) -> None:
        self.reads: list[tuple[int, int]] = []

    def __call__(self, path, mode="rb"):
        rec = self.reads

        class F(io.FileIO):
            def read(self, size=-1):
                pos = self.tell(); out = super().read(size); rec.append((pos, len(out) if out else 0)); return out

            def readinto(self, b):
                pos = self.tell(); n = super().readinto(b); rec.append((pos, n or 0)); return n
        return F(path, "r")


def overlaps(reads, lo, hi) -> bool:
    return any(n > 0 and pos < hi and pos + n > lo for pos, n in reads)


def test_member_layout_rejects_compressed_archives(tmp_path):
    p = tmp_path / "dev.npz"; write_dev_cache(p, ["s0", "s1"], ["c0", "c1"], True, compressed=True)
    with pytest.raises(SystemExit):
        rs.member_layout(p, "samples")


def test_calibration_reads_never_touch_canary_selection_rows(tmp_path):
    p = tmp_path / "dev.npz"; sel, cal = ["s0", "s1", "s2"], ["c0", "c1", "c2", "c3"]; d = write_dev_cache(p, sel, cal, True)
    rec = Recorder(); acc = rs.DevAccess({"calibration"})
    got = rs.dev_half(p, "calibration", cal, acc, opener=rec)
    assert got["identity"] == "dev-id" and list(got["ids"]) == cal
    for j, c in enumerate(cal):
        i = 3 + j; assert np.array_equal(got["samples"][c], d["samples"][d["s_off"][i]:d["s_off"][i + 1]])
    assert np.array_equal(got["window"], d["window"][d["w_off"][3]:]) and not (got["window"] > 1e29).any()
    for member, off_key in (("samples", "s_off"), ("window", "w_off")):
        lay = rs.member_layout(p, member); row = 512 * 4
        assert not overlaps(rec.reads, lay.offset, lay.offset + int(d[off_key][3]) * row), member


def test_half_block_requires_one_contiguous_block_in_order():
    ids = ["s0", "s1", "c0", "c1", "c2"]
    assert rs.half_block(ids, ["c0", "c1", "c2"]) == (2, 5) and rs.half_block(ids, ["s0", "s1"]) == (0, 2)
    for bad in (["c1", "c0", "c2"], ["c0", "c2"], ["c0", "c1", "c9"], ["s1", "c0"]):
        with pytest.raises(SystemExit):
            rs.half_block(ids, bad)


def test_selection_is_refused_before_the_barrier(tmp_path):
    p = tmp_path / "dev.npz"; write_dev_cache(p, ["s0", "s1"], ["c0"], True)
    with pytest.raises(SystemExit):
        rs.dev_half(p, "selection", ["s0", "s1"], rs.DevAccess({"calibration"}))
    with pytest.raises(SystemExit):
        rs.DevAccess({"calibration", "test"})


def test_encoders_and_eager_loaders_are_forbidden():
    import echo_routing.vit.target_mvit as tm
    import scripts.prepare_video_feature_temporal as pv
    import scripts.report_center_supervision as rcs
    import scripts.report_video_feature_temporal as rvt
    undo = rs.forbid_encoders()
    try:
        for fn in (lambda: tm.build_model(None, None), lambda: pv.build_source(None, 0, "cpu"), lambda: pv.extract(None, None, {}, "cpu", 1),
                   lambda: rcs.validation({}, {}, None), lambda: rvt.dev_features(None, {}, "cpu", 1), lambda: tm.TargetMViT.forward(None)):
            with pytest.raises(RuntimeError):
                fn()
    finally:
        undo()
    src = inspect.getsource(rs)
    for name in ("dev_features(", "rcs.validation(", "Inputs(", "build_source(", "pv.extract(", "np.load("):
        assert name not in src, name


# ----------------------------------------------------------------------------------------------- the git-object proof
def git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"; (r / "docs" / "results" / "x").mkdir(parents=True); git(r, "init", "-q")
    (r / "docs" / "results" / "x" / "choices.json").write_text('{"a": 1}\n'); (r / "README").write_text("r\n")
    git(r, "add", "."); git(r, "commit", "-q", "-m", "c1")
    return r, git(r, "rev-parse", "HEAD"), "docs/results/x/choices.json"


def test_proof_round_trip(repo):
    r, commit, rel = repo
    proof = rs.git_proof(r, commit, rel); data = (r / rel).read_bytes()
    blob = rs.verify_proof(json.loads(json.dumps(proof)), data, rel, commit)
    assert blob == git(r, "rev-parse", f"{commit}:{rel}")


def test_proof_rejects_changed_bytes_other_commits_and_tampered_objects(repo):
    r, commit, rel = repo; proof = rs.git_proof(r, commit, rel); data = (r / rel).read_bytes()
    with pytest.raises(SystemExit):
        rs.verify_proof(proof, data + b" ", rel, commit)
    with pytest.raises(SystemExit):
        rs.verify_proof(proof, data, rel, "0" * 40)
    with pytest.raises(SystemExit):
        rs.verify_proof(proof, data, "docs/results/x/other.json", commit)
    bad = json.loads(json.dumps(proof)); tree = next(k for k, v in bad["objects"].items() if v["kind"] == "tree")
    raw = bytearray(base64.b64decode(bad["objects"][tree]["raw"])); raw[-1] ^= 1; bad["objects"][tree]["raw"] = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(SystemExit):
        rs.verify_proof(bad, data, rel, commit)


def test_proof_needs_a_committed_unchanged_file(repo):
    r, commit, rel = repo
    (r / rel).write_text('{"a": 2}\n')
    with pytest.raises(SystemExit):
        rs.git_proof(r, commit, rel)                      # the working copy differs from the commit
    (r / "docs" / "results" / "x" / "new.json").write_text("{}\n")
    with pytest.raises(SystemExit):
        rs.git_proof(r, commit, "docs/results/x/new.json")


def test_barrier_rejects_missing_changed_or_foreign_choices(repo, tmp_path):
    r, commit, rel = repo; prov = {"code": "c", "config_sha256": "k", "runs": {"frame_s0": "i"}}
    data = json.dumps({"provenance": prov, "choices": {}}, indent=1).encode()
    (r / rel).write_bytes(data); git(r, "add", rel); git(r, "commit", "-q", "-m", "choices"); commit = git(r, "rev-parse", "HEAD")
    twin = tmp_path / "twin.json"; twin.write_bytes(data); proof = tmp_path / "proof.json"; proof.write_text(json.dumps(rs.git_proof(r, commit, rel)))
    import hashlib
    sha = hashlib.sha256(data).hexdigest()
    assert rs.barrier(r / rel, twin, proof, rel, sha, commit, prov)["provenance"] == prov
    with pytest.raises(SystemExit):
        rs.barrier(r / rel, twin, proof, rel, "f" * 64, commit, prov)
    with pytest.raises(SystemExit):
        rs.barrier(r / rel, twin, proof, rel, sha, commit, {**prov, "code": "other"})
    with pytest.raises(SystemExit):
        rs.barrier(r / rel, twin, tmp_path / "missing.json", rel, sha, commit, prov)
    twin.write_bytes(data + b"\n")
    with pytest.raises(SystemExit):
        rs.barrier(r / rel, twin, proof, rel, sha, commit, prov)
    with pytest.raises(SystemExit):
        rs.barrier(r / rel, twin, proof, rel, None, commit, prov)


def test_choice_records_follow_the_calibration_rule():
    goal = {"native_macro_f1_min": {"calibration": 0.98}, "boundary_recall_min": 0.95}
    grid = [{"order": i, "params": {"half_window": h, "threshold_quantile": q, "nms_radius": 5, "threshold": 0.1},
             "native_macro_f1": 0.99, "boundary_recall_multi": 0.96, "segment_f1_50_multi": 0.9 + 0.001 * (i == 7)}
            for i, (h, q) in enumerate((h, q) for h in (5, 7, 9, 11, 15) for q in (0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999))]
    rec = {"status": "feasible", "params": grid[7]["params"], "n_feasible": 40, "grid": grid}
    rs.rederive_choice(rec, goal)
    with pytest.raises(SystemExit):
        rs.rederive_choice({**rec, "params": grid[6]["params"]}, goal)


# ----------------------------------------------------------------------------------------------- tables
def test_edit_distance_and_transcripts():
    assert st.edit_distance((2, 3, 2), (2, 3, 2)) == 0 and st.edit_distance((2, 3, 2), (2, 2)) == 1
    assert st.edit_distance((), (1, 2)) == 2 and st.edit_distance((0, 1, 2), (2, 1, 0)) == 2
    assert st.transcript([1, 1, 2, 2, 1]) == (1, 2, 1)
    assert st.transcript_distance([2, 2, 2, 2], [2, 2, 3, 2]) == pytest.approx(2 / 3)


def synthetic_tables(seed: int, shift: float):
    rng = np.random.default_rng(seed); vids = [f"p{i // 3}_{i % 3}" for i in range(30)]
    nat = pd.DataFrame({"video_id": vids, "label": [i % 5 for i in range(30)], "majority": [(i % 5) if rng.random() > 0.2 + shift else (i + 1) % 5 for i in range(30)],
                        "cuts": rng.integers(0, 3, 30)})
    streams = [f"st{i}" for i in range(12)]
    ms = pd.DataFrame({"stream": streams, "seg_matched": rng.integers(2, 6, 12), "seg_pred": rng.integers(6, 9, 12), "seg_true": rng.integers(6, 8, 12)})
    mf = pd.DataFrame({"stream": np.repeat(streams, 2), "t_matched": rng.integers(1, 3, 24), "t_true": np.full(24, 3), "p_pred": rng.integers(2, 5, 24),
                       "false_split": rng.integers(0, 2, 24), "same_join": np.ones(24, int)})
    return {"native": nat, "multi_seg": ms, "multi_frag": mf}


def test_joint_bootstrap_reproduces_the_paired_bootstrap():
    a, b = synthetic_tables(1, 0.0), synthetic_tables(2, 0.1)
    want = paired_bootstrap(a, b); got = st.paired(st.system_draws(a), st.system_draws(b))
    for k, v in want.items():
        assert got[k]["delta"] == pytest.approx(v["delta"], abs=1e-12) and got[k]["ci95"] == pytest.approx(v["ci95"], abs=1e-12)
    fp_a = int(a["multi_seg"]["seg_pred"].sum() - a["multi_seg"]["seg_matched"].sum()); fp_b = int(b["multi_seg"]["seg_pred"].sum() - b["multi_seg"]["seg_matched"].sum())
    assert got["multi FP segments"]["delta"] == fp_a - fp_b
    fn_a = int(a["multi_seg"]["seg_true"].sum() - a["multi_seg"]["seg_matched"].sum()); fn_b = int(b["multi_seg"]["seg_true"].sum() - b["multi_seg"]["seg_matched"].sum())
    assert got["multi FN segments"]["delta"] == fn_a - fn_b


def test_segment_status_names_recovered_and_lost_references():
    labels = np.array([2] * 10 + [3] * 4 + [2] * 10)
    a = labels.copy(); b = np.array([2] * 24)
    res = st.segment_status(labels, a, b)
    assert res["status"] == ["a only", "a only", "a only"]                # one 24-sample prediction overlaps no reference at IoU 0.5
    assert res["counts_a"]["tp"] == 3 and res["counts_b"] == {"tp": 0, "fp": 1, "fn": 3, "pred": 1, "ref": 3}
    res = st.segment_status(labels, b, a)
    assert res["status"] == ["b only"] * 3 and len(res["new_unmatched"]) == 1 and len(res["removed_unmatched"]) == 0


def test_readings_follow_the_fixed_rules():
    POS, NONE = st.POS, st.NONE
    def ver(ob, bf, bo, oo):
        return {"O - B": {h: ob for h in st.HALVES}, "B - F": {h: bf for h in st.HALVES}, "O - F": {h: NONE for h in st.HALVES},
                "B - frozen Ordered": {h: bo for h in st.HALVES}, "O - frozen Ordered": {h: oo for h in st.HALVES}}
    ok = {"met": True, "constraint_failures": []}; bad = {"met": False, "constraint_failures": [{"seed": 0}]}
    r = st.readings(ver(NONE, NONE, NONE, NONE), {"B": ok, "O": ok}, {"transcript_or_recovery_gain": False, "tradeoff": False})
    assert r["applied"] == ["uncertain_or_no_qualifying_gain", "end"]
    r = st.readings(ver(POS, POS, NONE, NONE), {"B": ok, "O": ok}, {"transcript_or_recovery_gain": False, "tradeoff": False})
    assert "order_objective_effect" in r["applied"] and "segment_balancing_effect" in r["applied"] and "complete_system_progress" not in r["applied"]
    r = st.readings(ver(POS, NONE, NONE, POS), {"B": ok, "O": bad}, {"transcript_or_recovery_gain": True, "tradeoff": True})
    assert "order_objective_effect" not in r["applied"] and "transcript_only_or_local_effect" in r["applied"] and "trade_off" in r["applied"]
