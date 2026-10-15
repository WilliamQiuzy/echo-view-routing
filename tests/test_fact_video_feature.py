"""Training side of the FACT video feature round (configs/experiments/fact_video_feature_v1.yaml): the fixed recipe and
pins, the token rule, the official checkout's state, the bit-identical strict initial state, identities, unpadding and
padding isolation, queue identity rules, the budget projection, the timing fit and the smoke gate. CPU only; the
official FACT model is built from the local pinned checkout on short synthetic sequences; no real feature or label.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import scripts.train_fact_video_feature as tf
import scripts.train_video_feature_temporal as tv
from echo_routing.temporal import dev_data as dd
from echo_routing.temporal import fact_head as fh
from scripts.train_l1_asrf import batch_digest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "fact_video_feature_v1.yaml"
SHEET = ROOT / "docs" / "superpowers" / "plans" / "2026-10-15-fact-video-feature-training.md"
REPO = ROOT / "third_party" / "fact"
needs_fact = pytest.mark.skipif(not (REPO / "models" / "blocks.py").is_file(), reason="the pinned official FACT checkout is not present")
RULE = fh.TokenRule(8, 2.654333333333333, 1.5, 12)
DATA = {"epochs": 2, "updates_per_epoch": 5, "total_updates": 10, "streams_per_batch": 2, "natives_per_batch": 2, "native_window": 16,
        "native_per_epoch": 10}


def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())["fact_video_feature"]


# ----------------------------------------------------------------------------------------------- the fixed configuration
def test_config_holds_the_task_sheet_pins_and_the_fixed_recipe():
    c = cfg(); sheet = SHEET.read_text()
    table = dict(re.findall(r"\| `([^`]+)`[^|]*\| `([0-9a-f]{64})` \|", sheet))
    assert len(table) == 8
    pins = {v["path"]: v["sha256"] for k, v in c["source"].items() if isinstance(v, dict) and "sha256" in v}
    assert {p: pins[p] for p in table} == table
    import hashlib
    assert c["task_sheet"]["sha256"] == hashlib.sha256(SHEET.read_bytes()).hexdigest()
    assert c["queue"] == ["FACT_s0", "FACT_s1", "FACT_s2"] and c["seeds"] == [0, 1, 2]
    assert (c["budget"]["gpu_hours_max"], c["budget"]["gpu_reserve_hours"], c["budget"]["cpu_hours_max"]) == (12.0, 1.25, 4.0)
    f = c["fact"]
    assert f["commit"] == fh.FACT_PINNED and f["base_config"] == "gtea.yaml" and f["trans"] is False and f["match"] == "o2o" and f["bg_class"] == []
    assert (f["expected_longest_transcript"], f["expected_ntoken"], f["token_ratio"]) == (8, 12, 1.5)
    assert f["optimizer"] == {"name": "Adam", "lr": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "grad_clip": 10.0}
    assert f["schedule"] == {"name": "step", "decay_at_completed_updates": 11719, "factor": 0.1}
    d = c["data"]
    assert (d["epochs"], d["updates_per_epoch"], d["total_updates"], d["streams_per_batch"], d["natives_per_batch"], d["native_window"]) == (50, 375, 18750, 4, 4, 256)
    assert set(c["readings"]) == {"complete_system_progress", "learned_system_gain_without_strong_reference_win", "one_half_or_local_effect",
                                  "trade_off", "uncertain_or_no_qualifying_gain", "end"}
    assert c["checks"]["smoke"]["updates"] == 2000 and c["checks"]["reproducibility"]["tolerance"] == 1.0e-6


def test_token_rule_gate():
    f = cfg()["fact"]
    tf.check_rule(RULE, f)
    for bad in (fh.TokenRule(7, 2.654333333333333, 1.5, 10), fh.TokenRule(8, 2.65, 1.5, 12), fh.TokenRule(8, 2.654333333333333, 1.5, 13)):
        with pytest.raises(SystemExit):
            tf.check_rule(bad, f)
    streams = [np.array([0] * 3 + [1] * 2 + [0] * 4), np.array([2] * 5)]
    r = fh.token_rule(streams, 4, 1.5)
    assert (r.max_transcript, r.ntoken) == (3, 4) and abs(r.average_transcript - (3 + 1 + 4) / 6) < 1e-12


# ----------------------------------------------------------------------------------------------- the official checkout
def git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_official_checkout_must_be_at_the_commit_without_tracked_changes(tmp_path):
    r = tmp_path / "official"; (r / "models").mkdir(parents=True); git(r, "init", "-q")
    (r / "models" / "blocks.py").write_text("x = 1\n"); git(r, "add", "."); git(r, "commit", "-q", "-m", "c"); head = git(r, "rev-parse", "HEAD")
    st = tf.official_state(r, head)
    assert st["head"] == head and st["tracked_changes"] == [] and st["files"] == 1 and len(st["digest"]) == 64
    (r / "untracked.txt").write_text("u\n")
    assert tf.official_state(r, head)["digest"] == st["digest"]                 # untracked files are not official sources
    (r / "models" / "blocks.py").write_text("x = 2\n")
    with pytest.raises(SystemExit):
        tf.official_state(r, head)
    git(r, "checkout", "--", "models/blocks.py")
    with pytest.raises(SystemExit):
        tf.official_state(r, "0" * 40)


# ----------------------------------------------------------------------------------------------- initial states
@needs_fact
def test_initial_state_is_a_bit_identical_strictly_loaded_copy(tmp_path):
    head = fh.FACTHead(REPO, 512, RULE, num_classes=5, base_config="gtea.yaml")
    torch.manual_seed(0); src = tmp_path / "FACT_s0.pt"; torch.save(head.build().state_dict(), src)
    rec = tf.copy_init(src, tmp_path / "inits" / "FACT_s0.pt", tmp_path)
    assert (tmp_path / rec["path"]).read_bytes() == src.read_bytes()
    info = tf.strict_init(head, tmp_path / rec["path"])
    assert info["parameters"] > 10_000_000 and info["tensors"] > 50
    assert tf.copy_init(src, tmp_path / "inits" / "FACT_s0.pt", tmp_path) == rec
    st = torch.load(src, weights_only=True); k = next(iter(st)); bad = dict(st); bad[k + "_renamed"] = bad.pop(k); torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(SystemExit):
        tf.strict_init(head, tmp_path / "bad.pt")
    shp = dict(st); k2 = next(n for n, v in st.items() if v.ndim == 2); shp[k2] = torch.zeros(shp[k2].shape[0] + 1, shp[k2].shape[1]); torch.save(shp, tmp_path / "shape.pt")
    with pytest.raises(SystemExit):
        tf.strict_init(head, tmp_path / "shape.pt")
    (tmp_path / "other.pt").write_bytes(b"other")
    with pytest.raises(SystemExit):
        tf.copy_init(tmp_path / "other.pt", tmp_path / "inits" / "FACT_s0.pt", tmp_path)


def test_identity_covers_every_input():
    base = tf.identity_payload(cfg(), 0, {"head": "c", "digest": "d"}, "expanded", RULE.__dict__,
                               {"features": "f", "train_cache_sha256": "t", "sequence_file": "q", "plan": "p", "init": "i"}, "code")
    ident = tf.identity_of(base)
    for path, value in ((("seed",), 1), (("official", "digest"), "e"), (("expanded_config",), "x"), (("inputs", "plan"), "p2"),
                        (("inputs", "init"), "i2"), (("inputs", "train_cache_sha256"), "t2"), (("code",), "c2"), (("config", "fact", "optimizer", "lr"), 2e-4)):
        alt = json.loads(json.dumps(base)); node = alt
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        assert tf.identity_of(alt) != ident, path


# ----------------------------------------------------------------------------------------------- a synthetic A01 training cache
def library(rng):
    seqs = []
    for i in range(10):
        n = int(rng.integers(18, 30)); seqs.append({"id": f"nat{i:02d}", "kind": "native", "labels": np.full(n, i % 5, np.int64), "joins": np.zeros(0, np.int64)})
    pattern = [(2, 3, 2), (0, 0, 1), (4, 1), (2, 3), (1, 2, 1, 0), (3, 3), (0, 4, 0), (2, 2, 3), (1, 0), (4, 3, 4)]
    for i, p in enumerate(pattern):
        lens = rng.integers(5, 12, len(p)); labels = np.concatenate([np.full(k, c, np.int64) for k, c in zip(lens, p)])
        seqs.append({"id": f"bank{i:02d}", "kind": "bank", "labels": labels, "joins": np.cumsum(lens)[:-1].astype(np.int64)})
    return seqs


@pytest.fixture()
def trainset(tmp_path):
    from echo_routing.evaluate.complementarity import sha256_file
    seqs = library(np.random.default_rng(0)); d = tmp_path / "features" / "train"; d.mkdir(parents=True); n = np.array([q["labels"].size for q in seqs])
    joins = [q["joins"] for q in seqs]
    np.savez(d / "sequences.npz", ids=np.asarray([q["id"] for q in seqs]), kinds=np.asarray([q["kind"] for q in seqs]), n=n,
             labels=np.concatenate([q["labels"] for q in seqs]), join_offsets=np.concatenate([[0], np.cumsum([j.size for j in joins])]),
             joins=np.concatenate(joins), refs_digest=np.asarray(["x"] * len(seqs)))
    rng = np.random.default_rng(1)
    np.savez(d / "ordered_s0.npz", identity="feat-id", ids=np.asarray([q["id"] for q in seqs]), n=n, samples=rng.standard_normal((int(n.sum()), 512)).astype(np.float32))
    ctx = SimpleNamespace(out=tmp_path, vt={"data": DATA}, prep={"features": {"0": {"identity": "feat-id"}}, "sequence_file": {"sha256": sha256_file(d / "sequences.npz")}})
    data = tv.TrainSet(ctx, 0)
    return ctx, data, {"epochs": [[batch_digest(x) for x in data.epoch(ctx, e)] for e in range(DATA["epochs"])]}


def test_every_batch_unpads_to_its_source_sequences(trainset):
    ctx, data, plan = trainset
    rep = tf.check_batches(data, plan, ctx)
    assert rep["epochs_checked"] == 2 and rep["batches_unpadded"] == 10 and rep["sequences_compared"] == 40      # 5 batches of 2 streams + 2 crops per epoch
    bad = json.loads(json.dumps(plan)); bad["epochs"][0][0] = "f" * 16
    with pytest.raises(SystemExit):
        tf.check_batches(data, bad, ctx)


def test_feature_spot_check_reads_the_source_cache(trainset, tmp_path):
    ctx, data, _ = trainset
    rep = tf.spot_check(tmp_path / "features" / "train" / "ordered_s0.npz", data, n_sequences=4)
    assert rep["equal"] and rep["rows_compared"] > 0


@needs_fact
def test_padding_never_changes_the_loss_or_the_prediction(trainset):
    ctx, data, _ = trainset
    head = fh.FACTHead(REPO, 512, RULE, num_classes=5, base_config="gtea.yaml"); torch.manual_seed(0); model = head.build()
    seqs = data.epoch(ctx, 0)[0]; batch = dd.collate(seqs); pad = batch["valid"] == 0
    assert pad.any()
    alt = dict(batch); x = batch["x"].clone(); x[pad[:, None, :].expand_as(x)] = 77.0; lab = batch["labels"].clone(); lab[pad] = 3
    alt["x"], alt["labels"] = x, lab
    assert tf.batch_loss(head, model, batch, seed=1) == tf.batch_loss(head, model, alt, seed=1)
    xs, ys = fh.FACTHead.unpad(alt)
    assert all(torch.equal(a, torch.from_numpy(s.feat)) and torch.equal(b, torch.from_numpy(s.labels)) for a, b, s in zip(xs, ys, seqs))
    rep = tf.padding_check(head, model, batch, seqs, "cpu", 1)             # the check's own helper: eval-mode predictions
    assert rep["padded_positions"] == int(pad.sum()) and rep["loss_diff"] == 0.0 and rep["predictions_equal"]


@needs_fact
def test_inference_ignores_labels_ids_and_padding(trainset):
    import scripts.fact_integration as fi
    ctx, data, _ = trainset
    head = fh.FACTHead(REPO, 512, RULE, num_classes=5, base_config="gtea.yaml"); torch.manual_seed(0); model = head.build()
    rep = fi.isolation(head, model, list(data.check[:4]) + list(data.streams[:2]), "cpu", seed=1)
    assert rep["changed_predictions"] == 0 and rep["max_abs_logit_difference"] == 0 and rep["adapter_decode_differs"] == 0 and rep["comparisons"] == 24


@needs_fact
def test_updates_reproduce_and_resume_through_disk(trainset, tmp_path):
    ctx, data, _ = trainset
    head = fh.FACTHead(REPO, 512, RULE, num_classes=5, base_config="gtea.yaml"); torch.manual_seed(0); init = tmp_path / "init.pt"; torch.save(head.build().state_dict(), init)
    f = cfg()["fact"]; b = data.epoch(ctx, 0)[:4]
    r1 = tf.run_updates(head, init, b, f, 10, "cpu", 0); r2 = tf.run_updates(head, init, b, f, 10, "cpu", 0)
    assert r1[3] == r2[3] and all(torch.equal(r1[0].state_dict()[k], r2[0].state_dict()[k]) for k in r1[0].state_dict())
    rep = tf.resume_check(head, init, b, f, 10, "cpu", 0, tmp_path / "resume.pt", reference=r1)
    assert rep["params_max_abs"] <= 1e-6 and rep["losses_max_abs"] <= 1e-6 and rep["bound"] and rep["lr_equal"]


@needs_fact
def test_monitor_reports_terms_and_fit(trainset):
    ctx, data, _ = trainset
    head = fh.FACTHead(REPO, 512, RULE, num_classes=5, base_config="gtea.yaml"); torch.manual_seed(0); model = head.build()
    mon = tf.monitor(head, model, data.check, "cpu")
    for k in ("frame_accuracy", "segment_f1_50", "transcripts_exact", "loss_total", "loss_frame", "loss_token"):
        assert k in mon
    assert 0 <= mon["transcripts_exact"] <= len(data.check)


# ----------------------------------------------------------------------------------------------- queue, budget, timing, smoke
def test_queue_skips_only_finished_runs_of_the_same_identity(tmp_path):
    ck = tmp_path / "final.pt"; ck.write_bytes(b"w"); import hashlib
    man = tmp_path / "run_manifest.json"
    assert tf.run_state(man, "a") is None
    man.write_text(json.dumps({"identity": "a", "final_checkpoint": str(ck), "final_sha256": hashlib.sha256(b"w").hexdigest(), "completed_updates": 18750}))
    assert tf.run_state(man, "a") == "done" and tf.run_state(man, "b") == "other"
    man.write_text(json.dumps({"identity": "a", "final_checkpoint": str(ck), "final_sha256": hashlib.sha256(b"w").hexdigest(), "completed_updates": 375}))
    assert tf.run_state(man, "a") == "other"


def test_projection_counts_post_training_work_and_the_reserve():
    p = tf.projection(0.5, {"FACT_s0": 3.0, "FACT_s1": 3.1, "FACT_s2": 2.9}, ["FACT_s0", "FACT_s1", "FACT_s2"], 0.6, 1.25, 12.0)
    assert abs(p["total"] - (0.5 + 9.0 + 0.6 + 1.25)) < 1e-12 and p["within_cap"]
    assert not tf.projection(2.0, {"FACT_s0": 3.5, "FACT_s1": 3.5, "FACT_s2": 3.5}, ["FACT_s0", "FACT_s1", "FACT_s2"], 0.5, 1.25, 12.0)["within_cap"]


def test_time_fit_is_linear_in_batch_samples():
    sizes = np.array([400, 600, 800, 1000]); times = 0.1 + 0.0005 * sizes
    a, b = tf.fit_time(times, sizes)
    assert abs(a - 0.1) < 1e-9 and abs(b - 0.0005) < 1e-12
    a, b = tf.fit_time(np.array([0.5, 0.4]), np.array([100, 200]))          # a negative slope falls back to the mean
    assert (a, b) == (0.45, 0.0)
    assert abs(tf.run_hours(0.1, 0.0005, 10, 7000, 2, 18.0) - (0.1 * 10 + 0.0005 * 7000 + 2 * 18.0) / 3600) < 1e-12


@pytest.mark.parametrize("change,ok", [({}, True), ({"transcript_exact": 15}, False), ({"segment_f1_50_min": 0.99}, False),
                                        ({"match_one_to_one": 15}, False), ({"matched_token_class_correct_min": 0.9}, False)])
def test_smoke_gate(change, ok):
    out = {"sequences": 16, "transcript_exact": 16, "segment_f1_50_min": 1.0, "match_one_to_one": 16, "matched_token_class_correct_min": 1.0, **change}
    iso = [{"changed_predictions": 0, "max_abs_logit_difference": 0.0, "adapter_decode_differs": 0}] * 2
    assert tf.smoke_passed(out, iso)["all"] is ok
    assert tf.smoke_passed({**out, **change}, [{"changed_predictions": 1, "max_abs_logit_difference": 0.0, "adapter_decode_differs": 0}, iso[0]])["all"] is False


def test_smoke_check_points():
    due = [n for n in range(1, 2001) if tf.smoke_due(n, 500, 250, 2000)]
    assert due == [500, 750, 1000, 1250, 1500, 1750, 2000]


@needs_fact
def test_training_stream_fit_counts(trainset):
    import scripts.fact_video_feature_post as post
    ctx, data, _ = trainset
    head = fh.FACTHead(REPO, 512, RULE, num_classes=5, base_config="gtea.yaml"); torch.manual_seed(0); model = head.build()
    fit = post.stream_fit(head, model, list(data.streams), "cpu")
    refs = sum(len(np.unique(np.flatnonzero(np.diff(s.labels)) + 1)) + 1 for s in data.streams)
    assert fit["streams"] == len(data.streams) and fit["tp"] + fit["fn"] == refs and sum(fit["missed"]["by_class"].values()) == fit["fn"]
    assert sum(fit["missed"]["by_length"].values()) == fit["fn"] and sum(fit["tokens_minus_segments"].values()) == len(data.streams)
    assert 0.0 <= fit["frame_accuracy"] <= 1.0


def test_ledger_books_every_run_once(tmp_path):
    p = tmp_path / "ledger.jsonl"
    tf.ledger_append(p, "train FACT_s0", "gpu", "2026-10-15T08:00:05Z", 3600.0, True)
    assert not tf.ensure_run_line(p, "FACT_s0", "2026-10-15T08:00:00Z", 3700.0, "queue")
    assert tf.ensure_run_line(p, "FACT_s1", "2026-10-15T09:10:00Z", 600.0, "queue")
    assert abs(tf.ledger_hours(p, "gpu") - (3600.0 + 600.0) / 3600) < 1e-12


def test_training_code_never_builds_an_encoder_or_extracts_features():
    import inspect
    src = inspect.getsource(tf)
    for name in ("build_source(", "pv.extract(", "pv.encode(", "build_model(", "dev_features(", ".embed(", "validation(", "Inputs("):
        assert name not in src, name
