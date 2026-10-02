"""Round-2 protocol guards (docs/superpowers/plans/2026-10-02-model-development-round2.md, Task 1)."""
from __future__ import annotations

import pandas as pd
import pytest

from echo_routing.temporal.dev_run import check_index, feature_dim_of
from scripts.model_development_queue import plan_jobs, run_identity

CFG = {"experiment": "round2", "temporal_training": {"lr": 0.0003}}


def _ident(job: str = "M5", **kw) -> str:
    args = {"experiment": "round2", "config": CFG, "protocol_sha": "p" * 64, "features_sha": "f" * 64, "job": job, "seed": 0, "lr": None}
    args.update(kw)
    return run_identity(args["experiment"], args["config"], args["protocol_sha"], args["features_sha"], args["job"], args["seed"], args["lr"])


def test_identity_changes_with_every_input():
    base = _ident()
    assert base != _ident(features_sha="g" * 64)
    assert base != _ident(protocol_sha="q" * 64)
    assert base != _ident(config={**CFG, "temporal_training": {"lr": 0.001}})
    assert base != _ident(seed=1) and base != _ident(job="M3p") and base != _ident(experiment="round1")
    assert base == _ident()


def test_done_jobs_are_skipped_only_with_the_same_identity():
    ids = {"M5": _ident("M5"), "M3p": _ident("M3p")}
    status = {"M5": {"state": "done", "identity": ids["M5"]}, "M3p": {"state": "failed", "identity": ids["M3p"]}}
    assert plan_jobs(status, ["M5", "M3p"], ids) == ["M3p"]
    other_features = {"M5": _ident("M5", features_sha="g" * 64), "M3p": _ident("M3p", features_sha="g" * 64)}
    with pytest.raises(SystemExit):
        plan_jobs(status, ["M5", "M3p"], other_features)          # an E0 status file can never satisfy an E1 queue
    with pytest.raises(SystemExit):
        plan_jobs({"M5": {"state": "done", "run_dir": "x"}}, ["M5"], ids)   # round-1 status files carry no identity
    assert plan_jobs({}, ["M5"], ids) == ["M5"]


def _proto() -> dict:
    return {"train": {"clips": ["t1", "t2"]}, "validation": {"selection": {"clips": ["s1"]}, "calibration": {"clips": ["c1"]}}}


def test_cache_index_must_cover_the_protocol_and_exclude_test():
    ok = pd.DataFrame({"video_id": ["t1", "t2", "s1", "c1"], "split": ["train", "train", "validation", "validation"]})
    check_index(ok, _proto())
    with pytest.raises(SystemExit):
        check_index(ok[ok["video_id"] != "c1"], _proto())
    with pytest.raises(SystemExit):
        check_index(pd.concat([ok, pd.DataFrame({"video_id": ["x9"], "split": ["test"]})]), _proto())


def test_feature_dimension_comes_from_the_manifest_and_is_checked():
    assert feature_dim_of({"feature_dim": 512}, 512) == 512
    assert feature_dim_of({}, 768) == 768                        # round-1 manifests declare no dimension
    with pytest.raises(SystemExit):
        feature_dim_of({"feature_dim": 512}, 768)


def test_run_directories_never_collide(tmp_path, monkeypatch):
    import scripts._common as common

    class P:
        runs_root = tmp_path
    monkeypatch.setattr(common, "load_paths", lambda: P())
    a = common.new_run_dir({"seed": 0}, "same"); b = common.new_run_dir({"seed": 0}, "same")
    assert a != b and a.is_dir() and b.is_dir()
