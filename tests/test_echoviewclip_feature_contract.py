"""E1 feature-cache contract (round 2, Task 3): probe sampling, cache records, alignment with round 1, test exclusion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from echo_routing.features.cache import VideoFeatures, load_video_features, save_video_features
from scripts.extract_echoviewclip_features import jobs_from
from scripts.train_feature_probe import cache_arrays, check_alignment, epoch_draw


def test_epoch_draw_takes_16_samples_per_clip_and_balances_classes():
    n_samples = [40, 8, 60, 30, 5, 70]; labels = [0, 0, 0, 0, 1, 1]
    pairs = epoch_draw(n_samples, labels, k=16, n_clips=600, rng=np.random.default_rng(0))
    assert len(pairs) == 600 * 16
    assert all(0 <= p < n_samples[c] for c, p in pairs)
    by_clip: dict[int, int] = {}
    for c, _ in pairs:
        by_clip[c] = by_clip.get(c, 0) + 1
    assert all(v % 16 == 0 for v in by_clip.values())
    share_class1 = sum(v for c, v in by_clip.items() if labels[c] == 1) / len(pairs)
    assert 0.4 < share_class1 < 0.6                        # 2 of 6 clips, but half of the draws
    short = [p for c, p in pairs if c == 4]
    assert len(short) >= 16 and len(set(short)) <= 5        # a 5-sample clip is sampled with replacement


def test_cache_record_is_probe_output_on_the_embedding():
    probe = torch.nn.Linear(8, 5)
    emb = np.random.default_rng(1).normal(size=(6, 8)).astype(np.float32)
    vf = cache_arrays("v1", np.arange(0, 18, 3), np.arange(0, 18, 3) / 30.0, emb, probe)
    assert vf.frame_idx.dtype == np.int32 and vf.feat.shape == (6, 8) and vf.prob.shape == (6, 5)
    assert np.allclose(vf.prob, torch.softmax(torch.from_numpy(vf.logit), 1).numpy(), atol=1e-6)
    with torch.no_grad():
        assert np.allclose(vf.logit, probe(torch.from_numpy(emb)).numpy(), atol=1e-6)


def test_round_trip_through_the_cache_reader(tmp_path):
    probe = torch.nn.Linear(8, 5)
    emb = np.random.default_rng(2).normal(size=(4, 8)).astype(np.float32)
    vf = cache_arrays("v2", np.array([0, 3, 6, 9]), np.array([0, 0.1, 0.2, 0.3]), emb, probe)
    back = load_video_features(save_video_features(tmp_path, vf))
    assert np.array_equal(back.frame_idx, vf.frame_idx) and np.allclose(back.t_sec, vf.t_sec)
    assert np.allclose(back.prob, vf.prob) and np.allclose(back.logit, vf.logit)
    assert np.allclose(back.feat, emb, atol=1e-2)           # features are stored as float16


def test_alignment_with_round_one_is_enforced():
    v3 = VideoFeatures("v", np.array([0, 3, 6], np.int32), np.array([0, .1, .2], np.float32), np.zeros((3, 4)), np.zeros((3, 5)), np.zeros((3, 5)))
    check_alignment("v", np.array([0, 3, 6]), v3)
    with pytest.raises(SystemExit):
        check_alignment("v", np.array([0, 3, 7]), v3)


def _fake(tmp_path, split_of_c1="validation", shift_gamma=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    proto = {"train": {"clips": ["t1"]}, "validation": {"selection": {"clips": ["s1"]}, "calibration": {"clips": ["c1"]}}}
    rows = [{"video_id": v, "split": sp, "frames_dir": str(tmp_path / v)} for v, sp in (("t1", "train"), ("s1", "validation"), ("c1", split_of_c1))]
    man = tmp_path / "manifest.csv"; pd.DataFrame(rows).to_csv(man, index=False)
    for var in ("orig", "gamma090"):
        d = tmp_path / "v3" / var; d.mkdir(parents=True, exist_ok=True)
        for v in ("t1", "s1", "c1"):
            idx = np.array([0, 3, 6]) + (1 if (shift_gamma and var == "gamma090" and v == "s1") else 0)
            np.savez(d / f"{v}.npz", frame_idx=idx)
    return proto, man, tmp_path / "v3"


def test_extractor_jobs_follow_round_one_and_never_read_test(tmp_path):
    proto, man, v3 = _fake(tmp_path)
    jobs = jobs_from(proto, man, v3)
    assert [j["video_id"] for j in jobs] == ["t1", "s1", "c1"] and jobs[1]["frame_idx"].tolist() == [0, 3, 6]
    proto, man, v3 = _fake(tmp_path / "b", split_of_c1="test")
    with pytest.raises(SystemExit):
        jobs_from(proto, man, v3)
    proto, man, v3 = _fake(tmp_path / "c", shift_gamma=True)
    with pytest.raises(SystemExit):
        jobs_from(proto, man, v3)
