"""Round-1 real-video helpers: screening strata, image-motion cycles, MIMIC burned-in ECG."""
from __future__ import annotations

import numpy as np
import pytest

from echo_routing.realvideo.cycles import estimate_cycle, phase_from_boundaries, phase_locking
from echo_routing.realvideo.ecg import parse_burned_ecg
from echo_routing.realvideo.screening import clip_scores, cohen_kappa, patient_of, session_of

NAMES = ["A", "B", "C"]


def onehot(labels: list[int], k: int = 3) -> np.ndarray:
    return np.eye(k)[np.asarray(labels)]


def test_sustained_change_is_found_with_its_direction():
    s = clip_scores(onehot([0] * 30 + [2] * 30), NAMES, hz=10)
    assert s["stratum"] == "sustained" and s["sides"] == "A>C"
    assert s["contrast"] == pytest.approx(1.0) and s["split_s"] == pytest.approx(3.0)


def test_repeated_returning_excursions():
    lab = [0] * 60
    lab[10:13] = [1] * 3; lab[30:33] = [1] * 3
    s = clip_scores(onehot(lab), NAMES, hz=10)
    assert s["stratum"] == "repeated" and s["n_ret"] == 2 and s["t_ret"] == pytest.approx(0.6)
    assert s["contrast"] < 0.5


def test_single_sample_flicker_needs_min_run_one():
    lab = [0] * 60
    lab[10] = 1; lab[30] = 1
    assert clip_scores(onehot(lab), NAMES, hz=10)["stratum"] == "none"
    assert clip_scores(onehot(lab), NAMES, hz=10, prm={"min_run": 1})["stratum"] == "repeated"


def test_mixed_and_short():
    lab = [0] * 20 + [2] * 40
    lab[35:37] = [0, 0]; lab[45:47] = [0, 0]
    assert clip_scores(onehot(lab), NAMES, hz=10)["stratum"] == "mixed"
    assert clip_scores(onehot([0] * 15), NAMES, hz=10)["stratum"] == "short"


def test_patient_and_export_keys():
    assert session_of("2022-04-29_18-53-17_005_320x240_6") == "2022-04-29_18-53-17"
    assert session_of("2022-03-24_16-19-44_320x240_002_0") == "2022-03-24_16-19-44"
    assert patient_of("2022-04-29_18-53-17_005_320x240_6") == "2022-04-29_18-53-17_005_320x240"
    assert patient_of("2022-03-24_16-19-44_320x240_002_0") == "2022-03-24_16-19-44_320x240_002"
    assert patient_of("2022-07-10_10-58-17_320x240_7") == "2022-07-10_10-58-17_320x240"
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def beating_blob(n: int = 120, period: float = 25.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:48, :64]
    frames = []
    for t in range(n):
        r = 8 + 4 * np.sin(2 * np.pi * t / period)
        img = np.where((yy - 24) ** 2 + (xx - 32) ** 2 < r ** 2, 200.0, 20.0)
        img[:5, :20] = 255.0  # static overlay text
        frames.append(img + rng.normal(0, 5, img.shape))
    return np.stack(frames)


def test_image_cycle_recovers_period_and_boundaries():
    est = estimate_cycle(beating_blob(), fps=30)
    assert est.period_s == pytest.approx(25 / 30, rel=0.03)
    assert est.reliable(min_peak=0.5, min_dip=0.5)
    assert np.allclose(np.diff(est.boundaries), 25, atol=1.5) and est.cv < 0.1


def test_noise_has_no_reliable_cycle():
    rng = np.random.default_rng(1)
    est = estimate_cycle(rng.normal(100, 20, (120, 48, 64)), fps=30)
    assert not est.reliable(min_peak=0.3, min_dip=0.3)


def test_one_second_clip_has_no_reliable_cycle():
    assert not estimate_cycle(beating_blob(n=30), fps=30).reliable(min_peak=0.3, min_dip=0.3)


def test_phase_is_linear_between_boundaries_and_extrapolated():
    ph = phase_from_boundaries(12, np.array([2, 7]), 5.0)
    assert ph[2] == pytest.approx(0.0) and ph[4] == pytest.approx(0.4) and ph[7] == pytest.approx(0.0)
    assert ph[0] == pytest.approx(0.6) and ph[11] == pytest.approx(0.8)


def test_phase_locked_samples_are_detected():
    phase = (np.arange(100) % 20) / 20
    obs, p = phase_locking(phase, (phase >= 0.1) & (phase < 0.2))
    assert obs > 0.9 and p < 0.01
    rng = np.random.default_rng(3)
    ps = [phase_locking(phase, np.isin(np.arange(100), rng.choice(100, 8, replace=False)), n_perm=200, seed=s)[1]
          for s in range(40)]
    assert 0.02 <= np.mean(np.array(ps) < 0.05) <= 0.2  # unrelated samples: roughly nominal rejection


def synthetic_ecg_cine(n: int = 58) -> np.ndarray:
    fr = np.zeros((n, 708, 1016, 3), dtype=np.uint8)
    teal, red = (0, 200, 180), (220, 30, 30)
    fr[:, 650, 50:471] = teal
    for x0 in (60, 260, 460):
        fr[:, 620:651, x0 - 2:x0 + 3] = teal
    for f in range(n):
        x = 50 + 7 * f
        fr[f, 630:637, x:x + 3] = red
    return fr


def test_burned_in_ecg_gives_r_waves_in_frame_units():
    e = parse_burned_ecg(synthetic_ecg_cine(), frame_time_s=0.0337)
    assert e.ok
    assert np.allclose(e.r_frames, (np.array([60, 260, 460]) - 51) / 7, atol=0.3)
    assert e.rr_s == pytest.approx(200 / (7 / 0.0337), rel=0.02)
    ph = e.phase(58)
    assert ph[int(round(e.r_frames[1]))] < 0.05 or ph[int(round(e.r_frames[1]))] > 0.95


def test_r_waves_followed_by_deep_s_waves_are_all_found():
    fr = synthetic_ecg_cine()
    for x0 in (60, 260, 460):
        fr[:, 651:676, x0 + 3:x0 + 7] = (0, 200, 180)  # S wave below baseline right after each R
    assert parse_burned_ecg(fr, frame_time_s=0.0337).r_frames.size == 3


def test_missing_trace_fails_cleanly():
    e = parse_burned_ecg(np.zeros((10, 100, 100, 3), dtype=np.uint8), frame_time_s=0.033)
    assert not e.ok and e.reason == "no trace"

