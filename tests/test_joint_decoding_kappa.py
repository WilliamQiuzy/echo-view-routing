"""Pure helpers of scripts/joint_decoding_kappa.py (the post-hoc switch-cost range check) on synthetic data, and its
configuration: only the kappa range differs from the original round."""
from pathlib import Path

import pytest

import scripts.joint_decoding_kappa as jk
from echo_routing.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_extends_only_the_kappa_range():
    new = load_config(ROOT / "configs/experiments/joint_boundary_decoding_kappa_v1.yaml")
    old = load_config(ROOT / "configs/experiments/joint_boundary_decoding_v1.yaml")
    assert new["joint_decoding"]["grid"]["kappa"] == [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0]
    assert new["joint_decoding"]["grid"]["lambda"] == old["joint_decoding"]["grid"]["lambda"]
    assert set(old["joint_decoding"]["grid"]["kappa"]) <= set(new["joint_decoding"]["grid"]["kappa"])
    rest = lambda d: {k: v for k, v in d["joint_decoding"].items() if k != "grid"}  # noqa: E731
    assert rest(new) == rest(old)                                                   # decoder, choice, goal, inputs unchanged
    assert "post-hoc" in new["kappa_supplement"]["status"] and new["experiment"] == "joint_boundary_decoding_kappa_v1"


def _rec(lam, kap, seg, native=0.99):
    return {"params": {"lambda": lam, "kappa": kap}, **{k: 0.5 for k in jk.FLAT}, "segment_f1_50_multi": seg, "native_macro_f1": native}


def test_original_settings_must_reproduce_exactly():
    recs = [_rec(0.0, 8.0, 0.97), _rec(0.0, 16.0, 0.98), _rec(0.5, 8.0, 0.96)]
    assert jk.reproduce_original_grid(recs, [_rec(0.0, 8.0, 0.97), _rec(0.5, 8.0, 0.96)], 0) == 2
    with pytest.raises(SystemExit, match="reproduce"):
        jk.reproduce_original_grid(recs, [_rec(0.0, 8.0, 0.9700000001)], 0)


def test_upper_limit_flag():
    ch = {"J0": {"params": {"lambda": 0.0, "kappa": 128.0}}, "J1": {"params": {"lambda": 0.5, "kappa": 64.0}}, "J2": {"params": None}}
    assert jk.upper_limit(ch, [0.0, 64.0, 128.0]) == {"J0": True, "J1": False, "J2": False}


def _ch(r, cat, videos=("a_1", "b_1")):
    return {"stream": "s", "r": r, "cat": cat, "videos": list(videos)}


def test_gains_and_losses_against_the_old_choice():
    asrf = [_ch(10, "merged"), _ch(20, "merged"), _ch(30, "found, both views right"), _ch(40, "no candidate")]
    old = [_ch(10, "found, both views right"), _ch(20, "found, both views right"), _ch(30, "found, both views right"), _ch(40, "no candidate")]
    new = [_ch(10, "found, both views right"), _ch(20, "merged", ("c_2", "d_2")), _ch(30, "found, a view wrong"), _ch(40, "found, both views right")]
    g = jk.gains_losses(new, old, asrf)
    assert g["found_by_old"]["count"] == 3 and g["kept"]["count"] == 2 and g["lost"] == {"count": 1, "recordings": 2, "sessions": 2}
    assert g["lost_categories"] == {"merged": 1} and g["found_only_now"]["count"] == 1
    assert g["recovered_by_old"]["count"] == 2 and g["recovered_kept"]["count"] == 1


def _stab(j0, j1, extra):
    return {"J0 - (L1 + B3)": {"stable": j0}, "J1 - (L1 + B3)": {"stable": j1}, "J1 - (J0)": {"stable": extra}}


def test_decision_table():
    goals = {"J0": {"met": False}, "J1": {"met": False}}
    assert jk.decisions(_stab(False, False, False), goals)["case"] == "neither_improves"
    assert jk.decisions(_stab(True, True, False), goals)["case"] == "j0_only"
    assert jk.decisions(_stab(True, True, True), goals)["case"] == "j1_extra"
    assert jk.decisions(_stab(False, True, False), goals)["case"] == "j1_improves_without_extra_gain_over_j0"
    assert jk.decisions(_stab(False, False, True), goals)["case"] == "j1_extra"     # J1 above J0 without beating L1 + B3: reported as local
