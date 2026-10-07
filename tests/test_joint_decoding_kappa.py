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


def test_decision_table_is_named_in_the_config():
    table = load_config(ROOT / "configs/experiments/joint_boundary_decoding_kappa_v1.yaml")["kappa_supplement"]["decisions"]
    assert set(jk.DECISION_CASES) <= set(table)
    goals = {"J0": {"met": False}, "J1": {"met": False}}
    cases = {(False, False, False): "neither_improves", (True, True, False): "j0_only", (True, False, False): "j0_only",
             (True, True, True): "j1_extra", (False, False, True): "j1_extra", (False, True, False): "j1_only"}
    for flags, case in cases.items():
        d = jk.decisions(_stab(*flags), goals, table)
        assert d["case"] == case and d["action"] == table[case]


# ----------------------------------------------------------------------------------------------- a toy half
import numpy as np  # noqa: E402

from echo_routing.compose.recipe_builder import Fragment, Recipe  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.evaluate.failure_units import stream_units  # noqa: E402
from echo_routing.features.cache import VideoFeatures  # noqa: E402
from echo_routing.temporal.dev_data import EvalSet, native_seq  # noqa: E402
from echo_routing.temporal.dev_run import HALVES  # noqa: E402
from echo_routing.temporal.joint_decoding import joint_viterbi  # noqa: E402
from scripts.joint_boundary_decoding import candidates  # noqa: E402
from scripts.report_l1_asrf import fragment_transfer  # noqa: E402

CFG = {"protocol": {"hz": 10.0}, "selection": {"boundary_tolerance_s": 0.5}}


def _vf(vid, n, label, noise=()):
    prob = np.full((n, 5), 0.02, np.float32); prob[:, label] = 0.92
    for i in noise:                                   # single frames read as another class
        prob[i] = 0.02; prob[i, (label + 1) % 5] = 0.92
    return VideoFeatures(vid, np.arange(n) * 3, np.arange(n) * 0.1, np.zeros((n, 4), np.float32), prob, np.zeros((n, 5), np.float32))


class _Store:
    def __init__(self):
        self.d = {"a_1": _vf("a_1", 40, 2, (5,)), "b_1": _vf("b_1", 40, 3), "c_2": _vf("c_2", 40, 0), "d_3": _vf("d_3", 40, 1, (20,))}

    def get(self, vid, variant):
        return self.d[vid]


def _toy():
    st = _Store()
    natives = tuple(native_seq(v, lab, st.get(v, "orig")) for v, lab in (("a_1", 2), ("b_1", 3), ("c_2", 0), ("d_3", 1)))
    pairs = (render(Recipe("p1", "cell", (Fragment("a_1", 2, "orig", 10, 30), Fragment("b_1", 3, "orig", 0, 20))), st.get),)
    multi = (render(Recipe("m1", "cell", (Fragment("a_1", 2, "orig", 0, 15), Fragment("b_1", 3, "orig", 5, 25), Fragment("c_2", 0, "orig", 0, 15))), st.get),
             render(Recipe("m2", "cell", (Fragment("d_3", 1, "orig", 0, 25), Fragment("c_2", 0, "orig", 10, 30), Fragment("b_1", 3, "orig", 0, 15))), st.get))
    ev = EvalSet("toy", natives, pairs, multi); probs = {s.seq_id: s.prob for s in ev.sequences()}
    b = {sid: np.full(len(p), 0.02) for sid, p in probs.items()}
    for s in (*pairs, *multi):
        b[s.recipe.recipe_id][s.semantic_boundaries] = 0.97
    cuts = {s.recipe.recipe_id: candidates(b[s.recipe.recipe_id]) for s in multi}
    return st, ev, probs, b, cuts


def test_curve_records_on_a_toy_half():
    st, ev, probs, b, cuts = _toy(); merged = {("m1", int(ev.multi[0].semantic_boundaries[0]))}
    r0, r1, r2 = jk.curve_records(CFG, [(0.0, 0.0), (0.0, 8.0), (2.0, 8.0)], probs, b, ev, cuts, merged)
    assert (r0["order"], r1["order"], r2["order"]) == (0, 1, 2) and r1["params"] == {"lambda": 0.0, "kappa": 8.0}
    assert r0["native_switches"] == 4 and r1["native_switches"] == 0                    # argmax keeps the single noisy frames
    assert r0["extra_changes"] == 2 and r1["extra_changes"] == r2["extra_changes"] == 0         # the frame at 20 of m2 lies within 5 of the change at 25
    assert r1["matched_units"] == r1["predicted_segments"] == 6 and r1["unmatched_predicted_segments"] == 0
    assert r0["predicted_segments"] - r0["matched_units"] == r0["unmatched_predicted_segments"] > 0
    assert all(r["merged_total"] == 1 and r["merged_found"] == 1 for r in (r0, r1, r2))


def test_evaluate_core_builds_comparisons_before_gains(monkeypatch):
    import scripts.report_l1_asrf as rl
    import scripts.report_model_development as rmd
    monkeypatch.setattr(rmd, "N_BOOT", 20); monkeypatch.setattr(rl, "N_BOOT", 20)
    st, ev, probs, b, cuts = _toy(); halves = {h: ev for h in HALVES}; seqs = {h: jk.seqs_of(ev, st) for h in HALVES}
    truth = {s.seq_id: s.labels.astype(int) for s in ev.sequences()}; argmax = {k: p.argmax(1) for k, p in probs.items()}
    smooth = {k: joint_viterbi(probs[k], b[k], 0.0, 8.0) for k in probs}
    preds = {n: {h: p for h in HALVES} for n, p in (("B + ASRF", truth), ("L1 + B3", smooth), ("J0 (kappa <= 8)", argmax), ("J1 (kappa <= 8)", argmax),
                                                     ("J0", smooth), ("J1", smooth))}
    transfer = {h: fragment_transfer({s.recipe.recipe_id: stream_units(s.recipe) for s in ev.multi}, lambda v, var: st.get(v, var).prob) for h in HALVES}
    res, _ = jk.evaluate_core(CFG, preds, halves, seqs, {h: cuts for h in HALVES}, transfer, ["a_1", "b_1", "c_2"])
    assert set(res["comparisons"]) == {f"{x} - ({y})" for x, y in jk.CONTRASTS}
    g = res["gains_losses"]["J0"]["selection"]
    assert g["units"] == res["comparisons"]["J0 - (J0 (kappa <= 8))"]["selection"]["segment_changes"]
    assert g["extra_changes"]["old"]["count"] == 2 and g["extra_changes"]["new"]["count"] == 0
    assert g["kept"]["count"] == g["found_by_old"]["count"] == 4 and g["lost"]["count"] == 0
    assert g["unmatched_predicted_segments"] == {"old": 4, "new": 0} and g["native_switches"] == {"old": 4, "new": 0}   # each noisy frame is one unmatched segment
    assert res["diagnostics"]["J1"]["three_clips"] == {"clips": 3, "right": 3, "sessions_right": 3}     # sessions a, b and c
    assert res["diagnostics"]["J0"]["selection"]["segments"] == {"matched": 6, "predicted": 6, "reference": 6, "unmatched_predicted": 0}
