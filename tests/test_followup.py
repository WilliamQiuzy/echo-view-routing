"""The three checks on development data that follow the comparison (docs/FOLLOWUP.md)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from echoview import Ensemble, reference, windows
from echoview.evaluation import benchmark, comparison, followup, metrics, scorecard
from echoview.evaluation.comparison import ROUTER
from echoview.model import CLASSES
from echoview.pipeline import probabilities_from_logits, softmax
from echoview.reference import B3

from .conftest import BrightnessModel
from .test_comparison import RECORDINGS, logits_from, sequences, write_store
from .test_evaluation_run import tiny  # noqa: F401  (a fixture: a tiny benchmark file and its Images folder)

A4C, A5C = CLASSES.index("A4C"), CLASSES.index("A5C")


def dense_from(seqs: tuple[benchmark.Sequence, ...], rng: np.random.Generator, noise: float = 0.3) -> dict[str, np.ndarray]:
    """Logits (samples, 3 seeds, 5) that name the reference view of every sample."""
    return {s.id: (8.0 * np.eye(5)[s.labels][:, None, :] + noise * rng.normal(size=(len(s), 3, 5))).astype(np.float32) for s in seqs}


def at_published_centres(seqs: tuple[benchmark.Sequence, ...], dense: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {s.id: dense[s.id][windows.window_centres(len(s))] for s in seqs}


def test_interpolation_runs_between_two_outputs_and_is_flat_outside_them():
    centres = windows.window_centres(12)
    assert centres.tolist() == [2, 7, 11]
    logits = np.zeros((3, 2, 5), np.float32)
    logits[0, :, 0], logits[1, :, 1], logits[2, :, 1] = 6.0, 6.0, 6.0
    logits[1, 1, 2] = 7.0  # the second seed reads another view at the middle output
    prob = followup.interpolated(logits, 12)
    by_window = softmax(logits)
    assert prob.shape == (12, 5)
    np.testing.assert_allclose(prob.sum(1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(prob[:3], np.repeat(by_window[0].mean(0)[None], 3, 0), rtol=1e-5)  # before the first output: flat
    np.testing.assert_allclose(prob[4], (0.6 * by_window[0] + 0.4 * by_window[1]).mean(0), rtol=1e-5)  # sample 4 lies 2 of 5 steps on
    np.testing.assert_allclose(prob[7], by_window[1].mean(0), rtol=1e-5)
    np.testing.assert_allclose(prob[11], by_window[2].mean(0), rtol=1e-5)
    np.testing.assert_allclose(followup.interpolated(logits[:1], 1), by_window[0].mean(0)[None], rtol=1e-5)  # one output: nothing to interpolate
    # at an output the interpolated arm is the published arm to the last bit: same softmax, same average
    np.testing.assert_array_equal(prob[centres], probabilities_from_logits(logits, 12)[centres])


def test_the_density_arms_come_from_the_dense_outputs():
    seqs, rng = sequences(), np.random.default_rng(1)
    dense = dense_from(seqs, rng)
    assert followup.dense_check(at_published_centres(seqs, dense), dense, seqs) == {"sequences": len(seqs), "largest_difference": 0.0}
    stored = {key: value + np.float32(5e-4) * rng.normal(size=value.shape).astype(np.float32) for key, value in at_published_centres(seqs, dense).items()}
    arms = followup.density_arms(stored, dense, seqs)
    assert list(arms) == ["every 0.5 s", "every 0.5 s, interpolated", "every 0.2 s", "every 0.1 s"]
    published = comparison.member_probabilities(stored, seqs, (0, 1, 2))
    from_dense = comparison.member_probabilities(at_published_centres(seqs, dense), seqs, (0, 1, 2))
    for s in seqs:
        centres = windows.window_centres(len(s))
        np.testing.assert_array_equal(arms["every 0.5 s"][s.id], published[s.id])  # the two arms at 0.5 s come from the stored logits,
        np.testing.assert_array_equal(arms["every 0.5 s, interpolated"][s.id][centres], published[s.id][centres])
        assert not np.array_equal(published[s.id], from_dense[s.id])  # not from the dense ones at the same centres
        assert arms["every 0.1 s"][s.id].argmax(1).tolist() == s.labels.tolist()  # an output for every sample
        every_second = windows.window_centres(len(s), 2)
        np.testing.assert_allclose(arms["every 0.2 s"][s.id], probabilities_from_logits(dense[s.id][every_second], len(s), 2))
        assert all(arm[s.id].shape == (len(s), 5) for arm in arms.values())
    assert followup.forward_passes(seqs) == {"every 0.5 s": 1.0, "every 0.5 s, interpolated": 1.0,
                                             "every 0.2 s": pytest.approx(2.5, abs=0.3), "every 0.1 s": pytest.approx(5.0, abs=0.6)}


def test_dense_outputs_that_do_not_hold_the_published_ones_are_refused():
    seqs, rng = sequences(), np.random.default_rng(2)
    dense = dense_from(seqs, rng)
    stored = at_published_centres(seqs, dense)
    close = {key: value + 5e-4 for key, value in stored.items()}
    assert followup.dense_check(close, dense, seqs)["largest_difference"] == pytest.approx(5e-4, rel=1e-3)
    off = dict(stored)
    off[seqs[0].id] = stored[seqs[0].id] + 0.01
    with pytest.raises(SystemExit, match="do not agree with the stored ones"):
        followup.dense_check(off, dense, seqs)
    short = dict(dense)
    short[seqs[0].id] = dense[seqs[0].id][:-1]
    with pytest.raises(SystemExit, match="one output for every sample"):
        followup.dense_check(stored, short, seqs)
    for where in (0, 1):  # a number that is not finite, at a published centre (sample 2) or beside it
        holed = dict(dense)
        holed[seqs[0].id] = dense[seqs[0].id].copy()
        holed[seqs[0].id][2 + where, 0, 0] = np.nan
        with pytest.raises(SystemExit, match="not finite"):
            followup.dense_check(stored, holed, seqs)
    holed = dict(stored)
    holed[seqs[0].id] = stored[seqs[0].id].copy()
    holed[seqs[0].id][0, 0, 0] = np.inf
    with pytest.raises(SystemExit, match="not finite"):
        followup.dense_check(holed, dense, seqs)


def test_a_perfect_system_on_each_grid_is_the_reference():
    s = sequences()[-1]
    half_second, fifth = followup.grid_labels(s.labels, 5), followup.grid_labels(s.labels, 2)
    np.testing.assert_array_equal(half_second, scorecard.grid_floor_labels(s.labels))
    np.testing.assert_array_equal(followup.grid_labels(s.labels, 1), s.labels)
    centres = windows.window_centres(len(s), 2)
    assert np.array_equal(fifth[centres], s.labels[centres]) and (fifth != s.labels).sum() <= len(s.view_changes)  # off by one sample at most
    assert (half_second != s.labels).sum() > (fifth != s.labels).sum()


def test_the_offset_moves_the_a5c_logit_only():
    logits = {"x": np.zeros((4, 3, 5), np.float32)}
    moved = followup.with_offset(logits, 1.5)["x"]
    assert moved[..., A5C].tolist() == [[1.5] * 3] * 4 and np.delete(moved, A5C, -1).max() == 0.0
    assert logits["x"].max() == 0.0  # the stored logits are not changed


def test_the_offset_is_chosen_on_single_recordings_and_stays_near_zero_when_nothing_is_gained():
    seqs, rng = sequences(), np.random.default_rng(3)
    logits = logits_from(seqs, rng, noise=0.0)
    chosen = followup.choose_offset(logits, seqs)
    assert chosen["chosen"] == 0.0 and [c["offset"] for c in chosen["curve"]] == list(followup.OFFSETS)
    for s in seqs:  # every A5C sample is now read as A4C by a margin of 0.8
        a5c = s.labels[windows.window_centres(len(s))] == A5C
        logits[s.id][a5c] = 0.0
        logits[s.id][a5c, :, A4C], logits[s.id][a5c, :, A5C] = 2.0, 1.2
    chosen = followup.choose_offset(logits, seqs)
    assert chosen["chosen"] == 1.0  # the nearest to zero of the offsets that read them right
    by_offset = {c["offset"]: c["native_macro_f1"] for c in chosen["curve"]}
    assert by_offset[1.0] == 1.0 > by_offset[0.5] == by_offset[0.0]


def test_among_equal_offsets_the_nearest_to_zero_wins_and_then_the_negative_one():
    def curve(*pairs):
        return [{"offset": b, "native_macro_f1": f} for b, f in pairs]

    assert followup.pick_offset(curve((-1.0, 0.9), (0.0, 0.9), (1.0, 0.9))) == 0.0
    assert followup.pick_offset(curve((-0.5, 0.9), (0.5, 0.9), (1.0, 0.9), (0.0, 0.8))) == -0.5
    assert followup.pick_offset(curve((-0.5, 0.9), (0.5, 0.9), (2.0, 0.95))) == 2.0


def test_the_mix_rescales_the_other_systems_rows():
    first = {"x": np.array([[0.8, 0.2, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0]])}
    second = {"x": np.array([[0.0, 0.1, 0.3, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0, 0.25]])}  # rows sum to 0.4 and 0.5
    mixed = followup.mixed(first, second, 0.5)["x"]
    np.testing.assert_allclose(mixed, [[0.4, 0.225, 0.375, 0.0, 0.0], [0.5, 0.25, 0.0, 0.0, 0.25]])
    np.testing.assert_allclose(followup.mixed(first, second, 0.0)["x"], first["x"])
    with pytest.raises(SystemExit, match="without any probability"):
        followup.mixed(first, {"x": np.zeros((2, 5))}, 0.5)


def test_the_weight_is_the_best_feasible_one_and_the_smaller_of_equals(monkeypatch):
    table = {0.0: ("feasible", 0.90), 0.1: ("feasible", 0.95), 0.2: ("feasible", 0.95), 0.3: ("infeasible", 0.99), 0.4: ("feasible", 0.94),
             0.5: ("feasible", 0.90)}
    seen = []

    def choose(probabilities, sequences, drop_floors):
        weight = round(float(probabilities["x"][0, 1]), 1)  # the mix of [1, 0, ...] with [0, 1, ...] carries its weight
        seen.append(drop_floors)
        status, f1 = table[weight]
        return {"status": status, "params": {"half_window": 7, "threshold": 0.1, "nms_radius": 5}, "calibration": {"segment_f1_50_multi": f1}}

    monkeypatch.setattr(followup.comparison, "choose", choose)
    first, second = {"x": np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])}, {"x": np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])}
    chosen = followup.choose_weight(first, second, ())
    assert chosen["chosen"] == 0.1 and [c["weight"] for c in chosen["curve"]] == list(followup.WEIGHTS)
    assert seen == [False] * len(followup.WEIGHTS)  # the floors are never dropped for a mix
    table.update({w: ("infeasible", f1) for w, (_, f1) in table.items()})
    assert followup.choose_weight(first, second, ())["chosen"] == 0.0  # no weight has a setting: nothing is mixed in


def difference(value: float, interval: list[float] | None) -> dict:
    return {"value": value, "interval": interval}


def test_the_words_follow_the_plan():
    yes, no = "candidate", "not supported on development data"

    def q1(distance, f1=0.0, macro=0.0):
        return followup.word_q1({"multi": {"cut_distance_mean_s": distance, "segment_f1_50": difference(f1, None)},
                                 "recordings": {"macro_f1": difference(macro, None)}})

    assert q1(difference(-0.03, [-0.05, -0.01])) == yes
    assert q1(difference(-0.03, [-0.05, -0.01]), f1=-0.005) == yes  # not lower by more than 0.005
    assert q1(difference(-0.03, [-0.05, -0.01]), f1=-0.0051) == no
    assert q1(difference(-0.03, [-0.05, 0.0])) == no  # the interval must lie wholly below zero
    assert q1(difference(-0.03, None)) == no
    assert q1(difference(-0.03, [-0.05, -0.01]), macro=-0.001) == no
    assert followup.word_q1(None) == no  # no setting met the floors

    def q2(chosen, macro, f1=0.0):
        return followup.word_q2(chosen, {"multi": {"segment_f1_50": difference(f1, None)}, "recordings": {"macro_f1": macro}})

    assert q2(1.0, difference(0.02, [0.001, 0.04])) == yes
    assert q2(0.0, difference(0.02, [0.001, 0.04])) == no  # no offset was chosen
    assert q2(1.0, difference(0.02, [0.0, 0.04])) == no
    assert q2(1.0, difference(0.02, [0.001, 0.04]), f1=-0.005) == yes
    assert q2(1.0, difference(0.02, [0.001, 0.04]), f1=-0.0051) == no
    assert q2(-0.5, difference(0.02, [0.001, 0.04])) == yes  # an offset of either sign
    assert q2(1.0, difference(0.02, None)) == no  # no interval: too few recordings to say
    assert followup.word_q2(1.0, None) == no

    def q3(chosen, f1, macro=0.0):
        return followup.word_q3(chosen, {"multi": {"segment_f1_50": f1}, "recordings": {"macro_f1": difference(macro, None)}})

    assert q3(0.2, difference(0.01, [0.002, 0.02])) == yes
    assert q3(0.0, difference(0.01, [0.002, 0.02])) == no
    assert q3(0.2, difference(0.01, [-0.002, 0.02])) == no
    assert q3(0.2, difference(0.01, [0.002, 0.02]), macro=-0.01) == no
    assert q3(0.2, difference(0.01, None)) == no
    assert followup.word_q3(0.2, None) == no


def two_halves() -> dict[str, tuple[benchmark.Sequence, ...]]:
    """Two halves with sequences of their own, as the development benchmark has; neither names an examination."""
    return {"calibration": tuple(benchmark.Sequence(f"c_{s.id}", s.kind, s.fragments) for s in sequences()),
            "selection": tuple(benchmark.Sequence(s.id, s.kind, s.fragments) for s in sequences())}


def study_inputs(tmp_path, monkeypatch, rng):
    halves = two_halves()
    both = (*halves["calibration"], *halves["selection"])
    names = {f"rec{k}": f"rec{k}" for k in range(RECORDINGS)}
    monkeypatch.setattr(followup.benchmark, "load", lambda: halves)
    monkeypatch.setattr(followup.comparison, "_names", lambda images, sequences: names)
    dense = dense_from(both, rng)
    stored = at_published_centres(both, dense)
    by_recording = {s.fragments[0].video: (0.5 * softmax(8.0 * np.eye(5)[s.labels] + rng.normal(size=(len(s), 5)))).astype(np.float32)
                    for s in halves["selection"] if s.kind == "native"}  # a single recording has one stored file, whichever half asks
    other = {s.id: by_recording[s.fragments[0].video] if s.kind == "native"
             else (0.5 * softmax(8.0 * np.eye(5)[s.labels] + rng.normal(size=(len(s), 5)))).astype(np.float32) for s in both}
    np.savez(tmp_path / "logits.npz", **stored)
    np.savez(tmp_path / "dense.npz", **dense)
    write_store(tmp_path / "EchoViewCLIP", both, other, names)
    selection = halves["selection"]
    f1 = metrics.evaluate(comparison.decoded(comparison.member_probabilities(stored, selection, (0, 1, 2)), B3), selection)["segment_f1_50_multi"]
    monkeypatch.setattr(followup.comparison, "PUBLISHED_SELECTION_F1", round(f1, 4))
    return halves, stored, dense, ["study", "--images", str(tmp_path), "--logits", str(tmp_path / "logits.npz"), "--dense", str(tmp_path / "dense.npz"),
                                   "--store", f"EchoViewCLIP={tmp_path / 'EchoViewCLIP'}", "--draws", "60"]


def test_the_study_scores_every_arm_against_the_published_system(tmp_path, monkeypatch, capsys):
    halves, stored, dense, command = study_inputs(tmp_path, monkeypatch, np.random.default_rng(4))
    calibration, selection = halves["calibration"], halves["selection"]
    settings_used = []
    decode = comparison.decoded

    def recorded(rows, setting):
        settings_used.append({k: setting[k] for k in ("half_window", "threshold", "nms_radius")})
        return decode(rows, setting)

    monkeypatch.setattr(followup.comparison, "decoded", recorded)
    out = tmp_path / "out" / "followup.json"
    assert followup.main([*command, "--out", str(out)]) == 0 and "wrote" in capsys.readouterr().out
    result, used = json.loads(out.read_text()), list(settings_used)
    assert result["benchmark"] == "development" and result["draws"] == 60 and set(result["sha256"]) == {"logits", "dense", "EchoViewCLIP"}
    assert result["dense_check"] == {"sequences": len(calibration) + len(selection), "largest_difference": 0.0}
    q1, rows = result["q1"], result["rows"]
    assert set(q1["arms"]) == {"every 0.5 s", "every 0.5 s, interpolated", "every 0.2 s", "every 0.1 s"}
    on_calibration, on_selection = followup.density_arms(stored, dense, calibration), followup.density_arms(stored, dense, selection)
    for name, arm in q1["arms"].items():
        chosen = comparison.choose(on_calibration[name], calibration, drop_floors=False)  # the setting comes from the calibration half
        assert arm["setting"]["status"] == "feasible" and arm["setting"]["params"] == chosen["params"] != B3
        assert (arm["word"] == "the published system") == (name == "every 0.5 s")
        if name != "every 0.5 s":  # and the numbers from the selection half, decoded with that setting and with the released one
            own = metrics.evaluate(decode(on_selection[name], chosen["params"]), selection)
            released = metrics.evaluate(decode(on_selection[name], B3), selection)
            assert rows[f"{name}, own setting"]["multi"]["segment_f1_50"]["value"] == pytest.approx(own["segment_f1_50_multi"])
            assert rows[f"{name}, released numbers"]["multi"]["segment_f1_50"]["value"] == pytest.approx(released["segment_f1_50_multi"])
            # the arm's own setting decodes twice: the calibration half when it is chosen, the selection half for its row
            assert used.count({k: chosen["params"][k] for k in ("half_window", "threshold", "nms_radius")}) == 2
    assert "every 0.5 s, own setting" not in rows and {"Perfect on the 0.5 s grid", "Perfect on the 0.2 s grid"} <= set(rows)
    published = metrics.evaluate(decode(comparison.member_probabilities(stored, selection, (0, 1, 2)), B3), selection)
    assert rows[ROUTER]["multi"]["segment_f1_50"]["value"] == pytest.approx(published["segment_f1_50_multi"])
    assert rows[ROUTER]["multi"]["segment_f1_50"]["support"] == sum(len(s.view_changes) + 1 for s in selection if s.kind == "multi")
    # the dense arm reads every sample of this made-up benchmark right, so its cuts are nearer than those of the 0.5 s grid
    dense_row = result["differences_from_published"]["every 0.1 s, own setting"]["multi"]["cut_distance_mean_s"]
    assert dense_row["value"] < 0 and rows["every 0.1 s, own setting"]["multi"]["cut_distance_mean_s"]["value"] == pytest.approx(0.0, abs=0.02)
    assert q1["arms"]["every 0.1 s"]["word"] == "candidate"
    floors = [rows[f"Perfect on the {step} s grid"]["multi"]["cut_distance_mean_s"]["value"] for step in ("0.2", "0.5")]
    assert floors[0] < floors[1]
    q2, q3 = result["q2"], result["q3"]
    assert q2["chosen"] == 0.0 and q2["word"] == "not supported on development data" and len(q2["curve"]["calibration"]) == len(followup.OFFSETS)
    assert {"offset", "accuracy", "macro_f1", "a4c_as_a5c", "a5c_as_a4c", "segment_f1_50"} <= set(q2["curve"]["selection"][0])
    # the released setting decodes the check of the published number, the published row, the three other arms, and every offset
    # three times: on the calibration half to choose it, and on both halves for its curve
    assert used.count(dict(B3)) == 1 + 1 + 3 + 3 * len(followup.OFFSETS)
    assert q3["chosen"] in followup.WEIGHTS and q3["word"] in ("candidate", "not supported on development data")
    assert len(q3["curve"]) == len(followup.WEIGHTS) and "selection" in q3["curve"][0]
    assert not re.search(r"rec\d|native_\d|multi_\d|pairs_\d", json.dumps(result))  # pooled numbers only
    with pytest.raises(SystemExit, match="does not replace"):
        followup.main([*command, "--out", str(out)])


def test_the_study_needs_the_published_number_and_runs_without_the_other_system(tmp_path, monkeypatch):
    _, _, _, command = study_inputs(tmp_path, monkeypatch, np.random.default_rng(5))
    without = [arg for arg in command if "EchoViewCLIP" not in arg and arg != "--store"]
    assert followup.main([*without, "--out", str(tmp_path / "two.json")]) == 0
    result = json.loads((tmp_path / "two.json").read_text())
    assert result["q3"] == {"word": "not run: no stored outputs of EchoViewCLIP were given"}
    with pytest.raises(SystemExit, match="no other system"):
        followup.main([*without, "--store", f"STFM={tmp_path / 'EchoViewCLIP'}", "--out", str(tmp_path / "no.json")])
    monkeypatch.setattr(followup.comparison, "PUBLISHED_SELECTION_F1", 0.5)
    with pytest.raises(SystemExit, match=re.escape("must give 0.5")):
        followup.main([*command, "--out", str(tmp_path / "no.json")])
    assert not (tmp_path / "no.json").exists()


def test_every_choice_is_made_on_the_calibration_half_and_reported_on_the_selection_half(monkeypatch):
    halves, rng = two_halves(), np.random.default_rng(6)
    calibration, selection = halves["calibration"], halves["selection"]
    both = (*calibration, *selection)
    dense = dense_from(both, rng, noise=0.0)
    for s in calibration:  # on the calibration half only, EchoRouter reads every A5C sample as A4C by a margin of 0.8
        a5c = s.labels == A5C
        dense[s.id][a5c] = 0.0
        dense[s.id][a5c, :, A4C], dense[s.id][a5c, :, A5C] = 2.0, 1.2
    stored = at_published_centres(both, dense)
    other = {s.id: (0.5 * np.eye(5)[s.labels] + 0.01).astype(np.float32) for s in both}  # the other system reads every sample right
    calls, words = [], {}
    score_rows = comparison.score_rows

    def recorded(rows, sequences, draws, seed):
        calls.append(([row.name for row in rows], sequences, draws, seed))
        return score_rows(rows, sequences, draws, seed)

    def spy(name, function):
        def inner(*args):
            words[name] = args
            return function(*args)
        return inner

    monkeypatch.setattr(followup.comparison, "score_rows", recorded)
    for name in ("word_q2", "word_q3"):
        monkeypatch.setattr(followup, name, spy(name, getattr(followup, name)))
    result = followup.study(stored, dense, other, halves, draws=40, seed=11)
    assert result["draws"] == 40 and result["seed"] == 11
    names, scored_on, draws, seed = calls[-1]  # the one scoring with intervals: the selection half, with the draws and the seed given
    assert scored_on is selection and (draws, seed) == (40, 11) and names[0] == ROUTER
    # Q1: with A5C read wrong on the calibration half no setting meets the floors there, so no arm has a row of its own
    assert all(arm["setting"]["status"] == "infeasible" for arm in result["q1"]["arms"].values())
    assert not any("own setting" in name for name in names) and "every 0.1 s, released numbers" in names
    assert {arm["word"] for name, arm in result["q1"]["arms"].items() if name != "every 0.5 s"} == {"not supported on development data"}
    # Q2: the offset that the calibration half needs is chosen, though the selection half needs none
    assert result["q2"]["chosen"] == 1.0 and "A5C offset +1.0" in names
    assert words["word_q2"] == (1.0, result["differences_from_published"]["A5C offset +1.0"])
    assert [c["macro_f1"] for c in result["q2"]["curve"]["selection"]] == [1.0] * len(followup.OFFSETS)
    assert result["q2"]["curve"]["calibration"][6]["offset"] == 0.0 and result["q2"]["curve"]["calibration"][6]["macro_f1"] < 1.0
    assert result["q2"]["word"] == "not supported on development data"  # on the selection half the offset changes nothing
    # Q3: the calibration half is read right only with the other system mixed in
    chosen = result["q3"]["chosen"]
    assert chosen > 0 and f"mixed with EchoViewCLIP, weight {chosen:.1f}" in names
    assert words["word_q3"] == (chosen, result["differences_from_published"][f"mixed with EchoViewCLIP, weight {chosen:.1f}"])
    assert result["q3"]["curve"][0]["weight"] == 0.0 and result["q3"]["curve"][0]["setting"]["status"] == "infeasible"
    # and every reported number is of the selection half, which EchoRouter reads right
    assert result["rows"][ROUTER]["recordings"]["accuracy"]["value"] == 1.0
    assert result["rows"][ROUTER]["recordings"]["accuracy"]["support"] == sum(s.kind == "native" for s in selection)


def test_the_word_of_an_arm_rests_on_its_row_with_a_setting_of_its_own(monkeypatch):
    halves, rng = two_halves(), np.random.default_rng(7)
    both = (*halves["calibration"], *halves["selection"])
    dense = dense_from(both, rng)
    given = []
    word_q1 = followup.word_q1
    monkeypatch.setattr(followup, "word_q1", lambda difference: given.append(difference) or word_q1(difference))
    result = followup.study(at_published_centres(both, dense), dense, None, halves, draws=30)
    differences = result["differences_from_published"]
    assert [id(d) for d in given] == [id(differences[f"{arm}, own setting"]) for arm in result["q1"]["arms"] if arm != "every 0.5 s"]
    assert result["q3"] == {"word": "not run: no stored outputs of EchoViewCLIP were given"}


def test_the_logits_command_asks_for_every_sample(tiny, tmp_path, monkeypatch):  # noqa: F811
    stand_in = Ensemble.from_models([BrightnessModel(0.0), BrightnessModel(4.0)], B3, device="cpu")
    monkeypatch.setattr(followup, "reference_system", lambda weights, device, batch: stand_in)
    out = tmp_path / "dense.npz"
    assert followup.main(["logits", "--images", str(tiny[1]), "--benchmark", str(tiny[0]), "--out", str(out)]) == 0
    halves = benchmark.load(tiny[0])
    scored = [s for half in benchmark.HALVES for s in halves[half] if s.kind != "exam"]
    with np.load(out) as stored:
        assert set(stored.files) == {s.id for s in scored}
        for s in scored:
            assert stored[s.id].shape == (len(s), 2, 5) and stored[s.id].argmax(-1)[:, 0].tolist() == s.labels.tolist()
    with pytest.raises(SystemExit, match="does not replace"):
        followup.main(["logits", "--images", str(tiny[1]), "--benchmark", str(tiny[0]), "--out", str(out)])


def test_only_the_published_system_is_studied(monkeypatch):
    class Other:
        is_reference = False

    monkeypatch.setattr(followup, "Ensemble", lambda *args, **kwargs: Other())
    with pytest.raises(SystemExit, match="not the published system"):
        followup.reference_system(None, "cpu", 4)


def test_the_option_decodes_with_the_settings_the_study_chose():
    """`echoview segment --read-every` (reference.DENSER_READ_OUT) against the published output of the study."""
    study = json.loads((Path(__file__).parents[1] / "docs" / "followup" / "followup.json").read_text())
    for arm, stride in followup.DENSER.items():
        chosen = study["q1"]["arms"][arm]
        assert chosen["word"] == "candidate" and chosen["setting"]["status"] == "feasible"
        assert reference.DENSER_READ_OUT[stride] == {key: chosen["setting"]["params"][key] for key in ("half_window", "threshold", "nms_radius")}
    assert set(reference.DENSER_READ_OUT) == set(followup.DENSER.values()) == {1, 2}
    published = study["q1"]["arms"]["every 0.5 s"]["setting"]["params"]
    assert published["half_window"] == B3["half_window"] and abs(published["threshold"] - B3["threshold"]) < 1e-3


# ------------------------------------------------------------------------- the confirmation on the test benchmark
CONFIRMED, AT_A_COST, NOT_CONFIRMED = "confirmed on the test benchmark", "nearer cuts at a cost", "not confirmed on the test benchmark"
PUBLISHED_KEYS = ("segment_f1_50_multi", "segment_f1_50_pairs", "native_accuracy", "native_macro_f1")
FIVE_ROWS = [ROUTER, "Perfect on the 0.5 s grid", "Perfect on the 0.2 s grid", "every 0.2 s", "every 0.1 s"]


def published_of(seqs: tuple[benchmark.Sequence, ...], stored: dict[str, np.ndarray]) -> dict:
    """The four published numbers of this made-up benchmark, as comparison.PUBLISHED holds them."""
    numbers = metrics.evaluate(comparison.decoded(comparison.member_probabilities(stored, seqs, (0, 1, 2)), B3), seqs)
    return {ROUTER: {key: float(numbers[key]) for key in PUBLISHED_KEYS}}


def first_read_of(seqs: tuple[benchmark.Sequence, ...], stored: dict[str, np.ndarray], draws: int, seed: int) -> dict:
    """What the comparison's read of this made-up benchmark holds, as its output file gives it back."""
    read = {"benchmark": "test", **comparison.compare(seqs, comparison.build_rows(seqs, stored, {}, {}), draws, seed)}
    return json.loads(json.dumps(comparison._clean(read), allow_nan=False))


def test_the_words_of_the_confirmation_follow_the_plan():
    def said(distance, hull, f1=0.0, macro=0.0, interval=None):
        decides = {"value": distance, "interval": hull if interval is None else interval, "hull": hull}
        return followup.word_confirm(decides, difference(f1, None), difference(macro, None))

    assert said(-0.05, [-0.08, -0.02]) == {"nearer_cuts": True, "segment_f1_kept": True, "recordings_kept": True, "word": CONFIRMED}
    assert said(-0.05, [-0.08, -0.02], f1=-0.005)["word"] == CONFIRMED  # not lower by more than 0.005
    assert said(-0.05, [-0.08, -0.02], f1=-0.0051) == {"nearer_cuts": True, "segment_f1_kept": False, "recordings_kept": True, "word": AT_A_COST}
    assert said(-0.05, [-0.08, -0.02], macro=-0.001) == {"nearer_cuts": True, "segment_f1_kept": True, "recordings_kept": False, "word": AT_A_COST}
    assert said(-0.05, [-0.08, -0.02], f1=-0.01, macro=-0.01)["word"] == AT_A_COST
    assert said(-0.05, [-0.08, 0.0])["word"] == NOT_CONFIRMED  # the interval must lie wholly below zero
    assert said(-0.05, [-0.08, 0.01], interval=[-0.07, -0.01])["word"] == NOT_CONFIRMED  # every unit must agree: the hull decides
    assert said(-0.05, None)["word"] == NOT_CONFIRMED
    assert said(0.0, [-0.08, -0.02])["word"] == NOT_CONFIRMED  # and the distance itself must be lower
    assert said(None, [-0.08, -0.02])["word"] == NOT_CONFIRMED
    assert said(-0.05, [-0.08, -0.02], f1=None) == {"nearer_cuts": True, "segment_f1_kept": False, "recordings_kept": True, "word": AT_A_COST}
    assert said(-0.05, [-0.08, -0.02], macro=None) == {"nearer_cuts": True, "segment_f1_kept": True, "recordings_kept": False, "word": AT_A_COST}
    # without nearer cuts nothing else is weighed, but what was kept is still said
    assert said(0.02, [-0.01, 0.05], f1=-0.01) == {"nearer_cuts": False, "segment_f1_kept": False, "recordings_kept": True, "word": NOT_CONFIRMED}


def test_a_perfect_system_on_the_half_second_grid_is_the_row_of_the_comparison():
    """The row "Perfect on the 0.5 s grid" must come out again, so both commands must build it the same way: on every
    sequence of both packaged benchmarks. Only the reference labels are used here."""
    packaged = [s for half in benchmark.load().values() for s in half] + list(benchmark.load_test())
    assert len(packaged) > 2000
    for s in packaged:
        assert np.array_equal(followup.grid_labels(s.labels, 5), scorecard.grid_floor_labels(s.labels))


def told_apart(seqs: tuple[benchmark.Sequence, ...], rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Stored and dense logits on which the published system and the arms err differently, so that a rule that reads
    the wrong part, the wrong number or one model alone gives another result:
    the published system reads the last piece of one multi stream as another view (the arms read it right);
    the arms read one single recording as another view (the published system reads it right);
    and in another multi stream the first seed alone reads the last piece wrong, outvoted by the other two."""
    dense = dense_from(seqs, rng)
    stored = at_published_centres(seqs, dense)
    multi = [s for s in seqs if s.kind == "multi"]
    native = next(s for s in seqs if s.kind == "native")

    def last_piece(s: benchmark.Sequence) -> np.ndarray:
        return windows.window_centres(len(s)) >= s.joins[-1]

    def reads(view: int) -> np.ndarray:
        return (8.0 * np.eye(5)[view]).astype(np.float32)

    wrong = (int(multi[0].labels[-1]) + 2) % 5
    stored[multi[0].id][last_piece(multi[0])] = reads(wrong)
    stored[multi[1].id][last_piece(multi[1]), 0] = reads((int(multi[1].labels[-1]) + 2) % 5)
    dense[native.id][:] = reads((int(native.labels[0]) + 1) % 5)
    return stored, dense


def test_the_confirmation_scores_the_two_released_steps_against_the_published_system(monkeypatch):
    seqs, rng = sequences(), np.random.default_rng(8)
    stored, dense = told_apart(seqs, rng)
    monkeypatch.setattr(comparison, "PUBLISHED", published_of(seqs, stored))
    first = first_read_of(seqs, stored, 60, 5)
    used, calls = [], []
    decode, score_rows = comparison.decoded, comparison.score_rows
    monkeypatch.setattr(followup.comparison, "decoded", lambda rows, setting: used.append(dict(setting)) or decode(rows, setting))

    def recorded(rows, sequences, draws, seed):
        calls.append(([r.name for r in rows], sequences, draws, seed))
        return score_rows(rows, sequences, draws, seed)

    monkeypatch.setattr(followup.comparison, "score_rows", recorded)
    pairs_asked = []

    def only_one(natives, one, other):
        pairs_asked.append((tuple(s.id for s in natives), one.name, other.name))
        return {"only_first_right": 3, "only_second_right": 1}

    monkeypatch.setattr(followup.comparison, "discordant", only_one)
    result = followup.confirm(stored, dense, seqs, first, draws=60, seed=5)
    single = tuple(s.id for s in seqs if s.kind == "native")
    assert pairs_asked == [(single, "every 0.2 s", ROUTER), (single, "every 0.1 s", ROUTER)]
    assert all(result["differences_from_published"][name]["recordings_only_one_reads_right"] == {"only_the_arm": 3, "only_the_published_system": 1}
               for name in followup.DENSER)
    # the released settings and no other: the published system once, each step once
    assert used == [dict(B3), dict(reference.DENSER_READ_OUT[2]), dict(reference.DENSER_READ_OUT[1])]
    # the published rows are scored and checked before any arm is; every scoring uses the benchmark, the draws and the seed given
    assert [names for names, *_ in calls] == [FIVE_ROWS[:2], FIVE_ROWS]
    assert all(scored_on is seqs and (draws, seed) == (60, 5) for _, scored_on, draws, seed in calls)
    assert result["draws"] == 60 and result["seed"] == 5 and result["first_read_rows_reproduced"] == FIVE_ROWS[:2]
    assert result["published_numbers_reproduced"] == comparison.PUBLISHED[ROUTER]
    rows, arms = result["rows"], followup.denser_arms(dense, seqs)
    assert list(rows) == FIVE_ROWS and list(arms) == ["every 0.2 s", "every 0.1 s"]
    clean = json.loads(json.dumps(comparison._clean(result), allow_nan=False))
    assert clean["rows"][ROUTER] == first["rows"][ROUTER] and clean["rows"][FIVE_ROWS[1]] == first["rows"][FIVE_ROWS[1]]
    assert (rows[ROUTER]["role"], rows[FIVE_ROWS[2]]["role"], rows["every 0.1 s"]["role"], rows["every 0.1 s"]["system"]) == ("A", "P", "D", ROUTER)
    for name, stride in followup.DENSER.items():
        expected = metrics.evaluate(decode(arms[name], reference.DENSER_READ_OUT[stride]), seqs)
        assert rows[name]["multi"]["segment_f1_50"]["value"] == pytest.approx(expected["segment_f1_50_multi"])
        assert rows[name]["pairs"]["segment_f1_50"]["value"] == pytest.approx(expected["segment_f1_50_pairs"])
        assert rows[name]["recordings"]["accuracy"]["value"] == pytest.approx(expected["native_accuracy"])
        assert {"streams", "examinations"} <= set(rows[name]["multi"]["cut_distance_mean_s"])  # this benchmark names its examinations
        entry, described = result["confirmation"][name], result["differences_from_published"][name]
        assert set(described) == {"multi", "pairs", "recordings", "recordings_only_one_reads_right"} and described["multi"]["segment_f1_50"]["level"] == 95.0
        distance = rows[name]["multi"]["cut_distance_mean_s"]["value"] - rows[ROUTER]["multi"]["cut_distance_mean_s"]["value"]
        assert entry["cut_distance"]["value"] == pytest.approx(distance) and entry["cut_distance"]["level"] == 97.5  # two arms share 5%
        assert set(entry["cut_distance"]) >= {"streams", "examinations", "hull"}
        assert entry["segment_f1_50"] == described["multi"]["segment_f1_50"] and entry["macro_f1"] == described["recordings"]["macro_f1"]
        # on these outputs another part or another number would give another result
        assert described["multi"]["segment_f1_50"]["value"] > 0 == described["pairs"]["segment_f1_50"]["value"]
        assert described["recordings"]["macro_f1"]["value"] < 0 and described["recordings"]["macro_f1"]["value"] != described["recordings"]["accuracy"]["value"]
        assert entry["recordings_kept"] is False and entry["segment_f1_kept"] is True
        assert entry["forward_passes"] == pytest.approx(followup.forward_passes(seqs)[name])
        said = followup.word_confirm(entry["cut_distance"], entry["segment_f1_50"], entry["macro_f1"])
        assert {key: entry[key] for key in said} == said
    # the step of 0.1 s reads every stream of this made-up benchmark right: its cuts are nearer than those of the 0.5 s grid,
    # and it pays with the single recording it reads wrong
    assert result["confirmation"]["every 0.1 s"]["cut_distance"]["value"] < 0
    assert result["confirmation"]["every 0.1 s"]["word"] in (AT_A_COST, NOT_CONFIRMED)
    assert (result["confirmation"]["every 0.1 s"]["word"] == AT_A_COST) == result["confirmation"]["every 0.1 s"]["nearer_cuts"]
    # the published row is the ensemble's: the first seed alone reads one piece of these outputs differently
    alone, together = (decode(comparison.member_probabilities(stored, seqs, members), B3) for members in ((0,), (0, 1, 2)))
    assert any(not np.array_equal(alone[s.id], together[s.id]) for s in seqs)
    assert not re.search(r"rec\d|native_\d|multi_\d|pairs_\d", json.dumps(clean))  # pooled numbers only


def test_the_published_rows_must_be_those_of_the_first_read(monkeypatch):
    seqs, rng = sequences(), np.random.default_rng(9)
    dense = dense_from(seqs, rng)
    stored = at_published_centres(seqs, dense)
    numbers = published_of(seqs, stored)
    monkeypatch.setattr(comparison, "PUBLISHED", numbers)
    first = first_read_of(seqs, stored, 50, 3)
    assert followup.confirm(stored, dense, seqs, first, draws=50, seed=3)["first_read_rows_reproduced"] == list(followup.HELD_ROWS)
    scored, decoded = [], []
    score_rows, decode = comparison.score_rows, comparison.decoded
    monkeypatch.setattr(followup.comparison, "score_rows", lambda rows, *rest: scored.append([r.name for r in rows]) or score_rows(rows, *rest))
    monkeypatch.setattr(followup.comparison, "decoded", lambda rows, setting: decoded.append(dict(setting)) or decode(rows, setting))

    def changed(edit) -> dict:
        copy = json.loads(json.dumps(first))
        edit(copy)
        return copy

    def an_interval(read):
        read["rows"][ROUTER]["multi"]["segment_f1_50"]["streams"][0] += 1e-9

    def a_value(read):
        read["rows"]["Perfect on the 0.5 s grid"]["pairs"]["cut_distance_mean_s"]["value"] += 1e-9

    def a_recordings_number(read):  # the part behind rule (c)
        read["rows"][ROUTER]["recordings"]["macro_f1"]["value"] += 1e-9

    def an_interval_over_examinations(read):  # the unit behind the hull
        read["rows"][ROUTER]["multi"]["cut_distance_mean_s"]["examinations"][1] += 1e-9

    def the_wider_interval(read):
        read["rows"]["Perfect on the 0.5 s grid"]["multi"]["cut_distance_mean_s"]["interval"][0] -= 1e-9

    def a_support(read):
        read["rows"][ROUTER]["pairs"]["segment_f1_50"]["support"] += 1

    for edit, message in ((an_interval, "does not come out again: row: EchoRouter"), (a_value, "does not come out again: row: Perfect on the 0.5 s grid"),
                          (a_recordings_number, "does not come out again: row: EchoRouter"),
                          (an_interval_over_examinations, "does not come out again: row: EchoRouter"),
                          (the_wider_interval, "does not come out again: row: Perfect on the 0.5 s grid"),
                          (a_support, "does not come out again: row: EchoRouter"),
                          (lambda read: read["rows"][ROUTER].update(role="D"), "does not come out again: row: EchoRouter"),
                          (lambda read: read.update(benchmark="selection"), "does not come out again: benchmark"),
                          (lambda read: read.update(draws=51), "does not come out again: draws"),
                          (lambda read: read.update(seed=4), "does not come out again: seed"),
                          (lambda read: read.pop("seed"), "not an output of the comparison"),
                          (lambda read: read["rows"].pop("Perfect on the 0.5 s grid"), "not an output of the comparison")):
        with pytest.raises(SystemExit, match=message):
            followup.confirm(stored, dense, seqs, changed(edit), draws=50, seed=3)
    with pytest.raises(SystemExit, match="not an output of the comparison"):
        followup.confirm(stored, dense, seqs, [first], draws=50, seed=3)
    # every stopped read stopped before an arm was decoded or scored
    assert scored and all(names == list(followup.HELD_ROWS) for names in scored) and all(setting == dict(B3) for setting in decoded)
    scored.clear()
    monkeypatch.setattr(comparison, "PUBLISHED", {ROUTER: {**numbers[ROUTER], "native_accuracy": numbers[ROUTER]["native_accuracy"] - 0.01}})
    with pytest.raises(SystemExit, match="do not give the published numbers"):
        followup.confirm(stored, dense, seqs, first, draws=50, seed=3)
    assert scored == []  # nothing at all was scored


def confirm_inputs(tmp_path, monkeypatch, rng):
    seqs = sequences()
    dense = dense_from(seqs, rng)
    stored = at_published_centres(seqs, dense)
    np.savez(tmp_path / "logits.npz", **stored)
    np.savez(tmp_path / "dense.npz", **dense)
    monkeypatch.setattr(followup, "DRAWS", 60)
    (tmp_path / "first.json").write_text(json.dumps(first_read_of(seqs, stored, 60, followup.SEED)))
    monkeypatch.setattr(followup.benchmark, "load_test", lambda: seqs)
    monkeypatch.setattr(comparison, "PUBLISHED", published_of(seqs, stored))
    monkeypatch.setattr(comparison, "TEST_LOGITS_SHA256", comparison.file_sha256(tmp_path / "logits.npz"))
    monkeypatch.setattr(comparison, "FIRST_READ_SHA256", comparison.file_sha256(tmp_path / "first.json"))
    monkeypatch.setattr(followup, "TEST_DENSE_SHA256", comparison.file_sha256(tmp_path / "dense.npz"))
    monkeypatch.setattr(followup, "CONFIRMATION_SHA256", None)  # this made-up benchmark has not been read for the question

    def command(out="confirmation.json", **other) -> list[str]:
        files = {"logits": "logits.npz", "dense": "dense.npz", "earlier": "first.json", **other}
        given = [arg for key, name in files.items() if name is not None for arg in (f"--{key}", str(tmp_path / name))]
        return ["confirm", *given, "--draws", "60", "--out", str(tmp_path / out)]

    return seqs, stored, dense, command


def test_the_confirm_command_reads_once(tmp_path, monkeypatch, capsys):
    seqs, stored, dense, command = confirm_inputs(tmp_path, monkeypatch, np.random.default_rng(10))
    out = tmp_path / "confirmation.json"
    with pytest.raises(SystemExit, match="the test benchmark is not read: the plan fixes 60 draws"):
        followup.main([*command()[:-3], "50", "--out", str(out)])
    monkeypatch.setattr(comparison, "TEST_LOGITS_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="the test benchmark is not read: the logits file is not the one the plan names"):
        followup.main(command())
    monkeypatch.setattr(comparison, "TEST_LOGITS_SHA256", comparison.file_sha256(tmp_path / "logits.npz"))
    (tmp_path / "other.json").write_text(json.dumps({**json.loads((tmp_path / "first.json").read_text()), "seed": 1}))
    with pytest.raises(SystemExit, match="the test benchmark is not read: --earlier must be the output of the comparison's first read"):
        followup.main(command(earlier="other.json"))
    off = dict(dense)
    off[seqs[0].id] = dense[seqs[0].id] + np.float32(0.01)
    np.savez(tmp_path / "off.npz", **off)
    beside = dict(dense)  # equal at every published position, another number beside one: the check of the logits cannot see it
    beside[seqs[0].id] = dense[seqs[0].id].copy()
    beside[seqs[0].id][3] += np.float32(0.5)
    assert 3 not in windows.window_centres(len(seqs[0])).tolist()
    np.savez(tmp_path / "beside.npz", **beside)
    np.savez(tmp_path / "few.npz", **{s.id: dense[s.id] for s in seqs[1:]})
    for name in ("off.npz", "beside.npz", "few.npz"):
        with pytest.raises(SystemExit, match="the test benchmark is not read: the dense file is not the one whose digest was committed before the read"):
            followup.main(command(dense=name))
    named = followup.TEST_DENSE_SHA256
    monkeypatch.setattr(followup, "TEST_DENSE_SHA256", None)
    with pytest.raises(SystemExit, match="the test benchmark is not read: the code does not name the dense file yet"):
        followup.main(command())
    monkeypatch.setattr(followup, "TEST_DENSE_SHA256", comparison.file_sha256(tmp_path / "off.npz"))
    with pytest.raises(SystemExit, match="do not agree with the stored ones"):
        followup.main(command(dense="off.npz"))
    monkeypatch.setattr(followup, "TEST_DENSE_SHA256", comparison.file_sha256(tmp_path / "few.npz"))
    with pytest.raises(SystemExit, match="lacks the logits of 1 of the"):
        followup.main(command(dense="few.npz"))
    monkeypatch.setattr(followup, "TEST_DENSE_SHA256", named)
    assert not out.exists() and not list(tmp_path.glob("*.partial"))  # a stopped read leaves no output
    assert followup.main(command()) == 0 and "wrote" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["benchmark"] == "test" and result["sequences"] == len(seqs) and result["draws"] == 60 and result["seed"] == followup.SEED
    assert result["sha256"] == {key: comparison.file_sha256(tmp_path / name) for key, name in
                                (("logits", "logits.npz"), ("dense", "dense.npz"), ("first_read", "first.json"))}
    assert result["settings"] == {"every 0.2 s": reference.DENSER_READ_OUT[2], "every 0.1 s": reference.DENSER_READ_OUT[1]}
    assert result["dense_check"] == {"sequences": len(seqs), "largest_difference": 0.0}
    assert result["first_read_rows_reproduced"] == FIVE_ROWS[:2] and list(result["rows"]) == FIVE_ROWS
    assert set(result["confirmation"]) == set(result["differences_from_published"]) == {"every 0.2 s", "every 0.1 s"}
    direct = followup.confirm(stored, dense, seqs, json.loads((tmp_path / "first.json").read_text()), draws=60, seed=followup.SEED)
    assert result["confirmation"] == json.loads(json.dumps(comparison._clean(direct["confirmation"])))
    assert not re.search(r"rec\d|native_\d|multi_\d|pairs_\d", out.read_text()) and "NaN" not in out.read_text()
    before = out.read_bytes()
    with pytest.raises(SystemExit, match="does not replace"):
        followup.main(command())
    assert out.read_bytes() == before


def test_after_the_read_the_confirm_command_only_reproduces_it(tmp_path, monkeypatch):
    command = confirm_inputs(tmp_path, monkeypatch, np.random.default_rng(11))[-1]
    out = tmp_path / "confirmation.json"
    with pytest.raises(SystemExit, match="there is no read of it to reproduce"):  # before the read nothing can be reproduced
        followup.main(command(out="never.json", reproduce="first.json"))
    assert followup.main(command()) == 0
    monkeypatch.setattr(followup, "CONFIRMATION_SHA256", comparison.file_sha256(out))  # from now on the read has been made
    with pytest.raises(SystemExit, match="the test benchmark is not read: it has been read for this question"):
        followup.main(command(out="never.json"))
    changed = json.loads(out.read_text())
    changed["rows"]["every 0.1 s"]["multi"]["segment_f1_50"]["value"] += 1e-9
    (tmp_path / "changed.json").write_text(json.dumps(changed))
    with pytest.raises(SystemExit, match="the test benchmark is not read: it has been read for this question"):  # another file than the read's output
        followup.main(command(out="never.json", reproduce="changed.json"))
    assert followup.main(command(out="again.json", reproduce="confirmation.json")) == 0
    assert json.loads((tmp_path / "again.json").read_text()) == json.loads(out.read_text())
    monkeypatch.setattr(followup, "CONFIRMATION_SHA256", comparison.file_sha256(tmp_path / "changed.json"))
    with pytest.raises(SystemExit, match="the read does not come out again"):
        followup.main(command(out="never.json", reproduce="changed.json"))
    assert not (tmp_path / "never.json").exists()


def test_the_logits_command_reads_the_test_benchmark_on_request(tiny, tmp_path, monkeypatch):  # noqa: F811
    stand_in = Ensemble.from_models([BrightnessModel(0.0), BrightnessModel(4.0)], B3, device="cpu")
    monkeypatch.setattr(followup, "reference_system", lambda weights, device, batch: stand_in)
    halves = benchmark.load(tiny[0])
    monkeypatch.setattr(followup.benchmark, "load_test", lambda: halves["selection"])  # stands in for the packaged test benchmark
    monkeypatch.setattr(followup.benchmark, "load", lambda path=None: pytest.fail("the development benchmark is not read with --test"))
    out = tmp_path / "dense_test.npz"
    assert followup.main(["logits", "--test", "--images", str(tiny[1]), "--out", str(out)]) == 0
    scored = [s for s in halves["selection"] if s.kind != "exam"]
    with np.load(out) as stored:
        assert set(stored.files) == {s.id for s in scored} and all(stored[s.id].shape == (len(s), 2, 5) for s in scored)
    with pytest.raises(SystemExit, match="--test reads the packaged test benchmark"):
        followup.main(["logits", "--test", "--benchmark", str(tiny[0]), "--images", str(tiny[1]), "--out", str(tmp_path / "no.npz")])


def test_the_code_names_the_dense_file_and_the_read_once_they_exist():
    """The digest of the forward passes' output is committed before the read (docs/followup/dense_test_check.json), and
    once docs/followup/confirmation.json exists the command must be held to it."""
    folder = Path(__file__).parents[1] / "docs" / "followup"
    check, read = folder / "dense_test_check.json", folder / "confirmation.json"
    if check.is_file():
        stage = json.loads(check.read_text())
        assert stage["sha256"] == {"logits": comparison.TEST_LOGITS_SHA256, "dense": followup.TEST_DENSE_SHA256}
        assert stage["labels_read"] is False and stage["largest_difference"] <= followup.CLOSE
    else:
        assert followup.TEST_DENSE_SHA256 is None and not read.is_file()
    if read.is_file():
        assert followup.CONFIRMATION_SHA256 == comparison.file_sha256(read)
        assert json.loads(read.read_text())["sha256"]["dense"] == followup.TEST_DENSE_SHA256
    else:
        assert followup.CONFIRMATION_SHA256 is None


def test_an_output_appears_whole_or_not_at_all(tmp_path, monkeypatch):
    out = tmp_path / "deep" / "result.json"
    comparison._write({"a": np.float64(1.5), "b": [np.int64(2), float("nan")]}, out)
    assert json.loads(out.read_text()) == {"a": 1.5, "b": [2, None]} and [p.name for p in out.parent.iterdir()] == ["result.json"]
    with pytest.raises(FileExistsError):  # never replaces a file
        comparison._write({"a": 2}, out)
    assert json.loads(out.read_text())["a"] == 1.5 and [p.name for p in out.parent.iterdir()] == ["result.json"]
    # a stop while the text is being written leaves no output behind, so the step is not taken for done
    real_fsync = comparison.os.fsync
    monkeypatch.setattr(comparison.os, "fsync", lambda handle: (_ for _ in ()).throw(OSError("the disk is gone")))
    with pytest.raises(OSError, match="the disk is gone"):
        comparison._write({"a": 3}, tmp_path / "stopped.json")
    assert not (tmp_path / "stopped.json").exists()
    monkeypatch.setattr(comparison.os, "fsync", real_fsync)
    # a file system without hard links: the same promise by another way
    monkeypatch.setattr(comparison.os, "link", lambda source, target: (_ for _ in ()).throw(OSError("no hard links here")))
    comparison._write({"a": 4}, tmp_path / "plain.json")
    assert json.loads((tmp_path / "plain.json").read_text()) == {"a": 4} and not list(tmp_path.glob("*.partial"))
    with pytest.raises(FileExistsError):
        comparison._write({"a": 5}, tmp_path / "plain.json")
    assert json.loads((tmp_path / "plain.json").read_text()) == {"a": 4} and not list(tmp_path.glob("*.partial"))
