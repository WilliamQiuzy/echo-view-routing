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
