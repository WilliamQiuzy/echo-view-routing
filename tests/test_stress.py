"""The published system on development data that is changed on purpose (docs/ABLATIONS.md, part 2)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from echoview import Ensemble, windows
from echoview.evaluation import ablations, benchmark, classify_read, comparison, ev9v, metrics, run, stress
from echoview.evaluation.stress import CLEAN, HALF_RATE, PUBLISHED, REVERSED, SHUFFLED, STAGE2, STILL
from echoview.pipeline import probabilities_from_logits
from echoview.reference import B3

from .conftest import BrightnessModel, flat_frames
from .test_evaluation_run import tiny  # noqa: F401  (a fixture: a tiny benchmark file and its Images folder)


def stand_in(*shifts: float) -> Ensemble:
    return Ensemble.from_models([BrightnessModel(shift) for shift in (shifts or (0.0, 0.0, 0.0))], B3, device="cpu")


def changed(name: str, planes: np.ndarray, index: int = 0) -> np.ndarray:
    return stress.frames_of(stress.IMAGE[name](planes, stress.generator(name, index)))


# ------------------------------------------------------------------------------------------------- conditions
def test_grey_levels_go_to_planes_and_back():
    frames = flat_frames([0, 100, 255])
    planes = stress.planes_of(frames)
    assert planes.shape == (3, 224, 224) and planes.dtype == np.float64 and planes[1, 0, 0] == 100.0
    np.testing.assert_array_equal(stress.frames_of(planes), frames)
    back = stress.frames_of(np.array([[[-5.0, 99.5, 300.0]]]))
    assert back.dtype == np.uint8 and back.shape == (1, 1, 3, 3) and back[0, 0, :, 0].tolist() == [0, 100, 255]  # kept in range, rounded to even


def test_every_image_condition_gives_frames_the_models_can_read():
    rng = np.random.default_rng(0)
    planes = rng.uniform(0, 255, (2, 224, 224))
    for name in stress.IMAGE:
        out = changed(name, planes)
        assert out.shape == (2, 224, 224, 3) and out.dtype == np.uint8, name
        assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all(), name
        np.testing.assert_array_equal(out, changed(name, planes))  # the same in every run
        assert not np.array_equal(out, stress.frames_of(planes)), name  # and a change
    assert len(stress.CONDITIONS) == len(set(stress.CONDITIONS)) == 1 + len(stress.IMAGE) + 4
    assert len({stress.slug(name) for name in stress.CONDITIONS}) == len(stress.CONDITIONS) and stress.slug("JPEG quality 30") == "jpeg_quality_30"


def test_the_image_conditions_do_what_their_names_say():
    flat = np.full((1, 224, 224), 100.0)
    def level(name: str, value: float = 100.0) -> int:
        return int(changed(name, np.full((1, 224, 224), value))[0, 112, 112, 0])

    assert level("gamma 0.5") == round(255 * (100 / 255) ** 0.5) and level("gamma 2.0") == round(255 * (100 / 255) ** 2)
    assert level("gamma 0.7", 255.0) == 255 and level("gamma 1.5", 0.0) == 0
    assert level("darker by 40") == 60 and level("darker by 40", 20.0) == 0 and level("brighter by 40") == 140 and level("brighter by 40", 240.0) == 255
    assert level("contrast 0.5", 255.0) == 191 and level("contrast 1.5", 255.0) == 255 and level("contrast 0.5", 127.5) in (127, 128)
    for name in ("blur 1 pixel", "blur 2 pixels", "112 pixels", "64 pixels", "zoomed in 1.2", "mirrored"):  # a flat frame stays flat
        assert (changed(name, flat)[..., 0] == 100).all(), name
    for name in ("JPEG quality 30", "JPEG quality 10"):
        assert np.abs(changed(name, flat)[..., 0].astype(int) - 100).max() <= 2, name
    for name, sigma in (("noise 10", 10.0), ("noise 25", 25.0), ("speckle 0.25", 25.0)):  # speckle 0.25 on level 100 has a spread of 25
        out = changed(name, flat)[0, :, :, 0].astype(float)
        assert abs(out.mean() - 100) < 1 and abs(out.std() - sigma) < 0.1 * sigma + 0.5, name
    assert not np.array_equal(changed("noise 10", flat, index=0), changed("noise 10", flat, index=1))  # another sequence, another noise
    ramp = np.tile(np.arange(224, dtype=float), (1, 224, 1))  # grey level = column
    mirrored = changed("mirrored", ramp)[0, :, :, 0]
    assert mirrored[5, 0] == 223 and mirrored[5, 223] == 0
    small = changed("zoomed out 0.8", flat)[0, :, :, 0]
    assert small[0, 0] == 0 and small[112, 112] == 100 and small[21, 112] == 0 and small[22, 112] == 100  # 179 pixels set in black, 22 above
    near = changed("zoomed in 1.2", ramp)[0, :, :, 0]
    assert near[112, 0] > 15 and near[112, 223] < 208  # the middle 187 columns fill the frame
    turned = changed("rotated 20 degrees", flat)[0, :, :, 0]
    assert turned[0, 0] == 0 and turned[223, 223] == 0 and turned[112, 112] == 100  # the corners come from outside the frame
    assert (changed("rotated 10 degrees", flat)[0, :, :, 0] == 0).sum() < (turned == 0).sum()
    sharp = np.zeros((1, 224, 224))
    sharp[0, 112, 112] = 255.0
    assert changed("blur 2 pixels", sharp)[0, 112, 112, 0] < changed("blur 1 pixel", sharp)[0, 112, 112, 0] < 255
    assert changed("64 pixels", sharp)[0, 112, 112, 0] < changed("112 pixels", sharp)[0, 112, 112, 0] < 255


def test_the_conditions_use_the_parameters_in_their_names():
    import io

    from PIL import Image
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(1)
    planes = np.rint(rng.uniform(0, 255, (2, 224, 224)))
    first = Image.fromarray(planes[0].astype(np.uint8))

    def one(name: str) -> np.ndarray:
        return changed(name, planes)[0, :, :, 0]

    for quality in (30, 10):
        held = io.BytesIO()
        first.save(held, format="JPEG", quality=quality)
        np.testing.assert_array_equal(one(f"JPEG quality {quality}"), np.asarray(Image.open(io.BytesIO(held.getvalue())).convert("L")))
    for side in (112, 64):
        direct = first.resize((side, side), Image.BILINEAR).resize((224, 224), Image.BILINEAR)
        np.testing.assert_array_equal(one(f"{side} pixels"), np.asarray(direct))
    for degrees in (10, 20):
        np.testing.assert_array_equal(one(f"rotated {degrees} degrees"), np.asarray(first.rotate(degrees, resample=Image.BILINEAR, fillcolor=0)))
    np.testing.assert_array_equal(one("zoomed in 1.2"), np.asarray(first.crop((18, 18, 205, 205)).resize((224, 224), Image.BILINEAR)))  # 187 of 224 pixels
    ground = Image.new("L", (224, 224), 0)
    ground.paste(first.resize((179, 179), Image.BILINEAR), (22, 22))
    np.testing.assert_array_equal(one("zoomed out 0.8"), np.asarray(ground))
    for name, sigma in (("blur 1 pixel", 1.0), ("blur 2 pixels", 2.0)):
        np.testing.assert_array_equal(one(name), np.clip(np.rint(gaussian_filter(planes[0], sigma=sigma)), 0, 255).astype(np.uint8))
    for name, value in (("gamma 0.5", 0.5), ("gamma 0.7", 0.7), ("gamma 1.5", 1.5), ("gamma 2.0", 2.0)):
        np.testing.assert_array_equal(one(name), np.rint(255.0 * (planes[0] / 255.0) ** value).astype(np.uint8))
    for name, value in (("contrast 0.5", 0.5), ("contrast 1.5", 1.5)):
        np.testing.assert_array_equal(one(name), np.clip(np.rint((planes[0] - 127.5) * value + 127.5), 0, 255).astype(np.uint8))
    np.testing.assert_array_equal(one("darker by 40"), np.clip(planes[0] - 40, 0, 255).astype(np.uint8))
    np.testing.assert_array_equal(one("brighter by 40"), np.clip(planes[0] + 40, 0, 255).astype(np.uint8))
    # every frame on its own: two equal frames get different noise, and a blur does not mix a black frame with a white one
    same = np.full((2, 224, 224), 100.0)
    for name in ("noise 10", "noise 25", "speckle 0.25"):
        out = changed(name, same)
        assert not np.array_equal(out[0], out[1]), name
    apart = np.stack([np.zeros((224, 224)), np.full((224, 224), 255.0)])
    for name in ("blur 1 pixel", "blur 2 pixels"):
        out = changed(name, apart)
        assert (out[0] == 0).all() and (out[1] == 255).all(), name


def test_what_a_window_shows_under_the_conditions_of_time():
    centres, plan, slots = windows.window_plan(40)
    still, marks = stress.shown_still(40)
    assert still.shape == plan.shape and (still == centres[:, None]).all() and marks.tolist() == slots.tolist()
    shuffled, moved = stress.shown_shuffled(40, np.random.default_rng(3))
    assert shuffled.shape == plan.shape and not np.array_equal(shuffled, plan)
    for row in range(len(plan)):
        assert sorted(shuffled[row].tolist()) == sorted(plan[row].tolist())  # the same sixteen samples
        assert shuffled[row][moved[row]] == plan[row][slots[row]] == centres[row]  # the marker went with the centre sample
    again, _ = stress.shown_shuffled(40, np.random.default_rng(3))
    np.testing.assert_array_equal(shuffled, again)


def test_one_sequence_under_every_kind_of_condition():
    system = stand_in(0.0, 20.0, -20.0)
    frames = flat_frames([100] * 23 + [150] * 17)  # A4C, then A5C
    rng = np.random.default_rng(0)
    clean, logits = stress.under(system, frames, CLEAN, rng)
    np.testing.assert_array_equal(clean, probabilities_from_logits(system.window_logits(frames), 40))
    assert logits.shape == (len(windows.window_centres(40)), 3, 5) and clean.shape == (40, 5)
    assert clean[:15].argmax(1).tolist() == [2] * 15 and clean[-10:].argmax(1).tolist() == [3] * 10
    # the stand-in reads the marked frame only, so a still window and a shuffled one give what the unchanged one gives
    for name in (STILL, SHUFFLED):
        np.testing.assert_allclose(stress.under(system, frames, name, np.random.default_rng(1))[0], clean, atol=1e-6)
    backwards, _ = stress.under(system, frames, REVERSED, rng)
    assert backwards.shape == (40, 5) and backwards[:15].argmax(1).tolist() == [2] * 15 and backwards[-10:].argmax(1).tolist() == [3] * 10
    flat = flat_frames([100] * 30)
    np.testing.assert_allclose(stress.under(system, flat, HALF_RATE, rng)[0], stress.under(system, flat, CLEAN, rng)[0], atol=1e-6)
    odd = flat_frames([0, 200] * 15)  # the views alternate: at half the rate every second frame is gone
    assert set(stress.under(system, odd, HALF_RATE, rng)[0].argmax(1).tolist()) == {0}
    assert set(stress.under(system, odd, CLEAN, rng)[0].argmax(1).tolist()) == {0, 4}
    brighter, _ = stress.under(system, flat, "brighter by 40", rng)
    assert set(brighter.argmax(1).tolist()) == {3} and set(stress.under(system, flat, CLEAN, rng)[0].argmax(1).tolist()) == {2}  # 140 reads as 150
    with pytest.raises(ValueError, match="no such condition"):
        stress.under(system, flat, "upside down", rng)


def test_the_models_are_shown_other_windows_only_when_they_fit_the_plan():
    system, frames = stand_in(0.0), flat_frames([100] * 30)
    _, plan, slots = windows.window_plan(30)
    np.testing.assert_array_equal(system.window_logits(frames, shown=(plan, slots)), system.window_logits(frames))
    for bad in ((plan[:-1], slots[:-1]), (plan, slots[:-1]), (plan + 30, slots), (plan, slots + 16), (-plan - 1, slots)):
        with pytest.raises(ValueError, match="the windows to show"):
            system.window_logits(frames, shown=bad)


# ---------------------------------------------------------------------------------------- the view it does not know
def test_how_sure_the_five_probabilities_are():
    rows = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.6, 0.4, 0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]])
    sure = stress.certainty(rows)
    assert sure["probability"].tolist() == [1.0, 0.6, 0.2] and sure["margin"].tolist() == pytest.approx([1.0, 0.2, 0.0])
    assert sure["negative_entropy"].tolist() == pytest.approx([0.0, 0.6 * np.log(0.6) + 0.4 * np.log(0.4), np.log(0.2)])
    assert all(np.all(np.diff(values) < 0) for values in sure.values())  # each is higher when surer


def test_a_view_the_system_does_not_know_beside_the_views_it_knows():
    known = np.array([[0.98, 0.01, 0.01, 0.0, 0.0, 1.0]] * 6 + [[0.55, 0.45, 0.0, 0.0, 0.0, 0.0]])
    ninth = np.array([[0.1, 0.8, 0.1, 0.0, 0.0, 1.0], [0.0, 0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.01, 0.99, 1.0], [0.3, 0.6, 0.1, 0.0, 0.0, 1.0]])
    found = stress.ninth_part(ninth, known, draws=200, seed=4)
    assert found["recordings"] == 4 and found["given"] == {"PLAX": 0, "PSAX": 3, "A4C": 0, "A5C": 0, "SC4C": 1} and found["seeds_differ"] == 1
    assert found["probability"]["lowest"] == 0.5 and found["probability"]["highest"] == 0.99 and found["probability"]["above_0.9"] == 1
    assert found["probability"]["median"] == pytest.approx(0.7)
    assert found["known"] == {"recordings": 7, "median_probability": 0.98, "above_0.9": 6, "seeds_differ": 1}
    # of the 28 pairs, the known recording is surer in 6 x 3 + 1 (0.55 against 0.5): 19
    assert found["known_above_ninth"]["probability"]["value"] == pytest.approx(19 / 28)
    assert set(found["known_above_ninth"]) == {"probability", "margin", "negative_entropy"}
    low, high = found["known_above_ninth"]["probability"]["interval"]
    assert 0 <= low < 19 / 28 < high <= 1


# -------------------------------------------------------------------------------------------- the two commands
def made_up_run(tiny, tmp_path, monkeypatch):  # noqa: F811
    path, images = tiny
    halves = {half: comparison._scored_kinds(sequences) for half, sequences in benchmark.load(path).items()}
    monkeypatch.setattr(ablations.benchmark, "load", lambda: benchmark.__dict__["_parts"](path, "", benchmark.HALVES))
    monkeypatch.setattr(stress.benchmark, "load_test", lambda: pytest.fail("this study does not read the test benchmark"))
    monkeypatch.setattr(stress.followup, "reference_system", lambda weights, device, batch: stand_in())
    monkeypatch.setattr(stress, "stage2_system", lambda weights, device, batch: stand_in(30.0, 30.0, 30.0))
    system = stand_in()
    stored = {}
    for sequences in halves.values():
        stored.update(run.all_logits(system, sequences, ev9v.FrameTable.build(sequences, images)))
    np.savez(tmp_path / "dev.npz", **stored)
    selection = halves["selection"]
    f1 = metrics.evaluate(comparison.decoded(comparison.member_probabilities(stored, selection, (0, 1, 2)), B3), selection)["segment_f1_50_multi"]
    monkeypatch.setattr(comparison, "PUBLISHED_SELECTION_F1", round(f1, 4))
    return halves, images


def test_the_run_writes_one_file_per_condition_and_the_report_scores_them(tiny, tmp_path, monkeypatch, capsys):  # noqa: F811
    halves, images = made_up_run(tiny, tmp_path, monkeypatch)
    work, second = tmp_path / "work", tmp_path / "work2"
    (tmp_path / "val.txt").write_text("cal_v3\nsel_v1\n\n")
    (tmp_path / "train.txt").write_text("cal_v0\n")
    base = ["run", "--images", str(images), "--device", "cpu"]
    with pytest.raises(SystemExit, match="--ninth takes NAME=LIST"):
        stress.main([*base, "--folder", str(work), "--ninth", str(tmp_path / "val.txt")])
    assert stress.main([*base, "--folder", str(work), "--ninth", f"validation={tmp_path / 'val.txt'}", "--ninth", f"training={tmp_path / 'train.txt'}"]) == 0
    printed = capsys.readouterr().out
    assert printed.count("done:") == len(stress.CONDITIONS) + 3 and "the ninth view, list validation, 2 recordings" in printed
    files = sorted(p.name for p in work.glob("*.npz"))
    assert files == sorted([f"{stress.slug(name)}.npz" for name in stress.CONDITIONS] + ["ninth_validation.npz", "ninth_training.npz", stress.KNOWN_FILE])
    assert not list(work.glob("*.partial*")) and json.loads((work / "system.json").read_text())["system"] == PUBLISHED
    selection = halves["selection"]
    with np.load(work / "unchanged.npz") as data:
        assert set(data.files) == {f"p__{s.id}" for s in selection} | {f"c__{s.id}" for s in selection if s.kind == "native"}
        native = next(s for s in selection if s.kind == "native")
        assert data[f"p__{native.id}"].shape == (len(native), 5) and data[f"c__{native.id}"].shape == (6,)
    # a file that exists is not computed again
    (work / "mirrored.npz").unlink()
    calls = []
    condition = stress.run_condition
    monkeypatch.setattr(stress, "run_condition", lambda system, sequences, table, name: calls.append(name) or condition(system, sequences, table, name))
    assert stress.main([*base, "--folder", str(work)]) == 0 and calls == ["mirrored"]
    # the stage-2 ensemble: every condition, and the unchanged calibration half
    assert (work / stress.KNOWN_FILE).is_file()  # the 10 single recordings of both halves, read as the ninth view is read
    with np.load(work / stress.KNOWN_FILE) as data:
        assert data["rows"].shape == (10, 6)
    with pytest.raises(SystemExit, match="holds the outputs of another system"):  # every system has a folder of its own
        stress.main([*base, "--folder", str(work), "--system", "stage2"])
    assert json.loads((work / "system.json").read_text())["system"] == PUBLISHED
    calls.clear()
    assert stress.main([*base, "--folder", str(second), "--system", "stage2"]) == 0
    assert calls == [CLEAN, *stress.CONDITIONS] and (second / stress.CALIBRATION_FILE).is_file()
    assert json.loads((second / "system.json").read_text())["system"] == STAGE2
    with np.load(second / stress.CALIBRATION_FILE) as data:
        assert {key[3:] for key in data.files if key.startswith("p__")} == {s.id for s in halves["calibration"]}

    out = tmp_path / "forward.json"
    report = ["report", "--logits", str(tmp_path / "dev.npz"), "--folder", str(work), "--draws", "40"]
    assert stress.main([*report, "--stage2", str(second), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert (result["benchmark"], result["half"], result["sequences"], result["draws"]) == ("development", "selection", len(selection), 40)
    assert result["unchanged_against_stored"] <= 1e-6 and list(result["systems"]) == [PUBLISHED, STAGE2]
    published = result["systems"][PUBLISHED]
    assert published["setting"] == {"status": "published", "params": dict(B3)} and list(published["conditions"]) == list(stress.CONDITIONS)
    assert published["system"]["parameters_per_model"] == [0, 0, 0] and published["system"]["models"] == 3
    clean = published["conditions"][CLEAN]
    assert clean["kind"] == "unchanged" and "minus_unchanged" not in clean and clean["classify"] == {"errors": 0, "mean_probability": pytest.approx(
        clean["classify"]["mean_probability"]), "seeds_differ": 0}
    assert clean["selection"]["multi"]["segment_f1_50"]["value"] == pytest.approx(comparison.PUBLISHED_SELECTION_F1, abs=5e-5)
    assert clean["selection"]["recordings"]["accuracy"]["value"] == 1.0
    mirrored = published["conditions"]["mirrored"]  # flat frames: nothing changes, in no draw
    assert mirrored["kind"] == "image" and mirrored["minus_unchanged"]["multi"]["segment_f1_50"]["value"] == 0.0
    assert mirrored["minus_unchanged"]["multi"]["segment_f1_50"]["interval"] == [0.0, 0.0]
    brighter = published["conditions"]["brighter by 40"]  # the stand-in reads the grey level: four of the five views move to the next one
    assert brighter["classify"]["errors"] == 4 and brighter["selection"]["recordings"]["accuracy"]["value"] == pytest.approx(0.2)
    assert brighter["minus_unchanged"]["recordings"]["accuracy"]["value"] == pytest.approx(-0.8)
    assert brighter["minus_unchanged"]["recordings"]["accuracy"]["level"] == 95.0
    assert published["conditions"][STILL]["kind"] == "time" and published["conditions"][STILL]["minus_unchanged"]["multi"]["segment_f1_50"]["value"] == 0.0
    assert set(published["ninth_view"]) == {"validation", "training"} and published["ninth_view"]["validation"]["recordings"] == 2
    assert published["ninth_view"]["training"]["given"]["PLAX"] == 1 and published["ninth_view"]["validation"]["known"]["recordings"] == 10
    check = published["reader_against_benchmark_frames"]  # the same recordings through the command's reader and through the benchmark's frames
    assert check["recordings"] == 10 and check["recordings_given_another_view"] == 0 and check["largest_difference_in_a_probability"] < 1e-3
    second_system = result["systems"][STAGE2]
    assert set(second_system["setting"]) == {"status", "floors_dropped", "params", "calibration"} and "ninth_view" not in second_system
    # stage 2's setting is the rule's choice on its own outputs for the calibration half; here every recording is misread, so the floors are dropped
    with np.load(second / stress.CALIBRATION_FILE) as data:
        own = {s.id: data[f"p__{s.id}"] for s in halves["calibration"]}
    chosen = comparison.choose(own, halves["calibration"], drop_floors=True)
    assert second_system["setting"]["params"] == chosen["params"] and second_system["setting"]["floors_dropped"] is True
    assert second_system["setting"]["params"] != dict(B3) or chosen["params"] == dict(B3)
    assert set(result["published_minus_stage2"]) == set(stress.CONDITIONS)
    apart = result["published_minus_stage2"][CLEAN]["multi"]["segment_f1_50"]
    mine, theirs = clean["selection"]["multi"]["segment_f1_50"]["value"], second_system["conditions"][CLEAN]["selection"]["multi"]["segment_f1_50"]["value"]
    assert apart["value"] == pytest.approx(mine - theirs)
    # a condition of stage 2 is set against stage 2's own unchanged frames: its stand-in reads four of the five views wrong as they are
    theirs_clean = second_system["conditions"][CLEAN]["selection"]["recordings"]["accuracy"]["value"]
    theirs_brighter = second_system["conditions"]["brighter by 40"]
    assert theirs_clean == pytest.approx(0.2) and result["published_minus_stage2"][CLEAN]["recordings"]["accuracy"]["value"] == pytest.approx(0.8)
    assert theirs_brighter["selection"]["recordings"]["accuracy"]["value"] != pytest.approx(brighter["selection"]["recordings"]["accuracy"]["value"] - 0.8)
    assert theirs_brighter["minus_unchanged"]["recordings"]["accuracy"]["value"] == pytest.approx(
        theirs_brighter["selection"]["recordings"]["accuracy"]["value"] - theirs_clean)
    with pytest.raises(SystemExit, match="was written for"):  # the two folders exchanged
        stress.main(["report", "--logits", str(tmp_path / "dev.npz"), "--folder", str(second), "--stage2", str(work), "--draws", "40",
                     "--out", str(tmp_path / "exchanged.json")])
    names = {s.id for half in halves.values() for s in half} | {f.video for half in halves.values() for s in half for f in s.fragments}
    assert not any(name in out.read_text() for name in names) and "NaN" not in out.read_text()  # pooled numbers only
    with pytest.raises(SystemExit, match="does not replace"):
        stress.main([*report, "--out", str(out)])
    # without the stage-2 folder the report holds the published system alone
    alone = tmp_path / "alone.json"
    assert stress.main([*report, "--out", str(alone)]) == 0
    single = json.loads(alone.read_text())
    assert list(single["systems"]) == [PUBLISHED] and "published_minus_stage2" not in single
    assert single["systems"][PUBLISHED]["conditions"]["brighter by 40"] == brighter


def test_the_report_stops_when_a_file_is_missing_or_the_unchanged_run_is_off(tiny, tmp_path, monkeypatch):  # noqa: F811
    _, images = made_up_run(tiny, tmp_path, monkeypatch)
    work = tmp_path / "work"
    monkeypatch.setattr(stress, "CONDITIONS", (CLEAN, "mirrored", REVERSED))
    assert stress.main(["run", "--images", str(images), "--device", "cpu", "--folder", str(work)]) == 0
    report = ["report", "--logits", str(tmp_path / "dev.npz"), "--folder", str(work), "--draws", "30"]
    kept = (work / "mirrored.npz").read_bytes()
    (work / "mirrored.npz").unlink()
    with pytest.raises(SystemExit, match=r"lacks mirrored\.npz"):
        stress.main([*report, "--out", str(tmp_path / "a.json")])
    (work / "mirrored.npz").write_bytes(kept)
    with np.load(work / "mirrored.npz") as data:
        few = {key: data[key] for key in data.files[1:]}
    np.savez(work / "mirrored.npz", **few)
    with pytest.raises(SystemExit, match="does not hold every sequence"):
        stress.main([*report, "--out", str(tmp_path / "a.json")])
    (work / "mirrored.npz").write_bytes(kept)
    with np.load(work / "unchanged.npz") as data:
        off = {key: data[key] for key in data.files}
    first = next(key for key in off if key.startswith("p__"))
    off[first] = np.roll(off[first], 1, axis=1)
    np.savez(work / "unchanged.npz", **off)
    with pytest.raises(SystemExit, match="do not give the stored outputs"):
        stress.main([*report, "--out", str(tmp_path / "a.json")])
    with pytest.raises(SystemExit, match="was written for 'None'"):  # a folder that no run has written
        stress.main([*report, "--stage2", str(tmp_path / "nothing"), "--out", str(tmp_path / "a.json")])
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "system.json").write_text(json.dumps({"system": STAGE2}))
    with pytest.raises(SystemExit, match=r"lacks unchanged\.npz"):
        stress.main([*report, "--stage2", str(tmp_path / "empty"), "--out", str(tmp_path / "a.json")])
    np.savez(work / "unchanged.npz", **{key: (np.roll(value, -1, axis=1) if key == first else value) for key, value in off.items()})  # as it was
    np.savez(work / "ninth_x.npz.partial.npz", rows=np.zeros((1, 6)))  # left by a run that was cut off: not a list
    assert stress.main([*report, "--out", str(tmp_path / "b.json")]) == 0
    assert "ninth_view" not in json.loads((tmp_path / "b.json").read_text())["systems"][PUBLISHED]
    np.savez(work / "ninth_validation.npz", rows=np.tile([0.1, 0.8, 0.1, 0.0, 0.0, 1.0], (2, 1)))
    with pytest.raises(SystemExit, match=r"lacks known\.npz"):  # recordings of the ninth view need the known ones through the same reader
        stress.main([*report, "--out", str(tmp_path / "c.json")])
    assert not (tmp_path / "a.json").exists() and classify_read.N == 5
