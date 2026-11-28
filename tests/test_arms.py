"""Stage 3 trained again, three ways, on the development benchmark (docs/ABLATIONS.md, part 4)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from echoview import weights as published_weights
from echoview.evaluation import ablations, arms, comparison, ev9v, run, stress
from echoview.evaluation.arms import AGAIN, CANDIDATE, MIRROR, MOSTLY_CHANGES, MOSTLY_LOOP, NONE, NOT_SUPPORTED, NOT_TOLD, NOTHING, OWN, WITHOUT
from echoview.evaluation.stress import CLEAN, PUBLISHED, STAGE2
from echoview.reference import B3

from .test_evaluation_run import tiny  # noqa: F401  (a fixture: a tiny benchmark file and its Images folder)
from .test_stress import made_up_run, stand_in

RECIPE = Path(__file__).resolve().parents[1] / "configs" / "stage3.json"
# Stand-ins that read the grey level of a frame; a member reads a view right while its shift plus the change of the
# frames stays below 25 grey levels. The three members of an arm differ, and so do the arms:
#                          unchanged   40 brighter   40 darker
#   published (0, 0, 0)      right        wrong        wrong
#   stage 2 (30, 30, 30)     wrong        wrong        right (but for the black view, which stays black)
#   again                    right        right        wrong
#   without input changes    right        wrong        wrong
#   with a mirror            right        wrong        right
# "40 darker" stands in for the mirrored image in these tests: flat grey frames look the same mirrored.
SHIFTS = {AGAIN: (-20.0, 30.0, -20.0), WITHOUT: (10.0, 10.0, 10.0), MIRROR: (20.0, 20.0, -20.0)}
BRIGHTER, DARKER = "brighter by 40", "darker by 40"
CODE = "abc1234"  # the commit that the runs of the arms name
FEW = (CLEAN, BRIGHTER, DARKER)
FOLDER = {AGAIN: "again", WITHOUT: "without", MIRROR: "mirror"}


def scored(value: float, draws: list[float], support: float = 100.0) -> comparison.Scored:
    return comparison.Scored({"segment_f1_50": value}, {"segment_f1_50": support}, {"streams": {"segment_f1_50": np.asarray(draws, float)}})


# ------------------------------------------------------------------------------------------- the two questions
def test_the_share_of_a_difference_and_its_sentence():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.004, 4000)
    again, stage2 = scored(0.90, 0.90 + noise), scored(0.60, 0.60 + rng.normal(0.0, 0.004, 4000))
    for without, expected, word in ((0.63, 0.9, MOSTLY_CHANGES), (0.87, 0.1, MOSTLY_LOOP), (0.75, 0.5, NOT_TOLD)):
        found = arms.share(again, scored(without, without + rng.normal(0.0, 0.004, 4000)), stage2, "segment_f1_50", 99.0)
        assert found["value"] == pytest.approx(expected) and found["level"] == 99.0 and found["interval"][0] < expected < found["interval"][1]
        assert found["interval"] == found["streams"] and found["draws_without_a_share"] == {"streams": 0} and arms.sentence(found) == word
        assert found["of"]["value"] == pytest.approx(0.30) and found["of"]["level"] == 99.0  # the whole difference: again minus stage 2
    # the share is the first minus the second over the first minus the third, draw by draw
    one = arms.share(scored(0.9, [0.9, 0.8, 0.9, 0.8, 0.9, 0.8]), scored(0.7, [0.7, 0.7, 0.8, 0.7, 0.7, 0.7]), scored(0.5, [0.5, 0.6, 0.5, 0.5, 0.5, 0.6]),
                     "segment_f1_50", 99.0)
    assert one["value"] == pytest.approx(0.5) and one["interval"][0] >= 0.25 - 1e-9 and one["interval"][1] <= 0.5 + 1e-9
    # where the whole difference is not shown, there is nothing to split
    level = arms.share(scored(0.60, 0.60 + noise), scored(0.59, 0.59 + noise), scored(0.60, 0.60 + rng.normal(0.0, 0.004, 4000)), "segment_f1_50", 99.0)
    assert level["value"] is None and level["interval"] is None and arms.sentence(level) == NOTHING
    # draws in which the whole difference is not positive have no share: they are left out of the interval and counted
    part = [0.61] * 700 + [0.60] * 300  # the share is 0.5 in 700 of the draws that count and 1 in 295; five draws do not count
    thin = arms.share(scored(0.62, [0.62] * 995 + [0.58] * 5), scored(0.61, part), scored(0.60, [0.60] * 1000), "segment_f1_50", 90.0)
    assert thin["draws_without_a_share"] == {"streams": 5} and thin["value"] == pytest.approx(0.5)
    assert thin["interval"][0] == pytest.approx(0.5) and thin["interval"][1] == pytest.approx(1.0)  # a draw without a share would have put -0.5 here
    # the interval of the share is taken at the level that is given: here 2% of the draws lie below one half
    first, third = scored(0.9, [0.9] * 1000), scored(0.4, [0.4] * 1000)
    second = scored(0.5, [0.75] * 20 + [0.5] * 980)  # a share of 0.3 in 20 draws and of 0.8 in the others
    wide, narrow = (arms.share(first, second, third, "segment_f1_50", level) for level in (99.0, 95.0))
    assert wide["interval"] == pytest.approx([0.3, 0.8]) and arms.sentence(wide) == NOT_TOLD
    assert narrow["interval"] == pytest.approx([0.8, 0.8]) and arms.sentence(narrow) == MOSTLY_CHANGES
    assert arms.sentence({"interval": [0.5, 0.9]}) == NOT_TOLD and arms.sentence({"interval": [0.51, 0.9]}) == MOSTLY_CHANGES
    assert arms.sentence({"interval": [0.1, 0.5]}) == NOT_TOLD and arms.sentence({"interval": [0.1, 0.49]}) == MOSTLY_LOOP
    assert arms.FAMILY_LEVEL == 99.0 and len(arms.FAMILY) == 5 and set(arms.FAMILY) <= set(stress.CONDITIONS)


def test_the_word_for_the_arm_with_a_mirror():
    family = ("noise 25", "blur 2 pixels")

    def found(gain=(0.05, 0.30), own=-0.01, here=(-0.004, (-0.01, 0.01)), accuracy=0.0, under=((0.0, (-0.02, 0.02)), (0.01, (-0.01, 0.03)))):
        def entry(value, interval):
            return {"multi": {"segment_f1_50": {"value": value, "interval": None if interval is None else list(interval)}},
                    "recordings": {"accuracy": {"value": accuracy}}}

        apart = {arms.MIRRORED: entry(0.2, gain), CLEAN: entry(*here), **{name: entry(*one) for name, one in zip(family, under)}}
        return arms.mirror_word(apart, {"multi": {"segment_f1_50": {"value": own}}}, 284, family)

    good = found()
    assert good["word"] == CANDIDATE and all(good["holds"].values()) and len(good["holds"]) == 5
    # each condition of the plan can fail alone, and then the word is not given
    failing = {"gains under the mirrored image": [found(gain=(0.0, 0.30)), found(gain=(-0.02, 0.30)), found(gain=None)],
               "stands the mirrored image": [found(own=-0.031), found(own=-0.15)],
               "not below on the unchanged frames": [found(here=(-0.02, (-0.03, -0.001))), found(here=(-0.011, (-0.03, 0.01)))],
               "at most one recording fewer right": [found(accuracy=-2 / 284)],
               "not below under the five conditions": [found(under=((-0.06, (-0.10, -0.02)), (0.01, (-0.01, 0.03)))),
                                                       found(under=((0.0, (-0.02, 0.02)), (-0.05, (-0.09, -0.001))))]}
    for condition, cases in failing.items():
        for case in cases:
            assert case["word"] == NOT_SUPPORTED and case["holds"][condition] is False
            assert sum(not holds for holds in case["holds"].values()) == 1
    # what is allowed: the margin of its own loss, the margin on the unchanged frames, one recording, an interval that includes zero
    assert found(own=-0.03)["word"] == CANDIDATE and found(here=(-0.01, (-0.03, 0.0)))["word"] == CANDIDATE
    assert found(accuracy=-1 / 284)["word"] == CANDIDATE and found(here=(0.004, (0.001, 0.01)))["word"] == CANDIDATE
    assert found(under=((-0.06, (-0.10, 0.0)), (0.0, None)))["word"] == CANDIDATE
    assert (arms.OWN_LOSS, arms.UNCHANGED, arms.MIRRORED) == (0.03, 0.01, "mirrored")


# ----------------------------------------------------------------------------------------- the trained models
def a_run(folder: Path, arm: str, seed: int, weight: float = 0.0, torch_version: str = "2.9.0", changed: dict | None = None,
          sessions: int | None = 1) -> None:
    """A finished training run as the training program leaves it, with the identity of the arm and the seed."""
    folder.mkdir(parents=True)
    identity = {**arms.recipe_of(arm, seed, RECIPE), **(changed or {})}
    done = {"updates": identity["updates"], "windows": 48000, "seconds": 900.0, "sessions": sessions, "identity": identity}
    (folder / "done.json").write_text(json.dumps({key: value for key, value in done.items() if value is not None}))
    (folder / "config.json").write_text(json.dumps({"torch": torch_version, "mirror": identity.get("mirror", 0.0)}))
    torch.save({"w": torch.tensor([3.0, 4.0 + weight + seed]), "steps": torch.tensor(7)}, folder / "final.pt")
    (folder / "train_log.jsonl").write_text("".join(json.dumps({"update": 50 * k, "loss": 1.0 / k, "accuracy": 1.0 - 0.1 / k}) + "\n" for k in range(1, 31)))


def test_an_arm_is_three_finished_runs_of_its_own_command(tmp_path):
    for arm, weight in ((AGAIN, 0.0), (WITHOUT, 0.25), (MIRROR, 0.5)):
        for seed in range(3):
            a_run(tmp_path / FOLDER[arm] / f"s{seed}", arm, seed, weight)
    found = arms.trained(tmp_path / "again", AGAIN, RECIPE)
    assert [entry["seed"] for entry in found] == [0, 1, 2] and all(entry["torch"] == "2.9.0" and entry["sessions"] == 1 for entry in found)
    assert found[0]["sha256"] == comparison.file_sha256(tmp_path / "again" / "s0" / "final.pt") and len({entry["sha256"] for entry in found}) == 3
    record = found[0]["training"]  # the end of the log: its last ten lines
    assert record["lines"] == 30 and record["last_update"] == 1500 and record["loss"] == pytest.approx(np.mean([1.0 / k for k in range(21, 31)]))
    assert record["accuracy"] == pytest.approx(np.mean([1.0 - 0.1 / k for k in range(21, 31)]))
    # the identities are those the training program writes: no mirror where there is none, the start by its word
    again, without, mirror = (arms.recipe_of(arm, 1, RECIPE) for arm in arms.ARMS)
    assert "mirror" not in again and "mirror" not in without and mirror["mirror"] == 0.5 and (again["strength"], without["strength"]) == (1.0, 0.0)
    assert {key for key in again if again[key] != without[key]} == {"strength"} and again["init"] == "stage2" and again["updates"] == 1500
    assert found[1]["identity"] == again and arms.trained(tmp_path / "mirror", MIRROR, RECIPE)[1]["identity"] == mirror
    assert arms.trained(tmp_path / "without", WITHOUT, RECIPE)[2]["identity"]["strength"] == 0.0
    # the runs of one arm are not the runs of another
    with pytest.raises(SystemExit, match=r"is not a run of the arm 'without input changes' for seed 0: \{'strength': \(1.0, 0.0\)\}"):
        arms.trained(tmp_path / "again", WITHOUT, RECIPE)
    with pytest.raises(SystemExit, match=r"is not a run of the arm 'again' for seed 0: \{'mirror': \(0.5, None\)\}"):
        arms.trained(tmp_path / "mirror", AGAIN, RECIPE)
    with pytest.raises(SystemExit, match="is not a run of the arm 'with a mirror'"):
        arms.trained(tmp_path / "without", MIRROR, RECIPE)
    # every option of the recipe counts, not only the ones that name the arm
    for name, change, message in (("swapped", {"seed": 2}, r"'seed': \(2, 1\)"), ("short", {"updates": 750}, r"'updates': \(750, 1500\)"),
                                  ("start", {"init": "stage2_s1.pt"}, r"'init': \('stage2_s1.pt', 'stage2'\)"),
                                  ("degrade", {"degrade": 0.0}, r"'degrade': \(0.0, 1.0\)"), ("rate", {"lr_enc": 1e-4}, r"'lr_enc': \(0.0001, 1e-05\)"),
                                  ("plans", {"plans": ["other.npz"]}, r"'plans': \(\['other.npz'\], None\)"), ("plain", {"amp": 0}, r"'amp': \(0, 1\)")):
        for seed in range(3):
            a_run(tmp_path / name / f"s{seed}", AGAIN, seed, changed=change if seed == 1 else None)
        with pytest.raises(SystemExit, match=message):
            arms.trained(tmp_path / name, AGAIN, RECIPE)
    # a run that was cut and went on says so, and a run that does not say in how many sessions it was made is refused
    for name, sessions in (("continued", 3), ("silent", None)):
        for seed in range(3):
            a_run(tmp_path / name / f"s{seed}", AGAIN, seed, sessions=sessions if seed == 2 else 1)
    assert [entry["sessions"] for entry in arms.trained(tmp_path / "continued", AGAIN, RECIPE)] == [1, 1, 3]
    with pytest.raises(SystemExit, match="does not say in how many sessions the run was trained"):
        arms.trained(tmp_path / "silent", AGAIN, RECIPE)
    # a run that is not finished, a missing weight file, a missing log, the same file twice
    with pytest.raises(SystemExit, match="holds no finished training run"):
        arms.trained(tmp_path / "nothing", AGAIN, RECIPE)
    for name, gone, message in (("no_weights", "final.pt", r"lacks final\.pt"), ("no_log", "train_log.jsonl", "the log of the training run is not there")):
        for seed in range(3):
            a_run(tmp_path / name / f"s{seed}", AGAIN, seed)
        (tmp_path / name / "s2" / gone).unlink()
        with pytest.raises(SystemExit, match=message):
            arms.trained(tmp_path / name, AGAIN, RECIPE)
    for seed in range(3):
        a_run(tmp_path / "twice" / f"s{seed}", AGAIN, seed, weight=-float(seed))  # every file holds the same numbers
    with pytest.raises(SystemExit, match="three different files, and none of them a published one"):
        arms.trained(tmp_path / "twice", AGAIN, RECIPE)
    with pytest.raises(SystemExit, match="the recipe of stage 3 is not there"):
        arms.trained(tmp_path / "again", AGAIN, tmp_path / "no_recipe.json")


def test_a_published_file_is_no_model_of_an_arm(tmp_path, monkeypatch):
    for seed in range(3):
        a_run(tmp_path / "again" / f"s{seed}", AGAIN, seed)
    for table, changed in (("START_SHA256", lambda digest: {**published_weights.START_SHA256, 1: digest}),
                           ("WEIGHT_SHA256", lambda digest: {**published_weights.WEIGHT_SHA256, 1: {"final.pt": digest}})):
        with monkeypatch.context() as patch:
            patch.setattr(published_weights, table, changed(comparison.file_sha256(tmp_path / "again" / "s1" / "final.pt")))
            with pytest.raises(SystemExit, match="none of them a published one"):
                arms.trained(tmp_path / "again", AGAIN, RECIPE)
    assert len(arms.trained(tmp_path / "again", AGAIN, RECIPE)) == 3


def test_how_far_the_weights_of_an_arm_lie_from_their_start_and_from_the_arm_again(tmp_path, monkeypatch):
    start = {"w": torch.tensor([3.0, 4.0]), "steps": torch.tensor(7)}  # a length of 5; a counter is no weight
    assert arms.distance({"w": torch.tensor([3.0, 4.0]), "steps": torch.tensor(9)}, start) == 0.0
    assert arms.distance({"w": torch.tensor([3.0, 4.5]), "steps": torch.tensor(7)}, start) == pytest.approx(0.1)
    assert arms.distance({"w": torch.tensor([0.0, 0.0]), "steps": torch.tensor(7)}, start) == pytest.approx(1.0)
    # every seed against its own start and against the model of the same seed of the arm "again"
    for seed in range(3):
        a_run(tmp_path / "again" / f"s{seed}", AGAIN, seed)            # w = (3, 4 + seed)
        a_run(tmp_path / "mirror" / f"s{seed}", MIRROR, seed, 0.5)     # w = (3, 4.5 + seed)
        torch.save({"w": torch.tensor([3.0, 4.0 + 2 * seed]), "steps": torch.tensor(0)}, tmp_path / f"start{seed}.pt")
    asked = []
    monkeypatch.setattr(published_weights, "start_weights", lambda seed, weights_dir=None: asked.append((seed, weights_dir)) or tmp_path / f"start{seed}.pt")
    found = arms.weight_distances(tmp_path / "mirror", tmp_path / "weights", tmp_path / "again")
    assert asked == [(seed, tmp_path / "weights") for seed in range(3)] and [entry["seed"] for entry in found] == [0, 1, 2]
    for seed, entry in enumerate(found):
        own_start, again = np.array([3.0, 4.0 + 2 * seed]), np.array([3.0, 4.0 + seed])
        assert entry["from_its_start"] == pytest.approx(abs(4.5 + seed - own_start[1]) / np.linalg.norm(own_start))
        assert entry["from_again"] == pytest.approx(0.5 / np.linalg.norm(again))
    assert len({round(entry["from_its_start"], 6) for entry in found}) == 3  # a start or a partner of another seed would give another number
    assert all("from_again" not in entry for entry in arms.weight_distances(tmp_path / "again", None, None))


def test_the_models_of_an_arm_are_loaded_in_the_order_of_the_seeds(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(arms, "load_model", lambda path, where: loaded.append(Path(path)) or f"model of {Path(path).parent.name}")
    monkeypatch.setattr(arms.Ensemble, "from_models", classmethod(lambda cls, models, b3, device="auto", batch=0: (models, b3, device, batch)))
    models, setting, device, batch = arms.arm_system(tmp_path / "runs", "cpu", 3)
    assert loaded == [tmp_path / "runs" / f"s{seed}" / "final.pt" for seed in range(3)] and models == ["model of s0", "model of s1", "model of s2"]
    assert setting == B3 and (device, batch) == ("cpu", 3)


def test_the_report_takes_nine_models_trained_with_one_version_and_one_commit(tmp_path):
    def folders(hashes=None, versions=None, code=None, against=None):
        out = {}
        for position, system in enumerate(arms.SYSTEMS):
            out[system] = tmp_path / f"{len(list(tmp_path.iterdir()))}"
            out[system].mkdir()
            about = {"system": system}
            if system in arms.ARMS:
                mine = (hashes or {}).get(system, [f"{system} {k}" for k in range(3)])
                about.update({"weights_sha256": mine, "models": [{"torch": (versions or {}).get(system, "2.9.0")} for _ in mine],
                              "code": (code or {}).get(system, "abc1234")})
                if system != AGAIN:
                    about["set_against_sha256"] = (against or {}).get(system, [f"{AGAIN} {k}" for k in range(3)])
            (out[system] / "system.json").write_text(json.dumps({key: value for key, value in about.items() if value is not None}))
            assert position < 5
        return out

    assert set(arms.about_the(folders())) == set(arms.SYSTEMS)
    with pytest.raises(SystemExit, match="nine models, each a file of its own"):  # the arm without input changes given the files of the arm again
        arms.about_the(folders(hashes={WITHOUT: [f"{AGAIN} {k}" for k in range(3)]}))
    with pytest.raises(SystemExit, match="nine models, each a file of its own"):
        arms.about_the(folders(hashes={MIRROR: ["a", "b"]}))
    with pytest.raises(SystemExit, match=r"not trained with one version of PyTorch: \['2.10.0', '2.9.0'\]"):
        arms.about_the(folders(versions={MIRROR: "2.10.0"}))
    with pytest.raises(SystemExit, match="not trained with one version of PyTorch"):  # runs that do not say which version trained them
        arms.about_the(folders(versions={arm: None for arm in arms.ARMS}))
    with pytest.raises(SystemExit, match=r"do not name one commit of this package: \['abc1234', 'def5678'\]"):
        arms.about_the(folders(code={MIRROR: "def5678"}))
    for missing in (None, ""):
        with pytest.raises(SystemExit, match="do not name one commit of this package"):
            arms.about_the(folders(code={arm: missing for arm in arms.ARMS}))
    with pytest.raises(SystemExit, match="the arm 'with a mirror' was set against other models than those of the arm 'again' given here"):
        arms.about_the(folders(against={MIRROR: ["other 0", "other 1", "other 2"]}))
    with pytest.raises(SystemExit, match="the arm 'without input changes' was set against other models"):
        arms.about_the(folders(against={WITHOUT: None}))
    exchanged = folders()
    exchanged[AGAIN], exchanged[MIRROR] = exchanged[MIRROR], exchanged[AGAIN]
    with pytest.raises(SystemExit, match="was written for 'with a mirror', and is given as 'again'"):
        arms.about_the(exchanged)


def test_the_report_is_held_to_part_two_number_for_number(monkeypatch):
    names = ("unchanged", "mirrored")
    monkeypatch.setattr(arms, "CONDITIONS", names)

    def block(shift: float = 0.0, interval=(-0.01, 0.01)):
        return {name: {"multi": {"segment_f1_50": {"value": 0.03 + shift, "interval": None if interval is None else list(interval)}},
                       "recordings": {"a5c_as_a4c": {"value": 0.0, "interval": None}}} for name in names}

    arms.held_to_part_two(block(), {"published_minus_stage2": block()})
    arms.held_to_part_two(block(1e-12, (-0.01 + 1e-12, 0.01)), {"published_minus_stage2": block()})  # the last bits of a sum
    for mine in (block(1e-6), block(interval=(-0.01 + 1e-6, 0.01)), block(interval=(-0.01, 0.01 - 1e-6)), block(interval=None)):
        with pytest.raises(SystemExit, match=r"under 'unchanged' \(multi, segment_f1_50\) is not what part 2 published"):
            arms.held_to_part_two(mine, {"published_minus_stage2": block()})
    with pytest.raises(SystemExit, match="is not what part 2 published"):  # an interval here where part 2 has none
        arms.held_to_part_two(block(), {"published_minus_stage2": block(interval=None)})
    only_second = block()
    only_second["mirrored"]["multi"]["segment_f1_50"]["interval"][1] += 1e-6
    with pytest.raises(SystemExit, match=r"under 'mirrored' \(multi, segment_f1_50\)"):  # every condition is held, not the first alone
        arms.held_to_part_two(only_second, {"published_minus_stage2": block()})
    assert arms.CLOSE == 1e-9


def test_the_commands_that_train_the_arms_give_the_identities_the_runs_are_held_to():
    """What the driver adds to the command of the plan (where the frames, the output and the checkpoints are) is not
    part of a run's identity: a run made by these commands is the run that `trained` asks for."""
    from echoview.training import train

    for arm, options in arms.ARMS.items():
        for seed in range(3):
            command = ["--config", str(RECIPE), *options, "--seed", str(seed), "--cache", "/data/frame_cache", "--init", "stage2",
                       "--out", f"/data/runs/{FOLDER[arm]}/s{seed}", "--checkpoint-every", "100", "--resume"]
            identity = train.run_identity(train.parse_args(command))
            assert identity == arms.recipe_of(arm, seed, RECIPE) and identity["seed"] == seed and identity["init"] == "stage2"
            assert ("mirror" in identity) == (arm == MIRROR) and identity["strength"] == (0.0 if arm == WITHOUT else 1.0)
            assert not {"cache", "out", "resume", "checkpoint_every"} & set(identity)


# ---------------------------------------------------------------------------------------------- the two commands
def five_runs(tiny, tmp_path, monkeypatch):  # noqa: F811
    """The published stand-in, the stage-2 stand-in and the stand-ins of the three arms, each run into a folder of its
    own under a few conditions, and the report of part 2 on the first two."""
    halves, images = made_up_run(tiny, tmp_path, monkeypatch)
    monkeypatch.setattr(stress, "CONDITIONS", FEW)
    monkeypatch.setattr(arms, "CONDITIONS", FEW)
    monkeypatch.setattr(arms, "MIRRORED", DARKER)
    arm_of, asked = {}, []
    monkeypatch.setattr(arms, "trained", lambda runs, arm, recipe: asked.append((str(runs), arm)) or [
        {"seed": k, "sha256": f"{arm} {k}", "torch": "2.9.0"} for k in range(3)])
    monkeypatch.setattr(arms, "weight_distances", lambda runs, weights, again: [{"seed": k, "from_its_start": 0.1, "again": str(again)} for k in range(3)])
    monkeypatch.setattr(arms, "arm_system", lambda runs, device, batch: stand_in(*SHIFTS[arm_of[str(runs)]]))
    base = ["--images", str(images), "--device", "cpu"]
    work = {PUBLISHED: tmp_path / "published", STAGE2: tmp_path / "stage2", **{arm: tmp_path / f"work_{FOLDER[arm]}" for arm in arms.ARMS}}
    assert stress.main(["run", *base, "--folder", str(work[PUBLISHED])]) == 0
    assert stress.main(["run", *base, "--folder", str(work[STAGE2]), "--system", "stage2"]) == 0
    for arm in arms.ARMS:
        arm_of[str(tmp_path / FOLDER[arm])] = arm
        again = [] if arm == AGAIN else ["--again", str(tmp_path / FOLDER[AGAIN])]
        assert arms.main(["run", "--arm", arm, "--runs", str(tmp_path / FOLDER[arm]), *again, *base, "--folder", str(work[arm]), "--code", CODE]) == 0
    # every arm is held to its own command, and the runs an arm is set against are held to the command of the arm "again"
    assert asked == [(str(tmp_path / "again"), AGAIN), (str(tmp_path / "without"), WITHOUT), (str(tmp_path / "again"), AGAIN),
                     (str(tmp_path / "mirror"), MIRROR), (str(tmp_path / "again"), AGAIN)]
    part2 = tmp_path / "forward.json"
    assert stress.main(["report", "--logits", str(tmp_path / "dev.npz"), "--folder", str(work[PUBLISHED]), "--stage2", str(work[STAGE2]), "--draws", "40",
                        "--out", str(part2)]) == 0
    return halves, images, work, part2


def report_of(tmp_path, work, part2, out, **other):
    folders = {**work, **other}
    return ["report", "--logits", str(tmp_path / "dev.npz"), "--part2", str(part2), "--published", str(folders[PUBLISHED]), "--stage2", str(folders[STAGE2]),
            "--again", str(folders[AGAIN]), "--without", str(folders[WITHOUT]), "--mirror", str(folders[MIRROR]), "--draws", "40", "--out", str(out)]


def test_the_arms_are_run_and_scored_beside_the_published_system_and_stage_2(tiny, tmp_path, monkeypatch, capsys):  # noqa: F811
    """One run of the five stand-ins carries every check below, because the runs are the slow part."""
    halves, images, work, part2 = five_runs(tiny, tmp_path, monkeypatch)
    selection, calibration = halves["selection"], halves["calibration"]
    natives = [s for s in selection if s.kind == "native"]

    # ---- the run of an arm: its window logits of both halves, one file per condition, and what it was run from
    assert capsys.readouterr().out.count("done: the window logits of the unchanged frames of both halves") == 3
    for arm in arms.ARMS:
        files = sorted(p.name for p in work[arm].glob("*.npz"))
        assert files == sorted([f"{stress.slug(name)}.npz" for name in FEW] + [arms.LOGITS_FILE])
        about = json.loads((work[arm] / "system.json").read_text())
        assert about["system"] == arm and about["weights_sha256"] == [f"{arm} {k}" for k in range(3)] and len(about["models"]) == 3
        assert about["code"] == CODE and about.get("set_against_sha256") == (None if arm == AGAIN else [f"{AGAIN} {k}" for k in range(3)])
        assert about["distances"][0]["again"] == ("None" if arm == AGAIN else str(tmp_path / FOLDER[AGAIN]))  # the other arms are set against "again"
        with np.load(work[arm] / arms.LOGITS_FILE) as data:
            assert set(data.files) == {s.id for s in (*calibration, *selection)} and data[selection[0].id].shape[1:] == (3, 5)
            direct = run.all_logits(stand_in(*SHIFTS[arm]), selection[:1], ev9v.FrameTable.build(selection[:1], images))
            assert np.allclose(data[selection[0].id], direct[selection[0].id])
    with pytest.raises(SystemExit, match="give its training runs with --again"):
        arms.main(["run", "--arm", MIRROR, "--runs", str(tmp_path / "mirror"), "--images", str(images), "--device", "cpu", "--folder", str(tmp_path / "x"),
                   "--code", CODE])

    # ---- the report, watched: what the rule is asked, what is decoded with which setting, what the word is read from
    monkeypatch.setattr(arms, "FAMILY", (BRIGHTER, CLEAN))
    choices, decodings, words, tables, taken, held = [], [], [], [], {}, []
    choose, decoded, word = comparison.choose, comparison.decoded, arms.mirror_word
    score_rows, minus, held_to_part_two = comparison.score_rows, ablations.minus, arms.held_to_part_two
    keys = ("half_window", "threshold", "nms_radius")

    def watched_minus(first, second, level):
        out = minus(first, second, level)
        taken[id(out)] = (first, second, level, out)  # the result is kept, so that its number names it for good
        return out

    monkeypatch.setattr(comparison, "score_rows", lambda *given: tables.append(score_rows(*given)) or tables[-1])
    monkeypatch.setattr(ablations, "minus", watched_minus)
    monkeypatch.setattr(arms, "held_to_part_two", lambda mine, part2: held.append(mine) or held_to_part_two(mine, part2))
    monkeypatch.setattr(comparison, "choose", lambda probabilities, sequences, drop_floors: choices.append((set(probabilities), sequences, drop_floors))
                        or choose(probabilities, sequences, drop_floors))
    monkeypatch.setattr(comparison, "decoded", lambda rows, setting: decodings.append(tuple(setting[key] for key in keys)) or decoded(rows, setting))
    monkeypatch.setattr(arms, "mirror_word", lambda *given: words.append(given) or word(*given))
    out = tmp_path / "arms.json"
    assert arms.main(report_of(tmp_path, work, part2, out)) == 0
    for module, name, real in ((comparison, "choose", choose), (comparison, "decoded", decoded), (arms, "mirror_word", word),
                               (comparison, "score_rows", score_rows), (ablations, "minus", minus), (arms, "held_to_part_two", held_to_part_two)):
        monkeypatch.setattr(module, name, real)
    result = json.loads(out.read_text())
    # what is held to part 2 is the published system minus the stage-2 ensemble as part 2 decoded it, not minus its row with the published setting
    assert len(held) == 1 and set(held[0]) == set(FEW) and len(tables) == 2  # the conditions, and then the seeds
    for name in FEW:
        first, second, level, _ = taken[id(held[0][name])]
        assert first is tables[0][f"{PUBLISHED} | {name}"] and second is tables[0][f"{arms.AS_IN_PART_2} | {name}"] and level == 95.0
        assert second is not tables[0][f"{STAGE2} | {name}"]
    assert result["code"] == CODE
    assert (result["benchmark"], result["half"], result["sequences"], result["draws"]) == ("development", "selection", len(selection), 40)
    assert list(result["systems"]) == [PUBLISHED, STAGE2, AGAIN, WITHOUT, MIRROR] == list(arms.SYSTEMS) and result["decoder"] == dict(B3)
    assert result["sha256"] == {"logits": comparison.file_sha256(tmp_path / "dev.npz"), "part2": comparison.file_sha256(part2)}
    assert result["unchanged_against_stored"] <= 1e-6 and set(result["unchanged_against_own_logits"]) == set(arms.ARMS)
    assert max(result["unchanged_against_own_logits"].values()) <= 1e-6
    systems = result["systems"]
    # the rule chooses on the calibration half, once for every system, and the floors may be dropped
    assert len(choices) == 5 and all(sequences == calibration and drop is True and ids == {s.id for s in calibration} for ids, sequences, drop in choices)
    # what is decoded with which setting, to the count: every condition of every system and every seed with the published setting, the
    # stage-2 ensemble once more under every condition with the setting of the rule (as part 2 decoded it), and one row of the rule for every system
    own = {system: tuple(systems[system]["unchanged_with"][OWN]["setting"][key] for key in keys) for system in arms.SYSTEMS}
    expected = Counter({tuple(B3[key] for key in keys): 5 * len(FEW) + 12 + 1})  # and once more for the check of the published number
    expected.update({own[STAGE2]: len(FEW)})
    for system in arms.SYSTEMS:
        expected.update({own[system]: 2})  # the rule scores its own choice once, and the row of the rule is decoded with it
    assert Counter(decodings) == expected and own[STAGE2] != tuple(B3[key] for key in keys)

    def value(system: str, name: str, part: str = "recordings", key: str = "accuracy") -> float:
        return systems[system]["conditions"][name]["selection"][part][key]["value"]

    # the stand-ins tell the systems apart, as the table at the top of this file says
    right, wrong = 1.0, pytest.approx(0.2)
    assert [value(system, CLEAN) for system in arms.SYSTEMS] == [right, wrong, right, right, right]
    assert [value(system, BRIGHTER) for system in arms.SYSTEMS] == [wrong, wrong, right, wrong, wrong]
    assert [value(system, DARKER) for system in arms.SYSTEMS] == [wrong, pytest.approx(0.8), wrong, wrong, right]  # black stays black: stage 2 misreads it
    for system in arms.SYSTEMS:
        assert list(systems[system]["conditions"]) == list(FEW) and systems[system]["system"]["system"] == system
        for name in (BRIGHTER, DARKER):  # a condition is set against the system's own unchanged frames
            assert systems[system]["conditions"][name]["minus_unchanged"]["recordings"]["accuracy"]["value"] == pytest.approx(
                value(system, name) - value(system, CLEAN))
        assert "minus_unchanged" not in systems[system]["conditions"][CLEAN]
        assert set(systems[system]["unchanged_with"]) == {OWN, NONE}
        assert systems[system]["unchanged_with"][NONE]["selection"]["recordings"]["accuracy"]["value"] == pytest.approx(value(system, CLEAN))
    dropped = {system: systems[system]["unchanged_with"][OWN]["setting"]["floors_dropped"] for system in arms.SYSTEMS}
    assert dropped[STAGE2] is True and dropped[AGAIN] is False  # stage 2's stand-in reads the single recordings wrong: no setting keeps the floors
    errors = {system: [systems[system]["conditions"][name]["classify"]["errors"] for name in FEW] for system in arms.SYSTEMS}
    assert errors[AGAIN] == [0, 0, 4] and errors[MIRROR] == [0, 4, 0] and errors[WITHOUT] == [0, 4, 4] == errors[PUBLISHED]  # `classify`'s counts, each its own

    # every difference is the first minus the second
    differences = result["differences"]
    assert list(differences) == ["again minus without input changes", "without input changes minus stage 2", "again minus stage 2", "with a mirror minus again",
                                 "with a mirror minus published", "published minus again", "published minus stage 2"]
    for title, (first, second) in zip(differences, arms.DIFFERENCES):
        assert title == f"{first} minus {second}" and set(differences[title]) == set(FEW)
        for name in FEW:
            for part, key in (("recordings", "accuracy"), ("multi", "segment_f1_50"), ("multi", "wrong_samples")):
                assert differences[title][name][part][key]["value"] == pytest.approx(value(first, name, part, key) - value(second, name, part, key))
    accuracy = {title: [differences[title][name]["recordings"]["accuracy"]["value"] for name in FEW] for title in differences}
    assert accuracy["again minus without input changes"] == [0.0, pytest.approx(0.8), 0.0]
    assert accuracy["with a mirror minus again"] == [0.0, pytest.approx(-0.8), pytest.approx(0.8)]
    assert accuracy["with a mirror minus published"] == [0.0, 0.0, pytest.approx(0.8)] and accuracy["published minus again"] == [0.0, pytest.approx(-0.8), 0.0]
    assert accuracy["again minus stage 2"] == [pytest.approx(0.8), pytest.approx(0.8), pytest.approx(-0.6)]

    # the first question: the share of "again minus stage 2" that "again minus without input changes" is, and the same with the published system
    family = result["input_changes"]
    assert family["level"] == 99.0 and list(family["conditions"]) == [BRIGHTER, CLEAN]
    f1 = {system: {name: value(system, name, "multi", "segment_f1_50") for name in FEW} for system in arms.SYSTEMS}
    brighter, clean = family["conditions"][BRIGHTER], family["conditions"][CLEAN]
    for found, name in ((brighter, BRIGHTER), (clean, CLEAN)):
        assert found["again_minus_without"]["level"] == 99.0 and found["again_minus_without"]["value"] == pytest.approx(f1[AGAIN][name] - f1[WITHOUT][name])
        assert found["sentence"] == arms.sentence(found["share"]) and found["published_minus_again"]["level"] == 99.0
        assert found["published_minus_again"]["value"] == pytest.approx(f1[PUBLISHED][name] - f1[AGAIN][name])
        assert found["with_the_published_system"]["sentence"] == arms.sentence(found["with_the_published_system"]["share"])
        assert found["the_two_trainings_agree"] == (found["sentence"] == found["with_the_published_system"]["sentence"])
        assert found["share"]["level"] == 99.0 and found["with_the_published_system"]["share"]["level"] == 99.0 and found["share"]["of"]["level"] == 99.0
    # 40 grey levels brighter the arm without input changes and stage 2 are equally wrong: the whole difference is the input changes'
    assert brighter["share"]["value"] == pytest.approx((f1[AGAIN][BRIGHTER] - f1[WITHOUT][BRIGHTER]) / (f1[AGAIN][BRIGHTER] - f1[STAGE2][BRIGHTER]))
    assert brighter["share"]["value"] == pytest.approx(1.0) and brighter["sentence"] == MOSTLY_CHANGES
    # the published stand-in is as wrong there as stage 2: with it in the place of "again" there is nothing to split, and the two do not agree
    assert brighter["with_the_published_system"]["sentence"] == NOTHING and brighter["the_two_trainings_agree"] is False
    assert brighter["stands"] is None  # two trainings, two sentences: neither stands
    # on the unchanged frames the arm without input changes is as right as the arm again: none of the difference is the input changes'
    assert clean["share"]["value"] == 0.0 and clean["sentence"] == MOSTLY_LOOP
    assert clean["share"]["of"]["value"] == pytest.approx(f1[AGAIN][CLEAN] - f1[STAGE2][CLEAN])
    assert clean["with_the_published_system"]["sentence"] == MOSTLY_LOOP and clean["the_two_trainings_agree"] is True and clean["stands"] == MOSTLY_LOOP

    # the second question: what the word is read from, and the word. 40 darker stands in for the mirrored image here
    assert len(words) == 1
    apart, own_loss, recordings, conditions = words[0]
    assert apart == differences["with a mirror minus again"] and own_loss == systems[MIRROR]["conditions"][DARKER]["minus_unchanged"]
    assert recordings == len(natives) and conditions == (BRIGHTER, CLEAN)
    mirror = result["mirror"]
    assert mirror["under_the_mirrored_image"] == differences["with a mirror minus again"][DARKER] and mirror["its_own_loss"] == own_loss
    assert mirror["on_the_unchanged_frames"] == differences["with a mirror minus again"][CLEAN]
    assert mirror["margins"] == {"its_own_loss": 0.03, "on_the_unchanged_frames": 0.01}
    # it gains where "again" fails. Its streams lose more than the margin against its own unchanged frames, and 40 brighter it is
    # below "again": two conditions fail, and it is no candidate
    lost = own_loss["multi"]["segment_f1_50"]["value"]
    assert lost < -0.03 and own_loss["recordings"]["accuracy"]["value"] == 0.0
    assert mirror["holds"] == {"gains under the mirrored image": True, "stands the mirrored image": False, "not below on the unchanged frames": True,
                               "at most one recording fewer right": True, "not below under the five conditions": False}
    assert mirror["word"] == NOT_SUPPORTED
    # with a family under which it is not below and a margin that its own loss keeps, every condition holds and it is a candidate
    with monkeypatch.context() as patch:
        patch.setattr(arms, "FAMILY", (CLEAN,))
        patch.setattr(arms, "OWN_LOSS", -lost + 0.001)
        assert arms.main(report_of(tmp_path, work, part2, tmp_path / "second.json")) == 0
    second = json.loads((tmp_path / "second.json").read_text())["mirror"]
    assert second["word"] == CANDIDATE and all(second["holds"].values()) and second["margins"]["its_own_loss"] == pytest.approx(-lost + 0.001)
    # and with the margin just under its loss, that one condition fails again
    with monkeypatch.context() as patch:
        patch.setattr(arms, "FAMILY", (CLEAN,))
        patch.setattr(arms, "OWN_LOSS", -lost - 0.001)
        assert arms.main(report_of(tmp_path, work, part2, tmp_path / "third.json")) == 0
    third = json.loads((tmp_path / "third.json").read_text())["mirror"]
    assert third["word"] == NOT_SUPPORTED and [name for name, holds in third["holds"].items() if not holds] == ["stands the mirrored image"]

    # the seeds: every seed alone with the published setting, and a seed minus the seed of the same number of the arm "again"
    seeds = result["seeds"]
    assert list(seeds["systems"]) == [f"{system} | seed {k}" for system in (PUBLISHED, AGAIN, WITHOUT, MIRROR) for k in range(3)]

    def alone(system: str, k: int) -> float:
        return seeds["systems"][f"{system} | seed {k}"]["recordings"]["accuracy"]["value"]

    assert [alone(AGAIN, k) for k in range(3)] == [right, wrong, right]  # the member with a shift of 30 reads four of five views wrong
    assert [alone(MIRROR, k) for k in range(3)] == [right, right, right] == [alone(PUBLISHED, k) for k in range(3)]
    assert list(seeds["minus_again"]) == [PUBLISHED, WITHOUT, MIRROR]
    for system in seeds["minus_again"]:
        gaps = [seeds["minus_again"][system][f"seed {k}"]["recordings"]["accuracy"]["value"] for k in range(3)]
        assert gaps == [0.0, pytest.approx(0.8), 0.0]  # the system's seed minus the seed of "again": only seed 1 of "again" is wrong
    names = {s.id for half in halves.values() for s in half} | {f.video for half in halves.values() for s in half for f in s.fragments}
    assert not any(name in out.read_text() for name in names) and "NaN" not in out.read_text()  # pooled numbers only

    # ---- the report stops when a folder, the logits or part 2 are not what they are given as, and never replaces an output
    with pytest.raises(SystemExit, match="does not replace"):
        arms.main(report_of(tmp_path, work, part2, out))
    with pytest.raises(SystemExit, match="was written for 'with a mirror', and is given as 'again'"):
        arms.main(report_of(tmp_path, work, part2, tmp_path / "a.json", **{AGAIN: work[MIRROR], MIRROR: work[AGAIN]}))
    with pytest.raises(SystemExit, match="was written for 'stage 2', and is given as 'published'"):
        arms.main(report_of(tmp_path, work, part2, tmp_path / "a.json", **{PUBLISHED: work[STAGE2], STAGE2: work[PUBLISHED]}))
    other = json.loads(part2.read_text())
    other["published_minus_stage2"][DARKER]["multi"]["segment_f1_50"]["value"] += 0.001
    (tmp_path / "other.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="is not what part 2 published"):
        arms.main(report_of(tmp_path, work, tmp_path / "other.json", tmp_path / "a.json"))
    other["published_minus_stage2"][DARKER]["multi"]["segment_f1_50"]["value"] -= 0.001 - 1e-12  # the last bits of a sum may differ
    (tmp_path / "close.json").write_text(json.dumps(other))
    assert arms.main(report_of(tmp_path, work, tmp_path / "close.json", tmp_path / "close_out.json")) == 0
    with pytest.raises(SystemExit, match="the output of part 2 is not there"):
        arms.main(report_of(tmp_path, work, tmp_path / "missing.json", tmp_path / "a.json"))
    other = json.loads(part2.read_text())
    other["sha256"]["logits"] = "0" * 64  # a part 2 that was made from other stored outputs
    (tmp_path / "elsewhere.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="was made from other stored outputs than"):
        arms.main(report_of(tmp_path, work, tmp_path / "elsewhere.json", tmp_path / "a.json"))
    real = (work[WITHOUT] / arms.LOGITS_FILE).read_bytes()
    (work[WITHOUT] / arms.LOGITS_FILE).write_bytes((work[AGAIN] / arms.LOGITS_FILE).read_bytes())  # the logits of another arm
    with pytest.raises(SystemExit, match="the two runs of the arm 'without input changes' on the unchanged frames do not agree"):
        arms.main(report_of(tmp_path, work, part2, tmp_path / "a.json"))
    (work[WITHOUT] / arms.LOGITS_FILE).unlink()
    with pytest.raises(SystemExit, match=r"lacks unchanged_logits\.npz"):
        arms.main(report_of(tmp_path, work, part2, tmp_path / "a.json"))
    (work[WITHOUT] / arms.LOGITS_FILE).write_bytes(real)
    with monkeypatch.context() as patch:
        patch.setattr(comparison, "PUBLISHED_SELECTION_F1", comparison.PUBLISHED_SELECTION_F1 + 0.01)
        with pytest.raises(SystemExit, match="must give"):
            arms.main(report_of(tmp_path, work, part2, tmp_path / "a.json"))
    assert not (tmp_path / "a.json").exists()
    with pytest.raises(SystemExit, match="in this order"):
        arms.report({AGAIN: work[AGAIN]}, {}, {}, {"calibration": (), "selection": ()})


def test_an_arm_keeps_a_folder_of_its_own_and_goes_on_where_it_stopped(tiny, tmp_path, monkeypatch):  # noqa: F811
    halves, images = made_up_run(tiny, tmp_path, monkeypatch)
    monkeypatch.setattr(stress, "CONDITIONS", FEW[:2])
    monkeypatch.setattr(arms, "CONDITIONS", FEW[:2])
    weights = {"hash": "first"}
    monkeypatch.setattr(arms, "trained", lambda runs, arm, recipe: [{"seed": k, "sha256": f"{weights['hash']} {k}"} for k in range(3)])
    monkeypatch.setattr(arms, "weight_distances", lambda runs, weights, again: [])
    monkeypatch.setattr(arms, "arm_system", lambda runs, device, batch: stand_in(*SHIFTS[AGAIN]))
    command = ["run", "--arm", AGAIN, "--runs", str(tmp_path / "again"), "--images", str(images), "--device", "cpu", "--folder", str(tmp_path / "work"),
               "--code", CODE]
    assert arms.main(command) == 0
    with pytest.raises(SystemExit, match=r"holds the outputs of another commit of this package \(abc1234\)"):
        arms.main([*command[:-1], "def5678"])
    with pytest.raises(SystemExit):  # the commit must be named
        arms.main(command[:-2])
    (tmp_path / "work" / "brighter_by_40.npz").unlink()
    calls = []
    condition = stress.run_condition
    monkeypatch.setattr(stress, "run_condition", lambda system, sequences, table, name: calls.append(name) or condition(system, sequences, table, name))
    assert arms.main(command) == 0 and calls == [BRIGHTER]  # a file that exists is not computed again
    with pytest.raises(SystemExit, match="holds the outputs of another system"):
        arms.main([*command[:2], WITHOUT, *command[3:], "--again", str(tmp_path / "again")])
    weights["hash"] = "second"
    with pytest.raises(SystemExit, match="other weight files"):
        arms.main(command)
    with pytest.raises(SystemExit, match="holds the outputs of another system"):
        stress.main(["run", "--images", str(images), "--device", "cpu", "--folder", str(tmp_path / "work")])
    assert ablations.CANDIDATE == CANDIDATE and len(halves) == 2


def test_the_plan_of_part_four_names_what_the_code_holds():
    """Section 12 of docs/ABLATIONS.md was written before any model of part 4 was trained: what it fixes is what the code holds."""
    from echoview.training import augment, train

    root = Path(__file__).resolve().parents[1]
    plan = (root / "docs" / "ABLATIONS.md").read_text()
    section = plan[plan.index("## 12. Part 4: stage 3 trained again, three ways"):plan.index("## 13. Results of part 4")]
    # the three arms and their commands
    assert arms.ARMS == {AGAIN: (), WITHOUT: ("--strength", "0"), MIRROR: ("--mirror", "0.5")}
    assert list(arms.ARMS) == ["again", "without input changes", "with a mirror"]
    for arm, options in arms.ARMS.items():
        command = " ".join(["echoview-train --config configs/stage3.json", *options])
        assert f"| {arm} | `{command}` |" in section
    recipe = json.loads(RECIPE.read_text())
    assert (recipe["updates"], recipe["strength"], recipe["objective"], recipe["batch"]) == (1500, 1.0, "within_view", 32) and "mirror" not in recipe
    assert "trained with this package for 1,500 updates" in section and train.DEFAULTS["mirror"] == 0.0 and arms.RECIPE == Path("configs") / "stage3.json"
    assert "(0 switches augmentation off)" in augment.Strength.__doc__ and "every random input change switched off" in section
    assert "every group of windows mirrored left to right with a probability of one half" in section
    assert "`--init stage2`" in section and "`--checkpoint-every 100 --resume`" in section and arms.recipe_of(AGAIN, 0, RECIPE)["init"] == "stage2"
    # what is measured, and how
    assert "`python -m echoview.evaluation.arms` holds the two commands, `run` and `report`. The output is `docs/ablations/arms.json`." in section
    assert "under each of the 26 conditions" in section and len(stress.CONDITIONS) == 27 and stress.CONDITIONS[0] == CLEAN
    assert "**The decoder setting is the published one for every system.**" in section and (OWN, NONE) == ("the setting by the rule", "no decoder")
    assert "a threshold of 0.185" in section and arms.AS_IN_PART_2 == "stage 2 as in part 2"
    part2 = json.loads((root / "docs" / "ablations" / "forward.json").read_text())
    assert f"{part2['systems']['stage 2']['setting']['params']['threshold']:.3f}" == "0.185"
    assert "**Seven differences for every condition,**" in section and len(arms.DIFFERENCES) == 7
    assert list(arms.DIFFERENCES) == [(AGAIN, WITHOUT), (WITHOUT, STAGE2), (AGAIN, STAGE2), (MIRROR, AGAIN), (MIRROR, PUBLISHED), (PUBLISHED, AGAIN),
                                      (PUBLISHED, STAGE2)]
    assert "to 0.001 in every probability" in section and stress.CLOSE == 1e-3 and "over the last 500 updates of the run" in section
    assert "nine different files and none a published one" in section and "one version of PyTorch" in section
    assert "must name one commit of this package" in section and "in how many sessions it was trained" in section
    assert 'set against the three models of the arm "again" that the report is given' in section
    assert "must have been made from the stored outputs that the report reads" in section and "neither stands" in section
    assert "All nine are trained on one machine, with one commit of this package." in section and "one session" not in section
    assert "The mirror is part of the geometry" in section and "the loss as the loop logs it, with its penalty" in section
    # the first question: the family, the level, the sentences, and the second training
    assert ("noise with a spread of 25 grey levels, blur of 2 pixels, 64 pixels, speckle, and a contrast of 0.5" in section
            and arms.FAMILY == ("noise 25", "blur 2 pixels", "64 pixels", "speckle 0.25", "contrast 0.5"))
    assert "these intervals are taken at 99%" in section and arms.FAMILY_LEVEL == 99.0
    for words in (MOSTLY_CHANGES, MOSTLY_LOOP, NOT_TOLD, NOTHING):
        assert f'"{words}"' in section
    assert "wholly above one half" in section and "wholly below one half" in section and "left out of the interval and counted" in section
    assert "with the published system in the place of the arm \"again\"" in section
    # the second question: the word and its five conditions
    assert f'is a "{CANDIDATE}" when' in section and f'the word is "{NOT_SUPPORTED}"' in section and arms.MIRRORED == "mirrored"
    for line in ("- under the mirrored image, the 95% interval of \"with a mirror\" minus \"again\" lies wholly above zero;",
                 "- under the mirrored image, the arm loses no more than 0.03 against its own unchanged frames;",
                 "- on the unchanged frames, the interval of \"with a mirror\" minus \"again\" does not lie wholly below zero, and the arm is no more "
                 "than 0.01 below;",
                 "- on the unchanged frames, at most one single recording fewer is read right by the majority rule;",
                 "- under none of the five conditions of the first question does the interval of \"with a mirror\" minus \"again\" lie wholly below zero."):
        assert line in section
    assert (arms.OWN_LOSS, arms.UNCHANGED, comparison.DESCRIBED) == (0.03, 0.01, 95.0)
    assert "The other 20 conditions are described" in section and len(stress.CONDITIONS) - 1 - 1 - len(arms.FAMILY) == 20
    # the scale of two trainings, as the training document gives it
    training = (root / "docs" / "TRAINING.md").read_text()
    assert "| Ensemble, decoded with the published setting | 0.9565 (0.9594) |" in training
    assert "| Ensemble, setting chosen again by the rule | 0.9432 (0.9594) |" in training
    assert "Each new seed is within 0.004 of the published seed on both halves" in training
    assert "within 0.004 of the published ones, and the three together within 0.003 with the published decoder setting and 0.016 below" in section
    assert (round(0.9594 - 0.9565, 3), round(0.9594 - 0.9432, 3)) == (0.003, 0.016)
    # the plan stands as it was committed, and only its line of status says that part 4 has been run since (tests/test_docs_arms.py holds the results)
    assert ("**Status: planned on 2026-11-26 and committed on 2026-11-27 (UTC), before any of its models was trained. Run on 2026-11-27 and 2026-11-28; "
            "the results are in section 13.**") in section
