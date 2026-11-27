"""Stage 3 trained again, three ways, on the development benchmark (docs/ABLATIONS.md, part 4).

Three arms, each three models (seeds 0, 1, 2) trained with this package from the published stage-2 model of the seed:

  again                    echoview-train --config configs/stage3.json                the published recipe, trained again
  without input changes    echoview-train --config configs/stage3.json --strength 0   every random input change switched off
  with a mirror            echoview-train --config configs/stage3.json --mirror 0.5   the recipe, and every group of windows
                                                                                      mirrored left to right half the time

None of these models is published. Two commands score them:

  python -m echoview.evaluation.arms run --arm <name> --runs <folder with s0, s1, s2> --images <EV9V Images> --folder <work>
      --code <the commit of this package that trained the models and runs them>
      the three models of one arm as an ensemble: the window logits of the unchanged frames of both development halves,
      and the selection half under every condition of part 2, one file per condition as `stress run` writes them
  python -m echoview.evaluation.arms report --logits <development logits of the published system> --part2 <forward.json>
      --published <work folder of `stress run`> --stage2 <that of `stress run --system stage2`>
      --again <work> --without <work> --mirror <work> --out arms.json
      every system under every condition with the same draws, decoded with the published setting

"again" minus "without input changes" is what the input changes add. "with a mirror" minus "again" is what the mirror
adds. The published system minus "again" is what differs between two trainings of one recipe. The test benchmark is
not read.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .. import weights as published_weights
from ..model import load_model
from ..pipeline import Ensemble, pick_device
from ..reference import B3, SEEDS
from . import ablations, benchmark, comparison, ev9v, stress
from . import run as evaluation_run
from .benchmark import Sequence, of_kind
from .comparison import DESCRIBED, DRAWS, ROUTER, SEED, Row
from .run import member_probabilities
from .stress import CLEAN, CONDITIONS, PUBLISHED, STAGE2, TIME

AGAIN, WITHOUT, MIRROR = "again", "without input changes", "with a mirror"
ARMS = {AGAIN: (), WITHOUT: ("--strength", "0"), MIRROR: ("--mirror", "0.5")}  # what each arm adds to the command of the published recipe
SYSTEMS = (PUBLISHED, STAGE2, AGAIN, WITHOUT, MIRROR)
RECIPE = Path("configs") / "stage3.json"
LOGITS_FILE = "unchanged_logits.npz"  # the window logits of the unchanged frames of both halves: (windows, seeds, views) for every sequence
AS_IN_PART_2 = "stage 2 as in part 2"  # the stage-2 ensemble with the setting the rule gives it, as part 2 decoded it
FAMILY = ("noise 25", "blur 2 pixels", "64 pixels", "speckle 0.25", "contrast 0.5")  # where part 2 found the published system farthest ahead of stage 2
FAMILY_LEVEL = 99.0  # five comparisons decide a sentence each, so their intervals are taken at 99%
MOSTLY_CHANGES, MOSTLY_LOOP, NOT_TOLD, NOTHING = "mostly the input changes", "mostly the loop without them", "not told apart", "no difference to split"
CANDIDATE, NOT_SUPPORTED = ablations.CANDIDATE, ablations.NOT_SUPPORTED
MIRRORED = "mirrored"  # the condition of part 2 under which the arm with a mirror is asked its question
OWN_LOSS = 0.03    # under the mirrored image the arm with a mirror may lose this much segment F1 against its own unchanged frames
UNCHANGED = 0.01   # and on the unchanged frames it may lie this far below the arm "again"
DIFFERENCES = ((AGAIN, WITHOUT), (WITHOUT, STAGE2), (AGAIN, STAGE2), (MIRROR, AGAIN), (MIRROR, PUBLISHED), (PUBLISHED, AGAIN), (PUBLISHED, STAGE2))
OWN, NONE = "the setting by the rule", "no decoder"
CLOSE = 1e-9  # two scorings of the same outputs on two machines may differ in the last bits of a sum


# ------------------------------------------------------------------------------------------- the trained models
def recipe_of(arm: str, seed: int, recipe: Path) -> dict:
    """The identity that the training program gives the run the plan names for an arm and a seed."""
    from ..training import train

    if not recipe.is_file():
        raise SystemExit(f"{recipe}: the recipe of stage 3 is not there; give it with --recipe")
    args = train.parse_args(["--config", str(recipe), *ARMS[arm], "--seed", str(seed), "--init", "stage2", "--cache", "cache", "--out", "out"])
    return train.run_identity(args)


def training_record(log: Path) -> dict:
    """The end of a training log: the loss and the accuracy on the training windows over its last ten lines."""
    if not log.is_file():
        raise SystemExit(f"{log}: the log of the training run is not there")
    lines = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    tail = lines[-10:]
    if not tail:
        raise SystemExit(f"{log}: the log of the training run is empty")
    return {"lines": len(lines), "last_update": tail[-1].get("update"), "loss": float(np.mean([line["loss"] for line in tail])),
            "accuracy": float(np.mean([line["accuracy"] for line in tail]))}


def trained(runs: Path, arm: str, recipe: Path = RECIPE) -> list[dict]:
    """What the three training runs of an arm say of themselves. Each must be the finished run that the plan names for
    the arm and the seed, option for option, started from the published stage-2 model of the seed."""
    found = []
    for seed in SEEDS:
        folder = runs / f"s{seed}"
        try:
            done = json.loads((folder / "done.json").read_text())
            config = json.loads((folder / "config.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"{folder} holds no finished training run ({error})") from error
        mine, expected = done.get("identity", {}), recipe_of(arm, seed, recipe)
        apart = {key: (mine.get(key), expected.get(key)) for key in sorted(set(mine) | set(expected)) if mine.get(key) != expected.get(key)}
        if apart:
            raise SystemExit(f"{folder} is not a run of the arm '{arm}' for seed {seed}: {apart}")
        if not (folder / "final.pt").is_file():
            raise SystemExit(f"{folder} lacks final.pt")
        sessions = done.get("sessions")  # one, and one more for every checkpoint the run went on from
        if not isinstance(sessions, int) or sessions < 1:
            raise SystemExit(f"{folder / 'done.json'} does not say in how many sessions the run was trained")
        found.append({"seed": seed, "identity": mine, "torch": config.get("torch"), "sessions": sessions,
                      "sha256": comparison.file_sha256(folder / "final.pt"), "training": training_record(folder / "train_log.jsonl")})
    hashes = [entry["sha256"] for entry in found]
    known = {digest for files in published_weights.WEIGHT_SHA256.values() for digest in files.values()} | set(published_weights.START_SHA256.values())
    if len(set(hashes)) != len(hashes) or known & set(hashes):
        raise SystemExit(f"{runs}: the weight files of an arm are three different files, and none of them a published one")
    return found


def distance(first: Mapping, second: Mapping) -> float:
    """How far two sets of weights lie apart, as a share of the length of the second."""
    import torch

    keys = [key for key in second if torch.is_floating_point(second[key])]
    apart = sum(float(((first[key].double() - second[key].double()) ** 2).sum()) for key in keys)
    length = sum(float((second[key].double() ** 2).sum()) for key in keys)
    return float(np.sqrt(apart / length))


def weight_distances(runs: Path, weights_dir: Path | None, again: Path | None) -> list[dict]:
    """For every seed: the arm's model against the stage-2 model it started from, and against the model of the arm
    "again" of the same seed when that arm's runs are given."""
    import torch

    out = []
    for seed in SEEDS:
        mine = torch.load(runs / f"s{seed}" / "final.pt", map_location="cpu", weights_only=True)
        start = torch.load(published_weights.start_weights(seed, weights_dir), map_location="cpu", weights_only=True)
        entry = {"seed": seed, "from_its_start": distance(mine, start)}
        if again is not None:
            entry["from_again"] = distance(mine, torch.load(again / f"s{seed}" / "final.pt", map_location="cpu", weights_only=True))
        out.append(entry)
    return out


def arm_system(runs: Path, device: str, batch: int) -> Ensemble:
    """The three models of an arm as one ensemble, in the order of the seeds."""
    where = pick_device(device)
    return Ensemble.from_models([load_model(runs / f"s{seed}" / "final.pt", where) for seed in SEEDS], B3, device=device, batch=batch)


# ------------------------------------------------------------------------------------------------------ report
def share(first: comparison.Scored, second: comparison.Scored, third: comparison.Scored, name: str, level: float) -> dict:
    """(first minus second) as a share of (first minus third) for one number, with a percentile interval from the same
    draws. Where the interval of first minus third does not lie above zero there is nothing to split. A draw in which
    first minus third is not positive has no share; such draws are left out of the interval and counted."""
    whole = comparison.difference(first, third, name, level)
    if whole["interval"] is None or whole["interval"][0] <= 0:
        return {"value": None, "level": level, "interval": None, "of": whole}
    by_unit, without = {}, {}
    for unit in first.replicates:
        part = first.replicates[unit][name] - second.replicates[unit][name]
        full = first.replicates[unit][name] - third.replicates[unit][name]
        usable = full > 0
        by_unit[unit] = comparison.percentiles(part[usable] / full[usable], level)
        without[unit] = int((~usable).sum())
    return {"value": float((first.values[name] - second.values[name]) / (first.values[name] - third.values[name])), "level": level, **by_unit,
            "interval": comparison.wider(by_unit), "draws_without_a_share": without, "of": whole}


def sentence(found: Mapping) -> str:
    """The sentence the plan fixed for a share: above one half, below it, or not told apart."""
    if found["interval"] is None:
        return NOTHING
    return MOSTLY_CHANGES if found["interval"][0] > 0.5 else MOSTLY_LOOP if found["interval"][1] < 0.5 else NOT_TOLD


def mirror_word(apart: Mapping[str, Mapping], own_loss: Mapping, recordings: int, family: tuple[str, ...]) -> dict:
    """The word for the arm with a mirror. `apart` holds "with a mirror" minus "again" for every condition, and
    `own_loss` the arm's own mirrored image minus its own unchanged frames. Every condition of the plan must hold."""
    f1 = ("multi", "segment_f1_50")
    gain, here = apart[MIRRORED][f1[0]][f1[1]], apart[CLEAN][f1[0]][f1[1]]
    holds = {
        "gains under the mirrored image": gain["interval"] is not None and gain["interval"][0] > 0,
        "stands the mirrored image": own_loss[f1[0]][f1[1]]["value"] >= -OWN_LOSS,
        "not below on the unchanged frames": not (here["interval"] is not None and here["interval"][1] < 0) and here["value"] >= -UNCHANGED,
        "at most one recording fewer right": apart[CLEAN]["recordings"]["accuracy"]["value"] * recordings >= -1.0 - 1e-9,
        "not below under the five conditions": not any(apart[name][f1[0]][f1[1]]["interval"] is not None and apart[name][f1[0]][f1[1]]["interval"][1] < 0
                                                       for name in family),
    }
    return {"word": CANDIDATE if all(holds.values()) else NOT_SUPPORTED, "holds": holds}


def held_to_part_two(mine: Mapping, part2: Mapping) -> None:
    """The published system minus the stage-2 ensemble as part 2 decoded it, scored again here, must be what part 2
    published for every condition."""
    theirs = part2["published_minus_stage2"]
    for name in CONDITIONS:
        for part in mine[name]:
            for key, entry in mine[name][part].items():
                other = theirs[name][part][key]
                pairs = [(entry["value"], other["value"])]
                if (entry["interval"] is None) != (other["interval"] is None):
                    raise SystemExit(f"the published system minus stage 2 under '{name}' ({part}, {key}) is not what part 2 published")
                if entry["interval"] is not None:
                    pairs += list(zip(entry["interval"], other["interval"]))
                if any(abs(a - b) > CLOSE for a, b in pairs):
                    raise SystemExit(f"the published system minus stage 2 under '{name}' ({part}, {key}) is not what part 2 published")


def about_the(folders: Mapping[str, Path]) -> dict:
    """What every work folder says it holds. A folder must hold the system it is given as, the nine models of the arms
    must be nine different files trained with one version of PyTorch, the three arms must name one commit of this
    package, and the other two arms must have been set against the models of the arm "again" that is given here."""
    about = {}
    for system, where in folders.items():
        about[system] = json.loads((where / "system.json").read_text()) if (where / "system.json").is_file() else {}
        if about[system].get("system") != system:
            raise SystemExit(f"{where} was written for '{about[system].get('system')}', and is given as '{system}'")
    hashes = [digest for arm in ARMS for digest in about[arm].get("weights_sha256", [])]
    if len(hashes) != len(ARMS) * len(SEEDS) or len(set(hashes)) != len(hashes):
        raise SystemExit("the three arms are nine models, each a file of its own")
    versions = {entry.get("torch") for arm in ARMS for entry in about[arm].get("models", [])}
    if len(versions) != 1 or None in versions:
        raise SystemExit(f"the nine models were not trained with one version of PyTorch: {sorted(str(v) for v in versions)}")
    code = {about[arm].get("code") for arm in ARMS}
    if len(code) != 1 or not next(iter(code)):
        raise SystemExit(f"the three arms do not name one commit of this package: {sorted(str(c) for c in code)}")
    for arm in ARMS:
        if arm != AGAIN and about[arm].get("set_against_sha256") != about[AGAIN]["weights_sha256"]:
            raise SystemExit(f"the arm '{arm}' was set against other models than those of the arm '{AGAIN}' given here")
    return about


def report(folders: Mapping[str, Path], stored: Mapping[str, np.ndarray], part2: Mapping, halves: Mapping[str, tuple[Sequence, ...]],
           draws: int = DRAWS, seed: int = SEED) -> dict:
    """Every system under every condition on the selection half, decoded with the published setting and scored in one
    go, so that every difference has the same draws. On the unchanged frames every system is also given with the
    setting that the rule gives it on the calibration half and with no decoder."""
    calibration, selection = halves["calibration"], halves["selection"]
    natives = of_kind(selection, "native")
    if tuple(folders) != SYSTEMS:
        raise SystemExit(f"the report takes the work folders of {', '.join(SYSTEMS)}, in this order")
    about = about_the(folders)
    held = {system: {name: stress._held(where, name, selection) for name in CONDITIONS} for system, where in folders.items()}
    members = tuple(range(next(iter(stored.values())).shape[1]))
    published = member_probabilities(dict(stored), selection, members)
    apart = max(float(np.abs(held[PUBLISHED][CLEAN][f"p__{s.id}"] - published[s.id]).max()) for s in selection)
    if apart > stress.CLOSE:
        raise SystemExit(f"the unchanged frames of the published system, run again, do not give the stored outputs: {apart:.5f} apart")
    # an arm's condition file of the unchanged frames must be what its own window logits give, on the selection half
    logits, own_apart = {}, {}
    for arm in ARMS:
        if not (folders[arm] / LOGITS_FILE).is_file():
            raise SystemExit(f"{folders[arm]} lacks {LOGITS_FILE}")
        logits[arm] = comparison._logits(folders[arm] / LOGITS_FILE, (*calibration, *selection))
        again = member_probabilities(dict(logits[arm]), selection, members)
        own_apart[arm] = max(float(np.abs(held[arm][CLEAN][f"p__{s.id}"] - again[s.id]).max()) for s in selection)
        if own_apart[arm] > stress.CLOSE:
            raise SystemExit(f"the two runs of the arm '{arm}' on the unchanged frames do not agree: {own_apart[arm]:.5f} apart")
    # the probabilities of the unchanged calibration half, where the rule chooses a setting for a system
    on_calibration = {PUBLISHED: member_probabilities(dict(stored), calibration, members),
                      STAGE2: stress._probabilities(stress._held(folders[STAGE2], None, calibration), calibration),
                      **{arm: member_probabilities(dict(logits[arm]), calibration, members) for arm in ARMS}}
    own = {system: comparison.choose(on_calibration[system], calibration, drop_floors=True) for system in SYSTEMS}
    rows = [Row(f"{system} | {name}", ROUTER, "D", comparison.decoded(stress._probabilities(held[system][name], selection), B3))
            for system in SYSTEMS for name in CONDITIONS]
    # part 2 decoded the stage-2 ensemble with the setting of the rule: those rows again, to hold this report to part 2
    rows += [Row(f"{AS_IN_PART_2} | {name}", ROUTER, "D", comparison.decoded(stress._probabilities(held[STAGE2][name], selection), own[STAGE2]["params"]))
             for name in CONDITIONS]
    for system in SYSTEMS:
        unchanged = stress._probabilities(held[system][CLEAN], selection)
        rows.append(Row(f"{system} | {OWN}", ROUTER, "D", comparison.decoded(unchanged, own[system]["params"])))
        rows.append(Row(f"{system} | {NONE}", ROUTER, "N", comparison.likeliest(unchanged)))
    scored = comparison.score_rows(rows, selection, draws, seed)
    clean = ablations.numbers(scored[f"{PUBLISHED} | {CLEAN}"])
    if round(clean["multi"]["segment_f1_50"]["value"], 4) != comparison.PUBLISHED_SELECTION_F1:
        raise SystemExit("the unchanged frames of the published system, run again, do not give the published number of the selection half")
    held_to_part_two({name: ablations.minus(scored[f"{PUBLISHED} | {name}"], scored[f"{AS_IN_PART_2} | {name}"], DESCRIBED) for name in CONDITIONS}, part2)
    result: dict = {"benchmark": "development", "half": "selection", "sequences": len(selection), "draws": draws, "seed": seed,
                    "decoder": dict(B3), "code": about[AGAIN]["code"], "unchanged_against_stored": apart, "unchanged_against_own_logits": own_apart,
                    "systems": {}}
    for system in SYSTEMS:
        conditions = {}
        for name in CONDITIONS:
            mine = scored[f"{system} | {name}"]
            conditions[name] = {"kind": "unchanged" if name == CLEAN else "time" if name in TIME else "image",
                                "selection": ablations.numbers(mine), "classify": stress._classified(held[system][name], natives)}
            if name != CLEAN:
                conditions[name]["minus_unchanged"] = ablations.minus(mine, scored[f"{system} | {CLEAN}"], DESCRIBED)
        setting = {"status": own[system]["status"], "floors_dropped": own[system]["floors_dropped"], **own[system]["params"]}
        result["systems"][system] = {
            "conditions": conditions, "system": about[system],
            "unchanged_with": {OWN: {"setting": setting, "selection": ablations.numbers(scored[f"{system} | {OWN}"])},
                               NONE: {"selection": ablations.numbers(scored[f"{system} | {NONE}"])}}}
    result["differences"] = {f"{first} minus {second}": {name: ablations.minus(scored[f"{first} | {name}"], scored[f"{second} | {name}"], DESCRIBED)
                                                         for name in CONDITIONS} for first, second in DIFFERENCES}
    # the first question: under the five conditions of the family, what share of "again minus stage 2" is "again minus without"
    family = {}
    for name in FAMILY:
        again, without, stage2, other = (scored[f"{system} | {name}"]["multi"] for system in (AGAIN, WITHOUT, STAGE2, PUBLISHED))
        found, second = share(again, without, stage2, "segment_f1_50", FAMILY_LEVEL), share(other, without, stage2, "segment_f1_50", FAMILY_LEVEL)
        agree = sentence(found) == sentence(second)
        family[name] = {"again_minus_without": comparison.difference(again, without, "segment_f1_50", FAMILY_LEVEL),
                        "share": found, "sentence": sentence(found),
                        # the same share with the published system in the place of the arm "again": a second training of the same recipe
                        "with_the_published_system": {"share": second, "sentence": sentence(second)},
                        "published_minus_again": comparison.difference(other, again, "segment_f1_50", FAMILY_LEVEL),
                        # where the two trainings give two sentences, neither stands
                        "the_two_trainings_agree": agree, "stands": sentence(found) if agree else None}
    result["input_changes"] = {"level": FAMILY_LEVEL, "conditions": family}
    # the second question: the arm with a mirror against the arm "again"
    with_mirror = result["differences"][f"{MIRROR} minus {AGAIN}"]
    own_loss = result["systems"][MIRROR]["conditions"][MIRRORED]["minus_unchanged"]
    result["mirror"] = {"under_the_mirrored_image": with_mirror[MIRRORED], "on_the_unchanged_frames": with_mirror[CLEAN], "its_own_loss": own_loss,
                        "margins": {"its_own_loss": OWN_LOSS, "on_the_unchanged_frames": UNCHANGED},
                        **mirror_word(with_mirror, own_loss, len(natives), FAMILY)}
    result["seeds"] = seeds_beside(logits, stored, selection, draws, seed)
    return result


def seeds_beside(logits: Mapping[str, Mapping[str, np.ndarray]], stored: Mapping[str, np.ndarray], selection: tuple[Sequence, ...],
                 draws: int, seed: int) -> dict:
    """Every seed alone on the unchanged frames of the selection half, for the published system and the three arms,
    decoded with the published setting. A seed of the published system or of another arm minus the seed of the same
    number of the arm "again" is a pair that shares its start weights and its sampling plan."""
    given = {PUBLISHED: stored, **logits}
    count = next(iter(stored.values())).shape[1]
    rows = [Row(f"{system} | seed {k}", ROUTER, "S", comparison.decoded(member_probabilities(dict(given[system]), selection, (k,)), B3))
            for system in given for k in range(count)]
    scored = comparison.score_rows(rows, selection, draws, seed)
    out: dict = {"systems": {row.name: ablations.numbers(scored[row.name]) for row in rows}, "minus_again": {}}
    for system in (PUBLISHED, WITHOUT, MIRROR):
        out["minus_again"][system] = {f"seed {k}": ablations.minus(scored[f"{system} | seed {k}"], scored[f"{AGAIN} | seed {k}"], DESCRIBED)
                                      for k in range(count)}
    return out


# ---------------------------------------------------------------------------------------------------- commands
def run_command(args: argparse.Namespace) -> None:
    import importlib.util

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # plain float32, as in the published numbers
    torch.backends.cudnn.allow_tf32 = False
    if importlib.util.find_spec("scipy") is None:  # the blur needs it; say so before the long part
        raise SystemExit('the conditions need SciPy: pip install "echo-view-routing[evaluation]" (or pip install scipy)')
    if args.arm != AGAIN and args.again is None:
        raise SystemExit(f"the arm '{args.arm}' is set against the arm '{AGAIN}': give its training runs with --again")
    models = trained(args.runs, args.arm, args.recipe)
    about = {"system": args.arm, "code": args.code, "models": models, "weights_sha256": [entry["sha256"] for entry in models]}
    if args.arm != AGAIN:  # the runs it is set against must be those of the arm "again"
        about["set_against_sha256"] = [entry["sha256"] for entry in trained(args.again, AGAIN, args.recipe)]
    args.folder.mkdir(parents=True, exist_ok=True)
    if (args.folder / "system.json").is_file():
        before = json.loads((args.folder / "system.json").read_text())
        if before.get("system") != args.arm:
            raise SystemExit(f"{args.folder} holds the outputs of another system; give every system a folder of its own")
        if before.get("weights_sha256") != about["weights_sha256"]:
            raise SystemExit(f"{args.folder} holds the outputs of other weight files; give every set of models a folder of its own")
        if before.get("code") != args.code:
            raise SystemExit(f"{args.folder} holds the outputs of another commit of this package ({before.get('code')}); one commit runs all of part 4")
    try:
        about["distances"] = weight_distances(args.runs, args.weights, None if args.arm == AGAIN else args.again)
    except published_weights.WeightsError as error:
        raise SystemExit(str(error)) from None
    halves = ablations.development_halves()
    selection, calibration = halves["selection"], halves["calibration"]
    system = arm_system(args.runs, args.device, args.batch)
    (args.folder / "system.json").write_text(json.dumps({**about, "device": str(system.device), "torch": torch.__version__}) + "\n")
    table = ev9v.FrameTable.build((*selection, *calibration), args.images, workers=args.workers, cache=args.cache)
    if not (args.folder / LOGITS_FILE).is_file():
        both = {}
        for sequences in (calibration, selection):
            both.update(evaluation_run.all_logits(system, sequences, table))
        stress._save(both, args.folder / LOGITS_FILE)
        print("done: the window logits of the unchanged frames of both halves", flush=True)
    for condition in CONDITIONS:
        path = args.folder / f"{stress.slug(condition)}.npz"
        if path.is_file():
            continue
        stress._save(stress.run_condition(system, selection, table, condition), path)
        print(f"done: {condition}", flush=True)


def report_command(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; this command does not replace an output")
    if not args.part2.is_file():
        raise SystemExit(f"{args.part2}: the output of part 2 is not there")
    halves = ablations.development_halves()
    stored = comparison._logits(args.logits, tuple(s for half in benchmark.HALVES for s in halves[half]))
    ablations.held_to_published(stored, halves["selection"])
    part2 = json.loads(args.part2.read_text())
    digest = comparison.file_sha256(args.logits)
    if part2.get("sha256", {}).get("logits") != digest:
        raise SystemExit(f"{args.part2} was made from other stored outputs than {args.logits}")
    folders = {PUBLISHED: args.published, STAGE2: args.stage2, AGAIN: args.again, WITHOUT: args.without, MIRROR: args.mirror}
    result = {"sha256": {"logits": digest, "part2": comparison.file_sha256(args.part2)}, **report(folders, stored, part2, halves, args.draws, SEED)}
    comparison._write(result, args.out)
    print(f"wrote {args.out}")


def main(argv: list[str] | None = None) -> int:
    top = argparse.ArgumentParser(prog="python -m echoview.evaluation.arms", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="the three models of one arm on the development benchmark under every condition")
    p.add_argument("--arm", choices=tuple(ARMS), required=True)
    p.add_argument("--runs", type=Path, required=True, help="the folder with the training runs s0, s1 and s2 of the arm")
    p.add_argument("--again", type=Path, default=None, help="the same folder of the arm 'again', for the other two arms")
    p.add_argument("--recipe", type=Path, default=RECIPE, help="the recipe of stage 3, to which the training runs are held")
    p.add_argument("--code", required=True, help="the commit of this package that trained the models and runs them; the report takes one for all arms")
    p.add_argument("--images", type=Path, required=True, help="the Images folder of EV9V")
    p.add_argument("--folder", type=Path, required=True, help="the work folder; a condition whose file exists is not computed again")
    p.add_argument("--weights", type=Path, default=None, help="the folder of the published weights, for the stage-2 start files; downloaded when omitted")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--workers", type=int, default=8, help="threads that decode frames")
    p.add_argument("--cache", type=Path, default=None, help="a folder that keeps the decoded frames for the next run")
    p.set_defaults(run=run_command)
    p = commands.add_parser("report", help="every system under every condition, and the two questions of part 4; no model runs")
    p.add_argument("--logits", type=Path, required=True, help="the window logits of the published system for the development benchmark")
    p.add_argument("--part2", type=Path, required=True, help="the output of part 2 (docs/ablations/forward.json), to which this report is held")
    p.add_argument("--published", type=Path, required=True, help="the work folder of `stress run` for the published system")
    p.add_argument("--stage2", type=Path, required=True, help="the work folder of `stress run --system stage2`")
    p.add_argument("--again", type=Path, required=True, help="the work folder of this module's run command for the arm 'again'")
    p.add_argument("--without", type=Path, required=True, help="that of the arm 'without input changes'")
    p.add_argument("--mirror", type=Path, required=True, help="that of the arm 'with a mirror'")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--out", type=Path, required=True, help="JSON file to write; an existing file is never replaced")
    p.set_defaults(run=report_command)
    args = top.parse_args(argv)
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
