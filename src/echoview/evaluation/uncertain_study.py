"""The study behind the uncertain output (docs/UNCERTAIN.md, section 4): the rule is chosen on the calibration half of
the EV9V development benchmark and then scored on the selection half, by two separate commands.

  python -m echoview.evaluation.uncertain_study choose --logits dev_logits.npz --out choice.json
  python -m echoview.evaluation.uncertain_study report --logits dev_logits.npz --choice choice.json --out report.json

`dev_logits.npz` holds the window logits of every development sequence (`echoview evaluate --save-logits`). `choose`
reads the calibration half only, `report` the selection half only. Both run on the CPU and write pooled numbers: no
sequence and no recording appears in their output by itself. Neither replaces a file that exists, unless `--force`
is given: the choice of the first command is what the second one has to follow.

  python -m echoview.evaluation.uncertain_study confirm --logits test_logits.npz --out test_report.json

`confirm` reads the test benchmark. `test_logits.npz` holds the window logits of exactly its sequences, in the format
of `echoview evaluate --benchmark test --save-logits`, and must give the benchmark's published numbers. The command
scores the published output, the band alone and the released rule there, chooses nothing and never replaces its
output file (docs/UNCERTAIN.md, section 9).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .. import decoder, uncertain
from ..pipeline import probabilities_from_logits
from ..reference import B3
from . import benchmark, metrics, scorecard, selective

BANDS = tuple(range(0, 9))
FACTORS = (1.0, 1.25, 1.5, 2.0, 3.0)
MARGINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
SEEDS = (False, True)
WITHIN = 0.90  # step 1: the share of matched changes the band must hold
MIN_WRONG, MIN_RECORDINGS = 4, 3  # step 2: the bar for the candidate
PUBLISHED_F1 = {"selection": 0.9594, "calibration": 0.9837}  # of the published system: the logits must give it
# The published numbers of the test benchmark (docs/EVALUATION.md). Each is a ratio of counts, so logits of the
# published system give them exactly, and logits that differ in one decision miss one of them by far more than this.
PUBLISHED_TEST = {"segment_f1_50_multi": 0.908908908908909, "segment_f1_50_pairs": 0.9847009735744089,
                  "native_accuracy": 0.98125, "native_macro_f1": 0.9561411223630938}
CLOSE = 1e-6
TESTS = {"weak cut, lower threshold": {"f_low": 1.0}, "weak cut, higher threshold": {"f_high": 1.0}, "margin": {"margin": 0.0},
         "seeds": {"seeds": False}}


class Decoded:
    """Everything the tests need about one sequence, computed once."""

    def __init__(self, sequence: benchmark.Sequence, prob: np.ndarray, seed_prob: np.ndarray, b3: dict | None = None) -> None:
        b3 = dict(B3 if b3 is None else b3)
        self.sequence, self.prob, self.confidence = sequence, prob, prob.max(1)
        scores = decoder.boundary_scores(prob, int(b3["half_window"]))
        self.decided = decoder.decode(prob, b3["half_window"], b3["threshold"], b3["nms_radius"], scores)
        self.low = {f: uncertain.weak_cut_mask(prob, self.decided, b3, f, 1.0, scores) for f in FACTORS}
        self.high = {f: uncertain.weak_cut_mask(prob, self.decided, b3, 1.0, f, scores) for f in FACTORS}
        self.margins = {m: uncertain.margin_mask(prob, self.decided, m) for m in MARGINS}
        self.seeds = uncertain.seed_mask(seed_prob, self.decided)
        self.bands = {k: uncertain.band_mask(self.decided, k) for k in BANDS}
        runs = scorecard.runs(sequence.pieces)
        self.middles = np.array([scorecard.midpoint(a, b) for a, b, _ in runs], int)
        self.piece_wrong = self.decided[self.middles] != sequence.labels[self.middles]
        self.piece_recording = [sequence.refs[a][0] for a, _, _ in runs]

    @classmethod
    def from_logits(cls, sequence: benchmark.Sequence, logits: np.ndarray) -> Decoded:
        n = len(sequence)
        seeds = np.stack([probabilities_from_logits(logits[:, [k]], n) for k in range(logits.shape[1])])
        return cls(sequence, probabilities_from_logits(logits, n), seeds)

    def mask(self, rule: uncertain.Rule) -> np.ndarray:
        """The mask uncertain.mark gives, from the parts computed once."""
        on_grid(rule)
        mask = self.bands[rule.band] | self.low[rule.f_low] | self.high[rule.f_high] | self.margins[rule.margin]
        return uncertain.tidy(mask | self.seeds if rule.seeds else mask)

    def band(self, rule: uncertain.Rule) -> np.ndarray:
        """The mask of the band alone, tidied."""
        on_grid(rule)
        return uncertain.tidy(self.bands[rule.band])


def on_grid(rule: uncertain.Rule) -> None:
    """The study computes the parts of the mask for its own settings only."""
    if rule.band not in BANDS or rule.f_low not in FACTORS or rule.f_high not in FACTORS or rule.margin not in MARGINS:
        raise ValueError(f"{rule} is not one of the study's settings: band {BANDS[0]} to {BANDS[-1]}, factors {FACTORS}, margins {MARGINS}")


def row(rule: uncertain.Rule, wrong: int, wrong_recordings: int, right: int, samples: int) -> dict:
    """One setting of step 2. `points` is the score times five, a whole number: settings are ordered by it, so that
    two settings with the same score are a tie whatever floating point makes of `right / 5`."""
    return {"rule": asdict(rule), "wrong_declined": wrong, "wrong_declined_recordings": wrong_recordings, "right_declined": right,
            "points": selective.WORTH * wrong - right, "score": wrong - right / selective.WORTH, "uncertain_samples": samples}


def band_for(half: list[Decoded]) -> dict:
    """Step 1: the smallest band that holds 90% of the matched view changes of the multi and examination-order streams."""
    matched = [scorecard.cut_offsets(d.sequence.labels, d.decided) for d in half if d.sequence.kind in ("multi", "exam")]
    offsets = [abs(o) for found, _ in matched for o in found]
    within = {k: float(np.mean([o <= k for o in offsets])) if offsets else 0.0 for k in BANDS}
    return {"view_changes": sum(total for _, total in matched), "matched_changes": len(offsets), "share_within": within,
            "band": next((k for k in BANDS if within[k] >= WITHIN), None)}


def score(half: list[Decoded], rule: uncertain.Rule) -> dict:
    """Step 2 for one setting: the pieces beyond the band that it declines, wrong ones and right ones, and the score."""
    wrong_recordings, wrong, right, samples = set(), 0, 0, 0
    for d in half:
        mask = d.mask(rule)
        samples += int(mask.sum())
        counted = mask[d.middles] & ~d.band(rule)[d.middles]  # declined, and not by the band alone
        wrong += int((counted & d.piece_wrong).sum())
        right += int((counted & ~d.piece_wrong).sum())
        wrong_recordings.update(name for name, flag in zip(d.piece_recording, counted & d.piece_wrong) if flag)
    return row(rule, wrong, len(wrong_recordings), right, samples)


def choose(half: list[Decoded]) -> dict:
    """Steps 1 and 2 on the calibration half."""
    step1 = band_for(half)
    if step1["band"] is None:
        return {"step1": step1, "stopped": f"no band up to {BANDS[-1]} samples holds {WITHIN:.0%} of the matched view changes"}
    grid = [uncertain.Rule(step1["band"], f_low, f_high, margin, seeds)
            for f_low, f_high, margin, seeds in itertools.product(FACTORS, FACTORS, MARGINS, SEEDS)]
    rows = [score(half, rule) for rule in grid]
    best = max(range(len(rows)), key=lambda k: (rows[k]["points"], -rows[k]["uncertain_samples"], -k))
    candidate = rows[best]
    reasons = []
    if candidate["points"] <= 0:
        reasons.append("no setting has a score above 0")
    if candidate["wrong_declined"] < MIN_WRONG or candidate["wrong_declined_recordings"] < MIN_RECORDINGS:
        reasons.append(f"the best setting declines {candidate['wrong_declined']} wrong pieces from {candidate['wrong_declined_recordings']} "
                       f"recordings; the bar is {MIN_WRONG} from {MIN_RECORDINGS}")
    return {"step1": step1, "band": step1["band"], "settings": len(rows), "candidate": candidate, "goes_on": not reasons,
            "why_not": reasons, "all_settings": rows}


def lost_without(half: list[Decoded], rule: uncertain.Rule) -> dict:
    """For each test of the rule that is on: the samples the rule would lose if it were switched off, and how many of
    them the published system has wrong."""
    out = {}
    for name, off in TESTS.items():
        without = uncertain.Rule(**(asdict(rule) | off))
        if without == rule:
            continue
        lost = wrong = 0
        for d in half:
            only = d.mask(rule) & ~d.mask(without)
            lost += int(only.sum())
            wrong += int((only & (d.decided != d.sequence.labels)).sum())
        out[name] = {"samples": lost, "wrong": wrong, "share_wrong": wrong / lost if lost else None}
    return out


def evaluate(half: list[Decoded], rule: uncertain.Rule, draws: int) -> dict:
    """The two-part report of one rule on one half, with the pieces counted by recording."""
    sequences = tuple(d.sequence for d in half)
    decided = {d.sequence.id: d.decided for d in half}
    masks, bands = {d.sequence.id: d.mask(rule) for d in half}, {d.sequence.id: d.band(rule) for d in half}
    rows = [row for d in half for row in selective.pieces(d.sequence, d.decided, masks[d.sequence.id], bands[d.sequence.id])]
    by_kind = {kind: selective.piece_summary([r for r in rows if r["kind"] == kind], draws) for kind in benchmark.KINDS
               if any(r["kind"] == kind for r in rows)}
    return {"rule": asdict(rule), "samples": selective.report(sequences, decided, masks, bands, {d.sequence.id: d.confidence for d in half}, draws),
            "pieces": {"all": selective.piece_summary(rows, draws), **by_kind}, "each_test": lost_without(half, rule)}


def band_curve(half: list[Decoded]) -> list[dict]:
    """Coverage and error among the labelled samples for the band alone at every width, by kind and for all sequences."""
    out = []
    for k in BANDS:
        rule = uncertain.Rule(band=k)
        totals: dict[str, dict[str, float]] = {}
        for d in half:
            mask = d.mask(rule)
            wrong = d.decided != d.sequence.labels
            for kind in (d.sequence.kind, "all"):
                t = totals.setdefault(kind, {"n": 0, "labelled": 0, "wrong_labelled": 0})
                t["n"] += mask.size
                t["labelled"] += int((~mask).sum())
                t["wrong_labelled"] += int((wrong & ~mask).sum())
        out.append({"band": k, **{kind: {"coverage": t["labelled"] / t["n"], "error_labelled": t["wrong_labelled"] / max(1, t["labelled"])}
                                  for kind, t in totals.items()}})
    return out


def final(half: list[Decoded], choice: dict, draws: int) -> dict:
    """Step 3 on the selection half: the published output, the band alone, the band at every width, and the candidate
    of step 2 if it went on, with the decision about it."""
    band = uncertain.Rule(band=int(choice["band"]))
    out = {"published": evaluate(half, uncertain.OFF, draws), "band_alone": evaluate(half, band, draws), "bands": band_curve(half)}
    if not choice["goes_on"]:
        out["decision"] = {"released_rule": asdict(band), "tests_for_whole_pieces": "not adopted on the calibration half", "why": choice["why_not"]}
        return out
    rule = uncertain.Rule(**choice["candidate"]["rule"])
    out["candidate"] = evaluate(half, rule, draws)
    pieces = out["candidate"]["pieces"]["all"]
    low = pieces["score_p5_p95"][0]
    passed = pieces["score"] > 0 and low is not None and low > 0
    out["decision"] = {"released_rule": asdict(rule if passed else band), "score": pieces["score"], "score_p5": low, "passed": passed,
                       "tests_for_whole_pieces": "adopted" if passed else "not shown on the selection half"}
    return out


def confirm(sequences: list[Decoded], draws: int) -> dict:
    """The released rule on the test benchmark: the published output, the band alone, the band at every width and the
    rule, with the test of step 3 as the verdict about the tests for whole pieces. Nothing is chosen here."""
    rule = uncertain.released()
    on_grid(rule)
    out = {"published": evaluate(sequences, uncertain.OFF, draws), "band_alone": evaluate(sequences, uncertain.Rule(band=rule.band), draws),
           "bands": band_curve(sequences), "released": evaluate(sequences, rule, draws)}
    pieces = out["released"]["pieces"]["all"]
    low = pieces["score_p5_p95"][0]
    out["confirmation"] = {"rule": asdict(rule), "score": pieces["score"], "score_p5": low,
                           "confirmed": bool(pieces["score"] > 0 and low is not None and low > 0), "draws": draws}
    return out


def decode(sequences: tuple[benchmark.Sequence, ...], path: Path, where: str = "the benchmark", exact: bool = False) -> list[Decoded]:
    """The sequences decoded from a file of window logits. With `exact` the file may hold no other array."""
    try:
        with np.load(path) as stored:
            missing = [s.id for s in sequences if s.id not in stored.files]
            if missing:
                raise SystemExit(f"{path} lacks the logits of {len(missing)} of the {len(sequences)} sequences of {where}")
            if exact and len(stored.files) != len(sequences):
                raise SystemExit(f"{path} holds {len(stored.files)} arrays; {where} has {len(sequences)} sequences")
            return [Decoded.from_logits(s, stored[s.id]) for s in sequences]
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"cannot read the logits file {path}: {error}") from error


def published_numbers(decoded: list[Decoded]) -> dict:
    """The numbers of the published output that tell whether logits are the published system's."""
    scored = tuple(d.sequence for d in decoded if d.sequence.kind != "exam")
    numbers = metrics.evaluate({d.sequence.id: d.decided for d in decoded}, scored)
    return {name: numbers[name] for name in PUBLISHED_TEST}


def load(path: Path, half: str) -> list[Decoded]:
    """The decoded sequences of a development half, or of the test benchmark for `half` "test". Logits that do not
    give the published numbers stop the command before anything is scored."""
    if half == "test":
        decoded = decode(benchmark.load_test(), path, "the test benchmark", exact=True)
        numbers = published_numbers(decoded)
        off = {name: numbers[name] for name, value in PUBLISHED_TEST.items() if not abs(numbers[name] - value) < CLOSE}
        if off:
            raise SystemExit(f"the test benchmark: these logits do not give the published numbers. They give {off}; published: {PUBLISHED_TEST}")
        return decoded
    decoded = decode(benchmark.load()[half], path, f"the {half} half")
    f1 = published_numbers(decoded)["segment_f1_50_multi"]
    if round(f1, 4) != PUBLISHED_F1[half]:
        raise SystemExit(f"{half}: these logits give a segment F1@50 of {f1:.4f}; the published system has {PUBLISHED_F1[half]}")
    return decoded


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("at least 1 draw is needed")
    return value


def read_choice(path: Path) -> dict:
    """The file `choose` wrote, checked as far as `report` relies on it."""
    try:
        choice = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read the choice file {path}: {error}") from error
    if not isinstance(choice, dict) or "step1" not in choice:
        raise SystemExit(f"{path} is not a file written by `choose`")
    if "band" not in choice:
        raise SystemExit("the choice file holds no band: step 1 stopped the study")
    try:
        on_grid(uncertain.Rule(band=choice["band"]))
        if choice["goes_on"]:
            on_grid(uncertain.Rule(**choice["candidate"]["rule"]))
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{path} does not hold a choice of this study: {error}") from error
    return choice


def write(out: Path, content: dict) -> None:
    """Writes the whole file or nothing."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.write_text(json.dumps(content, indent=1) + "\n")
    partial.replace(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("choose", "report", "confirm"):
        p = commands.add_parser(name)
        p.add_argument("--logits", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--draws", type=positive, default=2000)
        if name != "confirm":  # the one read of the test benchmark is not made twice
            p.add_argument("--force", action="store_true", help="replace the output file if it exists")
    commands.choices["report"].add_argument("--choice", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = [args.logits] + ([args.choice] if args.command == "report" else [])
    if any(args.out.resolve() == path.resolve() for path in inputs):
        raise SystemExit("--out names one of the input files")
    if args.out.exists() and not getattr(args, "force", False):
        raise SystemExit(f"{args.out} exists; it is not replaced" + (" without --force" if hasattr(args, "force") else ""))
    if args.command == "choose":
        half = load(args.logits, "calibration")
        out = choose(half)
        if "band" in out:  # the calibration half's own report, marked as the half the choice was made on
            out["calibration_report_chosen_here"] = {"band_alone": evaluate(half, uncertain.Rule(band=out["band"]), args.draws),
                                                     "bands": band_curve(half)}
            if out["goes_on"]:
                out["calibration_report_chosen_here"]["candidate"] = evaluate(half, uncertain.Rule(**out["candidate"]["rule"]), args.draws)
        headline = {key: out.get(key) for key in ("band", "candidate", "goes_on", "why_not", "stopped")}
    elif args.command == "report":
        choice = read_choice(args.choice)
        out = final(load(args.logits, "selection"), choice, args.draws)
        headline = out["decision"]
    else:
        out = confirm(load(args.logits, "test"), args.draws)
        out["confirmation"]["logits_sha256"] = hashlib.sha256(args.logits.read_bytes()).hexdigest()
        headline = out["confirmation"]
    write(args.out, out)
    print(json.dumps(headline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
