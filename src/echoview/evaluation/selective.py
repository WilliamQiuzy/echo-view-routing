"""The two-part report of an output with uncertain samples (docs/UNCERTAIN.md, section 5).

Part A is about the samples that keep a view: how many they are and how often they are right. Part B is about the
uncertain samples: where they fall, what a forced answer would have been there, and which of the published system's
errors they take away. References stand beside the numbers: the published system (nothing declined), chance (the same
number of samples declined at random, which leaves the error where it was), the ideal (the same number declined,
wrong ones first) and low confidence (the same number declined, least confident first).

Numbers about samples are ratios of counts that add up over sequences: `counts` gives the counts of one sequence,
`ratios` turns summed counts into the numbers of the report, and their intervals come from a bootstrap over
sequences. Numbers about pieces are counted per recording, because one recording is reused in many sequences: `pieces`
lists the pieces of a sequence with their recording, and `piece_summary` draws whole recordings again. A number whose
numerator or denominator is SMALL or less has no interval: its counts are what there is to say.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from . import bootstrap, scorecard
from .benchmark import KINDS, Sequence

NEAR = 5  # samples: "at a view change" and "at a join" mean within 0.5 s
VIEWS = 5
A4C, A5C = 2, 3
ERROR_KINDS = scorecard.CATEGORIES  # edge, spurious, absorbed, misread
WORTH = 5  # one wrong piece declined is worth this many right pieces declined (docs/UNCERTAIN.md, step 2)
SMALL = 5  # a number resting on a count this small is given by its counts, without an interval
PIECE_NUMBERS = ("piece_accuracy_published", "piece_accuracy_labelled")  # about pieces: their intervals are piece_summary's, over recordings


def near(positions: np.ndarray, n: int) -> np.ndarray:
    """True for the samples within NEAR samples of a position: the NEAR before it and the NEAR from it on."""
    flag = np.zeros(n, bool)
    for position in np.asarray(positions, int):
        flag[max(0, position - NEAR):position + NEAR] = True
    return flag


def _checked(sequence: Sequence, decided: np.ndarray, mask: np.ndarray, band: np.ndarray | None) -> tuple:
    decided, mask = np.asarray(decided, int), np.asarray(mask, bool)
    band = np.zeros(mask.shape, bool) if band is None else np.asarray(band, bool)
    n = sequence.labels.size
    if decided.shape != (n,) or mask.shape != (n,) or band.shape != (n,):
        raise ValueError(f"sequence {sequence.id} has {n} samples; the labels and the masks must have as many")
    if decided.size and (decided.min() < 0 or decided.max() >= VIEWS):
        raise ValueError("`decided` must hold the decoder's view of every sample; uncertain samples are given by the mask")
    return decided, mask, band


def pieces(sequence: Sequence, decided: np.ndarray, mask: np.ndarray, band: np.ndarray | None = None) -> list[dict]:
    """One row per reference piece: its recording, whether the published answer at its middle sample is wrong, whether
    that sample is declined, whether the band alone leaves it labelled (`beyond_band`), and whether no sample of the
    piece is left labelled (`empty`)."""
    decided, mask, band = _checked(sequence, decided, mask, band)
    truth, rows = sequence.labels, []
    for start, end, _ in scorecard.runs(sequence.pieces):
        middle = scorecard.midpoint(start, end)
        rows.append({"recording": sequence.refs[start][0], "kind": sequence.kind, "wrong": bool(decided[middle] != truth[middle]),
                     "declined": bool(mask[middle]), "beyond_band": bool(not band[middle]), "empty": bool(mask[start:end].all())})
    return rows


def counts(sequence: Sequence, decided: np.ndarray, mask: np.ndarray, band: np.ndarray | None = None) -> dict[str, float]:
    """The counts of one sequence. `decided` is the published decoder's view of every sample, `mask` is True where
    the output is uncertain, `band` is True where the band alone would make it uncertain."""
    decided, mask, band = _checked(sequence, decided, mask, band)
    truth = sequence.labels
    n = truth.size
    wrong, kept = decided != truth, ~mask
    c: dict[str, float] = {"n": n, "labelled": int(kept.sum()), "uncertain": int(mask.sum()), "wrong": int(wrong.sum()),
                           "wrong_labelled": int((wrong & kept).sum()), "wrong_uncertain": int((wrong & mask).sum()),
                           "right_uncertain": int((~wrong & mask).sum())}
    for view in range(VIEWS):
        c[f"reference_{view}"] = int((truth == view).sum())
        for answer in range(VIEWS):
            c[f"kept_{view}_{answer}"] = int((kept & (truth == view) & (decided == answer)).sum())
    # the middle sample of every reference piece: right or wrong by the published answer, kept or declined
    rows = pieces(sequence, decided, mask, band)
    c.update(pieces=len(rows), pieces_unlabelled=sum(row["empty"] for row in rows))
    for answer in ("right", "wrong"):
        for fate in ("kept", "declined"):
            c[f"piece_{answer}_{fate}"] = sum((row["wrong"] == (answer == "wrong")) and (row["declined"] == (fate == "declined")) for row in rows)
    # the labelled runs the system returns, by their middle sample
    returned = [(a, b, view) for a, b, view in scorecard.runs(np.where(mask, -1, decided)) if view >= 0]
    c["segments_kept"] = len(returned)
    c["segments_kept_right"] = sum(int(truth[scorecard.midpoint(a, b)] == view) for a, b, view in returned)
    # where the uncertain samples fall
    changes = sequence.view_changes
    at_change = near(changes, n)
    intervals = [(a, b) for a, b, flag in scorecard.runs(mask.astype(int)) if flag]
    c.update(intervals=len(intervals), intervals_no_change=sum(int(not at_change[a:b].any()) for a, b in intervals),
             uncertain_near_change=int((mask & at_change).sum()), changes=int(changes.size),
             changes_located=sum(int(mask[position - 1] or mask[position]) for position in changes))
    same = [int(j) for j in sequence.joins if j not in set(changes.tolist())]
    c.update(same_joins=len(same), same_joins_uncertain=sum(int(mask[max(0, j - NEAR):j + NEAR].any()) for j in same))
    # single recordings, by whether the published system reads the recording right: the majority of its samples, as the scorecard has it
    native = sequence.kind == "native" and n > 0
    read_right = native and int(np.bincount(decided, minlength=VIEWS).argmax()) == int(truth[0])
    for answer in ("right", "wrong"):
        mine = native and read_right == (answer == "right")
        c[f"recordings_{answer}"] = int(mine)
        c[f"recordings_{answer}_uncertain"] = int(mine and mask.any())
    # which of the published system's wrong samples are declined, by kind of error
    piece_ok, segment_ok, *_ = scorecard._midpoint_tables(truth, sequence.pieces, decided, VIEWS)
    kinds = {"edge": wrong & piece_ok & segment_ok, "spurious": wrong & piece_ok & ~segment_ok,
             "absorbed": wrong & ~piece_ok & segment_ok, "misread": wrong & ~piece_ok & ~segment_ok,
             "a4c_a5c": wrong & np.isin(truth, (A4C, A5C)) & np.isin(decided, (A4C, A5C))}
    for name, flag in kinds.items():
        c[name], c[f"{name}_declined"] = int(flag.sum()), int((flag & mask).sum())
    # what the other tests add to the band
    added = mask & ~band
    c.update(added=int(added.sum()), added_wrong=int((added & wrong).sum()),
             after_band_labelled=int((~band).sum()), after_band_wrong=int((wrong & ~band).sum()))
    return c


def _share(a, b):
    """a / b where b is not 0, else nan; works for numbers and for arrays of bootstrap draws."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1.0), np.nan)[()]


def _parts(c: Mapping[str, float]) -> dict:
    """(numerator, denominator) of every number of the report, from summed counts; a list of pairs for a number per view."""
    n, wrong = np.asarray(c["n"], float), np.asarray(c["wrong"], float)
    kept_by_answer = [sum(c[f"kept_{view}_{answer}"] for view in range(VIEWS)) for answer in range(VIEWS)]
    out = {
        # part A: the labelled part
        "coverage": (c["labelled"], n),
        "error_published": (wrong, n),  # also the error under chance
        "error_labelled": (c["wrong_labelled"], c["labelled"]),
        "error_ideal": (np.maximum(0.0, wrong - c["uncertain"]), c["labelled"]),
        "purity": [(c[f"kept_{view}_{view}"], kept_by_answer[view]) for view in range(VIEWS)],
        "recovery": [(c[f"kept_{view}_{view}"], c[f"reference_{view}"]) for view in range(VIEWS)],
        "piece_accuracy_published": (c["piece_right_kept"] + c["piece_right_declined"], c["pieces"]),
        "piece_accuracy_labelled": (c["piece_right_kept"], c["piece_right_kept"] + c["piece_wrong_kept"]),
        "segment_accuracy": (c["segments_kept_right"], c["segments_kept"]),
        # part B: the uncertain part
        "uncertain_share": (c["uncertain"], n),  # also the share of the errors chance would decline
        "error_if_forced": (c["wrong_uncertain"], c["uncertain"]),
        "errors_declined": (c["wrong_uncertain"], wrong),
        "price": (c["right_uncertain"], n - wrong),
        "uncertain_near_change": (c["uncertain_near_change"], c["uncertain"]),
        "intervals_no_change": (c["intervals_no_change"], c["intervals"]),
        "changes_located": (c["changes_located"], c["changes"]),
        "same_joins_uncertain": (c["same_joins_uncertain"], c["same_joins"]),
        "right_recordings_uncertain": (c["recordings_right_uncertain"], c["recordings_right"]),
        "wrong_recordings_uncertain": (c["recordings_wrong_uncertain"], c["recordings_wrong"]),
        "error_added": (c["added_wrong"], c["added"]),
        "error_after_band": (c["after_band_wrong"], c["after_band_labelled"]),
    }
    for name in (*ERROR_KINDS, "a4c_a5c"):
        out[f"{name}_declined"] = (c[f"{name}_declined"], c[name])
    return out


def ratios(c: Mapping[str, float]) -> dict:
    """The numbers of the report from summed counts."""
    return {name: [_share(*pair) for pair in parts] if isinstance(parts, list) else _share(*parts) for name, parts in _parts(c).items()}


def _small(pair: tuple) -> bool:
    """True when the number rests on a count of SMALL or less."""
    return bool(min(float(pair[0]), float(pair[1])) <= SMALL)


def _least_confident_kept(confidence: np.ndarray, declined: int) -> np.ndarray:
    """The samples that are left when the `declined` least confident ones are declined; ties go by position."""
    return np.argsort(np.asarray(confidence, float), kind="stable")[max(0, int(declined)):]


def error_at_equal_coverage(wrong: np.ndarray, confidence: np.ndarray, declined: int) -> float:
    """The error among the samples that are left when the `declined` least confident samples are declined."""
    kept = _least_confident_kept(confidence, declined)
    return float(np.asarray(wrong, bool)[kept].mean()) if kept.size else float("nan")


def error_at_equal_coverage_drawn(wrong: np.ndarray, confidence: np.ndarray, owner: np.ndarray, weights: np.ndarray, declined: np.ndarray) -> np.ndarray:
    """`error_at_equal_coverage` in every bootstrap draw. Sample i belongs to sequence `owner[i]` and counts
    `weights[d, owner[i]]` times in draw d, of which the `declined[d]` least confident copies are declined."""
    order = np.argsort(np.asarray(confidence, float), kind="stable")
    wrong, owner = np.asarray(wrong, float)[order], np.asarray(owner, int)[order]
    out = np.full(len(weights), np.nan)
    for d, (row, gone) in enumerate(zip(np.asarray(weights, float), np.asarray(declined, float))):
        copies = row[owner]
        kept = np.clip(np.cumsum(copies) - gone, 0.0, copies)  # the copies of each sample that lie beyond the first `gone`
        total = kept.sum()
        if total > 0:
            out[d] = float(kept @ wrong) / total
    return out


def _plain(value):
    """Numbers and lists of numbers as JSON can hold them: nan becomes None."""
    if isinstance(value, list):
        return [_plain(v) for v in value]
    value = float(value)
    return None if np.isnan(value) else value


def _interval(drawn: np.ndarray, percentiles: tuple = (2.5, 97.5)) -> list:
    """Percentiles of the draws in which the number exists; [None, None] when it exists in none."""
    drawn = np.asarray(drawn, float)
    drawn = drawn[~np.isnan(drawn)]
    return [None, None] if drawn.size == 0 else np.percentile(drawn, percentiles).tolist()


def _pooled(rows: list[dict[str, float]], lengths: list[int], weights: np.ndarray) -> tuple[dict, dict]:
    """The pooled counts, numbers and intervals of some sequences, and the numbers in every draw of `weights`."""
    keys = list(rows[0])
    matrix = np.array([[row[key] for key in keys] for row in rows], float)
    total = dict(zip(keys, matrix.sum(0)))
    resampled = weights @ matrix  # (draws, keys)
    drawn = ratios({key: resampled[:, k] for k, key in enumerate(keys)})
    intervals = {}
    for name, parts in _parts(total).items():
        if name in PIECE_NUMBERS:
            continue
        if isinstance(parts, list):
            intervals[name] = [None if _small(pair) else _interval(values) for pair, values in zip(parts, drawn[name])]
        else:
            intervals[name] = None if _small(parts) else _interval(drawn[name])
    longest = np.asarray(lengths, float)
    described = {"count": len(lengths), "median_s": _plain(np.median(longest) / 10.0) if lengths else None,
                 "p90_s": _plain(np.percentile(longest, 90) / 10.0) if lengths else None,
                 "longest_s": _plain(longest.max() / 10.0) if lengths else None}
    return {"sequences": len(rows), "counts": {key: int(value) for key, value in total.items()},
            "ratios": {name: _plain(value) for name, value in ratios(total).items()},
            "intervals": intervals, "interval_lengths": described}, drawn


def report(sequences: tuple[Sequence, ...], decided: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray],
           bands: Mapping[str, np.ndarray] | None = None, confidence: Mapping[str, np.ndarray] | None = None,
           draws: int = bootstrap.DRAWS, seed: int = bootstrap.SEED) -> dict:
    """The report about samples by kind of sequence and for all sequences together.

    `bands` holds the mask of the band alone for every sequence (for "beyond the band"), `confidence` the largest
    probability of every sample (for the low-confidence reference: its error, and the rule's error minus its error).
    The intervals are 95% percentile intervals of a bootstrap over sequences; one generator serves the kinds in a
    fixed order, then all sequences. A number that rests on a count of SMALL or less has None for its interval.
    """
    generator = np.random.default_rng(seed)
    rows = {s.id: counts(s, decided[s.id], masks[s.id], None if bands is None else bands[s.id]) for s in sequences}
    lengths = {s.id: [b - a for a, b, flag in scorecard.runs(np.asarray(masks[s.id], bool).astype(int)) if flag] for s in sequences}
    out = {}
    for kind in (*KINDS, "all"):
        members = [s for s in sequences if kind in (s.kind, "all")]
        if not members:
            continue
        weights = bootstrap.resampling_weights(generator, len(members), draws)
        out[kind], drawn = _pooled([rows[s.id] for s in members], [x for s in members for x in lengths[s.id]], weights)
        if confidence is None:
            continue
        wrong = np.concatenate([np.asarray(decided[s.id]) != s.labels for s in members])
        sure = np.concatenate([np.asarray(confidence[s.id], float) for s in members])
        owner = np.concatenate([np.full(s.labels.size, k) for k, s in enumerate(members)])
        declined = out[kind]["counts"]["uncertain"]
        kept = _least_confident_kept(sure, declined)
        low = float(wrong[kept].mean()) if kept.size else float("nan")
        low_drawn = error_at_equal_coverage_drawn(wrong, sure, owner, weights, weights @ np.array([rows[s.id]["uncertain"] for s in members], float))
        labelled = out[kind]["ratios"]["error_labelled"]
        few = _small((wrong[kept].sum(), kept.size))
        out[kind]["error_low_confidence"] = _plain(low)
        out[kind]["error_labelled_minus_low_confidence"] = None if labelled is None else _plain(labelled - low)
        out[kind]["intervals"]["error_low_confidence"] = None if few else _interval(low_drawn)
        out[kind]["intervals"]["error_labelled_minus_low_confidence"] = (
            None if few or out[kind]["intervals"]["error_labelled"] is None else _interval(drawn["error_labelled"] - low_drawn))
    return out


def piece_summary(rows: list[dict], draws: int = bootstrap.DRAWS, seed: int = bootstrap.SEED, worth: float = WORTH) -> dict:
    """The pieces by their middle sample, counted per recording.

    The four counts (right or wrong by the published answer, kept or declined) are given for all pieces and for the
    pieces beyond the band, each with the number of different recordings behind it. The score is
    `wrong declined - right declined / worth` over the pieces beyond the band; its interval (5th and 95th percentile)
    comes from a bootstrap that draws whole recordings again, and so do the 95% intervals of the two accuracies.
    """
    recordings = sorted({row["recording"] for row in rows})
    index = {name: k for k, name in enumerate(recordings)}

    def table(only_beyond: bool) -> dict:
        out = {}
        for answer in ("right", "wrong"):
            for fate in ("kept", "declined"):
                cell = [r for r in rows if (r["wrong"] == (answer == "wrong")) and (r["declined"] == (fate == "declined"))
                        and (r["beyond_band"] or not only_beyond)]
                out[f"{answer}_{fate}"] = {"pieces": len(cell), "recordings": len({r["recording"] for r in cell})}
        return out

    cells = np.zeros((len(recordings), 4))  # all pieces: right kept, right declined, wrong kept, wrong declined
    beyond = np.zeros((len(recordings), 2))  # the pieces beyond the band: wrong declined, right declined
    for row in rows:
        cells[index[row["recording"]], 2 * int(row["wrong"]) + int(row["declined"])] += 1
        if row["beyond_band"] and row["declined"]:
            beyond[index[row["recording"]], 0 if row["wrong"] else 1] += 1
    score = lambda sums: sums[..., 0] - sums[..., 1] / worth
    accuracy = {"published": lambda sums: (sums[..., 0] + sums[..., 1], sums.sum(-1)), "labelled": lambda sums: (sums[..., 0], sums[..., 0] + sums[..., 2])}
    weights = bootstrap.resampling_weights(np.random.default_rng(seed), len(recordings), draws) if recordings else np.zeros((0, 0))
    empty = [row for row in rows if row["empty"]]
    out = {"pieces": len(rows), "recordings": len(recordings), "empty_pieces": len(empty), "empty_recordings": len({row["recording"] for row in empty}),
           "all": table(False), "beyond_band": table(True)}
    for name, parts in accuracy.items():  # the share of pieces read right at the middle: of all pieces, and of those left labelled
        out[f"accuracy_{name}"] = _plain(_share(*parts(cells.sum(0)))) if recordings else None
        out[f"accuracy_{name}_interval"] = None if not recordings or _small(parts(cells.sum(0))) else _interval(_share(*parts(weights @ cells)))
    out.update(score=float(score(beyond.sum(0))) if recordings else 0.0,
               score_p5_p95=_interval(score(weights @ beyond) if recordings else np.array([]), (5.0, 95.0)), worth=worth)
    return out
