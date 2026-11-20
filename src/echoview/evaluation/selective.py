"""The two-part report of an output with uncertain samples (docs/UNCERTAIN.md, section 5).

Part A is about the samples that keep a view: how many they are and how often they are right. Part B is about the
uncertain samples: where they fall, what a forced answer would have been there, and which of the published system's
errors they take away. References stand beside the numbers: the published system (nothing declined), chance (the same
number of samples declined at random, which leaves the error where it was), the ideal (the same number declined,
wrong ones first) and low confidence (the same number declined, least confident first).

Numbers about samples are ratios of counts that add up over sequences: `counts` gives the counts of one sequence,
`ratios` turns summed counts into the numbers of the report, and their intervals come from a bootstrap over
sequences. Numbers about pieces are counted per recording, because one recording is reused in many sequences: `pieces`
lists the pieces of a sequence with their recording, and `piece_summary` draws whole recordings again.
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
WORTH = 5.0  # one wrong piece declined is worth this many right pieces declined (docs/UNCERTAIN.md, step 2)
WITH_INTERVAL = ("coverage", "error_published", "error_labelled", "error_if_forced", "errors_declined", "price", "changes_located")


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
    # single recordings, by whether the published system reads them right at the middle
    for answer in ("right", "wrong"):
        mine = sequence.kind == "native" and rows[0]["wrong"] == (answer == "wrong")
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


def ratios(c: Mapping[str, float]) -> dict:
    """The numbers of the report from summed counts."""
    kept_by_answer = [sum(c[f"kept_{view}_{answer}"] for view in range(VIEWS)) for answer in range(VIEWS)]
    out = {
        # part A: the labelled part
        "coverage": _share(c["labelled"], c["n"]),
        "error_published": _share(c["wrong"], c["n"]),  # also the error under chance
        "error_labelled": _share(c["wrong_labelled"], c["labelled"]),
        "error_ideal": _share(np.maximum(0.0, np.asarray(c["wrong"], float) - c["uncertain"]), c["labelled"]),
        "purity": [_share(c[f"kept_{view}_{view}"], kept_by_answer[view]) for view in range(VIEWS)],
        "recovery": [_share(c[f"kept_{view}_{view}"], c[f"reference_{view}"]) for view in range(VIEWS)],
        "piece_accuracy_published": _share(c["piece_right_kept"] + c["piece_right_declined"], c["pieces"]),
        "piece_accuracy_labelled": _share(c["piece_right_kept"], c["piece_right_kept"] + c["piece_wrong_kept"]),
        "segment_accuracy": _share(c["segments_kept_right"], c["segments_kept"]),
        # part B: the uncertain part
        "uncertain_share": _share(c["uncertain"], c["n"]),  # also the share of the errors chance would decline
        "error_if_forced": _share(c["wrong_uncertain"], c["uncertain"]),
        "errors_declined": _share(c["wrong_uncertain"], c["wrong"]),
        "price": _share(c["right_uncertain"], np.asarray(c["n"], float) - c["wrong"]),
        "uncertain_near_change": _share(c["uncertain_near_change"], c["uncertain"]),
        "intervals_no_change": _share(c["intervals_no_change"], c["intervals"]),
        "changes_located": _share(c["changes_located"], c["changes"]),
        "same_joins_uncertain": _share(c["same_joins_uncertain"], c["same_joins"]),
        "right_recordings_uncertain": _share(c["recordings_right_uncertain"], c["recordings_right"]),
        "wrong_recordings_uncertain": _share(c["recordings_wrong_uncertain"], c["recordings_wrong"]),
        "error_added": _share(c["added_wrong"], c["added"]),
        "error_after_band": _share(c["after_band_wrong"], c["after_band_labelled"]),
    }
    for name in (*ERROR_KINDS, "a4c_a5c"):
        out[f"{name}_declined"] = _share(c[f"{name}_declined"], c[name])
    return out


def error_at_equal_coverage(wrong: np.ndarray, confidence: np.ndarray, declined: int) -> float:
    """The error among the samples that are left when the `declined` least confident samples are declined."""
    wrong, confidence = np.asarray(wrong, bool), np.asarray(confidence, float)
    kept = np.argsort(confidence, kind="stable")[int(declined):]
    return float(wrong[kept].mean()) if kept.size else float("nan")


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


def _pooled(rows: list[dict[str, float]], lengths: list[int], generator: np.random.Generator, draws: int) -> dict:
    keys = list(rows[0])
    matrix = np.array([[row[key] for key in keys] for row in rows], float)
    total = dict(zip(keys, matrix.sum(0)))
    point = ratios(total)
    resampled = bootstrap.resampling_weights(generator, len(rows), draws) @ matrix  # (draws, keys)
    drawn = ratios({key: resampled[:, k] for k, key in enumerate(keys)})
    longest = np.asarray(lengths, float)
    described = {"count": len(lengths), "median_s": _plain(np.median(longest) / 10.0) if lengths else None,
                 "p90_s": _plain(np.percentile(longest, 90) / 10.0) if lengths else None,
                 "longest_s": _plain(longest.max() / 10.0) if lengths else None}
    return {"sequences": len(rows), "counts": {key: int(value) for key, value in total.items()},
            "ratios": {name: _plain(value) for name, value in point.items()},
            "intervals": {name: _interval(drawn[name]) for name in WITH_INTERVAL}, "interval_lengths": described}


def report(sequences: tuple[Sequence, ...], decided: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray],
           bands: Mapping[str, np.ndarray] | None = None, confidence: Mapping[str, np.ndarray] | None = None,
           draws: int = bootstrap.DRAWS, seed: int = bootstrap.SEED) -> dict:
    """The report about samples by kind of sequence and for all sequences together.

    `bands` holds the mask of the band alone for every sequence (for "beyond the band"), `confidence` the largest
    probability of every sample (for the low-confidence reference). The intervals are 95% percentile intervals of a
    bootstrap over sequences; one generator serves the kinds in a fixed order, then all sequences.
    """
    generator = np.random.default_rng(seed)
    rows = {s.id: counts(s, decided[s.id], masks[s.id], None if bands is None else bands[s.id]) for s in sequences}
    lengths = {s.id: [b - a for a, b, flag in scorecard.runs(np.asarray(masks[s.id], bool).astype(int)) if flag] for s in sequences}
    out = {}
    for kind in (*KINDS, "all"):
        members = [s for s in sequences if kind in (s.kind, "all")]
        if not members:
            continue
        out[kind] = _pooled([rows[s.id] for s in members], [x for s in members for x in lengths[s.id]], generator, draws)
        if confidence is not None:
            wrong = np.concatenate([np.asarray(decided[s.id]) != s.labels for s in members])
            sure = np.concatenate([np.asarray(confidence[s.id], float) for s in members])
            out[kind]["error_low_confidence"] = _plain(error_at_equal_coverage(wrong, sure, out[kind]["counts"]["uncertain"]))
    return out


def piece_summary(rows: list[dict], draws: int = bootstrap.DRAWS, seed: int = bootstrap.SEED, worth: float = WORTH) -> dict:
    """The pieces by their middle sample, counted per recording.

    The four counts (right or wrong by the published answer, kept or declined) are given for all pieces and for the
    pieces beyond the band, each with the number of different recordings behind it. The score is
    `wrong declined - right declined / worth` over the pieces beyond the band; its interval (5th and 95th percentile)
    comes from a bootstrap that draws whole recordings again.
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

    per_recording = np.zeros((len(recordings), 2))  # wrong declined, right declined, beyond the band
    for row in rows:
        if row["beyond_band"] and row["declined"]:
            per_recording[index[row["recording"]], 0 if row["wrong"] else 1] += 1
    score = lambda sums: sums[..., 0] - sums[..., 1] / worth
    drawn = score(bootstrap.resampling_weights(np.random.default_rng(seed), len(recordings), draws) @ per_recording) if recordings else np.array([])
    return {"pieces": len(rows), "recordings": len(recordings), "empty_pieces": sum(row["empty"] for row in rows),
            "all": table(False), "beyond_band": table(True),
            "score": float(score(per_recording.sum(0))) if recordings else 0.0, "score_p5_p95": _interval(drawn, (5.0, 95.0)), "worth": worth}
