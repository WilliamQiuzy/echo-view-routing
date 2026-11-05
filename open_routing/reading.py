"""Blinded AI reading: identifiers, frame choice, agreement and the pre-written signal rule. Readings are pseudo-labels."""
import numpy as np

ANSWERS = ('A4C', 'A5C', 'uncertain', 'other')
LABEL_ANSWERS = ('A4C', 'A5C')
FRAMES = 4
SPAN = (.1, .9)
SIGNAL_RATIO = 3
Z = 1.959963984540054


def blind_ids(n, seed):
    """Random hexadecimal identifiers that reveal nothing about order or stratum."""
    rng = np.random.default_rng(seed)
    out = set()
    while len(out) < n:
        out.add(f'{int(rng.integers(0, 1 << 32)):08x}')
    return rng.permutation(sorted(out)).tolist()


def frame_picks(n_frames):
    """Evenly spaced frames across the central part of the clip."""
    if n_frames < 1:
        raise ValueError('an empty clip cannot be read')
    last = n_frames - 1
    return sorted(int(round(v)) for v in np.linspace(SPAN[0] * last, SPAN[1] * last, FRAMES))


def cohen_kappa(a, b):
    """Chance-corrected agreement. Undefined when expected agreement is one."""
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError('two equal nonempty reading lists required')
    categories = sorted(set(a) | set(b))
    observed = np.mean([x == y for x, y in zip(a, b)])
    expected = sum(a.count(c) * b.count(c) for c in categories) / len(a) ** 2
    return None if expected == 1 else float((observed - expected) / (1 - expected))


def wilson(successes, total):
    """Wilson 95% interval of a proportion."""
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + Z ** 2 / total
    centre = (p + Z ** 2 / (2 * total)) / denominator
    half = Z * np.sqrt(p * (1 - p) / total + Z ** 2 / (4 * total ** 2)) / denominator
    return [float(max(0, centre - half)), float(min(1, centre + half))]


def both_disagree(label, first, second):
    """Both readers gave a definite answer other than the official label. 'uncertain' never counts."""
    return all(r not in (label, 'uncertain') for r in (first, second))


def label_signal(flagged, flagged_total, control, control_total):
    """The flagged stratum's rate is at least three times the control's, and the Wilson intervals do not overlap."""
    rate, base = flagged / flagged_total, control / control_total
    return bool(rate >= SIGNAL_RATIO * base and wilson(flagged, flagged_total)[0] > wilson(control, control_total)[1])


def repeat_pairs(entries):
    """(original image, repeated image) of the same recording. Recording identity decides, so key order cannot matter."""
    originals = {e['video_id']: b for b, e in entries.items() if e['repeat_of'] is None}
    return [(originals[e['video_id']], b) for b, e in entries.items() if e['repeat_of'] is not None]


def unpack_answers(raw):
    """Split a reader's records into answers, outflow-tract frame counts and view-text sightings."""
    for record in raw.values():
        if not isinstance(record['lvot_frames'], int) or not 0 <= record['lvot_frames'] <= FRAMES or not isinstance(record['text_seen'], bool):
            raise ValueError('undeclared reader record')
    return ({k: v['answer'] for k, v in raw.items()}, {k: v['lvot_frames'] for k, v in raw.items()}, sorted(k for k, v in raw.items() if v['text_seen']))


def inconsistent_records(answers, frames):
    """Records that break the written criterion: A5C with no outflow frame, or A4C with one."""
    return sorted(k for k, a in answers.items() if (a == 'A5C' and not frames[k]) or (a == 'A4C' and frames[k]))


def check_answers(answers, expected_ids):
    """Every image must have exactly one declared answer."""
    if set(answers) != set(expected_ids):
        raise ValueError('reader answers do not cover exactly the given images')
    if any(v not in ANSWERS for v in answers.values()):
        raise ValueError('undeclared reader answer')
    return dict(answers)
