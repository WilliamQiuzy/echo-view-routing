"""The plan's reading rules and the paired bootstrap of external recall. Pure logic."""
import numpy as np

NATIVE_FLOOR = {'selection': .973, 'calibration': .980}
BOUNDARY_RECALL_FLOOR = .95
HALVES = ('selection', 'calibration')
N_BOOT = 5000
BOOT_SEED = 2026110403


def direction(intervals):
    """+1 when every seed's interval lies above zero, -1 when every one lies below, otherwise 0."""
    if not intervals:
        raise ValueError('at least one interval is required')
    if all(low > 0 for low, _ in intervals):
        return 1
    if all(high < 0 for _, high in intervals):
        return -1
    return 0


def floors_hold(rows):
    """rows: one {half, boundary_recall, native_macro_f1} per seed and half."""
    if not rows:
        raise ValueError('at least one row is required')
    return all(r['boundary_recall'] >= BOUNDARY_RECALL_FLOOR and r['native_macro_f1'] >= NATIVE_FLOOR[r['half']] for r in rows)


def ev9v_better(by_half, floor_ok):
    """by_half: half -> the per-seed intervals of segment F1 against A08."""
    signs = [direction(by_half[h]) for h in HALVES]
    return bool(floor_ok and max(signs) == 1 and min(signs) != -1)


def external_better(by_endpoint):
    """by_endpoint: endpoint -> the per-seed intervals of recall against A08."""
    if not by_endpoint:
        raise ValueError('at least one endpoint is required')
    signs = [direction(v) for v in by_endpoint.values()]
    return bool(max(signs) == 1 and min(signs) != -1)


def reading(ev9v_signs, is_ev9v_better, is_external_better, open_set_improved):
    if is_ev9v_better:
        return 'ev9v_better'
    if (is_external_better or open_set_improved) and min(ev9v_signs) == -1:
        return 'trade_off'
    if is_external_better:
        return 'external_better'
    if open_set_improved:
        return 'open_set_improved'
    return 'no_gain'


def stop_after_first_seed(dev_intervals, external_intervals, open_set_usable):
    """An arm stops only when a consistent gain has become impossible in all three places."""
    if len(dev_intervals) != len(HALVES) or not external_intervals:
        raise ValueError('both halves and at least one external endpoint are required')
    both_below = all(high < 0 for _, high in dev_intervals)
    no_external_gain = not any(low > 0 for low, _ in external_intervals)
    return bool(both_below and no_external_gain and not open_set_usable)


def still_possible_after_first_seed(dev_by_half, first_seed_floors_hold, external_by_endpoint, open_set_usable):
    """Which readings further seeds could still reach. A consistent gain needs every seed's interval above zero, the first seed's included."""
    return {'ev9v_better': bool(first_seed_floors_hold and any(low > 0 for low, _ in dev_by_half.values())),
            'external_better': any(low > 0 for low, _ in external_by_endpoint.values()),
            'open_set_improved': bool(open_set_usable)}


def fallback_kind(geometry):
    """The strongest fallback that any layout segment of one recording used."""
    if geometry.get('failed'):
        return 'failed'
    found = {segment['fallback'] for segment in geometry['segments']}
    for kind in ('dynamic_envelope_fallback', 'geometry_fallback'):
        if kind in found:
            return kind
    return 'none'


def _union(first, second):
    repeated = first.keys() & second.keys()
    if repeated:
        raise ValueError(f'a result appears twice: {sorted(repeated)}')
    return {**first, **second}


def merge_dev(files):
    """Development result files into one. A row may come from one file only, so no seed is ever read from two runs."""
    if not files:
        raise ValueError('at least one file is required')
    results = {half: {} for half in HALVES}
    for stored in files:
        results = {half: _union(results[half], stored['results'][half]) for half in HALVES}
    return {'results': results}


def merge_external(files):
    """External result files into one, by tag, under the same rule."""
    if not files:
        raise ValueError('at least one file is required')
    tags = {}
    for stored in files:
        tags = _union(tags, stored['tags'])
    return {'tags': tags}


def parse_cpu_log(text):
    """Lines of /usr/bin/time: stage, user, system and wall seconds. A retried stage adds up, because that CPU was really spent, and the status line of a failed run is skipped. Anything else is an error, so no time goes missing unnoticed."""
    stages = {}
    for line in text.splitlines():
        if line.startswith('Command '):
            continue
        fields = line.split('\t')
        if len(fields) != 4:
            raise ValueError(f'unexpected CPU log line: {line!r}')
        stage, user, system, wall = fields
        before = stages.get(stage, {'cpu_hours': 0., 'wall_hours': 0.})
        stages = {**stages, stage: {'cpu_hours': before['cpu_hours'] + (float(user) + float(system)) / 3600, 'wall_hours': before['wall_hours'] + float(wall) / 3600}}
    if not stages:
        raise ValueError('the CPU log has no stage')
    return {'stages': stages, 'cpu_hours': sum(v['cpu_hours'] for v in stages.values())}


def all_seed_pairs(systems, references, clusters):
    """Every system seed against every reference seed. Seed numbers only share a manifest, so this shows how far a gain rests on which seeds are paired."""
    if not systems or not references:
        raise ValueError('seeds are required on both sides')
    pairs = [{'system_seed': s, 'reference_seed': r, **paired_recall_delta(a, b, clusters)} for s, a in systems.items() for r, b in references.items()]
    return {'pairs': pairs, 'above_zero': sum(p['ci95'][0] > 0 for p in pairs), 'below_zero': sum(p['ci95'][1] < 0 for p in pairs)}


def paired_recall_delta(a, b, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """a, b: 0/1 correct per recording. Whole clusters are resampled, in the same draw for both systems."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    clusters = np.asarray(clusters)
    if not (len(a) == len(b) == len(clusters)) or not len(a):
        raise ValueError('equal non-empty lengths are required')
    names, inverse = np.unique(clusters, return_inverse=True)
    k = len(names)
    sum_a, sum_b, sizes = np.bincount(inverse, a, k), np.bincount(inverse, b, k), np.bincount(inverse, minlength=k).astype(float)
    weights = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=n_boot).astype(float)
    totals = weights @ sizes
    draws = (weights @ sum_a - weights @ sum_b) / totals
    return {'a': float(a.mean()), 'b': float(b.mean()), 'delta': float(a.mean() - b.mean()), 'ci95': [float(x) for x in np.percentile(draws, [2.5, 97.5])], 'n': int(len(a)), 'clusters': int(k)}


def distinguished(read_4ch, read_2ch):
    """One patient per position: 1 when the patient's 4CH recording is read as the target and the 2CH recording is not."""
    read_4ch, read_2ch = np.asarray(read_4ch, float), np.asarray(read_2ch, float)
    if len(read_4ch) != len(read_2ch):
        raise ValueError('both views are required for every patient')
    return read_4ch * (1. - read_2ch)


def auroc_of(positive, negative):
    """The chance that a positive scores above a negative, a tie counting half."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if not len(positive) or not len(negative):
        raise ValueError('both groups are required')
    ranked = np.sort(negative)
    wins = np.searchsorted(ranked, positive, side='left') + np.searchsorted(ranked, positive, side='right')
    return float(wins.sum() / (2 * len(positive) * len(negative)))


def paired_auroc_delta(a_positive, a_negative, b_positive, b_negative, n_boot=N_BOOT, seed=BOOT_SEED):
    """AUROC of system a minus system b. Position i holds one patient in all four arrays, and whole patients are resampled."""
    arrays = [np.asarray(values, float) for values in (a_positive, a_negative, b_positive, b_negative)]
    if len({len(values) for values in arrays}) != 1 or not len(arrays[0]):
        raise ValueError('all four arrays must hold the same patients')
    a_p, a_n, b_p, b_n = arrays
    n = len(a_p)
    rng = np.random.default_rng(seed)
    deltas = [auroc_of(a_p[i], a_n[i]) - auroc_of(b_p[i], b_n[i]) for i in rng.integers(0, n, (n_boot, n))]
    low, high = np.percentile(deltas, [2.5, 97.5])
    a, b = auroc_of(a_p, a_n), auroc_of(b_p, b_n)
    return {'a': a, 'b': b, 'delta': a - b, 'ci95': [float(low), float(high)], 'n': n}
