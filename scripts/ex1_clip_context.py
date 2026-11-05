#!/usr/bin/env python3
"""EX1 E2 readout: exam-context decoding at clip level. D1 = B0 session-fold outputs; D2 = A08 development outputs. CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv

import numpy as np

from echo_routing.config import load_paths
from echo_routing.temporal.dev_data import FeatureStore
from exam_level import clip_models as cm, io, metrics as mx, streams as st

ARMS = ('B', 'P', 'S', 'T', 'O0', 'O', 'OT')
DECLARED_CONTROLS = (('T', 'P'), ('T', 'S'), ('O', 'O0'))
EXPLORATORY_CONTROLS = (('S', 'P'), ('OT', 'T'), ('OT', 'O'))
BAG_PERMUTATIONS = 5
EXPECTED_D1 = {'clips': 3364, 'sessions': 502}
PUBLISHED_D1 = {'a5c_voted_a4c': 90, 'a4c_voted_a5c': 6, 'a5c': 217, 'a4c': 858}
PERMUTE_SEED = 2026110399
N_CLASSES = 5


def make_session(name, clips):
    """clips: (video_id, label, evidence, position vote) in any order. Output is in ascending numeric k."""
    ordered = sorted(clips, key=lambda c: st.clip_index(c[0]))
    return {'session': name, 'timestamp': st.timestamp_of(ordered[0][0]), 'ids': [c[0] for c in ordered],
            'y': np.array([c[1] for c in ordered], np.int64), 'L': np.stack([c[2] for c in ordered]),
            'vote': np.array([c[3] for c in ordered], np.int64)}


def position_vote(prob):
    """The published clip rule on raw positions: majority of position argmax, ties to the lower class."""
    return int(np.bincount(np.asarray(prob).argmax(1), minlength=N_CLASSES).argmax())


def permutation_of(session, index, draw=0):
    return np.random.default_rng([PERMUTE_SEED, index, draw]).permutation(len(session['y']))


def base_labels(session):
    return session['L'].argmax(1)


def apical_samples(sessions):
    """Per session: A5C-minus-A4C evidence on clips both truly apical and called apical by the base rule."""
    ds, ys = [], []
    for s in sessions:
        base = base_labels(s)
        keep = np.isin(s['y'], (cm.A4C, cm.A5C)) & np.isin(base, (cm.A4C, cm.A5C))
        if keep.any():
            ds.append(s['L'][keep, cm.A5C] - s['L'][keep, cm.A4C])
            ys.append((s['y'][keep] == cm.A5C).astype(np.int64))
    return ds, ys


def fit_arms(fit_sessions, label_sequences):
    """All parameters come from sessions other than those decoded."""
    pi = np.bincount(np.concatenate(label_sequences), minlength=N_CLASSES) + 1.
    pi = pi / pi.sum()
    chains = {'P': cm.prior_chain(pi), 'S': cm.fit_bag_chain(label_sequences), 'T': cm.fit_chain(label_sequences)}
    evidences = [s['L'] for s in fit_sessions]
    labels = [s['y'] for s in fit_sessions]
    shuffled = [permutation_of(s, i) for i, s in enumerate(fit_sessions)]
    taus = {'P': cm.fit_temperature(evidences, labels, *chains['P']),
            'T': cm.fit_temperature(evidences, labels, *chains['T']),
            'S': cm.fit_temperature([s['L'][p] for s, p in zip(fit_sessions, shuffled)],
                                    [s['y'][p] for s, p in zip(fit_sessions, shuffled)], *chains['S'])}
    ds, ys = apical_samples(fit_sessions)
    five = pi[cm.A5C] / (pi[cm.A4C] + pi[cm.A5C])
    return {'chains': chains, 'taus': taus, 'offset': cm.fit_offset(ds, ys), 'no_offset': cm.fit_offset(ds, ys, with_offset=False),
            'log_pi': np.log([1 - five, five])}


def decode_arms(sessions, fitted):
    """Per-arm clip labels, one array per session, and how many sessions needed the enumeration fallback."""
    out, fallback = {arm: [] for arm in ARMS}, {'O0': [], 'O': [], 'OT': []}
    chains, taus = fitted['chains'], fitted['taus']
    for index, s in enumerate(sessions):
        L, base = s['L'], base_labels(s)
        out['B'].append(base)
        out['P'].append(cm.viterbi(L / taus['P'], *chains['P']))
        out['T'].append(cm.viterbi(L / taus['T'], *chains['T']))
        out['S'].append(cm.decode_permuted(L, permutation_of(s, index), taus['S'], *chains['S']))
        for arm, kwargs in (('O0', {'log_pi': fitted['log_pi']}), ('O', {'log_pi': fitted['log_pi']}), ('OT', {'chain': chains['T']})):
            params = fitted['no_offset'] if arm == 'O0' else fitted['offset']
            labels, used = cm.decode_apical(L, base, params, **kwargs)
            out[arm].append(labels)
            if used:
                fallback[arm].append(index)
    return out, fallback


def bag_spread(sessions, fitted, y, base):
    """Net change of arm S under several random orders, because one order alone carries permutation noise."""
    nets = []
    for draw in range(BAG_PERMUTATIONS):
        labels = np.concatenate([cm.decode_permuted(s['L'], permutation_of(s, i, draw), fitted['taus']['S'], *fitted['chains']['S'])
                                 for i, s in enumerate(sessions)])
        nets.append(int(((base != y) & (labels == y)).sum() - ((base == y) & (labels != y)).sum()))
    return nets


def non_apical_in_decoding(sessions):
    """Clips the base rule calls apical whose truth is another view. They enter the shared offset as outliers."""
    return sum(int((np.isin(base_labels(s), (cm.A4C, cm.A5C)) & ~np.isin(s['y'], (cm.A4C, cm.A5C))).sum()) for s in sessions)


def excluding_fallback(sessions, decoded, fallback):
    """O-family results with the sessions that needed coordinate ascent removed."""
    out = {}
    for arm, skipped in fallback.items():
        keep = [i for i in range(len(sessions)) if i not in set(skipped)]
        if not keep:
            continue
        y = np.concatenate([sessions[i]['y'] for i in keep])
        out[arm] = arm_summary(y, np.concatenate([decoded['B'][i] for i in keep]), np.concatenate([decoded[arm][i] for i in keep]))
    return out


def one_hot_confusion(y, p):
    rows = np.zeros((len(y), N_CLASSES, N_CLASSES))
    rows[np.arange(len(y)), y, p] = 1
    return rows


def arm_summary(y, base, pred):
    conf = one_hot_confusion(y, pred).sum(0).astype(int)
    fix = (base != y) & (pred == y)
    harm = (base == y) & (pred != y)
    n5, n4 = int((y == cm.A5C).sum()), int((y == cm.A4C).sum())
    apical_fix, apical_harm = int((fix & (y == cm.A5C)).sum()), int((harm & (y == cm.A4C)).sum())
    ratio = (apical_fix / n5) / (apical_harm / n4) if n5 and n4 and apical_harm else None
    return {'clips': len(y), 'accuracy': float((y == pred).mean()), 'macro_f1': float(mx.macro_f1(conf)),
            'a5c_correct': int(conf[cm.A5C, cm.A5C]), 'a5c': n5, 'a4c_correct': int(conf[cm.A4C, cm.A4C]), 'a4c': n4,
            'a4c_to_a5c': int(conf[cm.A4C, cm.A5C]), 'a5c_to_a4c': int(conf[cm.A5C, cm.A4C]), 'fixes': int(fix.sum()), 'harms': int(harm.sum()),
            'net': int(fix.sum() - harm.sum()), 'a5c_fixed': apical_fix, 'a4c_harmed': apical_harm,
            'fix_to_harm_rate_ratio': ratio, 'break_even_ratio': n4 / n5 if n5 else None, 'clips_changed': int((pred != base).sum())}


def compare(y, clusters, pred_a, pred_b):
    """A minus B: macro-F1 and net clip change, with whole-timestamp bootstrap."""
    a, b = one_hot_confusion(y, pred_a), one_hot_confusion(y, pred_b)
    net_rows = np.stack([((pred_b != y) & (pred_a == y)).astype(float), ((pred_b == y) & (pred_a != y)).astype(float)], 1)
    return {'macro_f1': mx.paired_delta(a, b, clusters, mx.macro_f1),
            'net': mx.paired_delta(net_rows, np.zeros_like(net_rows), clusters, lambda v: v[..., 0] - v[..., 1])}


def evaluate_cell(dataset, cell, sessions, decoded, fallback):
    y = np.concatenate([s['y'] for s in sessions])
    clusters = [s['timestamp'] for s in sessions for _ in s['y']]
    flat = {arm: np.concatenate(v) for arm, v in decoded.items()}
    arms = {arm: arm_summary(y, flat['B'], flat[arm]) for arm in ARMS}
    against_base = {arm: compare(y, clusters, flat[arm], flat['B']) for arm in ARMS if arm != 'B'}
    votes = np.concatenate([s['vote'] for s in sessions])
    summary = {'dataset': dataset, 'cell': cell, 'sessions': len(sessions), 'timestamps': len(set(clusters)), 'arms': arms, 'against_base': against_base,
               'published_rule_position_vote': arm_summary(y, flat['B'], votes), 'session_auc': auc_with_interval(sessions),
               'non_apical_clips_in_o_family': non_apical_in_decoding(sessions),
               'fallback': {arm: {'sessions': len(v), 'clips': sum(len(sessions[i]['y']) for i in v)} for arm, v in fallback.items()},
               'excluding_fallback': excluding_fallback(sessions, decoded, fallback)}
    private = {'declared': {f'{a}-{b}': compare(y, clusters, flat[a], flat[b]) for a, b in DECLARED_CONTROLS},
               'exploratory': {f'{a}-{b}': compare(y, clusters, flat[a], flat[b]) for a, b in EXPLORATORY_CONTROLS}}
    return summary, private


def auc_with_interval(sessions, n_boot=1000):
    """Is the A5C-minus-A4C evidence better at ranking clips inside one exam than across exams?"""
    d, y, name, stamp = [], [], [], []
    for s in sessions:
        keep = np.isin(s['y'], (cm.A4C, cm.A5C))
        d += (s['L'][keep, cm.A5C] - s['L'][keep, cm.A4C]).tolist()
        y += (s['y'][keep] == cm.A5C).astype(int).tolist()
        name += [s['session']] * int(keep.sum())
        stamp += [s['timestamp']] * int(keep.sum())
    d, y, name, stamp = np.array(d), np.array(y), np.array(name), np.array(stamp)
    point = cm.session_auc(d, y, name)
    fives, fours = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    wins = (d[fives][:, None] > d[fours][None, :]) + .5 * (d[fives][:, None] == d[fours][None, :])
    same = name[fives][:, None] == name[fours][None, :]
    keys, inverse = np.unique(stamp, return_inverse=True)
    rng = np.random.default_rng(PERMUTE_SEED)
    deltas = []
    for _ in range(n_boot):
        weight = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys))).astype(float)[inverse]
        pair = weight[fives][:, None] * weight[fours][None, :]
        within, cross = (pair * same).sum(), (pair * ~same).sum()
        if within and cross:
            deltas.append((wins * pair * same).sum() / within - (wins * pair * ~same).sum() / cross)
    return {**point, 'within_minus_cross_ci95': np.quantile(deltas, [.025, .975]).tolist() if deltas else None}


def load_d1():
    """B0 outputs from models that never trained on the clip's session. Fold k is decoded with fold k's model."""
    fold_of = io.read_json(io.FOLDS)
    models = io.read_json(io.FOLD_MODELS)
    stores = {int(k): FeatureStore(load_paths().cache_root, m['cache_key']) for k, m in models.items()}
    grouped = {}
    votes = {'a5c_voted_a4c': 0, 'a4c_voted_a5c': 0, 'a5c': 0, 'a4c': 0}
    for row in io.manifest_rows('train'):
        session = st.session_of(row['video_id'])
        if not row['family5_label'] or session not in fold_of:
            continue
        label, prob = int(float(row['family5_index'])), stores[int(fold_of[session])].get(row['video_id'], 'orig').prob
        grouped.setdefault(session, []).append((row['video_id'], label, cm.clip_evidence(prob), position_vote(prob)))
        vote = int(np.bincount(prob.argmax(1), minlength=N_CLASSES).argmax())
        votes['a5c'] += label == cm.A5C
        votes['a4c'] += label == cm.A4C
        votes['a5c_voted_a4c'] += label == cm.A5C and vote == cm.A4C
        votes['a4c_voted_a5c'] += label == cm.A4C and vote == cm.A5C
    if votes != PUBLISHED_D1 or len(grouped) != EXPECTED_D1['sessions'] or sum(map(len, grouped.values())) != EXPECTED_D1['clips']:
        raise SystemExit(f'B0 session-fold outputs do not match the published library: {votes}, {len(grouped)}')
    folds = {}
    for name, clips in sorted(grouped.items()):
        folds.setdefault(int(fold_of[name]), []).append(make_session(name, clips))
    return folds


def train_label_sequences(sessions=None):
    """Five-class label sequences in k order from the training split, or from given sessions."""
    if sessions is not None:
        return [s['y'] for s in sessions]
    grouped = {}
    for row in io.manifest_rows('train'):
        if row['family5_label']:
            grouped.setdefault(st.session_of(row['video_id']), []).append((st.clip_index(row['video_id']), int(float(row['family5_index']))))
    return [np.array([y for _, y in sorted(v)], np.int64) for _, v in sorted(grouped.items())]


def run_d1():
    folds = load_d1()
    decoded = {arm: [] for arm in ARMS}
    ordered, per_fold, pooled_fallback, spreads = [], [], {'O0': [], 'O': [], 'OT': []}, []
    for k in sorted(folds):
        fit = [s for j, v in folds.items() if j != k for s in v]
        fitted = fit_arms(fit, train_label_sequences(fit))
        result, fallback = decode_arms(folds[k], fitted)
        y_fold, base_fold = np.concatenate([s['y'] for s in folds[k]]), np.concatenate(result['B'])
        spreads.append(bag_spread(folds[k], fitted, y_fold, base_fold))
        per_fold.append({'fold': k, 'taus': fitted['taus'], 'offset': vars(fitted['offset']), 'no_offset': vars(fitted['no_offset'])})
        for arm, skipped in fallback.items():
            pooled_fallback[arm] += [len(ordered) + i for i in skipped]
        ordered += folds[k]
        for arm in ARMS:
            decoded[arm] += result[arm]
    summary, private = evaluate_cell('D1_B0_session_fold', 'pooled_3_folds', ordered, decoded, pooled_fallback)
    stamps = {}
    for k, sessions in folds.items():
        for s in sessions:
            stamps.setdefault(s['timestamp'], set()).add(k)
    return {**summary, 'per_fold': per_fold, 'bag_net_per_fold_over_5_orders': spreads,
            'timestamps_crossing_folds': sum(len(v) > 1 for v in stamps.values())}, private


def dev_sessions(half, seed):
    cache = io.A08Cache(half, seed)
    grouped = {}
    for q in io.natives(half):
        positions = cache.positions(q['id'])
        grouped.setdefault(st.session_of(q['id']), []).append((q['id'], int(q['labels'][0]), cm.clip_evidence(positions), position_vote(positions)))
    return [make_session(name, clips) for name, clips in sorted(grouped.items())]


def known_boundary_f1(half, sessions, decoded):
    """Segment F1 on exam streams when clip boundaries are known and each clip gets one label. Archive routing only."""
    streams = np.load(io.RUN / 'streams' / f'exam_{half}.npy', allow_pickle=True).tolist()
    out = {}
    for arm in ('B', 'T', 'O', 'OT'):
        rows = []
        for q, s, labels in zip(streams, sessions, decoded[arm]):
            if [c[0] for c in q['clips']] != s['ids']:
                raise SystemExit('exam stream and session clip order differ')
            rows.append(mx.stream_counts(q['labels'], st.expand_clip_labels(q['clips'], labels)))
        out[arm] = mx.pooled(mx.as_rows(rows))['segment_f1']
    return out


def run_d2():
    transitions = train_label_sequences()
    cells, private = [], []
    for seed in io.SEEDS:
        for half, other in (('selection', 'calibration'), ('calibration', 'selection')):
            sessions = dev_sessions(half, seed)
            fitted = fit_arms(dev_sessions(other, seed), transitions)
            decoded, fallback = decode_arms(sessions, fitted)
            summary, controls = evaluate_cell('D2_A08_dev', f'{half}_s{seed}', sessions, decoded, fallback)
            cells.append({**summary, 'seed': seed, 'half': half, 'taus': fitted['taus'], 'offset': vars(fitted['offset']), 'no_offset': vars(fitted['no_offset']),
                          'known_boundary_segment_f1': known_boundary_f1(half, sessions, decoded)})
            private.append({'cell': summary['cell'], **controls})
    return cells, private


def stop_rule(d1, d2):
    """The plan's rule 1 and rule 3, applied mechanically."""
    passed = []
    for arm in ('T', 'O', 'OT'):
        row = d1['against_base'][arm]
        if row['net']['delta'] > 0 and row['macro_f1']['ci95'][0] > 0:
            passed.append(arm)
    total_net = {arm: sum(c['arms'][arm]['net'] for c in d2) for arm in ARMS}
    return {'d1_arms_passing': passed, 'route_continues': bool(passed), 'd2_total_net_over_6_cells': total_net,
            'arms_not_recommended_by_d2': [a for a in passed if total_net[a] < 0]}


def flat_rows(cell):
    for arm, row in cell['arms'].items():
        yield {'dataset': cell['dataset'], 'cell': cell['cell'], 'arm': arm, **row}


def main():
    d1, controls_d1 = run_d1()
    d2, controls_d2 = run_d2()
    rule = stop_rule(d1, d2)
    io.write_json(io.RUN / 'public/clip_context.json', {'D1': d1, 'D2': d2, 'stop_rule': rule})
    if rule['route_continues']:
        io.write_json(io.RUN / 'public/clip_context_controls.json', {'D1': controls_d1, 'D2': controls_d2})
    rows = [r for c in [d1, *d2] for r in flat_rows(c)]
    with (io.RUN / 'public/clip_context.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('clip context readout written')


if __name__ == '__main__':
    main()
