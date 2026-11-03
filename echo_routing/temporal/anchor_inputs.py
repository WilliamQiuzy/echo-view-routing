"""A59 split-pure construction and complete-context cache reuse qualification."""
import numpy as np

from echo_routing.compose.recipe_builder import Pool, make_multi_recipes, recipe_to_dict
from echo_routing.evaluate.failure_units import session_of
from echo_routing.temporal.anchor_edits import labels
from echo_routing.temporal.windowed import window_centres


def native_group(sequences):
    if not sequences or len({q['id'] for q in sequences}) != len(sequences):
        raise ValueError('nonempty unique native group required')
    for q in sequences:
        y = labels(q['labels'])
        if (q['kind'] != 'native' or len(q['refs']) != len(y) or not np.all(y == y[0])
                or any(r[0] != q['id'] or float(r[2]) != 1. for r in q['refs'])):
            raise ValueError('unaltered full native source footprint required')
    return sequences


def construct_bank(natives, count, domain, domain_index, seed=2026110359):
    native_group(natives)
    domains = {'fit': 0, 'inner_cal': 1, 'confirmation_selection': 2, 'confirmation_calibration': 3}
    if seed != 2026110359 or domains.get(domain) != domain_index or int(count) != count or count < 1:
        raise ValueError('prospective fixed A59 seed/domain/count required')
    by_id = {q['id']: q for q in natives}
    pool = Pool.from_index([q['id'] for q in natives], [int(q['labels'][0]) for q in natives], [len(q['refs']) for q in natives])
    recipes = make_multi_recipes(pool, count, np.random.default_rng([seed, 59, domain_index]),
                                n_fragments=(4, 8), edit_variant='gamma090', edit_prob=.5,
                                min_len=4, max_len=80, reuse=True, p_same_next=.35)
    if len(recipes) != count or any(not 4 <= len(r.fragments) <= 8 for r in recipes):
        raise ValueError('fixed construction failed; do not retry or select recipes')
    sequences, fragment_lengths = [], []
    for index, recipe in enumerate(recipes):
        refs, truth, joins = [], [], []
        for fragment in recipe.fragments:
            if refs:
                joins.append(len(refs))
            source = by_id[fragment.video_id]
            gamma = .9 if fragment.variant == 'gamma090' else 1.
            refs.extend((v, int(frame), gamma) for v, frame, _ in source['refs'][fragment.start:fragment.end])
            length = fragment.end - fragment.start
            truth.extend([fragment.label] * length)
            fragment_lengths.append(length)
        sequences.append({'id': f'a59_{domain}_{index:04d}', 'kind': 'multi', 'refs': refs,
                          'labels': np.asarray(truth, np.int64), 'joins': np.asarray(joins, np.int64)})
    summary = {'seed': seed, 'RNG_domain': [seed, 59, domain_index], 'streams': len(sequences),
               'requested_fragment_support': [4, 80], 'realized_fragments': len(fragment_lengths),
               'realized_fragment_minmax': [min(fragment_lengths), max(fragment_lengths)],
               'realized_fragment_quantiles': np.quantile(fragment_lengths, [.1, .5, .9]).tolist(),
               'same_source_repeated_within_stream': 0, 'source_footprint_outside_group': 0,
               'generation_retries_or_metric_selection': 0}
    return sequences, [recipe_to_dict(r) for r in recipes], summary


def verify_native_groups(groups):
    seen_sources, seen_sessions = set(), set()
    records = {}
    for name, sequences in groups.items():
        native_group(sequences)
        sources = {q['id'] for q in sequences}
        sessions = {session_of(v) for v in sources}
        if sources & seen_sources or sessions & seen_sessions:
            raise ValueError('whole source/session crosses native roles')
        seen_sources |= sources
        seen_sessions |= sessions
        records[name] = {'natives': len(sources), 'sessions': len(sessions),
                         'class_counts': np.bincount([int(q['labels'][0]) for q in sequences], minlength=5).tolist()}
    return {'source_overlap': 0, 'session_overlap': 0, 'roles': records,
            'session_is_not_confirmed_patient': True}


def native_reuse_index(old, new):
    if len({q['id'] for q in old}) != len(old) or len({q['id'] for q in new}) != len(new):
        raise ValueError('unique old/new sequence identities required')
    lookup = {q['id']: (i, q) for i, q in enumerate(old) if q['kind'] == 'native'}
    mapping = {}
    for i, q in enumerate(new):
        if q['kind'] != 'native' or q['id'] not in lookup:
            continue
        old_index, source = lookup[q['id']]
        if list(map(tuple, source['refs'])) != list(map(tuple, q['refs'])) or len(q['refs']) != len(q['labels']):
            raise ValueError('same ID does not establish identical full-context cache axis')
        mapping[i] = old_index
    return mapping


def workload(sequences):
    lengths = [len(q['refs']) for q in sequences]
    if not lengths or any(n < 1 or n != len(q['labels']) for n, q in zip(lengths, sequences)):
        raise ValueError('complete nonempty original axes required')
    return {'sequences': len(sequences), 'positions': sum(lengths),
            'a08_stride5_windows': sum(len(window_centres(n, 5)) for n in lengths),
            'jepa_chunks': sum((n + 31) // 32 for n in lengths), 'max_length': max(lengths)}
