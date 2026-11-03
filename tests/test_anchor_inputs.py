import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module('echo_routing.temporal.anchor_inputs')


def natives(prefix='s'):
    return [{'id': f'{prefix}{c}_{k}', 'kind': 'native',
             'refs': [(f'{prefix}{c}_{k}', i * 3, 1.) for i in range([4, 7, 19, 100][k % 4])],
             'labels': np.full([4, 7, 19, 100][k % 4], c, np.int64)}
            for c in range(5) for k in range(8)]


def test_new_bank_reuses_exact_native_footprints_and_pinned_recipe_without_retry():
    m = api()
    native = natives()
    seqs, recipes, summary = m.construct_bank(native, 120, 'fit', 0)
    repeat, repeated_recipes, repeated_summary = m.construct_bank(native, 120, 'fit', 0)
    assert len(seqs) == len(recipes) == 120
    assert recipes == repeated_recipes and summary == repeated_summary
    assert summary['requested_fragment_support'] == [4, 80]
    for q, r, same in zip(seqs, recipes, repeat):
        assert q['refs'] == same['refs']
        np.testing.assert_array_equal(q['labels'], same['labels'])
        assert 4 <= len(r['fragments']) <= 8
        ids = [f['video_id'] for f in r['fragments']]
        assert len(set(ids)) == len(ids)
        cursor = 0
        lookup = {s['id']: s for s in native}
        for f in r['fragments']:
            source = lookup[f['video_id']]
            length = f['end'] - f['start']
            assert 4 <= length <= min(80, len(source['refs']))
            gamma = .9 if f['variant'] == 'gamma090' else 1.
            assert q['refs'][cursor:cursor + length] == [(v, i, gamma) for v, i, _ in source['refs'][f['start']:f['end']]]
            assert np.all(q['labels'][cursor:cursor + length] == f['label'])
            cursor += length
        assert cursor == len(q['refs']) == len(q['labels'])
    with pytest.raises(ValueError):
        m.construct_bank(native, 120, 'fit', 0, seed=1)


def test_native_cache_reuse_requires_identical_complete_refs_and_input_axis():
    m = api()
    old = natives()
    new = [dict(q) for q in old[:3]]
    mapping = m.native_reuse_index(old, new)
    assert mapping == {0: 0, 1: 1, 2: 2}
    changed = [dict(new[0])]
    changed[0]['refs'] = list(changed[0]['refs'])
    changed[0]['refs'][0] = (changed[0]['id'], 0, .9)
    with pytest.raises(ValueError):
        m.native_reuse_index(old, changed)
    mixed = {'id': 'constructed', 'kind': 'multi', 'refs': list(old[0]['refs']), 'labels': old[0]['labels']}
    assert m.native_reuse_index(old, [mixed]) == {}


def test_full_role_footprints_and_sessions_remain_disjoint_and_count_original5_units():
    m = api()
    a, b = natives('fit'), natives('inner')
    assert m.verify_native_groups({'fit': a, 'inner_cal': b})['session_overlap'] == 0
    with pytest.raises(ValueError):
        m.verify_native_groups({'fit': a, 'inner_cal': a})
    q = {'id': 'axis', 'kind': 'multi', 'refs': [('x', i, 1.) for i in range(19)], 'labels': np.zeros(19, int)}
    assert m.workload([q]) == {'sequences': 1, 'positions': 19, 'a08_stride5_windows': 4, 'jepa_chunks': 1, 'max_length': 19}
