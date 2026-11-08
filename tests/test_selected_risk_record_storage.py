"""Synthetic selected-record codec/atomic IO only; no data, NN or SGD."""
import copy
import hashlib
import importlib
import struct

import numpy as np
import pytest

m = importlib.import_module('echo_routing.temporal.selected_risk_record_storage')
SESSIONS = [f's{i:03d}' for i in range(402)]


def api(name):
    assert callable(getattr(m, name, None)), 'new lossless record codec missing: '+name
    return getattr(m, name)


def records():
    r = dict(canonical_id='a'*64, kind='multi', D=np.bool_(True), X=False,
             u=np.float64(.125), H=np.int64(9), R=np.int32(2), dE=7, dFP=np.int16(-1),
             safe=False, v=-1.125, delta_M=-1, delta_P=2, unsafe=True, changed_K=14,
             neutral=3, short16_lost=2, short16_recovered=1, Jexclusive_short16_recovered=0,
             event_signature='b'*64, draw_signature='c'*64,
             responsible_sessions=['s001', 's003'], harm_transitions=[[2, 3], [3, 2]],
             new_FP16_segments=0, min_overlap=np.float32(.75), replacement_class=3, old=True, jepa=True)
    s = dict(r, canonical_id='d'*64, D=True, X=True, H=0, R=4, dE=-4, dFP=0,
             safe=True, unsafe=False, changed_K=5, neutral=1, short16_lost=0,
             Jexclusive_short16_recovered=1, event_signature=None, responsible_sessions=[], harm_transitions=[],
             u=-0., v=-0., min_overlap=.25, old=False)
    p = dict(r, canonical_id='e'*64, H=0, R=0, dE=0, dFP=1, changed_K=1, neutral=1,
             short16_lost=0, short16_recovered=0, Jexclusive_short16_recovered=0,
             event_signature='0'*64, responsible_sessions=['s401'], harm_transitions=[],
             new_FP16_segments=1, u=1e-7, v=-1.0000001, min_overlap=1., replacement_class=1, jepa=False)
    return [r, s, p, copy.deepcopy(r)]  # duplicates/order must survive, never dedup here.


def assert_exact(before, after):
    assert len(before) == len(after)
    for a, b in zip(before, after):
        assert set(a) == set(b)
        assert list(a) == list(b)  # even observable Python dict insertion order survives.
        for k, v in a.items():
            assert type(v) is type(b[k]), k
            if isinstance(v, (float, np.floating)):
                assert np.asarray(v).tobytes() == np.asarray(b[k]).tobytes(), k
            else: assert v == b[k], k


def test_all28_fields_none_signed_zero_scalar_dtypes_sessions_and_duplicate_rows_roundtrip():
    rows = records(); encoded = api('encode_records')(rows, SESSIONS)
    assert encoded.dtype.itemsize <= 512 and not encoded.dtype.hasobject
    assert_exact(rows, api('decode_records')(encoded, SESSIONS))


@pytest.mark.parametrize('change', ['drop', 'unknown', 'overflow', 'bool_count', 'float_count', 'nan', 'hash',
                                   'session', 'too_many_sessions', 'duplicate_pair', 'unsorted_pair', 'out_of_class', 'unknown_kind'])
def test_schema_or_range_loss_is_rejected_not_silently_cast(change):
    rows = records(); r = rows[0]
    if change == 'drop': del r['u']
    if change == 'unknown': r['new_undocumented_field'] = 1
    if change == 'overflow': r['H'] = np.uint64(2**63)
    if change == 'bool_count': r['R'] = True
    if change == 'float_count': r['dFP'] = 1.5
    if change == 'nan': r['u'] = float('nan')
    if change == 'hash': r['canonical_id'] = 'short'
    if change == 'session': r['responsible_sessions'] = ['not_in_FIT']
    if change == 'too_many_sessions': r['responsible_sessions'] = SESSIONS[:9]
    if change == 'duplicate_pair': r['harm_transitions'] = [[2, 3], [2, 3]]
    if change == 'unsorted_pair': r['harm_transitions'] = [[3, 2], [2, 3]]
    if change == 'out_of_class': r['harm_transitions'] = [[0, 5]]
    if change == 'unknown_kind': r['kind'] = 'pair'
    with pytest.raises(ValueError): api('encode_records')(rows, SESSIONS)


def test_atomic_immutable_NPZ_external_SHA_and_dictionary_bind_every_field(tmp_path):
    path = tmp_path/'selected.npz'; rows = records()
    result = api('write_records')(path, rows, SESSIONS)
    assert result['records'] == len(rows) and result['row_bytes'] <= 512
    assert result['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert_exact(rows, api('read_records')(path, expected_sha256=result['sha256'], expected_sessions=SESSIONS))
    original = path.read_bytes()
    with pytest.raises(FileExistsError): api('write_records')(path, rows[:1], SESSIONS)
    assert path.read_bytes() == original
    with pytest.raises(ValueError): api('read_records')(path, expected_sha256='f'*64, expected_sessions=SESSIONS)
    with pytest.raises(ValueError): api('read_records')(path, expected_sha256=result['sha256'], expected_sessions=SESSIONS[::-1])
    assert not list(tmp_path.glob('*.tmp'))


def test_maximum_selected_pool_has_value_independent_fixed_storage_bound(tmp_path):
    bound = api('storage_bound')(4*(3000+265), SESSIONS)
    assert bound['row_bytes'] <= 512 and bound['maximum_records'] == 13060
    assert bound['table_bytes'] < 7_000_000
    assert bound['atomic_coexistence_upper_bytes'] == 2*bound['table_bytes']
    # Fixed-width, uncompressed ZIP: payload values must not affect file size.
    rows = records(); result = api('write_records')(tmp_path/'rows.npz', rows, SESSIONS)
    assert result['bytes'] == api('storage_bound')(len(rows), SESSIONS)['table_bytes']


def test_corrupt_schema_type_tags_and_harm_mask_are_rejected():
    encoded = api('encode_records')(records(), SESSIONS)
    bad = encoded.copy(); bad['harm_mask'][0] = 2**31
    with pytest.raises(ValueError): api('decode_records')(bad, SESSIONS)
    bad = encoded.copy(); bad['type_tags'][0, 0] = 255
    with pytest.raises(ValueError): api('decode_records')(bad, SESSIONS)
    bad = encoded.copy(); bad['session_indices'][0, 0] = 65535
    with pytest.raises(ValueError): api('decode_records')(bad, SESSIONS)
    with pytest.raises(ValueError): api('encode_records')([], SESSIONS[:-1])


def test_exact_approved_effects_record_output_roundtrips_without_field_adapter():
    from echo_routing.temporal.selected_risk_effects import record
    from echo_routing.temporal.proposal_actions import canonical
    base = np.array([0, 0, 1, 1, 2, 2]); context = np.zeros(96); context[91] = .75
    rows = []
    for truth in [base, np.array([0, 0, 2, 2, 2, 2])]:
        q = {'id': 'synthetic', 'kind': 'multi', 'labels': truth,
             'refs': [('s001_clip', i, 1.) for i in range(len(base))]}
        action = canonical(base, 2, 4, [2, 2], old=True, jepa=False)
        rows.append(record(q, base, action, context, D=True, X=False))
    assert len(rows[0]) == 28
    assert_exact(rows, api('decode_records')(api('encode_records')(rows, SESSIONS), SESSIONS))
