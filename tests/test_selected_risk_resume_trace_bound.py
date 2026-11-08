"""Pure byte-domain proof and one toy journal row; no model or optimizer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest


def api():
    path = Path(__file__).resolve().parents[1]/'echo_routing/temporal/selected_risk_resume_trace_bound.py'
    spec = importlib.util.spec_from_file_location('private_selected_risk_resume_trace_bound', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.formal_trace_raw_bound


def test_complete_schema_digit_monotonicity_and_strict_integer_domain():
    bound = api()
    valid = [1, 2, 10, 11, 100, 101, 2**63 - 1]
    results = [bound(n) for n in valid]
    expected_fields = {
        'u', 'candidate_BCE', 'loss', 'lr', 'physical_candidates',
        'unsafe_targets', 'finite_gradients', 'gradient_norm',
        'nonzero_gradient_tensors', 'candidate_indices', 'source_kind_blocks',
        'optimizer_update_wall_seconds', 'identity',
    }
    for n, result in zip(valid, results):
        assert result['candidate_upper'] == n
        assert set(result['field_upper_bytes']) == expected_fields
        assert result['candidate_index_digit_upper'] == len(str(n - 1))
        assert result['per_head_raw_bytes'] == 6000 * result['per_row_upper']
        assert result['legacy_8192'] == {'per_row_upper': 8192, 'per_head_raw_bytes': 6000 * 8192}
        assert result['float_token_upper'] == 32
        assert result['field_upper_bytes']['finite_gradients'] == len('false')
        assert result['field_upper_bytes']['source_kind_blocks'] == len('[0, 1]')
        assert {p['path'] for p in result['source_pins']} == {
            'echo_routing/temporal/event_risk_model.py',
            'echo_routing/temporal/selected_risk_resume_formal.py',
            'echo_routing/temporal/proposal_training.py',
        }
        assert all(set(p) == {'path', 'sha256'} and re.fullmatch('[0-9a-f]{64}', p['sha256']) for p in result['source_pins'])
    assert [r['per_row_upper'] for r in results] == sorted(r['per_row_upper'] for r in results)
    assert bound(11)['per_row_upper'] - bound(10)['per_row_upper'] == 64
    assert bound(101)['per_row_upper'] - bound(100)['per_row_upper'] == 64
    for invalid in [True, False, 0, -1, 1.0, 1.5, None, '1', 2**63]:
        with pytest.raises((TypeError, ValueError)):
            bound(invalid)
    # Callers cannot mutate the literal pins held by later calls.
    first = bound(1); first['source_pins'][0]['sha256'] = '0'*64
    assert bound(1)['source_pins'][0]['sha256'] != '0'*64


def test_actual_qualified_journal_one_maximal_synthetic_row_matches_structural_bound(tmp_path):
    from echo_routing.temporal import proposal_training
    bound = api(); n = 2**63 - 1; result = bound(n)
    root = Path(proposal_training.__file__).resolve().parents[2]
    for pin in result['source_pins']:
        assert hashlib.sha256((root/pin['path']).read_bytes()).hexdigest() == pin['sha256']
    identity = {key: 'f'*64 for key in ['code', 'config', 'sources', 'input', 'draw_plan', 'normalizer', 'target_scales']}
    extreme = -sys.float_info.max
    row = {
        'u': 0, 'candidate_BCE': extreme, 'loss': extreme, 'lr': extreme,
        'physical_candidates': 64, 'unsafe_targets': 64, 'finite_gradients': False,
        'gradient_norm': extreme,
        'nonzero_gradient_tensors': {key: False for key in ['network.0.weight', 'network.0.bias', 'network.2.weight', 'network.2.bias']},
        'candidate_indices': [n - 1]*64, 'source_kind_blocks': [0, 1],
        'optimizer_update_wall_seconds': extreme,
    }
    journal = proposal_training.UpdateJournal(tmp_path/'toy_journal', identity)
    journal.begin(0); journal.finish(row)
    raw = journal.trace.read_bytes()
    assert raw == (json.dumps({**row, 'identity': identity}, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    assert journal.cursor == 1 and len(raw.splitlines()) == 1 and not journal.pending.exists()
    # Only u=0 was written; u=5999 requires three further decimal characters.
    # Five finite-float slots each receive the declared independent 32-char room.
    float_size = len(json.dumps(extreme, allow_nan=False))
    assert result['per_row_upper'] == len(raw) + (4 - 1) + 5*(32 - float_size)
    assert len(raw) < result['per_row_upper']
    for value in [-0.0, 0.0, sys.float_info.max, sys.float_info.min, -sys.float_info.min, 5e-324, -5e-324]:
        assert len(json.dumps(value, allow_nan=False)) <= result['float_token_upper']
