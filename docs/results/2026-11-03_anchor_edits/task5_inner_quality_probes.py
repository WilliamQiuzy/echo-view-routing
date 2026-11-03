"""A59 fixed-inner quality probes; synthetic CPU engineering updates only."""
import importlib
import numpy as np
import pytest
from echo_routing.temporal.anchor_training import UtilityRegressor, draw_plan, optimizer


def inputs():
    data = {'features': {'A': np.arange(196, dtype=np.float64).reshape(2, 98) / 196},
            'targets': np.ones(2, np.float32),
            'kind_indices': {'native': np.array([0]), 'constructed': np.array([1])},
            'candidate_rows': [{'feature_positions': 1}, {'feature_positions': 1}]}
    norm = {'mean': np.zeros(98), 'std': np.ones(98)}
    identity = {k: str(i) * 64 for i, k in enumerate(['code', 'config', 'sources', 'input', 'draw_plan', 'normalizer'], 1)}
    return data, norm, identity, draw_plan(1, 1, 2026110359)


def test_completed_inner_result_pin_revalidated_without_second_callback(tmp_path):
    m = importlib.import_module('scripts.train_anchor_edits')
    data, norm, identity, plan = inputs()
    calls = []
    result_path = tmp_path / 'inner_result.json'
    def callback(model, cursor, path):
        calls.append(cursor)
        claim = m.read(tmp_path / 'inner_pending.json')
        assert claim == {'identity': identity, 'weight_sha256': m.sha(path), 'step': cursor}
        m.write(result_path, {'completed_inner_result': True})
        return {'path': str(result_path), 'sha256': m.sha(result_path)}
    model = UtilityRegressor('A', 2026110359)
    m.train_one(model, optimizer(model), data, plan, norm, identity, tmp_path,
                stop=2, latest_every=2, milestones=(2,), on_inner=callback)
    assert calls == [2] and not (tmp_path / 'inner_pending.json').exists()
    result_path.write_text('corrupted result')
    resumed = UtilityRegressor('A', 2026110359)
    with pytest.raises(ValueError, match='frozen|changed'):
        m.train_one(resumed, optimizer(resumed), data, plan, norm, identity, tmp_path,
                    stop=2, latest_every=2, milestones=(2,), on_inner=callback)
    assert calls == [2]


def test_inner_claim_directory_sync_failure_never_reaches_callback(tmp_path, monkeypatch):
    m = importlib.import_module('scripts.train_anchor_edits')
    data, norm, identity, plan = inputs()
    calls = []
    def fail_sync(path):
        raise OSError('synthetic inner claim directory-sync failure')
    monkeypatch.setattr(m, 'fsync_directory', fail_sync)
    model = UtilityRegressor('A', 2026110359)
    with pytest.raises(OSError):
        m.train_one(model, optimizer(model), data, plan, norm, identity, tmp_path,
                    stop=2, latest_every=2, milestones=(2,), on_inner=lambda *args: calls.append('called'))
    assert calls == [] and (tmp_path / 'inner_pending.json').exists()
    resumed = UtilityRegressor('A', 2026110359)
    with pytest.raises(ValueError, match='ambiguous|pending'):
        m.train_one(resumed, optimizer(resumed), data, plan, norm, identity, tmp_path,
                    stop=2, latest_every=2, milestones=(2,), on_inner=lambda *args: calls.append('called'))
    assert calls == []
