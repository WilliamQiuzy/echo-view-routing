"""A59 inner scheduling probe; two synthetic CPU engineering updates, no formal work."""
import importlib
import numpy as np
import pytest
from echo_routing.temporal.anchor_training import UtilityRegressor, draw_plan, optimizer


def test_interrupted_started_inner_readout_cannot_be_silently_reexecuted(tmp_path):
    m = importlib.import_module('scripts.train_anchor_edits')
    data = {'features': {'A': np.zeros((2, 98), np.float64)},
            'targets': np.ones(2, np.float32),
            'kind_indices': {'native': np.array([0]), 'constructed': np.array([1])},
            'candidate_rows': [{'feature_positions': 1}, {'feature_positions': 1}]}
    norm = {'mean': np.zeros(98), 'std': np.ones(98)}
    identity = {k: str(i) * 64 for i, k in enumerate(['code', 'config', 'sources', 'input', 'draw_plan', 'normalizer'], 1)}
    plan = draw_plan(1, 1, 2026110359)
    calls = []
    def interrupted_inner(model, cursor, path):
        calls.append(cursor)
        raise OSError('synthetic interruption after inner readout work started')
    model = UtilityRegressor('A', 2026110359)
    with pytest.raises(OSError):
        m.train_one(model, optimizer(model), data, plan, norm, identity, tmp_path,
                    stop=2, latest_every=2, milestones=(2,), on_inner=interrupted_inner)
    assert calls == [2] and (tmp_path / 'step2.pt').exists()
    def resumed_inner(model, cursor, path):
        calls.append(cursor)
        return {'physical_readout_reexecuted': True}
    resumed = UtilityRegressor('A', 2026110359)
    with pytest.raises((RuntimeError, ValueError), match='pending|ambiguous|inner|replay'):
        m.train_one(resumed, optimizer(resumed), data, plan, norm, identity, tmp_path,
                    stop=2, latest_every=2, milestones=(2,), on_inner=resumed_inner)
    assert calls == [2], 'an already-started inner readout was silently repeated'
