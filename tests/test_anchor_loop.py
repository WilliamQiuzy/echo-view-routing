import importlib

import numpy as np
import torch

from echo_routing.temporal.anchor_training import UtilityRegressor, draw_plan, optimizer


def test_prefrozen_draw_loop_counts_updates_exposure_and_clean_checkpoint_resume(tmp_path):
    m = importlib.import_module('scripts.train_anchor_edits')
    rng = np.random.default_rng(59)
    data = {'features': {'A': rng.normal(size=(12, 98)).astype(np.float64)},
            'targets': rng.normal(size=12).astype(np.float32),
            'kind_indices': {'native': np.arange(4), 'constructed': np.arange(4, 12)},
            'candidate_rows': [{'feature_positions': i + 1} for i in range(12)]}
    normalizer = {'mean': np.zeros(98), 'std': np.ones(98)}
    identity = {k: str(i) * 64 for i, k in enumerate(['code', 'config', 'sources', 'input', 'draw_plan', 'normalizer'], 1)}
    plan = draw_plan(4, 8, 2026110359)
    seen = []
    model = UtilityRegressor('A', 2026110359)
    m.train_one(model, optimizer(model), data, plan, normalizer, identity, tmp_path, stop=2,
                latest_every=2, milestones=(2, 4), on_inner=lambda model, step, path: seen.append(step))
    resumed = UtilityRegressor('A', 2026110359)
    result = m.train_one(resumed, optimizer(resumed), data, plan, normalizer, identity, tmp_path, stop=4,
                         latest_every=2, milestones=(2, 4), on_inner=lambda model, step, path: seen.append(step))
    assert seen == [2, 4] and result['actual_Adam_update'] == result['committed_physical_updates'] == 4
    assert result['physical_candidate_exposures'] == 256
    expected = sum(sum(data['candidate_rows'][int(i)]['feature_positions'] for i in
                       np.r_[plan['native'][u], plan['constructed'][u] + 4]) for u in range(4))
    assert result['physical_feature_position_exposures'] == expected
    direct = UtilityRegressor('A', 2026110359)
    m.train_one(direct, optimizer(direct), data, plan, normalizer, identity, tmp_path / 'direct', stop=4,
                latest_every=2, milestones=(2, 4))
    assert all(torch.equal(a, b) for a, b in zip(resumed.parameters(), direct.parameters()))
    again = m.train_one(resumed, optimizer(resumed), data, plan, normalizer, identity, tmp_path, stop=4,
                        latest_every=2, milestones=(2, 4), on_inner=lambda model, step, path: seen.append(step))
    assert again['new_updates_this_process'] == 0 and seen == [2, 4]
