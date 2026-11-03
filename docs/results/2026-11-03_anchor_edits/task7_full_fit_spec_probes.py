"""Full-FIT entrypoint key-integrity probe; synthetic IO, no graph/optimizer stage."""
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from echo_routing.temporal.anchor_features import fit_normalizer
from echo_routing.temporal.anchor_runtime import prepare_training
from echo_routing.temporal.anchor_training import draw_plan


def test_full_fit_entrypoint_must_not_erase_seed_alias_before_validator(tmp_path, monkeypatch):
    m = importlib.import_module('scripts.check_anchor_full_fit')
    qs = [{'id': 'native', 'kind': 'native', 'labels': np.zeros(7, int)},
          {'id': 'multi', 'kind': 'multi', 'labels': np.array([0, 1, 1, 0, 0, 0, 0])}]
    a = [np.eye(5)[[0, 1, 0, 0, 1, 0, 0]], np.eye(5)[[0, 1, 1, 0, 0, 0, 0]]]
    b = [np.zeros(7, int), np.zeros(7, int)]
    j = [np.full((7, 5), .2)] * 2
    data = prepare_training(qs, a, j, b)
    norms = {arm: fit_normalizer(data['features'][arm]) for arm in ['A', 'J']}
    correct = draw_plan(2, 1, 2026110359)
    bad_canonical = copy.deepcopy(correct)
    bad_canonical['native'][100, 0] = 1 - bad_canonical['native'][100, 0]
    other = draw_plan(2, 1, 2026110360)
    source = {'a08': 'synthetic', 'jepa': 'synthetic'}
    def pin(name): return {'path': name, 'sha256': 'a' * 64}
    inputs = {'dataset': pin('dataset'), 'source_cache_identities': source,
              'normalizers': {arm: pin('norm_' + arm) for arm in ['A', 'J']},
              'plans': {'2026110359': pin('canonical_bad'), '02026110359': pin('alias_good'),
                        '2026110360': pin('second_good')}}
    arrays = {'dataset': data, 'canonical_bad': bad_canonical, 'alias_good': correct, 'second_good': other}
    monkeypatch.setattr(m, 'verify_freeze', lambda name: {'objects': {}, 'git_commit': 'synthetic'})
    monkeypatch.setattr(m, 'verify_pin', lambda value: Path(value['path']))
    monkeypatch.setattr(m, 'read', lambda path: inputs if Path(path).name == 'training_inputs.json' else norms[Path(path).name[-1]])
    monkeypatch.setattr(m.np, 'load', lambda path, **kwargs: SimpleNamespace(item=lambda: arrays[str(path)]))
    monkeypatch.setattr(m, 'role_values', lambda role: (qs, a, j, b, source))
    actual_validate = m.validate
    def stop_after_arithmetic(*args, **kwargs):
        actual_validate(*args, **kwargs)
        raise RuntimeError('aliased entrypoint incorrectly passed arithmetic; stop before engineering optimizer work')
    monkeypatch.setattr(m, 'validate', stop_after_arithmetic)
    with pytest.raises(ValueError, match='seed|alias|canonical'):
        m.main()
