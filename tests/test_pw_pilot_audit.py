import importlib.util
from pathlib import Path


def test_independent_scalar_metric_recount_handles_ties_and_inversion():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_pw_small_pilot.py'
    spec = importlib.util.spec_from_file_location('pw_pilot_independent', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.scalar_metrics([0., 0., 0., 0.])['AUROC'] == .5
    result = module.scalar_metrics([2., -2., 1., -1.])
    assert result['correct'] == 0 and result['AUROC'] == 0 and result['pair_order_points'] == 0
    result = module.scalar_metrics([-2., 2., -1., 1.])
    assert result['correct'] == 4 and result['AUROC'] == 1 and result['pair_order_points'] == 2
