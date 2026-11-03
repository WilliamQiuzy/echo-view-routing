import importlib

import numpy as np


def test_constant_actual_panel_is_reported_separately_from_gradient_control(tmp_path):
    m = importlib.import_module('scripts.check_anchor_head_runtime')
    x = np.tile(np.arange(98, dtype=np.float64), (2, 1))
    result = m.probe_head('A', x, np.array([1., 2.], np.float32),
                          {'native': np.array([0]), 'constructed': np.array([1])}, tmp_path,
                          'a' * 64)
    assert result['source_panel_nonconstant_normalized_features'] == 0
    assert result['actual_panel_later_tensor_liveness']['network.0.weight'] is False
    assert all(result['nondegenerate_control_later_tensor_liveness'].values())
    assert result['repeat_and_2plus2_disk_resume_exact']
    assert result['engineering_optimizer_updates'] == 36
    assert result['engineering_mutable_models'] == 6
    assert result['gradient_control_used_as_study_training_data'] is False
