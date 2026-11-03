import importlib

import numpy as np


def test_eligible_and_actual_feature_axis_exposure_separate_from_whole_source_context():
    m = importlib.import_module('echo_routing.temporal.anchor_exposure')
    qs = [{'id': 's0_0', 'kind': 'native', 'labels': np.zeros(50, int), 'refs': [('s0_0', i, 1.) for i in range(50)]},
          {'id': 's1_0', 'kind': 'native', 'labels': np.ones(10, int), 'refs': [('s1_0', i, 1.) for i in range(10)]},
          {'id': 'mix', 'kind': 'multi', 'labels': np.r_[np.zeros(50, int), np.ones(10, int)],
           'refs': [('s0_0', i, 1.) for i in range(50)] + [('s1_0', i, 1.) for i in range(10)]}]
    rows = [{'sequence_index': 0, 'span': (20, 22), 'kind': 'native', 'feature_positions': 34},
            {'sequence_index': 2, 'span': (0, 1), 'kind': 'constructed', 'feature_positions': 17}]
    out = m.summarize(qs, rows, [0, 0, 1])
    assert out['all_forwarded_native_sources'] == 2 and out['all_native_sessions'] == 2
    assert out['eligible_sequences'] == 2 and out['ineligible_sequences'] == 1
    assert out['eligible_whole_sequence_context_sources'] == 2
    assert out['eligible_feature_axis_sources'] == 1
    assert out['actual_physical_candidate_exposures'] == 3
    assert out['actual_unique_candidates'] == 2
    assert out['actual_feature_axis_position_exposures'] == 85
    assert out['actual_unique_sequence_axis_positions'] == 51
    assert out['actual_whole_sequence_context_sources'] == 2
    assert out['actual_feature_axis_sources'] == 1
    assert out['not_every_forwarded_source_is_a_feature_axis_optimizer_exposure']

