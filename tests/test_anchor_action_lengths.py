import pytest
from echo_routing.temporal.anchor_action_lengths import summarize


def test_full_axis_candidate_and_accepted_lengths_empty_sequence_and_weighting():
    result=summarize([{'T':10,'spans':[(0,2),(5,8)],'accepted':(5,8)},
                      {'T':4,'spans':[],'accepted':None}])
    assert result['positions']==14 and result['sequences']==2
    assert result['candidate_lengths']['sum']==5 and result['candidate_length_over_T']['median']==.25
    assert result['accepted_lengths']['sum']==3 and result['accepted_count']==1
    assert result['accepted_positions_over_all_positions']==3/14
    assert result['all_sequence_accepted_length_over_T']['mean']==.15
    assert result['unchanged_positions']==11 and result['no_action_sequences']==1
    empty=summarize([{'T':9,'spans':[(1,8)],'accepted':None}])
    assert empty['candidate_lengths']['sum']==7
    assert empty['accepted_lengths']['count']==0 and empty['accepted_lengths']['median'] is None
    assert empty['accepted_positions_over_all_positions']==0
    with pytest.raises(ValueError):summarize([{'T':9,'spans':[(1,10)],'accepted':None}])


def test_reporting_entrypoints_import_without_model_execution():
    import importlib
    for name in ['scripts.audit_anchor_action_lengths','scripts.supervise_anchor_reporting','scripts.close_anchor_reporting']:
        assert callable(importlib.import_module(name).main)
