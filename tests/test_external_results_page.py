import pytest
from scripts.build_external_results import validate_aggregate_inputs, validate_audit_binding


def test_incomplete_evaluation_cannot_be_packaged_as_finished_page():
    with pytest.raises(ValueError,match='incomplete'):
        validate_aggregate_inputs({}, {'passed':True}, {})


def test_unverified_result_cannot_be_packaged_even_with_full_denominators():
    native={d:{f'{dec}_s{s}':{'n_clips':n} for dec in ['argmax','frozen_B3'] for s in range(3)}
            for d,n in [('ev9v',800),('dynamic',1277),('lvh',343),('camus',50)]}
    with pytest.raises(ValueError,match='independent'):
        validate_aggregate_inputs(native, {'passed':False}, {})


def test_same_count_changed_results_are_rejected_under_stale_passed_audit(tmp_path):
    p=tmp_path/'native_results.json';p.write_text('{"n":800}')
    with pytest.raises(ValueError,match='stale'):
        validate_audit_binding({'hashes':{'native_results':'0'*64}}, {'native_results':p})


def test_public_export_rejects_per_source_metadata_paths():
    native={d:{f'{dec}_s{s}':{'n_clips':n} for dec in ['argmax','frozen_B3'] for s in range(3)}
            for d,n in [('ev9v',800),('dynamic',1277),('lvh',343),('camus',50)]}
    with pytest.raises(ValueError,match='aggregate hashes only'):
        validate_aggregate_inputs(native, {'passed':True,'hashes':{'source_metadata':{'private/source':'digest'}}}, {})
