import copy,importlib
import pytest


def test_terminal_complete_matrix_direct_wait_and_pid_reuse_are_separate():
    m=importlib.import_module('echo_routing.temporal.safety_terminal')
    s={'formal_models':10,'formal_optimizer_updates':60000,'metric_rows':314,'paired_comparison_count':240,'all314_metrics_independently_recounted':True};o={'all_maxone_G_and_noop':True,'domain_count':60,'overlapping_A60_G_oracles_exactly_reproduced':16};n={'passed':True,'cell_count':56};models={'passed':True,'all10finals24inner8taus56sets_frozen':True}
    assert m.validate_science(s,o,n,models)
    bad=copy.deepcopy(o);bad['domain_count']=59
    with pytest.raises(ValueError):m.validate_science(s,bad,n,models)
    row={'model_pid':42,'supervisor_pid':41,'process_start_ticks':100,'finished_epoch':200.,'observed_worker_exit_code':None,'child_exit_code_directly_waited':False}
    state=m.worker_status([row]);assert not state['all_new_worker_statuses_directly_waited'] and state['unknown_server_worker_OS_status_count']==1
    assert m.process_relation(row,42,{'start_ticks':100,'start_epoch':210.})=='reused_after_record'
    assert m.process_relation(row,42,{'start_ticks':101,'start_epoch':190.})=='reused_different_start_ticks'
    assert m.process_relation(row,42,{'start_ticks':100,'start_epoch':190.})=='possibly_same_owned_live_process'
    assert m.process_relation(row,42,None)=='absent'


def test_collector_cannot_declare_overall_completion_before_its_wait_and_root_integration():
    m=importlib.import_module('echo_routing.temporal.safety_terminal');state=m.collector_state()
    assert state['server_scientific_and_resource_collection_complete']
    assert state['collector_final_caller_OS_receipt_pending'] and state['protected_public_integration_pending']
    assert not state['all_authorized_stages_complete'] and state['phase']!='closed_complete'
