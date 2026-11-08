import pytest


def test_review_is_explicit_independent_same_snapshot_and_not_goal_success():
    from echo_routing.temporal.selected_risk_resume_finish import validate_review
    r=dict(approved=True,independent=True,scope='scientific_accuracy',runtime_authorization_sha256='a'*64,package_sha256='b'*64,summary_sha256='c'*64,reviewer_identity='independent reviewer',research_goal_succeeded=False)
    assert validate_review(r,'scientific_accuracy','a'*64,'b'*64,'c'*64)
    for key,value in [('approved',False),('independent',False),('package_sha256','d'*64)]:
        bad={**r,key:value}
        with pytest.raises(ValueError):validate_review(bad,'scientific_accuracy','a'*64,'b'*64,'c'*64)


def test_negative_never_labels_unassessed_coverage_failure_or_formal_complete():
    from echo_routing.temporal.selected_risk_resume_finish import negative_state
    state=negative_state('resource_stopped',None,0,0,0)
    assert state['coverage_evaluated'] is False and state['pass_branch']['outer']=='not_run'
    with pytest.raises(ValueError):negative_state('coverage_negative',None,48000,0,0)
    with pytest.raises(ValueError):negative_state('resource_stopped',None,0,1,1)


def test_full_closure_does_not_pass_incomplete_matrix_or_unknown_actual_caps():
    from echo_routing.temporal.selected_risk_resume_finish import validate_full
    p={'passed':True,'counts':dict(outer=238,comparisons=140,inner=34,support=56,native=34,regression=32,curves=24000),'actual_visual_QA_passed':True}
    assert validate_full(p,True)
    with pytest.raises(ValueError):validate_full(p,None)
    p['counts']['support']=55
    with pytest.raises(ValueError):validate_full(p,True)


def test_incomplete_retains_partial_formal_and_does_not_call_it_not_run_or_coverage_failure():
    from echo_routing.temporal.selected_risk_resume_finish import negative_state
    s=negative_state('incomplete',{'passed':True},48000,2700,2)
    assert s['status']=='incomplete' and s['formal_optimizer_updates']==2700 and s['C0_C1_instances']==2
    assert s['coverage_description']=='passed the fixed supply gates'
    assert s['coverage_passed'] is True and s['pass_branch']['formal']=='partial_or_completed_prefix_preserved'
    assert s['research_goal_succeeded'] is None and s['overall_complete'] is False
