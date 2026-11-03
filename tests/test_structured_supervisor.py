from scripts import supervise_structured_v2 as supervisor

def test_guardian_failure_cannot_become_stage_success_when_child_exits_zero():
    assert supervisor.stage_exit_status({'exit_code':0,'termination_reason':None,'guardian_exception':None})==0
    for row in [{'exit_code':0,'termination_reason':'guarderror','guardian_exception':None},{'exit_code':0,'termination_reason':None,'guardian_exception':'error'},{'exit_code':-15,'termination_reason':None,'guardian_exception':None}]:assert supervisor.stage_exit_status(row)!=0
