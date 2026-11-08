import pytest


def valid(phase='FIT'):
    roles=['F0_mixed','F1_mixed'] if phase=='FIT' else ['a63_confirmation_selection','a63_confirmation_calibration']
    n=1500 if phase=='FIT' else 120
    return {'complete':True,'phase':phase,'committed_payloads':4*n,'input_manifest_sha256':'a'*64,'runtime_authorization_sha256':'b'*64,'aux_updates_at_production':0 if phase=='FIT' else 48000,'formal_updates_at_production':0,'records':[{'source':s,'role':r,'committed_sequences':n,'complete':True,'path':'runs/a63_selected_action_risk_resume_v1/producer/'+s+'/'+r} for s in ['a08','jepa'] for r in roles]}


def test_complete_phase_source_set_not_passed_flag():
    from echo_routing.temporal.selected_risk_resume_inputs import producer_records
    v=valid();assert set(producer_records(v,'FIT','a'*64,'b'*64))=={'F0_mixed','F1_mixed'}
    for key,value in [('committed_payloads',5999),('complete',False),('aux_updates_at_production',1),('formal_updates_at_production',1),('runtime_authorization_sha256','c'*64)]:
        bad=valid();bad[key]=value
        with pytest.raises(ValueError):producer_records(bad,'FIT','a'*64,'b'*64)
    bad=valid();bad['records'][0]['path']='runs/a63_selected_action_risk_v1/old'
    with pytest.raises(ValueError):producer_records(bad,'FIT','a'*64,'b'*64)
    bad=valid();bad['records'][-1]=bad['records'][0]
    with pytest.raises(ValueError):producer_records(bad,'FIT','a'*64,'b'*64)


def test_confirm_cannot_borrow_pre_aux_or_wrong_payloads():
    from echo_routing.temporal.selected_risk_resume_inputs import producer_records
    v=valid('CONFIRM');assert len(producer_records(v,'CONFIRM','a'*64,'b'*64))==2
    v['aux_updates_at_production']=0
    with pytest.raises(ValueError):producer_records(v,'CONFIRM','a'*64,'b'*64)


def test_native_mapping_is_observable_only_and_no_fallback():
    from echo_routing.temporal.selected_risk_resume_inputs import native_indices
    class Bomb:
        def __array__(self,*a):raise AssertionError('truth accessed before selector seal')
        def __eq__(self,other):raise AssertionError('truth compared before selector seal')
    q={'id':'session_a_1','kind':'native','refs':[('session_a_1',0,1.)],'labels':Bomb()}
    assert native_indices([q],[dict(q)])==[0]
    bad=dict(q,refs=[('session_a_1',1,1.)])
    with pytest.raises(ValueError):native_indices([q],[bad])
