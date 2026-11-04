import importlib
import numpy as np
from echo_routing.temporal.proposal_actions import Action
from echo_routing.temporal.safety_runtime import policies


def test_gross_harm_and_repair_do_not_disappear_when_net_error_is_zero():
    m=importlib.import_module('echo_routing.temporal.safety_metrics');q=[{'id':'synthetic','kind':'native','labels':np.array([0,0])}]
    out=m.gross_frames(q,{'synthetic':np.array([0,1])},{'synthetic':np.array([1,0])})
    assert out['native']['H']==1 and out['native']['R']==1 and out['native']['dE']==0
    assert out['native']['changed_positions']==2 and out['native']['changed_sequences']==1


def test_unsafe_positive_safety_score_count_proxy_regret_and_fixed_risk_bins():
    m=importlib.import_module('echo_routing.temporal.safety_metrics');b=np.r_[np.full(20,2),np.full(18,3)];q={'id':'synthetic','kind':'native','labels':np.full(38,3)}
    action=Action(20,38,tuple([2]*18),jepa=True);scores=np.array([[1.,.1,17.8,.1,.01,.05]])
    result={**policies(b,[action],scores,.2),'actions':[action],'all_actions':[action],'G_rejected':[],'score_matrix':scores}
    out=m.action_summary([q],[b],[result],'C','S-L',.8917446116307443)
    assert out['accepted_actions']==1 and out['accepted_gross_harm_actions']==1 and out['positive_safety_score_false_accepts']==1
    assert out['accepted_effect_totals']['H']==18 and out['accepted_effect_totals']['u']>0
    assert out['pair_accuracy']['informative_pairs']==1 and out['pair_accuracy']['correct_pairs']==0
    assert np.isclose(out['after_veto_regret']['sum'],1.8917446116307443)
    assert out['risk_bins'][0]['candidates']==1 and out['risk_bins'][0]['unsafe_candidates']==1
    assert out['accepted_lengths']['changed_positions']['sum']==18


def test_new_gross_native_intervals_use_same_whole_session_draws():
    m=importlib.import_module('echo_routing.temporal.safety_metrics');left={'session_order':['s1','s2'],'session_gross':np.array([[1,0],[3,0]]),'stream_gross':{'multi':[('m',np.array([2,1]))],'pair':[('p',np.array([1,2]))]}}
    right={'session_order':['s1','s2'],'session_gross':np.zeros((2,2),int),'stream_gross':{'multi':[('m',np.zeros(2,int))],'pair':[('p',np.zeros(2,int))]}}
    r=m.gross_intervals(left,right,5000,9);draws=np.random.default_rng(9).integers(0,2,(5000,2));values=np.array([1,3])[draws].sum(1)
    assert np.array_equal(r['native_new_wrong_reference_positions'],np.quantile(values,[.025,.975]))
    assert r['native_repaired_reference_positions']==[0.,0.]
