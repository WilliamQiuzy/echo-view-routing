import numpy as np
from echo_routing.temporal.proposal_actions import propose,partition,decode
from echo_routing.temporal.proposal_metrics import action_summary

def test_full_axis_interval_vs_changed_positions_and_real_frame_harm():
    b=[np.array([1]+[0]*8+[1]),np.zeros(10,int)]
    aa=[np.eye(5)[x]*.8+.04 for x in b];jj=[np.eye(5)[np.zeros(10,int)]*.8+.04]*2
    qs=[{'id':str(i),'kind':'native','labels':x} for i,x in enumerate(b)]
    values=[]
    for x,a,j in zip(b,aa,jj):
        acts=propose(x,a.argmax(1),j.argmax(1));eligible,rejected=partition(x,acts);out=np.tile([1.,-1.,-1.],(len(eligible),1))
        values.append({**{p:decode(x,eligible,out,p) for p in ['O','E','R']},'all_actions':acts,'actions':eligible,'G_rejected':rejected,'outputs':out,'cached_feature_head_policy_seconds':0.})
    m=action_summary(qs,aa,jj,b,values,'R',.9)
    assert m['candidate_lengths']['whole_axis_count']==1 and m['accepted_lengths']['whole_axis_count']==1
    assert m['accepted_lengths']['interval_length']['sum']==10 and m['accepted_lengths']['changed_positions']['sum']==2
    assert m['accepted_interval_positions_fraction_of_role']==.5 and m['accepted_changed_positions_fraction_of_role']==.1
    assert m['accepted_harmful_wrong_positions_actions']==1 and m['accepted_new_JEPA_domain_actions']==1
    assert action_summary(qs,aa,jj,b,values,'O',.9)['accepted_actions']==0

def test_native_frame_error_count_interval_uses_actual_paired_session_draws():
    from types import SimpleNamespace
    from echo_routing.temporal.proposal_metrics import evaluate,paired
    from echo_routing.temporal.dense_runtime import metric_stream
    from echo_routing.temporal.structured_uncertainty import prepare_system
    def q(name,kind,y):return {'id':name,'kind':kind,'labels':np.asarray(y,int),'refs':[(name,i,1.) for i in range(len(y))],'joins':np.asarray([2] if kind!='native' else [],int)}
    qs=[q('alpha_0','native',[0]*5),q('beta_0','native',[1]*8),q('m','multi',[0,0,1,1]),q('p','pair',[0,0,1,1])]
    context={'sequences':qs,'eval_set':SimpleNamespace(natives=[SimpleNamespace(seq_id=x['id'],labels=x['labels']) for x in qs[:2]],multi=[metric_stream(qs[2])],pairs=[metric_stream(qs[3])])}
    left={x['id']:x['labels'].copy() for x in qs};right={x['id']:x['labels'].copy() for x in qs}
    left['alpha_0'][0]=2;right['alpha_0'][:2]=2;left['beta_0'][:4]=2
    lm,lt=evaluate(left,context);rm,rt=evaluate(right,context)
    assert lm['native_frame_errors']==5 and rm['native_frame_errors']==2
    result=paired(prepare_system(left,qs,lt),prepare_system(right,qs,rt),n_boot=32,seed=60)
    manual=np.array([-1,4])[np.random.default_rng(60).integers(0,2,(32,2))].sum(1)
    np.testing.assert_allclose(result['delta95']['native_frame_errors'],np.quantile(manual,[.025,.975]),rtol=0,atol=1e-15)
