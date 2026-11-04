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
