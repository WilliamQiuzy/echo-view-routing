"""A61 label-free shared utility/local-risk forward and same-choice E/N/C."""
import time,math
import numpy as np
import torch
from echo_routing.temporal.proposal_actions import propose,partition
from echo_routing.temporal.proposal_features import normalize
from echo_routing.temporal.safety_features import utility_features,position_features,normalize_positions
from echo_routing.temporal.safety_training import risk_summary
from echo_routing.temporal.safety_policy import decode


def policies(base,actions,scores,tau=None):
    scores=np.asarray(scores,np.float64)
    if scores.shape!=(len(actions),6) or not np.isfinite(scores).all():raise ValueError('complete C6 physical utility/gross/evidence matrix required')
    risks=[dict(zip(['predicted_H','predicted_R','predicted_neutral','pFP','r'],row[1:])) for row in scores]
    return {p:decode(base,actions,scores[:,0],risks,p,tau) for p in ['E','N']+(['C'] if tau is not None else [])}


@torch.no_grad()
def infer(utility,risk,normalizers,scale,prob_a,prob_j,base,tau=None):
    if not math.isfinite(scale) or scale<1:raise ValueError('frozen shared FIT utility RMS required')
    start=time.perf_counter();all_actions=propose(base,np.asarray(prob_a).argmax(1),np.asarray(prob_j).argmax(1));actions,rejected=partition(base,all_actions)
    if actions:
        x=np.stack([utility_features(prob_a,prob_j,base,a) for a in actions]);px=[position_features(prob_a,prob_j,base,a) for a in actions];K=[len(v) for v in px]
        xt=torch.from_numpy(normalize(x,normalizers['candidate']));pt=torch.from_numpy(normalize_positions(np.concatenate(px),normalizers['position']))
        scores=utility(xt).flatten().cpu().numpy().astype(np.float64)*scale;logits,fp=risk(xt,pt);offset=0;rows=[]
        for i,n in enumerate(K):
            result=risk_summary(logits[offset:offset+n],fp[i]);offset+=n
            rows.append([scores[i],*[result[k] for k in ['predicted_H','predicted_R','predicted_neutral','pFP','r']]])
        matrix=np.asarray(rows,np.float64)
    else:matrix=np.empty((0,6),np.float64)
    return {**policies(base,actions,matrix,tau),'actions':actions,'all_actions':all_actions,'G_rejected':rejected,'score_matrix':matrix,
            'cached_feature_heads_policy_seconds':time.perf_counter()-start,'full_system_inference_seconds':None,'producer_cost_excluded':True,'operational_required_producers':['A08','JEPA']}
