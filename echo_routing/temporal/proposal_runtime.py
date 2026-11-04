"""Label-free A60 inference; supervision is isolated in proposal_data."""
import time
import numpy as np
import torch
from echo_routing.temporal.proposal_actions import propose,partition,decode
from echo_routing.temporal.proposal_features import candidate_features,normalize,physical_outputs

@torch.no_grad()
def infer(model,normalizer,scales,prob_a,prob_j,base,arm):
    start=time.perf_counter()
    all_actions=propose(base,np.asarray(prob_a).argmax(1),np.asarray(prob_j).argmax(1))
    actions,rejected=partition(base,all_actions)
    if actions:
        features=np.stack([candidate_features(prob_a,prob_j,base,a,arm) for a in actions])
        outputs=physical_outputs(model(torch.from_numpy(normalize(features,normalizer))).cpu().numpy(),scales)
    else:
        outputs=np.empty((0,3),np.float64)
    result={p:decode(base,actions,outputs,p) for p in ['O','E','R']}
    result.update({'outputs':outputs,'actions':actions,'all_actions':all_actions,'G_rejected':rejected,
       'cached_feature_head_policy_seconds':time.perf_counter()-start,'full_system_inference_seconds':None,
       'producer_cost_excluded':True,'operational_required_producers':['A08','JEPA']})
    return result
