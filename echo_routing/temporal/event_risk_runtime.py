"""A62 label-free scalar event evidence on unchanged canonical A61 candidates."""
import hashlib,time
import numpy as np
import torch
from echo_routing.temporal.proposal_actions import propose,partition
from echo_routing.temporal.proposal_features import candidate_features
from echo_routing.temporal.event_risk_features import context_features,event_matrix


def C6_sha(scores):
    a=np.asarray(scores,np.float64)
    if a.ndim!=2 or a.shape[1]!=6 or not np.isfinite(a).all():raise ValueError('finite original C6 score matrix required')
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


@torch.no_grad()
def infer_events(heads,norms,prob_a,prob_j,base,old_scores,old_reference):
    if torch.cuda.is_initialized() or set(heads)!={'R1','R2'} or heads['R1'].seed!=heads['R2'].seed or any(heads[a].arm!=a for a in heads):raise ValueError('same-seed matched CPU event heads required')
    start=time.perf_counter();pa,pj=np.asarray(prob_a),np.asarray(prob_j);all_actions=propose(base,pa.argmax(1),pj.argmax(1));actions,rejected=partition(base,all_actions);old=np.asarray(old_scores,np.float64)
    if old.shape!=(len(actions),6) or not np.isfinite(old).all():raise ValueError('same complete canonical original C6 axis required')
    if actions:
        raw=np.stack([candidate_features(pa,pj,base,a,'J') for a in actions]);context=np.stack([context_features(pa,pj,base,a) for a in actions]);scores=[]
        for arm in ['R1','R2']:
            x=event_matrix(raw,None if arm=='R1' else context,arm,norms['candidate'],None if arm=='R1' else norms['context']);logit=heads[arm](torch.from_numpy(x))
            if not bool(torch.isfinite(logit).all()):raise ValueError('nonfinite action-event output')
            scores.append(logit.sigmoid().cpu().numpy().astype(np.float64))
        q=np.column_stack(scores)
    else:q=np.empty((0,2),np.float64)
    ref={**old_reference,'scores_sha256':C6_sha(old)}
    return {'old_scores':old,'q':q,'old_reference':ref,'actions':actions,'all_actions':all_actions,'G_rejected':rejected,'seconds':time.perf_counter()-start,'new_event_MLP_forwards':2*bool(actions),'old_fixed_MLP_forwards':0,'teacher_forwards':0}
