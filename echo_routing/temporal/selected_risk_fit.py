"""A63 fold-only features/targets/norms with ALL training axes in fold F_fit.

Membership/coverage uses its separate immutable original .8917446116307443;
this fold-specific operating point and common RMS only train auxiliary heads.
"""
import numpy as np
from echo_routing.temporal.anchor_edits import counts
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.proposal_features import fit_normalizer
from echo_routing.temporal.safety_targets import effects,position_labels
from echo_routing.temporal.safety_features import utility_features,position_features,fit_position_normalizer
from echo_routing.temporal.selected_risk_training import validate_training_sources

def prepare_aux_training(sequences,a08,jepa,baselines,train_sessions,held_sessions):
    validate_training_sources(sequences,train_sessions,held_sessions)
    if not len(sequences)==len(a08)==len(jepa)==len(baselines):raise ValueError('complete matched FIT axes required')
    pooled=np.asarray([counts(b,q['labels']) for q,b in zip(sequences,baselines)],np.int64)
    if not len(pooled):raise ValueError('complete complement native plus mixed bank required')
    m,p,t=map(int,pooled.sum(0));f=2*m/(p+t)
    features=[];positions=[];labels=[];targets=[];rows=[];groups=[];offsets=[0];costs=[]
    kinds={k:[] for k in ['native','constructed']};kg={k:[] for k in kinds};all_count=0;rejected_count=0
    for qi,(q,a,j,b) in enumerate(zip(sequences,a08,jepa,baselines)):
        actions=propose(b,np.asarray(a).argmax(1),np.asarray(j).argmax(1));eligible,rejected=partition(b,actions)
        all_count+=len(actions);rejected_count+=len(rejected);group=[];kind='native' if q['kind']=='native' else 'constructed'
        for action in eligible:
            index=len(rows);group.append(index);kinds[kind].append(index)
            x=utility_features(a,j,b,action);px=position_features(a,j,b,action);py=position_labels(b,action,q['labels']);cost=effects(b,action,q['labels'],f)
            if len(px)!=len(py):raise ValueError('changed-position feature/label axis differs')
            features.append(x);positions.append(px);labels.append(py);targets.append([cost['u'],cost['v']]);costs.append(cost);offsets.append(offsets[-1]+len(px))
            rows.append({'sequence_index':qi,'start':action.start,'end':action.end,'replacement':action.replacement,'old':action.old,'jepa':action.jepa,'kind':kind,'changed_positions':len(px),'T':len(b)})
        groups.append(np.asarray(group,np.int64))
        if group:kg[kind].append(qi)
    if not features:raise ValueError('complete G domain empty: no pretend optimizer steps')
    features=np.asarray(features,np.float64);targets=np.asarray(targets,np.float64);pos=np.concatenate(positions)
    scale=max(float(np.sqrt(np.mean(targets**2))),1.)
    # Pair target ties/signs use exact stored values. Assert common scaling preserves them.
    for group in groups:
        for col in [0,1]:
            raw=np.r_[targets[group,col],0.];scaled=raw/scale
            if not np.array_equal(np.sign(raw[:,None]-raw),np.sign(scaled[:,None]-scaled)):raise ValueError('utility scaling alters exact target pair ties/order')
    return {'F_fit':f,'FIT_operating_counts':{'M':m,'P':p,'T':t},'features':features,'position_features':pos,'position_labels':np.concatenate(labels),'position_groups':positions,'position_offsets':np.asarray(offsets,np.int64),'targets':targets,'effects':costs,'candidate_rows':rows,'groups':groups,'kind_indices':{k:np.asarray(v,np.int64) for k,v in kinds.items()},'kind_groups':{k:np.asarray(v,np.int64) for k,v in kg.items()},'normalizers':{'candidate':fit_normalizer(features),'position':fit_position_normalizer(pos)},'scale':scale,'all_candidates':all_count,'G_rejected_candidates':rejected_count}

