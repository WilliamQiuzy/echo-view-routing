"""A62 prespecified endpoints and label-assisted frozen-N-top support.

Support may only be read after outer readouts seal. It cannot replace the
original S-L top or original N. All identifiers and chosen axes stay private.
"""
import numpy as np
from echo_routing.temporal.anchor_oracle import fractional_oracle
from echo_routing.temporal.anchor_edits import counts
from echo_routing.temporal.proposal_actions import apply, partition
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments

SEEDS = (2026110461, 2026110462)
TABLES = ('main','a58_stress','a59_confirmation','a60_confirmation','a61_confirmation','a62_confirmation')
HALVES = ('selection','calibration')
POLICIES = ('E','N','L0','M0','R1','R2')
BASE = 'A08_s0_original_B3'
CONTRASTS = (('R2','Base'),('R1','Base'),('R2','R1'),('R1','M0'),('M0','L0'),('R2','M0'),('R2','N'),('R1','N'),('M0','N'),('N','E'))
SAFETIES = ('unrestricted','net-safe','gross-safe')
SUPPORT_FILTERS = ('event-safe','acceptance-safe')


def system(seed, policy):
    if seed not in SEEDS or policy not in POLICIES:
        raise ValueError('fixed seed and same-top policy required')
    return f'seed{seed}_{policy}_fixed6000'


def plan():
    rows=[]; inner=[]; comparisons=[]; native=[]; base_native=[]; new_oracle=[]; reused=[]; support_cells=[]; regression=[]
    for table in TABLES:
        for half in HALVES:
            base={'table':table,'half':half,'system':BASE,'role':'reference'}
            rows.append(base)
            if table=='main':base_native.append(dict(base))
            for seed in SEEDS:
                for policy in POLICIES:
                    row={'table':table,'half':half,'system':system(seed,policy),'role':'policy','seed':seed,'policy':policy,'step':6000}
                    rows.append(row)
                    if table=='main':native.append(dict(row))
                for left,right in CONTRASTS:
                    primary=table=='a62_confirmation' and (left,right) in [('R1','M0'),('R2','R1')]
                    comparisons.append({'table':table,'half':half,'seed':seed,'left':BASE if left=='Base' else system(seed,left),'right':BASE if right=='Base' else system(seed,right),'contrast':left+'-'+right,'bootstrap_replicates':5000,'primary_family':primary,'primary_quantiles':[.003125,.996875] if primary else None})
                support_cells.extend({'table':table,'half':half,'seed':seed,'safety':s,'domain':'noop_or_original_N_admitted_SL_top','read_after_outer_seal':True} for s in SUPPORT_FILTERS)
            for safety in SAFETIES:
                cell={'table':table,'half':half,'domain':'expanded','safety':safety,'guard':'G','max_actions':1,'read_after_outer_seal':True}
                (new_oracle if table=='a62_confirmation' else reused).append({**cell,'readonly_A61_reference':table!='a62_confirmation'})
    for population in ['CAL','AUDIT']:
        inner.append({'population':population,'system':BASE,'role':'reference'})
        inner.extend({'population':population,'system':system(seed,p),'role':'policy','seed':seed,'policy':p,'step':6000} for seed in SEEDS for p in POLICIES)
    regression.extend({'axis':axis,'seed':seed,'policy':p,'system':system(seed,p),'development_regression_not_independent_holdout':True} for axis in ['Case_A','Case_B'] for seed in SEEDS for p in POLICIES[2:])
    return {'tables':list(TABLES),'rows':rows,'inner_rows':inner,'comparisons':comparisons,'native_cells':native,'base_native_references':base_native,'new_oracle_cells':new_oracle,'reused_oracle_cells':reused,'support_cells':support_cells,'regression_records':regression,'outer_metric_rows':156,'inner_metric_rows':26,'metric_rows':182,'paired_comparisons':240,'bootstrap_replicates':5000,'bootstrap_seed':2026110461,'thresholds':6,'legacy_threshold_references':2,'new_event_models':4,'formal_updates':24000,'fullstate_records':16,'stress_native_pairs_reuse_main':True,'checkpoint_or_seed_selection':False,'primary_system':'R2','primary_comparisons':8,'new_source_streams':540,'new_source_payloads':1080}


def primary_intervals(samples, primary=False):
    """Both intervals use the identical already generated 5000 paired samples.

    Match the historical conditional-on-finite convention: all undefined is
    null, otherwise finite paired replicates and missing counts are disclosed.
    """
    if not isinstance(samples,dict) or not samples:
        raise ValueError('named complete paired samples required')
    out={'delta95':{},'undefined_bootstrap_replicates':{},'bootstrap_replicates':5000,'undefined_ratios':'null if all undefined; otherwise conditional on finite paired replicates with count disclosed'}
    if primary:out['delta99_375']={}
    for key,value in samples.items():
        v=np.asarray(value,dtype=float)
        if v.shape!=(5000,) or np.isinf(v).any():
            raise ValueError('same exact5000 samples, NaN only for undefined ratio required')
        finite=np.isfinite(v);out['undefined_bootstrap_replicates'][key]=int((~finite).sum())
        out['delta95'][key]=np.quantile(v[finite],[.025,.975]).tolist() if finite.any() else None
        if primary:out['delta99_375'][key]=np.quantile(v[finite],[.003125,.996875]).tolist() if finite.any() else None
    return out


def risk_summary(scores, unsafe):
    scores=np.asarray(scores,dtype=float);target=np.asarray(unsafe)
    if scores.ndim!=1 or target.shape!=scores.shape or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any() or not np.isin(target,[0,1]).all():
        raise ValueError('aligned unit-range event scores and binary unsafe events required')
    target=target.astype(bool)
    with np.errstate(divide='ignore',invalid='ignore'):
        loss=np.where(target,-np.log(scores),-np.log1p(-scores))
    bins=[]
    for k in range(5):
        mask=(scores>=k/5)&((scores<(k+1)/5) if k<4 else (scores<=1))
        bins.append({'lower':k/5,'upper':(k+1)/5,'upper_inclusive':k==4,'count':int(mask.sum()),'unsafe':int(target[mask].sum()),'mean_score':float(scores[mask].mean()) if mask.any() else None,'observed_unsafe_fraction':float(target[mask].mean()) if mask.any() else None})
    return {'count':len(scores),'unsafe':int(target.sum()),'BCE':float(loss.mean()) if len(loss) and np.isfinite(loss).all() else None,'infinite_BCE_rows':int(np.isinf(loss).sum()),'Brier':float(np.mean((scores-target.astype(float))**2)) if len(scores) else None,'bins':bins,'scope':'uncalibrated action-unsafe event score; not independent frame trials or clinical probability'}


def _top(value,base):
    actions=value['actions'];eligible,rejected=partition(base,actions)
    if rejected or len(eligible)!=len(actions):raise ValueError('complete canonical G-eligible scored candidates required')
    n=value['N'];top=n['top_index'];accepted=n['accepted_index']
    for i in [top,accepted]:
        if i is not None and (isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<len(actions)):
            raise ValueError('original canonical top cursor required')
    if accepted is not None and accepted!=top:raise ValueError('N cannot replace original top with fallback')
    return None if accepted is None else actions[accepted],accepted


def _zero():
    return {key:0 for key in ['H','R','dE','dFP','u','true_recovered','true_lost','short10_recovered','short10_lost','short16_recovered','short16_lost','JEPA_exclusive_short16_recovered']}


def _sum(rows):
    out=_zero()
    for row in rows:
        for key in out:out[key]+=row[key]
    return out


def _effect(base,action,truth,fit_f1):
    cost=effects(base,action,truth,fit_f1);true=labels_to_segments(truth)
    before=set(match_segments(labels_to_segments(base),true));after=set(match_segments(labels_to_segments(apply(base,action)),true))
    recovered=after-before;lost=before-after
    result={key:cost[key] for key in ['H','R','dE','dFP','u']}
    result.update(true_recovered=len(recovered),true_lost=len(lost))
    for maximum in [10,16]:
        result[f'short{maximum}_recovered']=sum(true[i].length<=maximum for i in recovered)
        result[f'short{maximum}_lost']=sum(true[i].length<=maximum for i in lost)
    result['JEPA_exclusive_short16_recovered']=result['short16_recovered'] if action.jepa and not action.old else 0
    return result,recovered,lost,true


def support(sequences, baselines, values, safety, fit_f1, private_details=False):
    """Exact multi-only F1 over {noop, originalNtop}, never another candidate.

    Native and pair top support are separate counts. Union counts include all
    available tops, while joint counts follow exact deterministic F1 choices.
    No learned threshold is claimed able to realize these label-assisted choices.
    """
    if safety not in SUPPORT_FILTERS or not len(sequences)==len(baselines)==len(values) or not np.isfinite(fit_f1) or not 0<=fit_f1<=1:
        raise ValueError('complete aligned declared support filter and fixedFIT F1 required')
    groups={kind:[] for kind in ['native','multi','pair']};ids=set()
    for q,base,value in zip(sequences,baselines,values):
        if q['id'] in ids or q['kind'] not in groups:raise ValueError('unique original axes and declared native/multi/pair kinds required')
        ids.add(q['id']);action,index=_top(value,base);record={'N':action is not None,'available':False,'index':None,'options':[],'effect':None,'N_effect':None,'recovered':set(),'true':[]}
        m,p,t=counts(base,q['labels']);record['options'].append({'M':m,'P':p,'T':t,'span':None})
        if action is not None:
            cost,recovered,lost,true=_effect(base,action,q['labels'],fit_f1)
            record['N_effect']=cost
            allowed=cost['H']==0 and cost['dFP']<=0 and (safety=='event-safe' or cost['short16_lost']==0)
            if allowed:
                m,p,t=counts(apply(base,action),q['labels']);record['options'].append({'M':m,'P':p,'T':t,'span':action.key})
                record.update(available=True,index=index,effect=cost,recovered=recovered,true=true)
        groups[q['kind']].append(record)
    out={'safety':safety,'scope':'label-assisted original N-admitted S-L top or noop; fixed candidate/G/maxONE; pooled objective multi only','q_threshold_representability_guaranteed':False,'union_is_jointly_reachable_claim':False,'unsafe_or_short_losing_policy_is_bounded_by_safe_oracle':False}
    private={}
    for kind,records in groups.items():
        available=[r for r in records if r['available']]
        row={'axes':len(records),'N_admitted_tops':sum(r['N'] for r in records),'available_tops':len(available),'all_N_top_effects':_sum(r['N_effect'] for r in records if r['N']),'positive_u_available_tops':sum(r['effect']['u']>0 for r in available),'all_available_top_effects':_sum(r['effect'] for r in available),'union_recovered_true_units':sum(len(r['recovered']) for r in available)}
        for maximum in [10,16]:row[f'union_recovered_short{maximum}_true_units']=sum(sum(r['true'][i].length<=maximum for i in r['recovered']) for r in available)
        if kind=='multi':
            if records:
                solved=fractional_oracle([r['options'] for r in records]);choices=solved.pop('choices')
                joint=[r for r,chosen in zip(records,choices) if chosen==1]
                row['oracle']={**solved,'scope':'exact pooled multi IoU50 F1; deterministic noop-first ties over frozenNtop-or-noop only'}
                row['joint_chosen_tops']=len(joint);row['joint_chosen_effects']=_sum(r['effect'] for r in joint)
                private['multi_chosen_accepted_indices']=[r['index'] if chosen==1 else None for r,chosen in zip(records,choices)]
            else:
                row.update(oracle=None,joint_chosen_tops=0,joint_chosen_effects=_zero());private['multi_chosen_accepted_indices']=[]
        else:row['pooled_F1_objective_member']=False
        out[kind]=row
    if private_details:out['private_details']=private
    return out


def evaluate(predictions, context, baselines):
    # Heavy historical task evaluator loads only in the existing server runtime.
    from echo_routing.temporal.safety_metrics import evaluate as inherited
    return inherited(predictions, context, baselines)


def prepare_system(predictions, baselines, sequences, tables):
    from echo_routing.temporal.safety_metrics import prepare_system as inherited
    return inherited(predictions, baselines, sequences, tables)


def paired(left, right, n_boot=5000, seed=2026110461, primary=False):
    """Historical whole-unit pooled-count backend with one shared draw per unit.

    Generate each session/stream/pair weight matrix once and reuse for all task,
    short-segment and gross-frame metrics and both confidence levels.
    """
    from echo_routing.temporal.structured_uncertainty import _weights, _ratio, _macro
    if n_boot!=5000 or left['session_order']!=right['session_order'] or not left['session_order'] or set(left['native'])!=set(right['native']) or left['true_lengths']!=right['true_lengths']:
        raise ValueError('exact5000 paired original units/truth identity required')
    def matrix(v,n,width):
        v=np.asarray(v)
        if v.shape!=(n,width) or not np.issubdtype(v.dtype,np.integer) or (v<0).any():
            raise ValueError('complete nonnegative exact integer whole-unit count matrix required')
        return v
    rng=np.random.default_rng(seed);delta={};n=len(left['session_order']);w=_weights(rng,n,n_boot)
    a=(w@matrix(left['session_confusion'].reshape(n,25),n,25)).reshape(n_boot,5,5)
    b=(w@matrix(right['session_confusion'].reshape(n,25),n,25)).reshape(n_boot,5,5)
    delta['native_macro_f1']=_macro(a)-_macro(b)
    delta['native_accuracy']=_ratio(np.trace(a,axis1=1,axis2=2),a.sum((1,2)))-_ratio(np.trace(b,axis1=1,axis2=2),b.sum((1,2)))
    ae=w@matrix(left['session_extra'],n,4);be=w@matrix(right['session_extra'],n,4)
    delta['native_internal_cuts_per_clip']=_ratio(ae[:,0],ae[:,3])-_ratio(be[:,0],be[:,3])
    delta['native_frame_accuracy']=_ratio(ae[:,2],ae[:,1])-_ratio(be[:,2],be[:,1])
    delta['native_frame_errors']=(ae[:,1]-ae[:,2])-(be[:,1]-be[:,2])
    gross=w@(matrix(left['session_gross'],n,2)-matrix(right['session_gross'],n,2))
    for i,key in enumerate(['new_wrong_reference_positions','repaired_reference_positions']):delta['native_'+key]=gross[:,i]
    delta['native_net_frame_error_delta']=gross[:,0]-gross[:,1]
    for kind in ['multi','pair']:
        la,lb=left['streams'][kind],right['streams'][kind];n=len(la)
        if not n or [r[0] for r in la]!=[r[0] for r in lb] or [r[0] for r in left['stream_gross'][kind]]!=[r[0] for r in la] or [r[0] for r in right['stream_gross'][kind]]!=[r[0] for r in la]:
            raise ValueError('complete paired stream/pair and gross order required')
        w=_weights(rng,n,n_boot);a=w@matrix(np.stack([r[1] for r in la]),n,18);b=w@matrix(np.stack([r[1] for r in lb]),n,18)
        suffix='multi' if kind=='multi' else 'pairs'
        delta['segment_f1_50_'+suffix]=_ratio(2*a[:,0],a[:,1]+a[:,2])-_ratio(2*b[:,0],b[:,1]+b[:,2])
        delta['boundary_recall_'+suffix]=_ratio(a[:,5],a[:,6])-_ratio(b[:,5],b[:,6])
        delta['boundary_precision_'+suffix]=_ratio(a[:,5],a[:,7])-_ratio(b[:,5],b[:,7])
        delta['same_class_cut_rate_'+suffix]=_ratio(a[:,8],a[:,9])-_ratio(b[:,8],b[:,9])
        delta[suffix+'_frame_accuracy']=_ratio(a[:,4],a[:,3])-_ratio(b[:,4],b[:,3])
        delta[suffix+'_frame_errors']=(a[:,3]-a[:,4])-(b[:,3]-b[:,4])
        for maximum,start in [(10,10),(16,14)]:
            prefix='' if kind=='multi' else 'pairs_'
            delta[f'{prefix}short_true_unmatched_{maximum}samples']=a[:,start+1]-b[:,start+1]
            delta[f'{prefix}short_spurious_segments_{maximum}samples']=a[:,start+3]-b[:,start+3]
        gross=w@(matrix(np.stack([r[1] for r in left['stream_gross'][kind]]),n,2)-matrix(np.stack([r[1] for r in right['stream_gross'][kind]]),n,2))
        for i,key in enumerate(['new_wrong_reference_positions','repaired_reference_positions']):delta[kind+'_'+key]=gross[:,i]
        delta[kind+'_net_frame_error_delta']=gross[:,0]-gross[:,1]
    exact={key:0 for key in ['native_left_only_correct','native_right_only_correct','native_both_wrong','native_both_correct']}
    for key,(lab,a) in left['native'].items():
        other,b=right['native'][key]
        if lab!=other:raise ValueError('native paired class differs')
        x,y=a==lab,b==lab
        key='native_both_correct' if x and y else 'native_both_wrong' if not x and not y else 'native_left_only_correct' if x else 'native_right_only_correct'
        exact[key]+=1
    for kind in ['multi','pair']:
        if set(left['matched'][kind])!=set(right['matched'][kind]):raise ValueError('paired matched stream identity differs')
        for label in ['recovered','lost']:
            exact[f'{kind}_true_segments_{label}']=0
            for maximum in [10,16]:exact[f'{kind}_short_true_segments_{label}_{maximum}samples']=0
        for key,a in left['matched'][kind].items():
            b=right['matched'][kind][key]
            for label,indices in [('recovered',a-b),('lost',b-a)]:
                exact[f'{kind}_true_segments_{label}']+=len(indices)
                for maximum in [10,16]:exact[f'{kind}_short_true_segments_{label}_{maximum}samples']+=sum(left['true_lengths'][key][i]<=maximum for i in indices)
    return {**primary_intervals(delta,primary=primary),'exact_disagreements':exact,'bootstrap':{'replicates':5000,'seed':seed,'native_sessions':len(left['session_order']),'multi_streams':len(left['streams']['multi']),'pair_streams':len(left['streams']['pair']),'units':'complete sessions / complete fixed streams / complete fixed pairs','weight_matrices_generated_once_per_unit_kind':True,'scope':'conditional fixed banks with shared sources; sessions not confirmed patients; no independent clinical population claim','primary_family_quantiles':[.003125,.996875] if primary else None}}
