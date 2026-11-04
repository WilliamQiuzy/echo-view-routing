"""A62 anonymous native cubes and two development-regression axes; no NN/I/O.

Raw sequence identifiers are never returned. RLE helper output is server-only.
Candidate potential harm remains distinct from actual gated output harm.
"""
import hashlib,json,math
import numpy as np
from echo_routing.temporal.event_risk_policy import decode,POLICIES
from echo_routing.temporal.event_risk_features import context_features
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.proposal_native_attribution import joint_counts,effects as native_effects
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.anchor_edits import majority,aligned
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments
from echo_routing.temporal.windowed import FAMILY5

TOTALS=('positions','Base_correct_to_wrong','Base_wrong_to_correct','both_correct','both_wrong_unchanged_class','both_wrong_changed_class','changed_positions','Base_frame_errors','new_frame_errors','frame_error_delta')
EXTRA=('added_cuts','newly_wrong_positions_extending_old_wrong_run','newly_wrong_positions_new_wrong_segment','newly_wrong_positions_in_long_wrong_segments','wrong_run_crossed_short16_to_long','new_wrong_positions_touch_long_wrong_segment')


def _fit(value):
    if isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('fixed finite FIT marginal reference required')


def _decode(base,value):
    if any(not (a.old or a.jepa) for a in value['actions']):raise ValueError('original old/JEPA candidate provenance required')
    old=np.asarray(value['old_scores'],float);q=np.asarray(value['q'],float)
    if old.shape!=(len(value['actions']),6) or q.shape!=(len(value['actions']),2):raise ValueError('complete originalC6/newC2 and original candidate axes required')
    taus={p:value[p]['tau'] for p in POLICIES[2:]};expected=decode(base,value['actions'],old,q[:,0],q[:,1],taus)
    for p,row in expected.items():
        actual=value[p]
        for name in ['top_index','accepted_index']:
            i=actual[name]
            if isinstance(i,(bool,np.bool_)) or i is not None and not isinstance(i,(int,np.integer)) or i!=row[name]:raise ValueError('same canonical top/no-fallback decision differs from frozen gate')
        if not np.array_equal(actual['prediction'],row['prediction']):raise ValueError('decoded whole-axis prediction differs from fixed score/gate')
    return expected


def _native_inputs(sequences,bases):
    if not sequences or len(sequences)!=len(bases):raise ValueError('complete nonempty aligned native axes required')
    for q,b in zip(sequences,bases):
        truth,b=aligned(q['labels'],b)
        if q['kind']!='native' or not np.all(truth==truth[0]):raise ValueError('original whole native constant propagated references required')


def _result(cube,totals,sequences,changed,accepted,groups,policy):
    if int(cube.sum())!=totals['positions'] or totals['Base_correct_to_wrong']-totals['Base_wrong_to_correct']!=totals['frame_error_delta']:raise ValueError('native joint-cube/gross algebra failed')
    return {'policy':policy,'true_Base_prediction_joint5cube':cube.tolist(),**totals,'native_sequences':len(sequences),'changed_native_sequences':changed,'majority_changed_native_sequences':0,'accepted_native_actions':accepted,'accepted_groups':groups,'true_Base_frame_confusion5':cube.sum(2).tolist(),'true_prediction_frame_confusion5':cube.sum(1).tolist(),'Base_prediction_frame_transition5':cube.sum(0).tolist(),'all_native_majorities_mechanically_equal_Base':True,'majority_equality_is_frame_safety_claim':False,'classes':list(FAMILY5),'new_source_forwards':0,'new_optimizer_updates':0,'reference_scope':'propagated cine-family reference positions; no expert visibility/clinical harm adjudication','native_and_pair_reuse_across_tables_is_independent_evidence':False}


def native_reference(sequences,bases,fit_f1):
    _fit(fit_f1);_native_inputs(sequences,bases);cube=np.zeros((5,5,5),np.int64);totals={k:0 for k in TOTALS}
    for q,b in zip(sequences,bases):
        row=joint_counts(q['labels'],b,b);cube+=np.asarray(row['true_Base_prediction_joint5cube'])
        for k in TOTALS:totals[k]+=row[k]
    return _result(cube,totals,sequences,0,0,[],'Base')


def native_cube(sequences,bases,values,policy,fit_f1):
    _fit(fit_f1);_native_inputs(sequences,bases)
    if policy not in POLICIES or len(values)!=len(sequences):raise ValueError('complete six-policy native value axes required')
    cube=np.zeros((5,5,5),np.int64);totals={k:0 for k in TOTALS};changed=accepted=0;groups={}
    for q,b,v in zip(sequences,bases,values):
        pred=_decode(b,v)[policy];p=pred['prediction'];row=joint_counts(q['labels'],b,p);cube+=np.asarray(row['true_Base_prediction_joint5cube'])
        for k in TOTALS:totals[k]+=row[k]
        changed+=row['changed_positions']>0
        if majority(b)!=majority(p):raise ValueError('native output changed whole-axis majority despite fixedG')
        index=pred['accepted_index']
        if index is None:
            if not np.array_equal(p,b):raise ValueError('no-op native differs from Base')
            continue
        a=v['actions'][index];cost=effects(b,a,q['labels'],fit_f1);extra=native_effects(b,p,q['labels'],fit_f1);mask=np.asarray(b)!=p;accepted+=1
        if cost['H']!=row['Base_correct_to_wrong'] or cost['R']!=row['Base_wrong_to_correct'] or cost['dE']!=row['frame_error_delta'] or cost['dFP']!=extra['dFP16'] or cost['u']!=extra['u'] or extra['newly_wrong_positions_extending_old_wrong_run']+extra['newly_wrong_positions_new_wrong_segment']!=cost['H']:raise ValueError('native action/whole-axis/joint/length effects disagree')
        provenance='both' if a.old and a.jepa else 'old_only' if a.old else 'JEPA_only';key=(int(q['labels'][0]),tuple(np.unique(np.asarray(b)[mask])),tuple(np.unique(p[mask])),provenance)
        if key not in groups:
            groups[key]={'true_class':FAMILY5[key[0]],'Base_changed_classes':[FAMILY5[int(c)] for c in key[1]],'prediction_changed_classes':[FAMILY5[int(c)] for c in key[2]],'provenance':provenance,'actions':0,'interval_length_sum':0,'changed_positions_sum':0,**{'actual_'+k:0 for k in ['u','v','H','R','dE','dFP','delta_M','delta_P']},**{k:0 for k in EXTRA},'predicted_utility_sum':0.,'old_predicted_H_sum':0.,'old_predicted_R_sum':0.,'old_pFP_sum':0.,'old_r_sum':0.,'q_R1_sum':0.,'q_R2_sum':0.}
        g=groups[key];g['actions']+=1;g['interval_length_sum']+=a.end-a.start;g['changed_positions_sum']+=int(mask.sum())
        for k in ['u','v','H','R','dE','dFP','delta_M','delta_P']:g['actual_'+k]+=cost[k]
        for k in EXTRA:g[k]+=int(extra[k])
        old=v['old_scores'][index];qscore=v['q'][index]
        for name,value in [('predicted_utility_sum',old[0]),('old_predicted_H_sum',old[1]),('old_predicted_R_sum',old[2]),('old_pFP_sum',old[4]),('old_r_sum',old[5]),('q_R1_sum',qscore[0]),('q_R2_sum',qscore[1])]:g[name]+=float(value)
    if changed!=accepted:raise ValueError('maxONE/native nonempty-action count differs from changed axes')
    return _result(cube,totals,sequences,int(changed),accepted,[groups[k] for k in sorted(groups)],policy)


def regression_records(axis,sequence,base,prob_a,prob_j,value,seed,fit_f1):
    _fit(fit_f1)
    if axis not in ['Case_A','Case_B'] or seed not in [2026110461,2026110462]:raise ValueError('only two anonymous development axes and fixedseeds allowed')
    decoded=_decode(base,value);top=decoded['E']['top_index'];original_N=decoded['N']['accepted_index'] is not None
    if top is None:cost=None;lost=None;summary=None;keysha=None;span=None
    else:
        a=value['actions'][top];cost=effects(base,a,sequence['labels'],fit_f1);context=context_features(prob_a,prob_j,base,a);summary={'minimum_overlap_over_full_run':float(context[91]),'minimum_full_Base_run_A08_margin':float(context[93])};keysha=hashlib.sha256(json.dumps(a.key,separators=(',',':')).encode()).hexdigest();span={'start':a.start,'end':a.end,'changed_positions':int(np.count_nonzero(apply(base,a)!=base)),'old':a.old,'jepa':a.jepa}
        true=labels_to_segments(sequence['labels']);before=set(match_segments(labels_to_segments(base),true));after=set(match_segments(labels_to_segments(apply(base,a)),true));lost=sum(true[i].length<=16 for i in before-after)
    rows=[]
    for policy in POLICIES[2:]:
        row=decoded[policy];accepted=row['accepted_index'] is not None
        rows.append({'axis':axis,'seed':int(seed),'policy':policy,'top_exists':top is not None,'original_N_admitted':original_N,'accepted':accepted,'gate_score':row['gate_score'],'tau':row['tau'],'candidate_key_sha256':keysha,'candidate_action':span,'H':None if cost is None else int(cost['H']),'R':None if cost is None else int(cost['R']),'dFP16':None if cost is None else int(cost['dFP']),'short16_lost':lost,'actual_H':int(cost['H']) if accepted else 0,'actual_R':int(cost['R']) if accepted else 0,'actual_dFP16':int(cost['dFP']) if accepted else 0,'actual_short16_lost':int(lost) if accepted else 0,'context_summary':summary,'scope':'anonymous development regression; candidate potential and actual gated effects separate; two axes not independent heldout cases'})
    return rows


def regression_rle(sequence,base,value):
    decoded=_decode(base,value)
    def rows(labels):return [{'start':s.start,'end':s.end,'label':int(s.label)} for s in labels_to_segments(labels)]
    return {'truth':rows(sequence['labels']),'Base':rows(base),'predictions':{p:rows(decoded[p]['prediction']) for p in POLICIES},'scope':'server-only development RLE; never export raw ID/media or per-case arrays in public artifacts'}
