"""A61 cached native joint transitions and accepted-action gross/gate attribution."""
import numpy as np
from echo_routing.temporal.proposal_native_attribution import joint_counts,effects as native_effects
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.windowed import FAMILY5


def provenance(a):return 'both' if a.old and a.jepa else 'old_only' if a.old else 'JEPA_only'


def diagnose(sequences,baselines,values,policy,arm,fit_f1,legacy=False):
    if not len(sequences)==len(baselines)==len(values):raise ValueError('complete native source/value axes required')
    cube=np.zeros((5,5,5),np.int64);totals={k:0 for k in ['positions','Base_correct_to_wrong','Base_wrong_to_correct','both_correct','both_wrong_unchanged_class','both_wrong_changed_class','changed_positions','Base_frame_errors','new_frame_errors','frame_error_delta']};changed=majority_changed=accepted=0;groups={}
    for q,b,value in zip(sequences,baselines,values):
        if q['kind']!='native' or not np.all(np.asarray(q['labels'])==q['labels'][0]):raise ValueError('native constant propagated reference only')
        p=value[policy]['prediction'];joint=joint_counts(q['labels'],b,p);cube+=np.asarray(joint['true_Base_prediction_joint5cube']);changed+=joint['changed_positions']>0;majority_changed+=majority(b)!=majority(p)
        for key in totals:totals[key]+=joint[key]
        index=value[policy]['accepted_index']
        if index is None:
            if not np.array_equal(p,b):raise ValueError('no-op native differs from Base')
            continue
        a=value['actions'][index];accepted+=1;cost=effects(b,a,q['labels'],fit_f1);extra=native_effects(b,p,q['labels'],fit_f1);mask=p!=b;key=(int(q['labels'][0]),tuple(np.unique(np.asarray(b)[mask]).tolist()),tuple(np.unique(p[mask]).tolist()),provenance(a))
        if key not in groups:
            groups[key]={'true_class':FAMILY5[key[0]],'Base_changed_classes':[FAMILY5[k] for k in key[1]],'prediction_changed_classes':[FAMILY5[k] for k in key[2]],'provenance':key[3],'actions':0,'interval_length_sum':0,'changed_positions_sum':0,**{'actual_'+k:0 for k in ['u','v','H','R','dE','dFP','delta_M','delta_P']},**{k:0 for k in ['added_cuts','newly_wrong_positions_extending_old_wrong_run','newly_wrong_positions_new_wrong_segment','newly_wrong_positions_in_long_wrong_segments','wrong_run_crossed_short16_to_long','new_wrong_positions_touch_long_wrong_segment']},'positive_safety_score_false_accepts':0,'predicted_utility_sum':0.,'predicted_H':None if legacy else 0.,'predicted_R':None if legacy else 0.,'predicted_pFP_sum':None if legacy else 0.,'risk_evidence_sum':None if legacy else 0.,'predicted_signed_dE_sum':0. if legacy else None,'predicted_signed_dFP16_sum':0. if legacy else None}
        row=groups[key];row['actions']+=1;row['interval_length_sum']+=a.end-a.start;row['changed_positions_sum']+=int(mask.sum())
        for k in ['u','v','H','R','dE','dFP','delta_M','delta_P']:row['actual_'+k]+=cost[k]
        for k in ['added_cuts','newly_wrong_positions_extending_old_wrong_run','newly_wrong_positions_new_wrong_segment','newly_wrong_positions_in_long_wrong_segments','wrong_run_crossed_short16_to_long','new_wrong_positions_touch_long_wrong_segment']:row[k]+=int(extra[k])
        if cost['H']!=joint['Base_correct_to_wrong'] or cost['R']!=joint['Base_wrong_to_correct'] or cost['dE']!=joint['frame_error_delta'] or cost['u']!=extra['u'] or cost['dFP']!=extra['dFP16'] or extra['newly_wrong_positions_extending_old_wrong_run']+extra['newly_wrong_positions_new_wrong_segment']!=cost['H']:raise ValueError('native joint/fullaxis/action-cost recount differs')
        score=value['outputs' if legacy else 'score_matrix'][index];row['predicted_utility_sum']+=float(score[0])
        if legacy:
            row['predicted_signed_dE_sum']+=float(score[1]);row['predicted_signed_dFP16_sum']+=float(score[2])
            if policy=='R' and (score[0]<=0 or score[1]>0 or score[2]>0):raise ValueError('original A60 R accepted a predicted-vetoed action')
        else:
            h,r,fp,evidence=map(float,score[[1,2,4,5]]);row['predicted_H']+=h;row['predicted_R']+=r;row['predicted_pFP_sum']+=fp;row['risk_evidence_sum']+=evidence;row['positive_safety_score_false_accepts']+=arm.startswith('S') and score[0]>0 and cost['v']<0
            if policy in ['N','C'] and (h-r>0 or fp>=.5):raise ValueError('accepted native violates frozen N prediction gate')
            if policy=='C' and not evidence<value['tau']:raise ValueError('accepted native violates strict frozen tau')
    if majority_changed or accepted!=changed or totals['Base_correct_to_wrong']-totals['Base_wrong_to_correct']!=totals['frame_error_delta']:raise ValueError('native majority/accepted/gross algebra failed')
    return {'true_Base_prediction_joint5cube':cube.tolist(),**totals,'native_sequences':len(sequences),'changed_native_sequences':int(changed),'majority_changed_native_sequences':int(majority_changed),'accepted_native_actions':int(accepted),'accepted_groups':list(groups.values()),'prediction_schema':'A60_original_signed_u_dE_dFP' if legacy else 'A61_u_or_v_grossH_R_pFP_uncalibrated_r','all_native_majorities_mechanically_equal_Base':True,'reference_scope':'propagated native class references,notexpertframevisibility/clinicalharm','new_source_forwards':0,'new_optimizer_updates':0}
