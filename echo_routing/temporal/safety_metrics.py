"""A61 complete-task gross harm, accepted actions, risk and within-query diagnostics."""
import numpy as np
from echo_routing.temporal.proposal_metrics import evaluate as historical_evaluate,paired as historical_paired,describe,distribution
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.anchor_edits import aligned
from echo_routing.temporal.structured_uncertainty import prepare_system as historical_prepare,_weights
from echo_routing.evaluate.failure_units import session_of,match_segments
from echo_routing.temporal.segments import labels_to_segments


def gross_one(truth,base,prediction):
    base,truth=aligned(base,truth);prediction,truth=aligned(prediction,truth)
    h=(base==truth)&(prediction!=truth);r=(base!=truth)&(prediction==truth);changed=base!=prediction
    return {'H':int(h.sum()),'R':int(r.sum()),'dE':int(h.sum()-r.sum()),'changed_positions':int(changed.sum()),'changed_sequences':int(changed.any()),'wrong_to_other_wrong':int(np.count_nonzero(changed&(base!=truth)&(prediction!=truth))),'H_by_true_class':np.bincount(truth[h],minlength=5).tolist(),'R_by_true_class':np.bincount(truth[r],minlength=5).tolist(),'reference_positions':len(truth)}


def gross_frames(sequences,baselines,predictions):
    out={kind:{'H':0,'R':0,'dE':0,'changed_positions':0,'changed_sequences':0,'wrong_to_other_wrong':0,'H_by_true_class':[0]*5,'R_by_true_class':[0]*5,'reference_positions':0} for kind in ['native','multi','pair']}
    for q in sequences:
        kind=q['kind'] if q['kind'] in out else 'pair';row=gross_one(q['labels'],baselines[q['id']],predictions[q['id']])
        for key,value in row.items():
            if isinstance(value,list):out[kind][key]=(np.asarray(out[kind][key])+value).tolist()
            else:out[kind][key]+=value
    return out


def evaluate(predictions,context,baselines):
    result,tables=historical_evaluate(predictions,context);gross=gross_frames(context['sequences'],baselines,predictions)
    result['gross_reference_position_effects']=gross
    for kind,row in gross.items():
        result[kind+'_new_wrong_reference_positions']=row['H'];result[kind+'_repaired_reference_positions']=row['R']
    return result,tables


def prepare_system(predictions,baselines,sequences,tables):
    bundle=historical_prepare(predictions,sequences,tables);sessions={};streams={'multi':[],'pair':[]}
    for q in sequences:
        row=gross_one(q['labels'],baselines[q['id']],predictions[q['id']]);v=np.array([row['H'],row['R']],np.int64)
        if q['kind']=='native':
            session=session_of(q['id']);sessions.setdefault(session,np.zeros(2,np.int64));sessions[session]+=v
        else:streams['multi' if q['kind']=='multi' else 'pair'].append((q['id'],v))
    bundle['session_gross']=np.stack([sessions[s] for s in bundle['session_order']]);bundle['stream_gross']=streams
    return bundle


def gross_intervals(left,right,n_boot=5000,seed=2026110461):
    if left['session_order']!=right['session_order']:raise ValueError('gross native session order differs')
    rng=np.random.default_rng(seed);out={};w=_weights(rng,len(left['session_order']),n_boot);d=w@(left['session_gross']-right['session_gross'])
    for i,name in enumerate(['new_wrong_reference_positions','repaired_reference_positions']):out['native_'+name]=np.quantile(d[:,i],[.025,.975]).tolist()
    for kind in ['multi','pair']:
        a,b=left['stream_gross'][kind],right['stream_gross'][kind]
        if [x[0] for x in a]!=[x[0] for x in b]:raise ValueError('gross whole stream/pair order differs')
        w=_weights(rng,len(a),n_boot);d=w@(np.stack([x[1] for x in a])-np.stack([x[1] for x in b]))
        for i,name in enumerate(['new_wrong_reference_positions','repaired_reference_positions']):out[kind+'_'+name]=np.quantile(d[:,i],[.025,.975]).tolist()
    return out


def paired(left,right,n_boot=5000,seed=2026110461):
    result=historical_paired(left,right,n_boot=n_boot,seed=seed);extra=gross_intervals(left,right,n_boot,seed)
    result['delta95'].update(extra);result['undefined_bootstrap_replicates'].update({k:0 for k in extra})
    return result


def action_summary(sequences,baselines,values,policy,arm,fit_f1):
    if policy not in ['E','N','C'] or arm not in ['U-P','U-L','S-P','S-L'] or not len(sequences)==len(baselines)==len(values):raise ValueError('complete declared fixed-policy source axes required')
    counts={k:0 for k in ['candidate_count','G_eligible_scored_candidates','G_vetoed_candidates','positive_utility_candidates','zero_utility_candidates','negative_utility_candidates','top_positive_sequences','vetoed_top_sequences','N_admitted_tops','C_extra_vetoes','accepted_actions','accepted_gross_harm_actions','accepted_net_harm_actions','accepted_short16_FP_increase_actions','accepted_unsafe_actions','positive_safety_score_false_accepts','accepted_negative_count_utility_actions','accepted_new_JEPA_domain_actions','new_JEPA_domain_candidates','old_provenance_candidates','JEPA_provenance_candidates','both_provenance_candidates','accepted_true_units_recovered','accepted_true_units_lost','accepted_new_JEPA_true_units_recovered','accepted_new_JEPA_short16_true_units_recovered']}
    for n in [10,16]:
        for label in ['recovered','lost']:counts[f'accepted_short{n}_true_units_{label}']=0
    bins=[{'lower':k/5,'upper':(k+1)/5,'upper_inclusive':k==4,'candidates':0,'unsafe_candidates':0,'actual_H':0,'actual_R':0,'predicted_H':0.,'predicted_R':0.,'risk_evidence_sum':0.} for k in range(5)]
    all_items=[];eligible_items=[];accepted_items=[];accepted_totals={k:0 for k in ['u','v','H','R','dE','dFP','delta_M','delta_P']};errors=[];riskerrors=[];top_regret=[];post_regret=[];pair_correct=pair_count=pair_ties=0;query_accuracies=[];query_mse=[]
    for q,b,value in zip(sequences,baselines,values):
        actions=value['actions'];scores=np.asarray(value['score_matrix']);chosen=value[policy]['accepted_index'];top=value[policy]['top_index'];C=len(actions)
        if scores.shape!=(C,6) or not np.isfinite(scores).all():raise ValueError('same C6 source score matrix required')
        counts['candidate_count']+=len(value['all_actions']);counts['G_eligible_scored_candidates']+=C;counts['G_vetoed_candidates']+=len(value['G_rejected']);all_items.extend((b,a) for a in value['all_actions']);eligible_items.extend((b,a) for a in actions)
        counts['positive_utility_candidates']+=int((scores[:,0]>0).sum());counts['zero_utility_candidates']+=int((scores[:,0]==0).sum());counts['negative_utility_candidates']+=int((scores[:,0]<0).sum());counts['top_positive_sequences']+=top is not None;counts['vetoed_top_sequences']+=top is not None and chosen is None;counts['N_admitted_tops']+=value[policy]['N_admitted'];counts['C_extra_vetoes']+=policy=='C' and value[policy]['N_admitted'] and chosen is None
        for a in value['all_actions']:
            counts['new_JEPA_domain_candidates']+=a.jepa and not a.old;counts['old_provenance_candidates']+=a.old;counts['JEPA_provenance_candidates']+=a.jepa;counts['both_provenance_candidates']+=a.old and a.jepa
        costs=[effects(b,a,q['labels'],fit_f1) for a in actions];target=np.asarray([c['v' if arm.startswith('S') else 'u'] for c in costs]);actual=np.r_[target,0.];pred=np.r_[scores[:,0],0.]
        best=max(actual);top_actual=0. if top is None else target[top];post_actual=0. if chosen is None else target[chosen];top_regret.append(best-top_actual);post_regret.append(best-post_actual)
        i,j=np.triu_indices(C+1,1);keep=actual[i]!=actual[j];i,j=i[keep],j[keep];truthsign=np.sign(actual[i]-actual[j]);predictsign=np.sign(pred[i]-pred[j]);correct=int((truthsign==predictsign).sum());ties=int((predictsign==0).sum());pair_count+=len(i);pair_correct+=correct;pair_ties+=ties
        if len(i):query_accuracies.append(correct/len(i))
        if C:
            error=scores[:,0]-target;errors.extend(error.tolist());query_mse.append(float(np.mean(error**2)))
        for index,(a,cost,score) in enumerate(zip(actions,costs,scores)):
            riskerrors.append([score[1]-cost['H'],score[2]-cost['R']]);k=min(int(score[5]*5),4);row=bins[k];row['candidates']+=1;row['unsafe_candidates']+=not cost['safe'];row['actual_H']+=cost['H'];row['actual_R']+=cost['R'];row['predicted_H']+=score[1];row['predicted_R']+=score[2];row['risk_evidence_sum']+=score[5]
            if chosen!=index:continue
            accepted_items.append((b,a));counts['accepted_actions']+=1;counts['accepted_gross_harm_actions']+=cost['H']>0;counts['accepted_net_harm_actions']+=cost['dE']>0;counts['accepted_short16_FP_increase_actions']+=cost['dFP']>0;counts['accepted_unsafe_actions']+=not cost['safe'];counts['positive_safety_score_false_accepts']+=arm.startswith('S') and score[0]>0 and cost['v']<0;counts['accepted_negative_count_utility_actions']+=cost['u']<0
            for key in accepted_totals:accepted_totals[key]+=cost[key]
            true=labels_to_segments(q['labels']);old=set(match_segments(labels_to_segments(b),true));new=set(match_segments(labels_to_segments(apply(b,a)),true));recovered,lost=new-old,old-new
            counts['accepted_true_units_recovered']+=len(recovered);counts['accepted_true_units_lost']+=len(lost)
            for maximum in [10,16]:
                counts[f'accepted_short{maximum}_true_units_recovered']+=sum(true[k].length<=maximum for k in recovered);counts[f'accepted_short{maximum}_true_units_lost']+=sum(true[k].length<=maximum for k in lost)
            if a.jepa and not a.old:
                counts['accepted_new_JEPA_domain_actions']+=1;counts['accepted_new_JEPA_true_units_recovered']+=len(recovered);counts['accepted_new_JEPA_short16_true_units_recovered']+=sum(true[k].length<=16 for k in recovered)
    for row in bins:
        n=row['candidates'];row['mean_uncalibrated_r']=row['risk_evidence_sum']/n if n else None;row['observed_unsafe_fraction']=row['unsafe_candidates']/n if n else None
        for key in ['predicted_H','predicted_R','risk_evidence_sum']:row[key]=float(row[key])
    error=np.asarray(errors);riskerror=np.asarray(riskerrors).reshape(-1,2);accepted=describe(accepted_items);positions=sum(len(b) for b in baselines)
    return {**{k:int(v) for k,v in counts.items()},'role_sequences':len(sequences),'role_positions':positions,'candidate_lengths':describe(all_items),'G_eligible_lengths':describe(eligible_items),'accepted_lengths':accepted,'accepted_interval_positions_fraction_of_role':accepted['interval_length']['sum']/positions,'accepted_changed_positions_fraction_of_role':accepted['changed_positions']['sum']/positions,'accepted_effect_totals':{k:float(v) if k in ['u','v'] else int(v) for k,v in accepted_totals.items()},'risk_bins':bins,'risk_bin_scope':'fixed5equalwidth bins of uncalibrated max positionpH/candidatepFP;not probability-of-any-harm or independent repeated-position samples','gross_count_MAE':np.abs(riskerror).mean(0).tolist() if len(riskerror) else None,'gross_count_MSE':(riskerror**2).mean(0).tolist() if len(riskerror) else None,'utility_target':'v_safety_surrogate' if arm.startswith('S') else 'u_fixedFIT_count_marginal','physical_utility_MAE':float(np.abs(error).mean()) if len(error) else None,'physical_utility_MSE':float((error**2).mean()) if len(error) else None,'query_mean_MSE':float(np.mean(query_mse)) if query_mse else None,'pair_accuracy':{'correct_pairs':pair_correct,'informative_pairs':pair_count,'prediction_ties':pair_ties,'pooled_accuracy':pair_correct/pair_count if pair_count else None,'query_mean_accuracy':float(np.mean(query_accuracies)) if query_accuracies else None,'queries_with_pairs':len(query_accuracies),'fixed_noop_included':True,'true_ties_excluded':True,'prediction_ties_count_as_incorrect':True},'top_action_regret':distribution(top_regret),'after_veto_regret':distribution(post_regret),'regret_scope':'maximum withinquery actualtarget includingfixednoop minusselected actualtarget;not pooledbankF1 or safelyattainable globalgain','majority_protection_scope':'mechanical wholeaxis vote only','reference_scope':'clip-propagated labels, not expert frame visibility','whole_axis_policy_cap':False,'source_risk_seed_shared_across_four_utility_arms':True}
