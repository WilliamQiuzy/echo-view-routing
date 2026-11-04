"""Full task metrics plus explicit complete-axis proposal/acceptance/harm counts."""
import numpy as np
from echo_routing.temporal.anchor_metrics import evaluate,paired as inherited_paired
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.proposal_features import targets
from echo_routing.temporal.anchor_edits import predicted_count
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments


def paired(left,right,n_boot=5000,seed=2026110360):return inherited_paired(left,right,n_boot=n_boot,seed=seed)

def distribution(values):
    v=np.asarray(values,float)
    return {'count':len(v),'sum':float(v.sum()),'min':float(v.min()) if len(v) else None,'max':float(v.max()) if len(v) else None,'median':float(np.median(v)) if len(v) else None,'q10_q90':np.quantile(v,[.1,.9]).tolist() if len(v) else None}

def describe(items):
    values={k:[] for k in ['interval_length','changed_positions','length_over_T','changed_over_length','changed_over_T','added_cuts']};whole=0
    for base,action in items:
        edited=apply(base,action);n=len(base);l=action.end-action.start;k=int(np.count_nonzero(edited!=base))
        for key,value in zip(values,[l,k,l/n,k/l,k/n,predicted_count(edited)-predicted_count(base)]):values[key].append(value)
        whole+=action.start==0 and action.end==n
    return {**{k:distribution(v) for k,v in values.items()},'whole_axis_count':int(whole),'atmost10_interval_count':sum(x<=10 for x in values['interval_length']),'atmost16_interval_count':sum(x<=16 for x in values['interval_length'])}


def action_summary(sequences,aa,jj,baselines,values,policy,fit_f1):
    if policy not in ['O','E','R'] or not len(sequences)==len(aa)==len(jj)==len(baselines)==len(values):raise ValueError('complete same-axis declared policy role required')
    all_items=[];eligible_items=[];accepted=[];errors=[];target_values=[]
    out={k:0 for k in ['G_vetoed_candidates','risk_vetoed_positive_utility_candidates','positive_utility_candidates','zero_utility_candidates','negative_utility_candidates','accepted_actions','accepted_harmful_utility_actions','accepted_harmful_wrong_positions_actions','accepted_harmful_short16_FP_actions','missed_beneficial_alternative_actions','accepted_new_JEPA_domain_actions','accepted_new_to_raw_A08_positions','new_JEPA_domain_candidates','accepted_new_JEPA_true_units_recovered','accepted_new_JEPA_short16_true_units_recovered','accepted_true_units_lost','old_provenance_candidates','JEPA_provenance_candidates','both_provenance_candidates']}
    positions=sum(len(b) for b in baselines)
    for q,a,j,b,v in zip(sequences,aa,jj,baselines,values):
        domain=[x for x in v['all_actions'] if policy!='O' or x.old];all_items.extend((b,x) for x in domain)
        out['G_vetoed_candidates']+=sum(policy!='O' or x.old for x in v['G_rejected'])
        out['new_JEPA_domain_candidates']+=sum(x.jepa and not x.old for x in domain)
        out['old_provenance_candidates']+=sum(x.old for x in domain);out['JEPA_provenance_candidates']+=sum(x.jepa for x in domain);out['both_provenance_candidates']+=sum(x.jepa and x.old for x in domain)
        out['risk_vetoed_positive_utility_candidates']+=len(v[policy]['risk_vetoed_indices'])
        selected=v[policy]['accepted'];true=labels_to_segments(q['labels']);base_matched=set(match_segments(labels_to_segments(b),true))
        for index,x in enumerate(v['actions']):
            if policy=='O' and not x.old:continue
            eligible_items.append((b,x));pred=v['outputs'][index];target=targets(b,x,q['labels'],fit_f1);errors.append(pred-target);target_values.append(target)
            out['positive_utility_candidates']+=pred[0]>0;out['zero_utility_candidates']+=pred[0]==0;out['negative_utility_candidates']+=pred[0]<0
            out['missed_beneficial_alternative_actions']+=target[0]>0 and (selected is None or selected.key!=x.key)
            if selected is not None and selected.key==x.key:
                accepted.append((b,x));out['accepted_actions']+=1
                out['accepted_harmful_utility_actions']+=target[0]<0;out['accepted_harmful_wrong_positions_actions']+=target[1]>0;out['accepted_harmful_short16_FP_actions']+=target[2]>0
                edited=apply(b,x);matched=set(match_segments(labels_to_segments(edited),true));out['accepted_true_units_lost']+=len(base_matched-matched)
                if x.jepa and not x.old:
                    out['accepted_new_JEPA_domain_actions']+=1;out['accepted_new_JEPA_true_units_recovered']+=len(matched-base_matched);out['accepted_new_JEPA_short16_true_units_recovered']+=sum(true[k].length<=16 for k in matched-base_matched)
                changed=edited!=b;out['accepted_new_to_raw_A08_positions']+=int(np.count_nonzero(changed & (edited!=a.argmax(1))))
    candidate=describe(all_items);accepted_description=describe(accepted);err=np.asarray(errors).reshape(-1,3);actual=np.asarray(target_values).reshape(-1,3)
    return {**{k:int(v) for k,v in out.items()},'candidate_count':len(all_items),'G_eligible_scored_candidates':len(eligible_items),'candidate_lengths':candidate,'G_eligible_lengths':describe(eligible_items),'accepted_lengths':accepted_description,'role_sequences':len(sequences),'role_positions':positions,'accepted_interval_positions_fraction_of_role':accepted_description['interval_length']['sum']/positions,'accepted_changed_positions_fraction_of_role':accepted_description['changed_positions']['sum']/positions,'physical_output_MAE':np.abs(err).mean(0).tolist() if len(err) else None,'physical_output_MSE':(err**2).mean(0).tolist() if len(err) else None,'physical_target_minmax':np.stack([actual.min(0),actual.max(0)]).tolist() if len(actual) else None,'signed_target_scope':'complete-axis supervised diagnosis at fixed FIT F1;nativeutility=countstabilityproxy','missed_beneficial_scope':'positiveFITmarginal candidate alternatives not chosen;not jointly attainable oracle','whole_axis_policy_cap':False,'scope':'complete listed role axes;length interval includes unchanged interior;changed positions counted separately'}
