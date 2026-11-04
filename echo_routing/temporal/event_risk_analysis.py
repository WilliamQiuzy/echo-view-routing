"""A62 declared action/event diagnostics, sharing one count pass across six gates."""
import numpy as np
from echo_routing.temporal.event_risk_policy import POLICIES
from echo_routing.temporal.event_risk_metrics import risk_summary
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.proposal_metrics import describe
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments

COUNT_NAMES=('accepted_actions','accepted_unsafe_actions','accepted_gross_harm_actions','accepted_net_harm_actions','accepted_short16_FP_increase_actions','accepted_negative_u_actions','accepted_true_units_recovered','accepted_true_units_lost','accepted_new_JEPA_domain_actions','accepted_new_JEPA_true_units_recovered','accepted_new_JEPA_short16_true_units_recovered','accepted_short10_true_units_recovered','accepted_short10_true_units_lost','accepted_short16_true_units_recovered','accepted_short16_true_units_lost','vetoed_top_sequences','top_positive_sequences','N_admitted_tops')
COST_NAMES=('u','v','H','R','dE','dFP','delta_M','delta_P')


def summaries(sequences,baselines,values,fit_f1):
    if not len(sequences)==len(baselines)==len(values) or not sequences:raise ValueError('complete nonempty matched source axes required')
    counts={p:{k:0 for k in COUNT_NAMES} for p in POLICIES};totals={p:{k:0 for k in COST_NAMES} for p in POLICIES};accepted={p:[] for p in POLICIES};all_items=[];eligible=[];events=[];qvalues=[];n_events=[];n_q=[];legacy=[];g_veto=0;Ntotal=sum(len(b) for b in baselines)
    for q,b,v in zip(sequences,baselines,values):
        actions=v['actions'];old=np.asarray(v['old_scores']);new=np.asarray(v['q']);y=np.asarray(q['labels'])
        if len(b)!=len(y) or old.shape!=(len(actions),6) or new.shape!=(len(actions),2):raise ValueError('aligned source/score/action axis required')
        all_items.extend((b,a) for a in v['all_actions']);eligible.extend((b,a) for a in actions);g_veto+=len(v['G_rejected']);costs=[effects(b,a,y,fit_f1) for a in actions];events.extend(not c['safe'] for c in costs);qvalues.extend(new.tolist());legacy.extend(old[:,5].tolist());n=v['N']['accepted_index']
        if n is not None:n_events.append(not costs[n]['safe']);n_q.append(new[n].tolist())
        true=labels_to_segments(y);base_match=set(match_segments(labels_to_segments(b),true));diagnostics={}
        for p in POLICIES:
            row=counts[p];top=v[p]['top_index'];selected=v[p]['accepted_index'];row['top_positive_sequences']+=top is not None;row['N_admitted_tops']+=n is not None;row['vetoed_top_sequences']+=top is not None and selected is None
            if selected is None:continue
            if selected!=top or selected!=v['E']['top_index']:raise ValueError('all gates must retain same original top with no fallback')
            a=actions[selected];c=costs[selected];row['accepted_actions']+=1;row['accepted_unsafe_actions']+=not c['safe'];row['accepted_gross_harm_actions']+=c['H']>0;row['accepted_net_harm_actions']+=c['dE']>0;row['accepted_short16_FP_increase_actions']+=c['dFP']>0;row['accepted_negative_u_actions']+=c['u']<0
            accepted[p].append((b,a))
            for key in COST_NAMES:totals[p][key]+=c[key]
            if selected not in diagnostics:
                new_match=set(match_segments(labels_to_segments(apply(b,a)),true));diagnostics[selected]=(new_match-base_match,base_match-new_match)
            recovered,lost=diagnostics[selected];row['accepted_true_units_recovered']+=len(recovered);row['accepted_true_units_lost']+=len(lost)
            for maximum in [10,16]:
                row[f'accepted_short{maximum}_true_units_recovered']+=sum(true[k].length<=maximum for k in recovered);row[f'accepted_short{maximum}_true_units_lost']+=sum(true[k].length<=maximum for k in lost)
            if a.jepa and not a.old:row['accepted_new_JEPA_domain_actions']+=1;row['accepted_new_JEPA_true_units_recovered']+=len(recovered);row['accepted_new_JEPA_short16_true_units_recovered']+=sum(true[k].length<=16 for k in recovered)
    qvalues=np.asarray(qvalues).reshape(-1,2);n_q=np.asarray(n_q).reshape(-1,2);all_desc=describe(all_items);eligible_desc=describe(eligible);risk={'R1':risk_summary(qvalues[:,0],events),'R2':risk_summary(qvalues[:,1],events)};n_risk={'R1':risk_summary(n_q[:,0],n_events),'R2':risk_summary(n_q[:,1],n_events)}
    out={}
    for p in POLICIES:
        desc=describe(accepted[p]);out[p]={**{k:int(v) for k,v in counts[p].items()},'role_sequences':len(sequences),'role_positions':Ntotal,'candidate_count':len(all_items),'G_eligible_scored_candidates':len(eligible),'G_vetoed_candidates':g_veto,'candidate_lengths':all_desc,'G_eligible_lengths':eligible_desc,'accepted_lengths':desc,'whole_axis_candidate_actions':all_desc['whole_axis_count'],'whole_axis_accepted_actions':desc['whole_axis_count'],'max_one_action_per_axis':True,'accepted_interval_positions_fraction_of_role':desc['interval_length']['sum']/Ntotal,'accepted_changed_positions_fraction_of_role':desc['changed_positions']['sum']/Ntotal,'accepted_effect_totals':{k:float(v) if k in ['u','v'] else int(v) for k,v in totals[p].items()},'event_risk_all_candidates':risk,'event_risk_original_N_top':n_risk,'old_r_evidence_fixed5bins':[{'lower':k/5,'upper':(k+1)/5,'count':sum(k/5<=x<((k+1)/5) or k==4 and x==1 for x in legacy)} for k in range(5)],'risk_scope':'q is uncalibrated action-unsafe event evidence; old r is max positionpH/pFP and not action-event probability','matched_counts_once_per_axis_six_same_top_gates':True,'reference_scope':'propagated cine-family references, not expert frame visibility/clinical harm'}
    return out
