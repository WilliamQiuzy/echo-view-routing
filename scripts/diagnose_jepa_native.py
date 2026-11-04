#!/usr/bin/env python3
"""One cached-only native recovery/harm attribution, no new model or policy."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scripts.proposal_common import *
from scripts.train_jepa_proposals import role_values
from scripts.readout_jepa_proposals import pair_cache
from echo_routing.temporal.proposal_native_attribution import joint_counts,effects
from echo_routing.temporal.proposal_readout_plan import HALVES,BASE,system
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.proposal_metrics import distribution


def main():
    freeze=verify_freeze('native_attribution_freeze.json');inputs=read(PUB/'training_inputs.json');out=read(PUB/'outer_metrics.json');models=read(PUB/'models_manifest.json')
    predictions=np.load(verify_pin(out['private_predictions']),allow_pickle=True).item();rows={(r['half'],r['system']):r for r in out['rows'] if r['table']=='main'};banks={h:role_values(h) for h in HALVES};cells=[]
    for model in models['models']:
        seed,arm=model['seed'],model['arm']
        for half in HALVES:
            qs,a,j,b,sources=banks[half];cache=pair_cache(model,half,qs,a,j,b,sources,readonly=True);values=[cache.load(i) for i in range(len(qs))]
            for policy in ['O','E','R']:
                name=system(seed,arm+'-'+policy);native=[i for i,q in enumerate(qs) if q['kind']=='native'];cube=np.zeros((5,5,5),np.int64);totals={k:0 for k in ['positions','Base_correct_to_wrong','Base_wrong_to_correct','both_correct','both_wrong_unchanged_class','both_wrong_changed_class','changed_positions','Base_frame_errors','new_frame_errors','frame_error_delta']};changed_clips=0;groups={}
                for i in native:
                    q=qs[i];base=b[i];pred=predictions['main'][half][name][q['id']];v=values[i];choice=v[policy]['accepted'];idx=v[policy]['accepted_index']
                    if not np.array_equal(pred,v[policy]['prediction']) or not np.array_equal(base,predictions['main'][half][BASE][q['id']]) or majority(base)!=majority(pred):raise ValueError('frozen Base/policy/journal/majority binding failed')
                    jc=joint_counts(q['labels'],base,pred);cube+=np.asarray(jc.pop('true_Base_prediction_joint5cube'));changed_clips+=np.any(base!=pred)
                    for k in totals:totals[k]+=jc[k]
                    if choice is None:continue
                    actual=effects(base,pred,q['labels'],inputs['F_fit']);score=v['outputs'][idx]
                    provenance='both' if choice.old and choice.jepa else 'old_only' if choice.old else 'JEPA_only'
                    key=(int(q['labels'][0]),provenance,'|'.join(map(str,sorted(set(choice.replacement)))))
                    record={**actual,'pred_u':float(score[0]),'pred_dE':float(score[1]),'pred_dFP16':float(score[2]),'interval_length':choice.end-choice.start,'changed_positions':int(np.count_nonzero(base!=pred)),'interval_over_T':(choice.end-choice.start)/len(base),'changed_over_T':float(np.count_nonzero(base!=pred))/len(base),'represented_whole_axis':choice.start==0 and choice.end==len(base)}
                    groups.setdefault(key,[]).append(record)
                grouped=[]
                for (truec,provenance,replacements),records in sorted(groups.items()):
                    entry={'true_family_index':truec,'provenance':provenance,'replacement_family_indices':replacements,'accepted_actions':len(records)}
                    for key in ['delta_M','delta_P','u','dE','dFP16','pred_u','pred_dE','pred_dFP16','interval_length','changed_positions','interval_over_T','changed_over_T','added_cuts','newly_wrong_positions_extending_old_wrong_run','newly_wrong_positions_new_wrong_segment','newly_wrong_positions_in_long_wrong_segments']:
                        entry[key]=distribution([r[key] for r in records])
                    predicates={'frame_harm_actions':lambda r:r['dE']>0,'frame_repair_actions':lambda r:r['dE']<0,'frame_harm_M_P_unchanged':lambda r:r['dE']>0 and r['delta_M']==r['delta_P']==0,'frame_harm_u_zero_but_predpositive':lambda r:r['dE']>0 and r['u']==0 and r['pred_u']>0,'frame_harm_without_short16_FP_increase':lambda r:r['dE']>0 and r['dFP16']<=0,'frame_harm_with_predicted_dE_nonpositive':lambda r:r['dE']>0 and r['pred_dE']<=0,'frame_harm_extending_old_wrong_run':lambda r:r['dE']>0 and r['newly_wrong_positions_extending_old_wrong_run']>0,'frame_harm_crossing_short16_to_long':lambda r:r['dE']>0 and r['wrong_run_crossed_short16_to_long'],'frame_harm_new_cut':lambda r:r['dE']>0 and r['added_cuts']>0,'represented_whole_axis_actions':lambda r:r['represented_whole_axis']}
                    entry.update({k:sum(bool(f(r)) for r in records) for k,f in predicates.items()});grouped.append(entry)
                metric=rows[(half,name)];base_metric=rows[(half,BASE)]
                if (totals['new_frame_errors']!=metric['native_frame_errors'] or totals['Base_frame_errors']!=base_metric['native_frame_errors'] or totals['frame_error_delta']!=totals['Base_correct_to_wrong']-totals['Base_wrong_to_correct'] or sum(g['accepted_actions'] for g in grouped)!=changed_clips or sum(g['dE']['sum'] for g in grouped)!=totals['frame_error_delta']):raise ValueError('native frame attribution failed sealed totals/parity')
                cells.append({'half':half,'system':name,'seed':seed,'arm':arm,'policy':policy,**totals,'native_clips':len(native),'changed_native_clips':int(changed_clips),'majority_changed_clips':0,'true_Base_prediction_joint5cube':cube.tolist(),'class_provenance_groups':grouped,'scope':'main native only;identical native predictions reused by all3stress tables;propagated clip references not dense clinical annotations'})
    if len(cells)!=24:raise ValueError('exact24 main policy-native cells required')
    value={'passed':True,'cells':cells,'cell_count':24,'source_outer_metrics_sha256':sha(PUB/'outer_metrics.json'),'private_predictions_sha256':out['private_predictions']['sha256'],'native_weak_reference_scope':'inherited clip family propagated along cine;not clinician-adjudicated frame truth','no_new_model_or_source_forwards':True,'new_optimizer_updates':0,'no_new_policy_threshold_or_risk_safe_oracle':True,'no_ids_or_percase_rows_exported':True,'same32oracle_and124144_outputs_unchanged':True,'freeze_git':freeze['git_commit'],'epoch':time.time()}
    write(PUB/'native_attribution.json',value);pd.json_normalize([{k:v for k,v in r.items() if k not in ['class_provenance_groups','true_Base_prediction_joint5cube']} for r in cells]).to_csv(PUB/'native_attribution.csv',index=False)
    print({'native_attribution_passed':True,'cells':24,'source_forward_updates':0},flush=True)
if __name__=='__main__':main()
