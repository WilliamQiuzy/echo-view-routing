#!/usr/bin/env python3
"""Allfour final models, CAL-only6 thresholds and unchanged AUDIT26 diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.event_risk_common import *
from scripts.event_risk_execution import bank,score_values,decoded,QINFO,FIXED_INFO
from echo_routing.temporal.event_risk_policy import calibrate
from echo_routing.temporal.event_risk_metrics import BASE,SEEDS,POLICIES,system,plan
from echo_routing.temporal.event_risk_analysis import summaries
from echo_routing.temporal.safety_policy import decode as old_decode
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.safety_metrics import gross_frames
from echo_routing.temporal.dense_runtime import metrics
from echo_routing.temporal.dense_analysis import short_metrics


def calibration_rows(population,values):
    if population!='CAL':raise ValueError('only CAL can produce thresholds')
    qs,pa,pj,bases=bank(population);fit=read(PUB/'training_inputs.json')['F_fit'];rows=[]
    for q,b,v in zip(qs,bases,values):
        score=v['old_scores'];risk=[dict(zip(['predicted_H','predicted_R','predicted_neutral','pFP','r'],s[1:])) for s in score];n=old_decode(b,v['actions'],score[:,0],risk,'N');i=n['top_index'];c=None if i is None else effects(b,v['actions'][i],q['labels'],fit)
        rows.append({'N':n['accepted_index'] is not None,'unsafe':False if c is None else not c['safe'],'r':None if i is None else float(score[i,5]),'q1':None if i is None else float(v['q'][i,0]),'q2':None if i is None else float(v['q'][i,1]),'kind':q['kind']})
    return rows


def inner_metric(qs,pred,bases):
    value=metrics(qs,pred)[0];value.update(short_metrics(qs,pred,16));g=gross_frames(qs,{q['id']:b for q,b in zip(qs,bases)},pred);cms={k:np.zeros((5,5),np.int64) for k in ['native','multi','pair']}
    for q in qs:np.add.at(cms[q['kind']],(q['labels'],pred[q['id']]),1)
    value.update(gross_reference_position_effects=g,decoded_position_confusion={k:v.tolist() for k,v in cms.items()},segment_f1_50_pairs=None,boundary_precision_pairs=None,boundary_recall_pairs=None,pair_reference_positions=0,inner_pair_bank_absent=True)
    for kind,r in g.items():value[kind+'_new_wrong_reference_positions']=r['H'];value[kind+'_repaired_reference_positions']=r['R']
    return value


def inner_rows(thresholds,legacy,readonly=False,prediction_output=None):
    fit=read(PUB/'training_inputs.json')['F_fit'];rows=[];predictions={}
    for population in ['CAL','AUDIT']:
        qs,pa,pj,bases=bank(population);predictions[population]={};basepred={q['id']:b for q,b in zip(qs,bases)};row={'population':population,'system':BASE,'role':'reference',**inner_metric(qs,basepred,bases)};rows.append(row);predictions[population][BASE]=basepred
        for seed in SEEDS:
            raw,info=score_values(seed,population,readonly=readonly);taus={r['policy']:r['tau'] for r in [*thresholds,*legacy] if r['seed']==seed};values=decoded(raw,bases,taus);stats=summaries(qs,bases,values,fit)
            for policy in POLICIES:
                pred={q['id']:v[policy]['prediction'] for q,v in zip(qs,values)};st=stats[policy];r={'population':population,'system':system(seed,policy),'role':'policy','seed':seed,'policy':policy,'step':6000,**inner_metric(qs,pred,bases),'action_statistics_source_role':st,'action_statistics_full_task':st,'risk_diagnostics':{a:{'N_top':st['event_risk_original_N_top'][a],'all_candidates':st['event_risk_all_candidates'][a]} for a in ['R1','R2']},'q_cache_pin':{k:info[k] for k in ['path','identity','journal_sha256','sequences']}};rows.append(r);predictions[population][r['system']]=pred
    if len(rows)!=26 or {(r['population'],r['system']) for r in rows}!={(r['population'],r['system']) for r in plan()['inner_rows']}:raise ValueError('complete26 CAL/AUDIT rows required, absentpair retainednull')
    if prediction_output is not None:save_array(prediction_output,predictions)
    return rows


def main():
    p=execution_parser(__doc__);p.add_argument('--CAL-only-thresholds',action='store_true');p.add_argument('--AUDIT-diagnostic',action='store_true');p.add_argument('--seal-before-outer',action='store_true');a=p.parse_args()
    if not authorize(a):return 0
    freeze=verify_freeze('calibration_freeze.json')
    if not a.CAL_only_thresholds or not a.AUDIT_diagnostic or not a.seal_before_outer or (PUB/'calibration.json').exists():raise ValueError('single CAL fit and unchangedAUDIT diagnostic explicit flags required')
    modelaudit=read(PUB/'model_audit.json')
    if not modelaudit['passed'] or modelaudit['formal_optimizer_updates']!=24000:raise ValueError('allfour finals audited before any new thresholds')
    thresholds=[];legacy=[];info={}
    for seed in SEEDS:
        raw,_=score_values(seed,'CAL');rows=calibration_rows('CAL',raw);c=calibrate(rows);info[str(seed)]={k:c[k] for k in ['total_axes','N_admitted','unsafe_N_top','calibration_limited']};info[str(seed)]['source_kind_counts']={k:sum(r['kind']==k for r in rows) for k in ['native','multi']}
        for policy in ['M0','R1','R2']:thresholds.append({'seed':seed,'policy':policy,**c[policy]})
        legacy.append({'seed':seed,'policy':'L0','tau':read(CFG)['legacy_SL_tau'][str(seed)],'original_A61_calibration_sha256':read(CFG)['prior_A61']['calibration.json']['sha256']})
    rows=inner_rows(thresholds,legacy,prediction_output=RUN/'inner_predictions.private.npy');write(PUB/'inner_metrics.json',{'rows':rows,'metric_rows':26,'new_threshold_fit_population':'CAL_only','AUDIT_used_for_model_or_threshold_selection':False});write(PUB/'calibration.json',{'thresholds':thresholds,'legacy_thresholds':legacy,'inner_records':rows,'calibration_information':info,'CAL_source_sessions':50,'AUDIT_source_sessions':50,'CAL_AUDIT_independent_patient_or_clinical_validation':False,'q_cache_records':QINFO,'fixed_head_cache_records':FIXED_INFO,'new_teacher_forwards':0,'new_optimizer_updates':0,'freeze_git':freeze['git_commit']});print({'newtaus':6,'legacyrefs':2,'inner_rows':26,'calibration_limited':{s:v['calibration_limited'] for s,v in info.items()}},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
