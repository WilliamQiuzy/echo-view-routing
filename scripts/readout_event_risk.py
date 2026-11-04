#!/usr/bin/env python3
"""Fixed156 A62 complete-task readouts; old E/N/L0 semantics preserved."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scripts.event_risk_common import *
from scripts.event_risk_execution import bank,contexts,role_for,score_values,decoded,preouter,old_records,QINFO,FIXED_INFO
from echo_routing.temporal.event_risk_metrics import plan,TABLES,HALVES,SEEDS,POLICIES,BASE,system,evaluate
from echo_routing.temporal.event_risk_analysis import summaries
from echo_routing.temporal.safety_tasks import compose,align
from echo_routing.temporal.structured_metrics import compose_stress_predictions
from echo_routing.temporal.anchor_edits import majority


def equivalent(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equivalent(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,np.floating)) or isinstance(b,(float,np.floating)):return a is not None and b is not None and bool(np.isclose(a,b,rtol=0,atol=1e-12,equal_nan=False))
    return a==b


def complete(readonly=False):
    preouter();cfg=read(CFG);m=read(PUB/'models_manifest.json');cal=read(PUB/'calibration.json');fit=read(PUB/'training_inputs.json')['F_fit'];ctx=contexts();rows=[];predictions={t:{h:{} for h in HALVES} for t in TABLES};tabs={};banks={};decoded_bank={};stats_bank={};fullgroups={}
    for table in TABLES:
        for half in HALVES:
            role=role_for(table,half);qs,pa,pj,bases=bank(role);banks[role]=(qs,pa,pj,bases);basepred={q['id']:b for q,b in zip(qs,bases)}
            if table!='main':basepred=compose_stress_predictions(predictions['main'][half][BASE],basepred,banks[half][0],qs)
            predictions[table][half][BASE]=basepred;metric,tab=evaluate(basepred,ctx[table][half],basepred);tabs[(table,half,BASE)]=tab;rows.append({'table':table,'half':half,'system':BASE,'role':'reference',**metric})
            for seed in SEEDS:
                taus={r['policy']:r['tau'] for r in cal['thresholds']+cal['legacy_thresholds'] if r['seed']==seed};raw,info=score_values(seed,role,readonly=readonly);values=decoded(raw,bases,taus);decoded_bank[(seed,role)]=values;stats=summaries(qs,bases,values,fit);stats_bank[(seed,role)]=stats
                if table=='main':full=(qs,bases,values)
                else:full=compose(banks[half][0],[banks[half][3],decoded_bank[(seed,half)]],qs,[bases,values],ctx[table][half]['sequences'])
                fq,fb,fv=full;fullgroups[(seed,table,half)]=full;fullstats=stats if table=='main' else summaries(fq,fb,fv,fit)
                for policy in POLICIES:
                    name=system(seed,policy);pred={q['id']:v[policy]['prediction'] for q,v in zip(qs,values)}
                    if any(majority(v[policy]['prediction'])!=majority(b) for b,v in zip(bases,values)):raise ValueError('whole-axis G majority invariant failed')
                    if table!='main':pred=compose_stress_predictions(predictions['main'][half][name],pred,banks[half][0],qs)
                    predictions[table][half][name]=pred;metric,tab=evaluate(pred,ctx[table][half],basepred);tabs[(table,half,name)]=tab
                    if table!='a62_confirmation' and policy in ['E','N','L0']:
                        oldp='C' if policy=='L0' else policy;old=next(r for r in old_records()['outer_metrics.json']['rows'] if r['table']==table and r['half']==half and r.get('seed')==seed and r.get('arm')=='S-L' and r.get('policy')==oldp)
                        if not equivalent(metric,{k:old[k] for k in metric}):raise ValueError('fixed A61 E/N/L0 full-task metrics changed')
                    st=stats[policy];rows.append({'table':table,'half':half,'system':name,'role':'policy','seed':seed,'policy':policy,'step':6000,**metric,'source_role':role,'tau':taus.get(policy),'action_statistics_source_role':st,'action_statistics_full_task':fullstats[policy],'risk_diagnostics':{a:{'N_top':st['event_risk_original_N_top'][a],'all_candidates':st['event_risk_all_candidates'][a]} for a in ['R1','R2']},'q_cache_pin':{k:info[k] for k in ['path','identity','journal_sha256','sequences']},'same_original_SL_top_and_N_no_fallback':True,'all_sequences_majority_retained':True,'max_one_action_per_axis':True,'native_pair_predictions_reused_identically':table!='main'})
            print({'table':table,'half':half,'rows':len(rows),'readonly_recount':readonly},flush=True)
    expected={(r['table'],r['half'],r['system']) for r in plan()['rows']};actual=[(r['table'],r['half'],r['system']) for r in rows]
    if len(rows)!=156 or set(actual)!=expected or torch.cuda.is_initialized():raise ValueError('complete exact156 CPU readouts required')
    return rows,predictions,tabs,ctx,banks,fullgroups


def main():
    p=execution_parser(__doc__);a=p.parse_args()
    if not authorize(a):return 0
    freeze=verify_freeze('readout_freeze.json')
    if (PUB/'outer_metrics.json').exists():raise ValueError('sealed readout cannot be overwritten')
    rows,pred,_,_,_,_=complete();path=RUN/'outer_predictions.private.npy';save_array(path,pred)
    write(PUB/'outer_metrics.json',{'rows':rows,'metric_rows':156,'policy_metric_rows':144,'reference_metric_rows':12,'models_manifest_sha256':sha(PUB/'models_manifest.json'),'calibration_sha256':sha(PUB/'calibration.json'),'readout_plan_sha256':sha(PUB/'readout_plan.json'),'producer_manifest_sha256':sha(PUB/'producer_manifest.json'),'source_adoption_sha256':sha(PUB/'source_adoption.json'),'private_predictions':pin(path),'joint_q_cache_cells':QINFO,'new_fixed_head_cache_cells':FIXED_INFO,'old_E_N_L0_all5tables_metric_exact':True,'new_teacher_forwards':0,'readout_freeze_git':freeze['git_commit'],'epoch':time.time()});write(RUN/'runtimecoordination.json',{'phase':'all156_outer26inner_complete_pending240paired','model_pid':None,'epoch':time.time(),'next_action':'independentoriginalaxisrecount240CI_then6new30refs48support24native16regression'});return 0
if __name__=='__main__':raise SystemExit(main())
