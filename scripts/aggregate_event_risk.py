#!/usr/bin/env python3
"""Independent156 original-axis/C6/q/gate recount; fixed240 shared5000 CIs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.event_risk_common import *
from scripts.readout_event_risk import complete,equivalent
from echo_routing.temporal.event_risk_metrics import plan,BASE,prepare_system,paired
from echo_routing.temporal.event_risk_acceptance import assess


def main():
    p=execution_parser(__doc__);a=p.parse_args()
    if not authorize(a):return 0
    freeze=verify_freeze('aggregate_freeze.json');sealed=read(PUB/'outer_metrics.json');rows,pred,tables,ctx,banks,full=complete(readonly=True)
    if not equivalent(rows,sealed['rows']):raise ValueError('all156 independently recounted metrics/length/effects/risk/gates differ')
    original=np.load(verify_pin(sealed['private_predictions']),allow_pickle=True).item()
    for t in pred:
        for h in pred[t]:
            for n in pred[t][h]:
                if set(pred[t][h][n])!=set(original[t][h][n]) or any(not np.array_equal(v,original[t][h][n][i]) for i,v in pred[t][h][n].items()):raise ValueError('sealed private decoded axes differ from immutable score-cache recount')
    systems={t:{h:{r['system']:r for r in rows if r['table']==t and r['half']==h} for h in pred[t]} for t in pred};prepared={};comparisons=[]
    def bundle(t,h,n):
        key=(t,h,n)
        if key not in prepared:prepared[key]=prepare_system(pred[t][h][n],pred[t][h][BASE],ctx[t][h]['sequences'],tables[key])
        return prepared[key]
    for item in plan()['comparisons']:
        t,h,l,r=item['table'],item['half'],item['left'],item['right'];x,y=systems[t][h][l],systems[t][h][r];delta={k:x[k]-y[k] if x[k] is not None and y[k] is not None else None for k in set(x)&set(y) if (x[k] is None or type(x[k]) in [int,float]) and (y[k] is None or type(y[k]) in [int,float])}
        stats=paired(bundle(t,h,l),bundle(t,h,r),primary=item['primary_family']);comparisons.append({**item,'delta':delta,**stats})
        if len(comparisons)%10==0:write(RUN/'runtimecoordination.json',{'phase':'paired240_same5000_draws_95_and99_375','completed':len(comparisons),'model_pid':os.getpid(),'epoch':time.time()})
    if len(comparisons)!=240:raise ValueError('complete fixed240 comparison matrix required')
    information=read(PUB/'calibration.json')['calibration_information'];accept=assess(rows,comparisons,information);write(PUB/'acceptance.json',accept)
    write(PUB/'summary.json',{'systems':systems,'rows':rows,'paired_comparisons':comparisons,'metric_rows':156,'inner_metric_rows':26,'paired_comparison_count':240,'formal_models':4,'formal_optimizer_updates':24000,'all156_original_axes_metrics_gates_action_and_event_summaries_independently_recounted':True,'all5000_draws_reused_for95_and99_375':True,'primary_policy':'R2','outer_metrics_sha256':sha(PUB/'outer_metrics.json'),'models_manifest_sha256':sha(PUB/'models_manifest.json'),'calibration_sha256':sha(PUB/'calibration.json'),'readout_plan_sha256':sha(PUB/'readout_plan.json'),'acceptance_sha256':sha(PUB/'acceptance.json'),'no_outer_selection_or_tuning':True,'old_A08_inner_exposure_not_OOF':True,'CAL_and_AUDIT_are_new_q_split_not_independent_A08_validation':True,'sessions_not_confirmed_patients':True,'constructed_bank_CIs_conditional_on_shared_sources':True,'propagated_family_labels_not_expert_visibility_or_clinical_harm':True,'aggregate_freeze_git':freeze['git_commit'],'epoch':time.time()});print({'outer_rows':156,'paired5000':240,'primary_safety_goal':accept.get('R2jointtarget')},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
