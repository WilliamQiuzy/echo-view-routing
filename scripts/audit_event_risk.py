#!/usr/bin/env python3
"""Independent A62 binary/trace audit or original-axis CAL/AUDIT recount;0NN."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_audit import audit_trace,validate_fullstates,FIXED_A61
from echo_routing.temporal.event_risk_model import EventRisk,optimizer


def independent_identity(row,inputs,formal):
    from scripts.check_event_risk import identity
    expected=identity(inputs,row['seed'],formal)
    if row['identity']!=expected:raise ValueError('completion identity differs from independently pinned formal source/config/input/draw/norm contract')
    return expected


def models():
    freeze=verify_freeze('model_audit_freeze.json');completion=read(PUB/'formal_completion.json');inputs=read(PUB/'training_inputs.json');data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item();rows=completion['models'];states_all=[];records=[];formal=verify_freeze('formal_freeze.json')
    if len(rows)!=4 or {(r['seed'],r['arm']) for r in rows}!={(s,a) for s in [2026110461,2026110462] for a in ['R1','R2']} or completion['formal_optimizer_updates']!=24000:raise ValueError('completefourfinal matrix required')
    for r in rows:
        expected_identity=independent_identity(r,inputs,formal)
        model=EventRisk(r['arm'],r['seed']);initial=copy.deepcopy(model.state_dict());opt=optimizer(model);folder=ROOT/r['path'];folder=folder.parent;plan=np.load(verify_pin(inputs['plans'][str(r['seed'])]),allow_pickle=True).item();trace=folder/'physical_updates.jsonl';physical=[json.loads(x) for x in trace.read_text().splitlines()];report=audit_trace(physical,plan,data['kind_indices'],data['unsafe'],r['seed'],expected_identity);states={}
        for cursor in [2000,4000,6000,'latest']:
            p=folder/('latest.pt' if cursor=='latest' else f'step{cursor}.pt');states[cursor]=torch.load(p,weights_only=True,map_location='cpu');states_all.append({'seed':r['seed'],'arm':r['arm'],'step':6000 if cursor=='latest' else cursor,'kind':'latest' if cursor=='latest' else 'milestone',**pin(p)})
        full=validate_fullstates(model,opt,states,expected_identity,data['normalizers'],initial)
        if sha(verify_pin(r))!=r['sha256'] or r['result']['journal_sha256']!=sha(trace):raise ValueError('final/full physicaltrace changed')
        records.append({**r,'independent_trace_audit':report,'independent_fullstates_audit':full})
    if len(states_all)!=16:raise ValueError('12milestones+4latest required')
    for r in FIXED_A61:verify_pin(r)
    write(PUB/'model_audit.json',{'passed':True,'models':records,'fixed_A61_models':list(FIXED_A61),'fullstates':states_all,'formal_optimizer_updates':24000,'physical_models':4,'new_model_forwards':0,'new_optimizer_updates':0,'all16binary_states_and24000rows_independently_recounted':True,'freeze_git':freeze['git_commit']});print({'audited_models':4,'states':16,'physical_updates':24000,'newNN':0},flush=True)


def preouter():
    verify_freeze('final_audit_freeze.json')
    from scripts.calibrate_event_risk import inner_rows,calibration_rows
    from scripts.event_risk_execution import bank,score_values
    from echo_routing.temporal.event_risk_policy import calibrate
    from echo_routing.temporal.event_risk_metrics import plan
    calibration=read(PUB/'calibration.json');modelaudit=read(PUB/'model_audit.json');recomputed=[]
    for seed in [2026110461,2026110462]:
        raw,_=score_values(seed,'CAL',readonly=True);result=calibrate(calibration_rows('CAL',raw));frozen={r['policy']:r['tau'] for r in calibration['thresholds'] if r['seed']==seed}
        if any(result[p]['tau']!=frozen[p] for p in ['M0','R1','R2']):raise ValueError('CAL-only firstunsafe threshold recount differs')
    recomputed=inner_rows(calibration['thresholds'],calibration['legacy_thresholds'],readonly=True)
    if recomputed!=calibration['inner_records'] or read(PUB/'readout_plan.json')!=plan():raise ValueError('all26 original-axis inner metrics/gates/endpoints must independently reproduce')
    for r in [*modelaudit['models'],*modelaudit['fixed_A61_models'],*modelaudit['fullstates']]:verify_pin(r)
    out={**modelaudit,'thresholds':calibration['thresholds'],'legacy_thresholds':calibration['legacy_thresholds'],'inner_records':recomputed,'calibration_sha256':sha(PUB/'calibration.json'),'readout_plan_sha256':sha(PUB/'readout_plan.json'),'audited_readout_plan_sha256':sha(PUB/'readout_plan.json'),'all26_CAL_AUDIT_rows_and6newtaus_recounted_noNN':True};write(PUB/'models_manifest.json',out);print({'models':4,'fixedrefs':4,'inner_rows':26,'taus':8,'newNN':0},flush=True)


def main():
    p=execution_parser(__doc__);p.add_argument('--stage',choices=['models','preouter'],default='models');a=p.parse_args()
    if not authorize(a):return 0
    (models if a.stage=='models' else preouter)();return 0
if __name__=='__main__':raise SystemExit(main())
