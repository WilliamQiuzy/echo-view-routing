#!/usr/bin/env python3
"""No-update cached FIT worst-geometry timing; label-free role workload bounds."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.proposal_common import *
from scripts.train_jepa_proposals import role_values
from echo_routing.temporal.proposal_actions import propose,partition
from echo_routing.temporal.proposal_training import ProposalRegressor,optimizer,restore
from echo_routing.temporal.proposal_runtime import infer


def geometry(qs,aa,jj,bb):
    rows=[]
    for q,a,j,b in zip(qs,aa,jj,bb):
        actions=propose(b,a.argmax(1),j.argmax(1));eligible,_=partition(b,actions)
        rows.append((len(b),len(actions),len(eligible),max([x.end-x.start for x in actions],default=0)))
    v=np.asarray(rows,np.int64)
    return v,{'sequences':len(qs),'positions':int(v[:,0].sum()),'all_candidates':int(v[:,1].sum()),'eligible_candidates':int(v[:,2].sum()),'max_length':int(v[:,0].max()),'max_candidates':int(v[:,1].max()),'max_T_times_candidates':int((v[:,0]*v[:,1]).max()),'max_interval_length':int(v[:,3].max()),'proposal_units_1plusC':int((1+v[:,1]).sum()),'new_head_scores_or_metrics_computed':False}


def main():
    freeze=verify_freeze('cost_probe_freeze.json');verify_freeze('fullFIT_check_freeze.json')
    inputs=read(PUB/'training_inputs.json');checks=read(PUB/'wholeFIT_checks.json');qs,aa,jj,bb,_=role_values('fit')
    if not checks['passed']:raise ValueError('actual FIT graph must pass first')
    v,fit_summary=geometry(qs,aa,jj,bb)
    indices=set(np.linspace(0,len(qs)-1,32,dtype=int).tolist())|{int(np.argmax(v[:,0])),int(np.argmax(v[:,1])),int(np.argmax(v[:,0]*v[:,1])),int(np.argmax(v[:,3]))}
    scale=np.asarray(read(verify_pin(inputs['target_scales']))['scales']);heads={};torch.set_num_threads(1)
    for arm in ['A','J']:
        norm=read(verify_pin(inputs['normalizers'][arm]));model=ProposalRegressor(arm,2026110360);opt=optimizer(model)
        payload=torch.load(RUN/'engineering/fullFIT'/arm/'timed20/latest.pt',map_location='cpu',weights_only=True)
        if restore(model,opt,payload,payload['identity'],norm,scale)!=20:raise ValueError('frozen engineering model step identity changed')
        times=[];units=[]
        for i in sorted(indices):
            result=infer(model,norm,scale,aa[i],jj[i],bb[i],arm);times.append(result['cached_feature_head_policy_seconds']);units.append(1+v[i,1])
        heads[arm]={'actual_sequences_timed':len(indices),'includes_actual_longest_T':True,'includes_actual_most_proposals':True,'includes_actual_max_T_times_proposals':True,'includes_actual_longest_action_interval':True,'maximum_cached_inference_seconds':max(times),'maximum_seconds_per_1plus_all_candidate':max(t/u for t,u in zip(times,units)),'formal_updates':0,'engineering_updates':0,'timing_excludes_frozen_producers':True}
    roles={'fit':fit_summary}
    for role in read(PUB/'input_manifest.json')['roles']:
        if role=='fit':continue
        if role.startswith('a60_confirmation_'):
            q=sequences(role);length=np.asarray([len(x['refs']) for x in q]);roles[role]={'sequences':len(q),'positions':int(length.sum()),'max_length':int(length.max()),'all_candidates_upper':int(2*length.sum()),'eligible_candidates_upper':int(2*length.sum()),'proposal_units_1plusC_upper':int((1+2*length).sum()),'source_forward_pending_until_weights_freeze':True,'bound':'old components <=T;J runs<=T;dedup/trim/G only reduce'}
        else:
            q,a,j,b,_=role_values(role);_,roles[role]=geometry(q,a,j,b)
    write(PUB/'FIT_inference_cost_bounds.json',{'passed':True,'FIT_geometry':fit_summary,'heads':heads,'role_geometry':roles,'source_only_outer_geometry_no_head_outputs_metrics_or_targets':True,'new_confirmation_labels_not_used_for_modelmetrics':True,'timing_extrapolation_margin':4,'length_extrapolation':'per-role 1+C units times max(1,role_maxT/FITmaxT);actual FIT maxT/maxC/maxTxC/maxinterval included;runtime guards remain hardcap enforcement','formal_updates':0,'engineering_updates':0,'extra_FIT_inference_models':2,'actual_cached_FIT_head_inference_calls':2*len(indices),'CUDA_initialized':torch.cuda.is_initialized(),'cost_probe_freeze_git':freeze['git_commit'],'epoch':time.time()})
    if torch.cuda.is_initialized():raise ValueError('cached CPU cost probe initialized CUDA')
    print({'cost_probe_passed':True,'heads':heads,'roles':roles},flush=True)
if __name__=='__main__':main()
