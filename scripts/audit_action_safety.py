#!/usr/bin/env python3
"""A61 independent complete10-model/24inner/8calibration/56-set audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.safety_common import *
from echo_routing.temporal.safety_audit import verify_models
from echo_routing.temporal.safety_training import SEEDS,ARMS
from echo_routing.temporal.safety_runtime import policies
from echo_routing.temporal.safety_policy import calibrate
from echo_routing.temporal.safety_targets import effects
from echo_routing.temporal.proposal_actions import propose,partition
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.dense_runtime import metrics
from echo_routing.temporal.dense_analysis import short_metrics


def equivalent(left,right):
    if isinstance(left,dict):return isinstance(right,dict) and set(left)==set(right) and all(equivalent(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return isinstance(right,(list,tuple)) and len(left)==len(right) and all(equivalent(a,b) for a,b in zip(left,right))
    if isinstance(left,(int,float,np.number)) and not isinstance(left,bool):return isinstance(right,(int,float,np.number)) and math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12)
    return left==right


def main():
    verify_freeze('audit_freeze.json');formal=verify_freeze('formal_freeze.json');completed=read(PUB/'formal_completion.json');inputs=read(PUB/'training_inputs.json')
    if (PUB/'models_manifest.json').exists() or not completed['passed'] or completed['formal_optimizer_updates']!=60000:raise ValueError('fresh complete60000 audit required')
    data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item();plans={k:np.load(verify_pin(p),allow_pickle=True).item() for k,p in inputs['plans'].items()}
    for row in completed['models']:
        expected={'code':digest(formal['objects']),'config':sha(CFG),'sources':digest(inputs['source_cache_identities']),'input':inputs['dataset']['sha256'],'draw_plan':inputs['plans'][str(row['seed'])]['sha256'],'normalizer':inputs['normalizers']['sha256'],'target_scales':digest({'scale':inputs['utility_scale']})}
        if row['identity']!=expected:raise ValueError('model identity not bound to independently loaded formal contract/input/plan')
    torch.set_num_threads(1);models=verify_models(completed['models'],data,plans,ROOT)
    iq,ia,ij,ib,sources=role_values('inner_cal')
    if len(iq)!=928:raise ValueError('whole928 inner provenance required')
    domains=[partition(b,propose(b,a.argmax(1),j.argmax(1)))[0] for a,j,b in zip(ia,ij,ib)]
    expected_ids={q['id'] for q in iq};risk_pins={r['seed']:r['sha256'] for r in models if r['role']=='risk'}
    cells=[];thresholds=[];final_scores={}
    def verify_labels(pred):
        if set(pred)!=expected_ids or any(np.asarray(pred[q['id']]).shape!=(len(q['labels']),) or not np.issubdtype(np.asarray(pred[q['id']]).dtype,np.integer) or np.min(pred[q['id']])<0 or np.max(pred[q['id']])>4 for q in iq):raise ValueError('inner private label full-axis/ID/class coverage changed')
        if any(majority(pred[q['id']])!=majority(b) for q,b in zip(iq,ib)):raise ValueError('all-sequence G invariant failed')
    for seed in SEEDS:
        for arm in ARMS:
            for cursor in [2000,4000,6000]:
                path=PUB/f'inner_seed{seed}_{arm}_step{cursor}.json';record=read(path);private=np.load(verify_pin(record['private_outputs']),allow_pickle=True).item();weight=RUN/'models'/f'seed{seed}'/arm/f'step{cursor}.pt'
                if record['source_cache_identities']!=sources or record['utility_weight_sha256']!=sha(weight) or record['risk_weight_sha256']!=risk_pins[seed] or record['sequences']!=928 or len(private['scores'])!=928 or set(record['policies'])!={'E','N'}:raise ValueError('inner checkpoint/risk/source/axis identity changed')
                predictions=private['labels'];accepted={p:0 for p in ['E','N']}
                for p in predictions:verify_labels(predictions[p])
                for q,b,domain,score in zip(iq,ib,domains,private['scores']):
                    decoded=policies(b,domain,score)
                    for policy in ['E','N']:
                        if not np.array_equal(decoded[policy]['prediction'],predictions[policy][q['id']]):raise ValueError('inner labels differ from original frozen scores/top-first policy')
                        accepted[policy]+=decoded[policy]['accepted'] is not None
                recounted={p:{**metrics(iq,pred)[0],**short_metrics(iq,pred,16)} for p,pred in predictions.items()}
                if not equivalent(recounted,record['policies']) or accepted!=record['accepted_actions']:raise ValueError('inner original-label metric/accept recount differs')
                cells.append({'seed':seed,'arm':arm,'step':cursor,'path':str(path.relative_to(ROOT)),'sha256':sha(path),'original928_labels_metrics_score_policies_recounted':True})
                if cursor==6000:final_scores[(seed,arm)]=private['scores']
    calibration=read(PUB/'calibration.json')
    expected_paths={c['path']:c['sha256'] for c in cells}
    if calibration['inner_sequences']!=928 or calibration['inner_policy_sets']!=56 or {p['path']:p['sha256'] for p in calibration['inner_records']}!=expected_paths:raise ValueError('all24 source-record/calibration coverage mismatch')
    if len(calibration['thresholds'])!=8 or {(p['seed'],p['arm']) for p in calibration['thresholds']}!={(s,a) for s in SEEDS for a in ARMS}:raise ValueError('exact8 threshold binding required')
    for pin in calibration['thresholds']:
        path=verify_pin(pin);record=read(path);seed,arm=pin['seed'],pin['arm'];scores=final_scores[(seed,arm)];rows=[]
        for q,b,domain,score in zip(iq,ib,domains,scores):
            result=policies(b,domain,score)['N'];top=result['top_index'];cost={} if top is None else effects(b,domain[top],q['labels'],data['F_fit']);rows.append({'N_admitted':result['N_admitted'],'r':result['r'],**cost})
        expected=calibrate(rows)
        if record['source_cache_identities']!=sources or record['risk_weight_sha256']!=risk_pins[seed] or record['utility_weight_sha256']!=sha(RUN/'models'/f'seed{seed}'/arm/'step6000.pt') or pin['tau']!=expected['tau'] or any(not equivalent(record[k],v) for k,v in expected.items()):raise ValueError('firstunsafe calibration/weight/source binding differs from independent recount')
        cached=np.load(verify_pin(record['private_outputs']),allow_pickle=True).item();verify_labels(cached['predictions'])
        if not equivalent(rows,cached['calibration_rows']):raise ValueError('calibration original-axis costs changed')
        for q,b,domain,score in zip(iq,ib,domains,scores):
            result=policies(b,domain,score,expected['tau'])['C']
            if not np.array_equal(result['prediction'],cached['predictions'][q['id']]):raise ValueError('cached C labels not strict frozen gate')
        recount={**metrics(iq,cached['predictions'])[0],**short_metrics(iq,cached['predictions'],16)}
        if not equivalent(recount,record['C_metrics']) or expected['C_unsafe']!=0:raise ValueError('calibration C original-label empirical unsafe/metric mismatch')
        thresholds.append({**pin,**expected,'original928_alltop_costs_and_strict_C_predictions_recounted':True})
    if len(cells)!=24 or len(thresholds)!=8 or torch.cuda.is_initialized():raise ValueError('whole model/inner/calibration CPU barrier incomplete')
    write(PUB/'models_manifest.json',{'passed':True,'models':models,'physical_models':10,'formal_optimizer_updates':60000,'utility_updates':48000,'risk_updates':12000,'inner_records':cells,'thresholds':thresholds,'inner_policy_sets':56,'all10finals24inner8taus56sets_frozen':True,'source_inner_cache_identities':sources,'training_inputs_sha256':sha(PUB/'training_inputs.json'),'formal_completion_sha256':sha(PUB/'formal_completion.json'),'calibration_sha256':sha(PUB/'calibration.json'),'readout_plan_sha256':sha(PUB/'readout_plan.json'),'all_fullaxis_majorities_equal_Base':True,'no_outer_outcome_used_for_model_or_threshold_choice':True,'binary_weights_Adam_RNG_and_percase_outputs_remain_server_only':True,'new_neural_forwards_during_audit':0,'CUDA_initialized':False})
    write(RUN/'runtimecoordination.json',{'phase':'all10finals24inner8taus56sets_audited_pending_actualGit_outer_seal','formal_optimizer_updates':60000,'model_pid':os.getpid(),'epoch':time.time(),'next_action':'actualGit seal model/calibration/endpoints then new240source/cache and314readouts;no new training'})
    print({'passed':True,'models':10,'updates':60000,'inner_records':24,'taus':8,'sets':56},flush=True)
if __name__=='__main__':main()
