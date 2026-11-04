#!/usr/bin/env python3
"""All four physical heads, full traces/Adam states and12 cached inner recounts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.proposal_common import RUN,PUB,CFG,read,write,sha,digest,verify_pin,verify_freeze,sequences
from scripts.train_jepa_proposals import role_values
from echo_routing.temporal.proposal_training import ProposalRegressor,optimizer,restore,UpdateJournal
from echo_routing.temporal.proposal_audit import exact_models,validate_trace
from echo_routing.temporal.dense_runtime import metrics
from echo_routing.temporal.dense_analysis import short_metrics
from echo_routing.temporal.structured_metrics import verify_reference
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.proposal_readout_plan import plan as readout_plan


def main():
    freeze=verify_freeze('audit_freeze.json');formal=verify_freeze('formal_freeze.json')
    done=read(PUB/'formal_completion.json');inputs=read(PUB/'training_inputs.json')
    exact_models(done['models'])
    if not done['passed'] or done['formal_optimizer_updates']!=24000 or done['inner_used_for_model_selection']:
        raise ValueError('complete fixed formal matrix required')
    data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item();scales=np.asarray(read(verify_pin(inputs['target_scales']))['scales'])
    qs,aa,jj,bb,source=role_values('inner_cal');models=[];inner_rows=[];curves=[]
    for row in done['models']:
        seed,arm=row['seed'],row['arm'];folder=RUN/'models'/f'seed{seed}'/arm
        normalizer=read(verify_pin(inputs['normalizers'][arm]));pp=inputs['plans'][str(seed)]
        plan=np.load(verify_pin(pp),allow_pickle=True).item()
        identity={'code':digest(formal['objects']),'config':sha(CFG),'sources':digest(inputs['source_cache_identities']),
                  'input':inputs['dataset']['sha256'],'draw_plan':pp['sha256'],'normalizer':inputs['normalizers'][arm]['sha256'],'target_scales':inputs['target_scales']['sha256']}
        if row['identity']!=identity or row['final_sha256']!=sha(ROOT/row['final_path']):raise ValueError('final identity changed')
        journal=UpdateJournal(folder,identity)
        if journal.resume_cursor()!=6000 or (folder/'inner_pending.json').exists():raise ValueError('unfinished physical/inner claim')
        trace=[json.loads(line) for line in journal.trace.read_text().splitlines()]
        live=validate_trace(trace,data,plan,identity)
        initial=ProposalRegressor(arm,seed).state_dict()
        if set(live)!=set(initial) or not all(live.values()):raise ValueError('missing history gradient liveness')
        milestones=[]
        for cursor in [2000,4000,6000]:
            path=folder/f'step{cursor}.pt';payload=torch.load(path,map_location='cpu',weights_only=True)
            model=ProposalRegressor(arm,seed);opt=optimizer(model)
            if restore(model,opt,payload,identity,normalizer,scales)!=cursor:raise ValueError('milestone Adam cursor changed')
            if any(not value['exp_avg_sq'].count_nonzero() for value in opt.state.values()):raise ValueError('zero whole tensor second moment')
            if any(torch.equal(initial[k],v) for k,v in model.state_dict().items()):raise ValueError('no whole tensor drift')
            record_path=PUB/f'inner_seed{seed}_{arm}_step{cursor}.json';record=read(record_path)
            marker=read(folder/f'inner_step{cursor}.json')
            if (record['weight_sha256']!=sha(path) or record['identity']!=identity or record['step']!=cursor
                or record['source_cache_identities']!=source or marker['callback_result']['sha256']!=sha(record_path)):
                raise ValueError('inner checkpoint/source/record binding changed')
            predictions=np.load(verify_pin(record['private_predictions']),allow_pickle=True).item()
            if set(predictions)!={'O','E','R'}:raise ValueError('inner complete shared O/E/R policies required')
            for policy,pred in predictions.items():
                if set(pred)!={q['id'] for q in qs}:raise ValueError('inner full sequence axes changed')
                for q,a,j,b in zip(qs,aa,jj,bb):
                    labels=np.asarray(pred[q['id']]);actions=partition(b,propose(b,a.argmax(1),j.argmax(1)))[0]
                    if labels.shape!=(len(q['labels']),) or labels.dtype.kind not in 'iu' or not np.isin(labels,np.arange(5)).all():raise ValueError('inner labels invalid')
                    changed=not np.array_equal(labels,b)
                    if changed and not any(np.array_equal(labels,apply(b,action)) for action in actions if policy!='O' or action.old):
                        raise ValueError('inner prediction outside single declared canonical guarded action domain')
                    if majority(labels)!=majority(b):raise ValueError('inner global guard failed')
                metric={**metrics(qs,pred)[0],**short_metrics(qs,pred,16)}
                verify_reference(metric,record['policies'][policy])
            info={'seed':seed,'arm':arm,'step':cursor,'path':str(path.relative_to(ROOT)),'sha256':sha(path),
                  'inner_record_path':str(record_path.relative_to(ROOT)),'inner_record_sha256':sha(record_path),
                  'identity':identity}
            milestones.append(info);inner_rows.append(info)
        latest=ProposalRegressor(arm,seed);latestopt=optimizer(latest)
        restore(latest,latestopt,torch.load(folder/'latest.pt',map_location='cpu',weights_only=True),identity,normalizer,scales)
        if any(not torch.equal(latest.state_dict()[k],v) for k,v in model.state_dict().items()):raise ValueError('final/latest model mismatch')
        for start in range(0,6000,100):
            block=trace[start:start+100]
            curves.append({'seed':seed,'arm':arm,'start_update':start+1,'end_update':start+100,
                **{key:float(np.mean([r[key] for r in block])) for key in ['loss','gradient_norm','lr']},
                **{key:None if block[0][key] is None else float(np.mean([r[key] for r in block])) for key in ['native_MSE','constructed_MSE']}})
        models.append({**row,'normalizer':inputs['normalizers'][arm],'target_scales':inputs['target_scales'],'history_all_tensors_live':live,'milestones':milestones,
                       'trace_sha256':sha(journal.trace),'actual_Adam_update':6000,'parameters':sum(p.numel() for p in model.parameters())})
    if len(inner_rows)!=12 or len(curves)!=240:raise ValueError('complete curve/inner matrix required')
    if torch.cuda.is_initialized():raise ValueError('CPU audit unexpectedly initialized CUDA')
    write(PUB/'models_manifest.json',{'passed':True,'milestone_payload_format':'full_model_Adam_RNG_normalizer_sharedscales;additional_private_retention_vs_weight_only_wording;training_recipe_unchanged;storage_preforecast','models':models,'inner_records':inner_rows,
        'physical_models':4,'formal_optimizer_updates':24000,'inner_record_count':12,'inner_policy_metric_sets':36,
        'all_models_and12_inner_records_frozen':True,'all12_inner_metrics_independently_recounted':True,
        'every_tensor_history_live_and_drift_and_Adam_second_moment_nonzero':True,
        'no_outer_evaluation_or_inner_selection':True,'formal_freeze_git':formal['git_commit'],
        'formal_completion_sha256':sha(PUB/'formal_completion.json'),'audit_freeze_git':freeze['git_commit'],'epoch':time.time()})
    write(PUB/'training_curves.json',{'bins':curves,'bin_count':240,'inner_diagnostic_only':True})
    write(PUB/'readout_plan.json',readout_plan())
    write(RUN/'runtimecoordination.json',{'phase':'all4heads_12inner_audited_pending_actualGit_outer_seal','model_pid':None,
          'formal_optimizer_updates':24000,'epoch':time.time(),'next_action':'actualGit freeze all4final weights+12inner records then fixed outer source cache/readout'})
    print({'audit_passed':True,'models':4,'formal_updates':24000,'inner_records':12},flush=True)

if __name__=='__main__':main()
