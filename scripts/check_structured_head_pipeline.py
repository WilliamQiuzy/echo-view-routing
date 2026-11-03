#!/usr/bin/env python3
"""New lifecycle-only real FIT check; reuses frozen engineering posteriors, no media/GPU."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from types import SimpleNamespace
from scripts.structured_common import ROOT,RUN,PUB,read,write,sha,sequences,verify_freeze
from echo_routing.temporal.structured_evidence import StructuredEvidence
from echo_routing.temporal.structured_training import optimizer
from echo_routing.temporal.structured_loop import run_updates,_equal
from echo_routing.temporal.structured_lifecycle import audit_payload
from echo_routing.temporal.structured_evaluation import decode_bank

def main():
    freeze=verify_freeze('head_check_freeze.json');verify_freeze('producer_freeze.json');torch.set_num_threads(1);torch.use_deterministic_algorithms(True);checks=read(PUB/'checks.json');path=RUN/'engineering/FIT_posteriors.private.npy';assert sha(path)==checks['FIT_engineering_probability_sha256'];values=np.load(path,allow_pickle=True).item();profiles=read(RUN/'engineering/profiles.private.json');assert sha(RUN/'engineering/profiles.private.json')==checks['FIT_engineering_profile_sha256'];fit=sequences('fit');records=[];physical=0
    for profile in ['median','longest']:
        indices=profiles[profile];qs=[fit[int(i)] for i in indices];batch=[tuple(torch.from_numpy(v) for v in [values['a08'][str(i)],values['jepa'][str(i)],np.asarray(fit[int(i)]['labels'],np.int64)]) for i in indices];bank=SimpleNamespace(sequences=qs,values=batch);plan=np.tile([0,1],(6,1))
        for arm in ['D0','D1','D2','D3']:
            prefix=RUN/'engineering/head_pipeline'/profile/arm;identity='realFITlifecycle|'+freeze['git_commit']+'|'+profile+'|'+arm;callback_seconds=0
            def run(name,stop):
                nonlocal callback_seconds,physical
                model=StructuredEvidence(arm,2026110358);opt=optimizer(model);folder=prefix/name;before=len((folder/'physical_updates.jsonl').read_text().splitlines()) if (folder/'physical_updates.jsonl').exists() else 0
                def callback(step,path):
                    nonlocal callback_seconds
                    start=time.time();decode_bank(model,bank,folder/f'inner_step{step}',sha(path));callback_seconds+=time.time()-start
                start=time.time();result=run_updates(model,opt,lambda _:batch,plan,folder,identity,stop=stop,checkpoint_every=2,milestones=[2,4,6],on_milestone=callback);elapsed=time.time()-start;physical+=result['physical_optimizer_updates']-before;payload=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True);audit_payload(model,payload,identity);return payload,elapsed
            startcallback=callback_seconds;reference,elapsed=run('reference',6);reference_callback=callback_seconds-startcallback;repeat,_=run('repeat',6);_,_=run('resume',2);resumed,_=run('resume',6);assert _equal(reference['model'],repeat['model']) and _equal(reference['optimizer'],repeat['optimizer']) and _equal(reference['model'],resumed['model']) and _equal(reference['optimizer'],resumed['optimizer']);moments=audit_payload(StructuredEvidence(arm,2026110358),reference,identity);assert all(moments['nonzero_second_moment_tensors'].values());records.append({'profile':profile,'arm':arm,'complete_update_seconds_excluding_MAP_upper':(elapsed-reference_callback)/6,'valid_positions_per_update':sum(len(q['labels']) for q in qs),'per_valid_position_seconds_complete_pipeline':(elapsed-reference_callback)/6/sum(len(q['labels']) for q in qs),'whole_repeat_and_disk2plus4_model_Adam_exact':True,'full_model_optimizer_recipe_cursor_audit_passed':True,'three_atomic_milestones_and_clean_prefix_no_extra_decode':True,'engineering_optimizer_updates':18})
    assert physical==144;result={'passed':True,'freeze_git':freeze['git_commit'],'records':records,'physical_engineering_optimizer_updates_this_stage':physical,'formal_optimizer_updates':0,'new_visual_producer_forwards':0,'new_media_reads':0,'backend':'cpu','torch_CPU_rng_restored':True,'NumPy_and_Python_global_RNG_not_used_by_frozen_plan_loop':True,'epoch':time.time()};write(PUB/'head_pipeline_checks.json',result);print(json.dumps({'head_pipeline_checks_passed':True,'physical_engineering_updates':physical,'maximum_pipeline_seconds_per_valid_position':max(r['per_valid_position_seconds_complete_pipeline'] for r in records)}),flush=True)
if __name__=='__main__':main()
