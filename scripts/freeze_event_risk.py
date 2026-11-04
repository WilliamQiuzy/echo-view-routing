#!/usr/bin/env python3
"""Build exact actual-Git A62 stage contracts locally; no models or remote writes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,ast,subprocess,time,resource,socket
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.safety_git import build,verify


from echo_routing.temporal.event_risk_contracts import source_closure as _source_closure,stage_spec,validate_coverage

def source_closure(paths):return _source_closure(ROOT,paths)


def make(name,paths,public_inputs):
    if not name.endswith('_freeze.json') or '/' in name:raise ValueError('plain stage freeze name required')
    target=RUN/name
    if target.exists():raise ValueError('executed/frozen contract immutable; use a distinct version')
    entries,required_inputs=stage_spec(name)
    if not set(entries)<=set(paths):raise ValueError('explicit exact required stage entry coverage missing/wrong')
    public_inputs=list(dict.fromkeys([*required_inputs,*public_inputs]))
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();cfg=read(CFG)
    explicit=[str(CFG.relative_to(ROOT)),cfg['card']['path'],cfg['authorization']['path'],cfg['basis']['path'],*public_inputs]
    objects={p:sha(ROOT/p) for p in source_closure(paths+explicit)}
    validate_coverage(ROOT,name,objects,explicit)
    proof=build(ROOT,commit,objects)
    if any(proof['files'][p]['sha256']!=h for p,h in objects.items()):raise ValueError('commit every exact source/input byte before stage freeze')
    p=RUN/'Git_proofs'/name;write(p,proof);contract={'git_commit':commit,'objects':objects,'actual_Git_object_proof':pin(p),'stage':name,'fixed_deadline_epoch':1793867561,'created_epoch':time.time(),'runtime_private_arrays_bound_by_committed_public_SHA_manifests':True};write(target,contract)
    write(PUB/(('outer_stage' if name=='outer_freeze.json' else name.removesuffix('_freeze.json'))+'_identity.json'),contract);return contract


def basis():
    cfg=read(CFG);paths=[v['path'] for v in cfg['prior_A61'].values()]+list(cfg['readonly_code_pins']);paths=sorted(set(paths));proof=build(ROOT,cfg['prior_scientific_git'],paths)
    expected={p:sha(ROOT/p) for p in paths};verify(proof,cfg['prior_scientific_git'],expected)
    p=RUN/'Git_proofs/prior_science.private.json';write(p,proof)
    row={'git_commit':cfg['prior_scientific_git'],'objects':expected,'actual_Git_object_proof':pin(p),'prior_sealed_science_not_retrained':True};write(PUB/'prior_science_binding.json',row);return row


def outer():
    paths=[str((PUB/f).relative_to(ROOT)) for f in ['models_manifest.json','calibration.json','readout_plan.json']];commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();proof=build(ROOT,commit,paths)
    if any(sha(ROOT/p)!=proof['files'][p]['sha256'] for p in paths):raise ValueError('commit allthree audited model/calibration/plan bytes before outer seal')
    p=RUN/'Git_proofs/outer_head_seal.private.json';write(p,proof);models=read(PUB/'models_manifest.json');out={'models_manifest_sha256':sha(PUB/'models_manifest.json'),'calibration_sha256':sha(PUB/'calibration.json'),'readout_plan_sha256':sha(PUB/'readout_plan.json'),'all_final_checkpoints':[{k:r[k] for k in ['path','sha256']} for r in models['models']+models['fixed_A61_models']],'git_commit':commit,'actual_Git_object_proof':pin(p)}
    from echo_routing.temporal.event_risk_audit import validate_barrier
    validate_barrier(models,out,out['models_manifest_sha256'],out['calibration_sha256'],out['readout_plan_sha256'],proof);write(PUB/'outer_identity.json',out);return out


def main():
    p=execution_parser(__doc__);p.add_argument('--name');p.add_argument('--entry',action='append',default=[]);p.add_argument('--input',action='append',default=[]);p.add_argument('--prior-science',action='store_true');p.add_argument('--outer-head-seal',action='store_true');a=p.parse_args()
    if not authorize(a):return 0
    result=basis() if a.prior_science else outer() if a.outer_head_seal else make(a.name,a.entry,a.input);print({'stage_contract':result.get('stage'),'actualGit':result['git_commit'],'objects':len(result.get('objects',{}))},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
