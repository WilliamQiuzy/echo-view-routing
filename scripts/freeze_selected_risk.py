#!/usr/bin/env python3
"""A63 local exact Git stage contracts with five explicit historical imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['GIT_NO_REPLACE_OBJECTS']='1'
import sys,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.selected_risk_common import *
from echo_routing.temporal.selected_risk_contracts import source_closure,stage_spec,validate_coverage
from echo_routing.temporal.safety_git import build,verify


def make(stage):
    target=RUN/'contracts'/(stage+'.json')
    if target.exists():raise ValueError('frozen/executed contracts immutable; no implicit overwrite')
    cfg=read(CFG);validate_config(cfg);refs=read(verify_pin(cfg['source_refs']));entries,inputs=stage_spec(stage)
    base=[relative(CFG),cfg['card']['path'],cfg['basis']['path'],cfg['authorization']['path'],cfg['source_refs']['path'],'scripts/supervise_selected_risk.py']
    basis=read(verify_pin(cfg['basis']));base += [relative(prior_path(p)) for p in basis['predecessor_public_and_source_pins']]
    objects={p:sha(ROOT/p) for p in source_closure(ROOT,entries+inputs+base)}
    validate_frozen_prior(objects)
    validate_coverage(ROOT,stage,objects,base);commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();current=dict(objects);historical={}
    for p,v in refs['historical_frozen_imports'].items():
        if p not in current:continue
        if current.pop(p)!=v['sha256']:raise ValueError('frozen historical dependency bytes changed')
        proof=build(ROOT,v['git_commit'],[p]);verify(proof,v['git_commit'],{p:v['sha256']})
        if proof['files'][p]['blob_oid']!=v['blob_oid']:raise ValueError('historical dependency raw blob differs')
        fp=RUN/'Git_proofs'/stage/(Path(p).name+'.json');write(fp,proof);historical[p]={**v,'proof':pin(fp)}
    proof=build(ROOT,commit,current);verify(proof,commit,current)
    pp=RUN/'Git_proofs'/stage/'current.private.json';write(pp,proof)
    out={'stage':stage,'git_commit':commit,'objects':objects,'historical_frozen_imports':historical,'actual_Git_object_proof':pin(pp),'fixed_T0_epoch':1793853840,'fixed_deadline_epoch':1793897040,'created_epoch':time.time(),'runtime_private_arrays_bound_by_committed_public_SHA':True};write(target,out);write(PUB/(stage+'_identity.json'),out);return out


def main():
    p=execution_parser(__doc__);p.add_argument('--stage',required=True);a=p.parse_args()
    if not authorize(a):return 0
    out=make(a.stage);print({'stage':a.stage,'actualGit':out['git_commit'],'objects':len(out['objects']),'historical_imports':len(out['historical_frozen_imports'])},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
