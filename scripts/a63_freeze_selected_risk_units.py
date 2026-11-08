#!/usr/bin/env python3
"""Build prospective A63 shared-object proofs, preserving all old raw contracts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['GIT_NO_REPLACE_OBJECTS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.a63_selected_risk_costunit_contract import *
from echo_routing.temporal.safety_git import build
from echo_routing.temporal.selected_risk_storage_engineering import pack_proof,expand_proof

def make(stage):
    target=RUN/'contracts'/(stage+'.json')
    if target.exists():raise ValueError('never overwrite an old executed/frozen contract')
    old=verify_freeze('metadata');cfg=read(CFG);validate_config(cfg);refs=read(verify_pin(cfg['source_refs']));objects={p:sha(ROOT/p) for p in required(stage)};validate_frozen_prior(objects)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();current=dict(objects);historical={}
    for p,v in refs['historical_frozen_imports'].items():
        if p not in current:continue
        if current.pop(p)!=v['sha256']:raise ValueError('historical source bytes drift')
        proof=build(ROOT,v['git_commit'],[p]);verify_Git(proof,v['git_commit'],{p:v['sha256']})
        if proof['files'][p]['blob_oid']!=v['blob_oid']:raise ValueError('historical blob differs')
        fp=RUN/'Git_proofs'/stage/(Path(p).name+'.json');write(fp,proof);historical[p]={**v,'proof':pin(fp)}
    full=build(ROOT,commit,current);pool=read(verify_pin(old['actual_Git_object_proof']));packed=pack_proof(full,pool)
    if expand_proof(packed,pool)!=full:raise ValueError('shared encoding changed any original actual Git proof byte')
    verify_Git(full,commit,current);pp=RUN/'Git_proofs'/stage/'current.shared.private.json';write(pp,packed)
    out={'stage':stage,'git_commit':commit,'objects':objects,'historical_frozen_imports':historical,'actual_Git_object_proof':pin(pp),'shared_object_pool':old['actual_Git_object_proof'],'proof_encoding':'A63_shared_Git_objects_v1','fixed_T0_epoch':1793853840,'fixed_deadline_epoch':1793897040,'created_epoch':time.time(),'runtime_private_arrays_bound_by_committed_public_SHA':True,'unshared_JSON_bytes':len(json.dumps(full,sort_keys=True)),'shared_JSON_bytes':pp.stat().st_size};write(target,out);write(PUB/(stage+'_identity.json'),out);verify_shared_freeze(stage);return out

def main():
    p=execution_parser(__doc__);p.add_argument('--stage',required=True);a=p.parse_args()
    if not authorize(a):return 0
    r=make(a.stage);print({'stage':a.stage,'git':r['git_commit'],'objects':len(r['objects']),'historical':len(r['historical_frozen_imports']),'unshared_bytes':r['unshared_JSON_bytes'],'shared_bytes':r['shared_JSON_bytes']},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
