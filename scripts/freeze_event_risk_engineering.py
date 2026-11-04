#!/usr/bin/env python3
"""Exact actual-Git independent engineering contract; no overwritten old freezes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_engineering import stage_spec,validate_coverage
from echo_routing.temporal.event_risk_contracts import source_closure
from echo_routing.temporal.safety_git import build


def make(name):
    target=RUN/'engineering_contracts'/name
    if target.exists():raise ValueError('independent engineering contract immutable')
    entries,inputs=stage_spec(name);cfg=read(CFG);base=[str(CFG.relative_to(ROOT)),cfg['card']['path'],cfg['authorization']['path'],cfg['basis']['path']]
    objects={p:sha(ROOT/p) for p in source_closure(ROOT,entries+inputs+base)}
    validate_coverage(ROOT,name,objects,base);commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();proof=build(ROOT,commit,objects)
    if any(proof['files'][p]['sha256']!=h for p,h in objects.items()):raise ValueError('commit every exact new source/input first')
    p=RUN/'Git_proofs'/'engineering'/name;write(p,proof)
    out={'stage':name,'git_commit':commit,'objects':objects,'actual_Git_object_proof':pin(p),'fixed_deadline_epoch':1793867561,'created_epoch':time.time(),'executed_metadata_and_cost_contracts_not_overwritten':True}
    write(target,out);write(PUB/('engineering_'+name.removesuffix('_freeze.json')+'_identity.json'),out);return out


if __name__=='__main__':
    parser=execution_parser(__doc__);parser.add_argument('--name',required=True);args=parser.parse_args()
    if authorize(args):
        v=make(args.name);print({'independent_contract':v['stage'],'actualGit':v['git_commit'],'objects':len(v['objects'])})
