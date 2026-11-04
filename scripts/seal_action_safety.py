#!/usr/bin/env python3
"""Local actual-Git proof builder; only minimal private proofs go to new RUN."""
import time,resource
_START=time.time()
import os,sys,argparse,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.safety_common import RUN,PUB,sha,read,write,digest
from echo_routing.temporal.safety_git import build
from echo_routing.temporal.safety_history import EXPECTED,validate as validate_history
from echo_routing.temporal.safety_barrier import validate as validate_outer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scope',choices=['history','outer']);parser.add_argument('--commit');args=parser.parse_args();receipt=PUB/(args.scope+'_Git_builder_resource_receipt.json')
    if receipt.exists():raise ValueError('builder receipt exists; do not silently repeat/rewrite')
    if args.scope=='history':
        path=RUN/'historical_Git_objects.private.json'
        if path.exists():raise ValueError('history proof exists; no overwrite')
        raw={n:(PUB/(n+'.json')).read_bytes() for n in EXPECTED};contracts={n:read(PUB/(n+'.json')) for n in EXPECTED}
        proofs={'contracts':{n:{'proof':build(ROOT,pin['commit'],contracts[n]['objects'])} for n,pin in EXPECTED.items()}}
        checked=validate_history(proofs,raw,ROOT);write(path,proofs)
        write(PUB/'historical_Git_proof_identity.json',{'path':str(path.relative_to(ROOT)),'sha256':sha(path),'original_contracts':{n:pin['identity_sha256'] for n,pin in EXPECTED.items()},'proofs_are_server_private_not_in_Git':True,'locally_verified':checked})
    else:
        if not args.commit:raise ValueError('explicit actual committed model/calibration/plan revision required')
        path=RUN/'preouter_Git_objects.private.json';target=PUB/'outer_identity.json'
        if path.exists() or target.exists():raise ValueError('preouter seal exists; no overwrite')
        names=['models_manifest.json','calibration.json','readout_plan.json'];paths=[str((PUB/n).relative_to(ROOT)) for n in names];proof=build(ROOT,args.commit,paths);models=read(PUB/'models_manifest.json')
        seal={'git_commit':args.commit,'models_manifest_sha256':sha(PUB/names[0]),'calibration_sha256':sha(PUB/names[1]),'readout_plan_sha256':sha(PUB/names[2]),'all_final_checkpoints':[{'path':r['path'],'sha256':r['sha256']} for r in models['models']],'actual_Git_object_proof':{'path':str(path.relative_to(ROOT)),'sha256':digest(proof)},'server_must_verify_all10_private_checkpoint_bytes_before_first_outer_forward':True}
        validate_outer(models,seal,*[sha(PUB/n) for n in names],proof)
        write(path,proof);seal['actual_Git_object_proof']['sha256']=sha(path);write(target,seal)
    own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(receipt,{'scope':'local actualGit administrative builder; no NN/optimizer;proof private RUN only','measurement_host':socket.gethostname(),'model_pid':os.getpid(),'started_epoch':_START,'sample_epoch':time.time(),'rusage_self_plus_waited_children_cpu_seconds_through_sample':own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,'final_receipt_write_exit_and_launcher_tail_unmetered':True,'observed_worker_exit_code':None})
    print({'scope':args.scope,'proof_path':str(path.relative_to(ROOT)),'actual_Git_proof_built':True},flush=True)
if __name__=='__main__':main()
