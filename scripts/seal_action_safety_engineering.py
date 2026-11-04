#!/usr/bin/env python3
"""Local minimal private proofs for two already-executed cost-stage versions."""
import time,resource,socket,os
_START=time.time()
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.safety_common import RUN,PUB,read,write,sha
from echo_routing.temporal.safety_git import build
from echo_routing.temporal.safety_engineering_history import EXPECTED,validate


def main():
    path=RUN/'engineering_Git_objects.private.json';receipt=PUB/'engineering_Git_builder_resource_receipt.json'
    if path.exists() or receipt.exists():raise ValueError('engineering proof exists; no repeated builder')
    raw={n:(PUB/(n+'.json')).read_bytes() for n in EXPECTED};identities={n:read(PUB/(n+'.json')) for n in EXPECTED}
    bundle={'contracts':{n:{'proof':build(ROOT,pin['commit'],identities[n]['objects'])} for n,pin in EXPECTED.items()}}
    checked=validate(bundle,raw,ROOT);write(path,bundle);write(PUB/'engineering_Git_proof_identity.json',{'path':str(path.relative_to(ROOT)),'sha256':sha(path),'original_contracts':{n:pin['identity_sha256'] for n,pin in EXPECTED.items()},'server_only_actualGit_proof':True,'locally_verified':checked})
    own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(receipt,{'scope':'local exact historical cost Git builder;0NN/SGD;minimal proof private only','measurement_host':socket.gethostname(),'model_pid':os.getpid(),'started_epoch':_START,'sample_epoch':time.time(),'rusage_self_plus_waited_children_cpu_seconds_through_sample':own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,'final_receipt_write_exit_and_launcher_tail_unmetered':True,'observed_worker_exit_code':None})
    print({'historical_cost_contracts':2,'bindings':checked['historical_object_bindings'],'private_proof_built':True},flush=True)
if __name__=='__main__':main()
