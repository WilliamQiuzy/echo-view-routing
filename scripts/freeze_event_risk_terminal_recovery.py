#!/usr/bin/env python3
"""Independent exact terminal-only recovery contract; never rewrite failed v1."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_contracts import source_closure
from echo_routing.temporal.event_risk_terminal_recovery import validate_failure
from echo_routing.temporal.safety_git import build


def main():
 p=execution_parser(__doc__);a=p.parse_args()
 if not authorize(a):return 0
 target=RUN/'terminal_recovery/v2_freeze.json'
 if target.exists():raise ValueError('immutable independent recovery contract already exists')
 cfg=read(CFG);binding=read(PUB/'terminal_failure_binding.json');validate_failure(ROOT,binding);old=read(PUB/'terminal_identity.json')
 entries=['scripts/close_event_risk_terminal_recovery.py','scripts/wait_event_risk_terminal_recovery.py'];inputs=[str(CFG.relative_to(ROOT)),cfg['card']['path'],cfg['authorization']['path'],cfg['basis']['path'],str((PUB/'terminal_failure_binding.json').relative_to(ROOT)),str((PUB/'local_cpu_receipts_terminal_recovery.json').relative_to(ROOT))]
 objects={p:sha(ROOT/p) for p in source_closure(ROOT,entries+inputs+list(binding['objects'])+list(old['objects']))};commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();proof=build(ROOT,commit,objects)
 if any(proof['files'][p]['sha256']!=h for p,h in objects.items()):raise ValueError('commit every exact recovery object before freeze')
 pp=RUN/'Git_proofs/terminal_recovery_v2.private.json';write(pp,proof);out={'stage':'terminal_recovery_v2','git_commit':commit,'objects':objects,'actual_Git_object_proof':pin(pp),'fixed_deadline_epoch':1793867561,'created_epoch':time.time(),'old_terminal_and_scientific_contracts_not_overwritten':True}
 write(target,out);write(PUB/'terminal_recovery_identity.json',out);print({'actualGit':commit,'objects':len(objects)});return 0
if __name__=='__main__':raise SystemExit(main())
