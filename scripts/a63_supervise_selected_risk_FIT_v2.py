#!/usr/bin/env python3
"""One FIT-only GPU stage using the unchanged owned guardian/entry meter."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.selected_risk_common import *
from scripts.supervise_selected_risk import execute,meter_entry,stage_exit_status
from scripts.a63_selected_risk_FIT_v2_contract import STAGE,ENTRY,verify_shared_freeze
from echo_routing.temporal.selected_risk_budget import totals

def main():
    p=execution_parser(__doc__);p.add_argument('--seconds',type=float,required=True);p.add_argument('stage',choices=[STAGE]);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if not authorize(a):return 0
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or not 0<a.seconds<=4500:raise ValueError('one explicit CUDA0 FIT source stage <=4500 occupied seconds required')
    if len(cmd)<3 or Path(cmd[1]).resolve()!=(ROOT/ENTRY).resolve() or '--execute' not in cmd[2:] or cmd[cmd.index('--phase')+1:]!=['FIT']:raise ValueError('one exact authorized FIT producer, no arbitrary child')
    verify_shared_freeze(STAGE)
    row=execute(STAGE,cmd,RUN,PUB,totals(actual_receipts()),a.seconds,True);print(json.dumps(row),flush=True);return stage_exit_status(row)
if __name__=='__main__':raise SystemExit(meter_entry(main,RUN,PUB))
