#!/usr/bin/env python3
"""Exact two-stage CPU extension front-end to the unchanged A63 guardian engine."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.selected_risk_common import *
from scripts.supervise_selected_risk import execute,meter_entry,stage_exit_status
from scripts.selected_risk_shared_contracts import EXTENSIONS
from echo_routing.temporal.selected_risk_budget import totals

def main():
    p=execution_parser(__doc__);p.add_argument('--seconds',type=float,required=True);p.add_argument('stage',choices=sorted(EXTENSIONS));p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if not authorize(a):return 0
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if len(cmd)<3 or Path(cmd[1]).resolve() not in {(ROOT/p).resolve() for p in EXTENSIONS[a.stage][0]} or '--execute' not in cmd[2:]:raise ValueError('one exact authorized CPU extension child, no arbitrary CLI')
    r=execute(a.stage,cmd,RUN,PUB,totals(actual_receipts()),a.seconds,False);print(json.dumps(r),flush=True);return stage_exit_status(r)
if __name__=='__main__':raise SystemExit(meter_entry(main,RUN,PUB))
