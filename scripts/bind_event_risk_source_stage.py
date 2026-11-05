#!/usr/bin/env python3
"""Bind a completed logical-source manifest to actual producer receipts locally."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_source_stage import build


def main():
    args=execution_parser(__doc__).parse_args()
    if not authorize(args):return 0
    verify_freeze('producer_freeze.json');producer=read(PUB/'producer_manifest.json');identity=read(PUB/'producer_identity.json');paths=sorted(PUB.glob('process_producer_*.json'));attempts=[read(p) for p in paths];pins={r['receipt_id']:pin(p) for r,p in zip(attempts,paths)}
    value=build(producer,sha(PUB/'producer_manifest.json'),identity,sha(PUB/'producer_identity.json'),attempts,pins);path=PUB/'source_stage_binding.json'
    if path.exists():raise ValueError('keep accounting binding immutable')
    write(path,value);print({'logical_source_stage':'sources','actual_source_stage':'producer','attempts':len(attempts),'new_forward':0});return 0


if __name__=='__main__':raise SystemExit(main())
