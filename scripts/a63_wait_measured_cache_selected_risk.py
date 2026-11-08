#!/usr/bin/env python3
"""Qualified whole-caller wait4, changing only the reviewed guardian entry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.selected_risk_resume_contract import stage_graph


def spawn_guardian(root,command,**kwargs):
    if len(command)<2 or command[1]!=str(Path(root)/'scripts/a63_resume_selected_risk.py'):
        raise ValueError('only the exact qualified guardian command may be substituted')
    return subprocess.Popen([command[0],str(Path(root)/'scripts/a63_resume_measured_cache_selected_risk.py'),*command[2:]],**kwargs)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--stage',required=True,choices=sorted(stage_graph()))
    p.add_argument('--runtime-spec');p.add_argument('--runtime-spec-sha');p.add_argument('--owner-sha');a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'preparation_only':True,'execution_authorized':False,'new_child_created':False,'stage':a.stage}));return 0
    if not all([a.runtime_spec,a.runtime_spec_sha,a.owner_sha]):p.error('original owner and raw runtime spec pins required')
    from echo_routing.temporal.selected_risk_resume_runtime import bootstrap
    from scripts.a63_resume_measured_cache_selected_risk import verify_extra_sources
    from scripts.a63_wait_resume_selected_risk import execute
    pin={'path':a.runtime_spec,'sha256':a.runtime_spec_sha};ctx,spec=bootstrap(ROOT,pin,a.owner_sha)
    if spec['stage']!=a.stage:raise ValueError('caller/raw spec stage differs')
    verify_extra_sources(ctx,spec)
    result=execute(ctx,spec,pin,spawn=lambda command,**kw:spawn_guardian(ROOT,command,**kw))
    print(json.dumps(result,sort_keys=True),flush=True);return result['stage_exit_status']


if __name__=='__main__':raise SystemExit(main())
