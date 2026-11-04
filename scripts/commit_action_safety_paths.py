#!/usr/bin/env python3
"""Default dry-run for explicit newly copied A61 paths; never copies payloads.

Example paths JSON: [{"path":"docs/results/2026-11-04_action_safety_ranking/x.json",
"sha256":"...","mode":"100644"}]. Apply additionally requires the exact index
SHA256 returned by the reviewed dry-run. No automatic partial-state recovery.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from echo_routing.temporal.safety_git_integration import integrate,IntegrationError


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--expected-head',required=True);p.add_argument('--expected-branch',required=True);p.add_argument('--paths-json',required=True);p.add_argument('--message',required=True);p.add_argument('--expected-index-sha256');p.add_argument('--apply',action='store_true');a=p.parse_args()
    try:
        files=json.loads(Path(a.paths_json).read_text())
        if not isinstance(files,list):raise ValueError('explicit JSON list of reviewed new A61 path/SHA256 records required')
        result=integrate(a.repo,a.expected_head,a.expected_branch,files,a.message,apply=a.apply,expected_index_sha256=a.expected_index_sha256)
    except IntegrationError as e:
        print(json.dumps(e.report,indent=2));return 2 if e.report['status']=='partial' else 1
    except Exception as e:
        print(json.dumps({'status':'rejected','error':repr(e),'no_integration_apply_entered':True}));return 1
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
