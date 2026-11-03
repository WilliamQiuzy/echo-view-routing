#!/usr/bin/env python3
"""Only fixed cached-report additions; same accounted queue and hard envelopes."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import supervise_structured_accounted as accounted
COMMANDS={'curve_seal':('scripts/export_structured_curves.py','seal',900),'package':('scripts/package_structured_evidence.py','run',1800)}
if __name__=='__main__':
    stage=sys.argv[1];script,mode,limit=COMMANDS[stage];r=accounted.supervisor.execute(stage,[str(ROOT/'.venv/bin/python'),script,mode],False,limit);print(json.dumps(r),flush=True);raise SystemExit(accounted.supervisor.stage_exit_status(r))
