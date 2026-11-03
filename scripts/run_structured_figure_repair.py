#!/usr/bin/env python3
"""Rendering-only repair under the existing accounted queue and sealed summary."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import supervise_structured_accounted as accounted
if __name__=='__main__':
    r=accounted.supervisor.execute('figure_layout',[str(ROOT/'.venv/bin/python'),'scripts/plot_structured_evidence_v2.py'],False,900);print(json.dumps(r),flush=True);raise SystemExit(accounted.supervisor.stage_exit_status(r))
