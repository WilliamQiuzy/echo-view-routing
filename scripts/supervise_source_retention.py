#!/usr/bin/env python3
"""Reuse the sole validated A55 supervisor engine for A56's namespace."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import supervise_annotation_coverage as supervisor
supervisor.OUT=ROOT/'runs/a56_source_retention_v1'
supervisor.COMMANDS={k:('scripts/train_source_retention.py',k,limit,gpu) for k,limit,gpu in [('prepare',600,False),('check',1200,True),('train',7200,True),('audit_models',600,False),('readout',600,True)]}
if __name__=='__main__':supervisor.main()
