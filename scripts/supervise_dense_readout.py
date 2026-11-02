#!/usr/bin/env python3
"""A57 final independent CPU audits and frozen once-per-sequence GPU reader."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import supervise_annotation_coverage as s
s.OUT=ROOT/'runs/a57_dense_sequence_v1'
s.COMMANDS={'audit_models':('scripts/audit_dense_sequence.py','run',7200,False),'decoder_calibration':('scripts/calibrate_dense_sequence.py','run',14400,False),'outer_metadata':('scripts/prepare_dense_outer.py','run',1200,False),'outer_readout':('scripts/readout_dense_sequence.py','run',28800,True),'aggregate':('scripts/aggregate_dense_sequence.py','run',14400,False),'plots':('scripts/plot_dense_sequence.py','run',1200,False)}
if __name__=='__main__':s.main()
