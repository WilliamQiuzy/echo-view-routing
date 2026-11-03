#!/usr/bin/env python3
"""MD stages share the original A57 exclusive lock, active PID and cost ledger."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import supervise_annotation_coverage as s
s.OUT=ROOT/'runs/a57_dense_sequence_v1'
s.COMMANDS={'MD_checks':('scripts/check_dense_multidepth.py','run',540,True),'MD_admission':('scripts/admit_dense_multidepth.py','run',1200,False),'MD_formal':('scripts/train_dense_multidepth.py','formal',43200,True),'MD_audit':('scripts/report_dense_multidepth.py','audit',3600,False),'MD_decoder':('scripts/report_dense_multidepth.py','calibrate',14400,False),'MD_readout':('scripts/report_dense_multidepth.py','readout',14400,True),'MD_aggregate':('scripts/report_dense_multidepth.py','aggregate',14400,False)}
if __name__=='__main__':s.main()
