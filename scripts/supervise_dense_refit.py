#!/usr/bin/env python3
"""All adaptive and qualification stages share the original exclusive A57 queue."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import supervise_annotation_coverage as s
s.OUT=ROOT/'runs/a57_dense_sequence_v1'
s.COMMANDS={'MD_checks':('scripts/check_dense_multidepth.py','run',540,True),'refit_checks':('scripts/check_dense_refit.py','run',540,True),'refit_admission':('scripts/admit_dense_refit.py','run',1200,False),'refit_formal':('scripts/train_dense_refit.py','formal',14400,True),'refit_audit':('scripts/report_dense_refit.py','audit',3600,False),'refit_readout':('scripts/report_dense_refit.py','readout',7200,True),'refit_aggregate':('scripts/report_dense_refit.py','aggregate',7200,False)}
if __name__=='__main__':s.main()
