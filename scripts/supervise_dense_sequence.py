#!/usr/bin/env python3
"""Reuse the validated single-process supervisor for the new A57 namespace."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import supervise_annotation_coverage as s
s.OUT=ROOT/'runs/a57_dense_sequence_v1'
s.COMMANDS={'check':('scripts/check_dense_sequence.py','run',3600,True),'pipeline_check':('scripts/train_dense_sequence.py','pipeline_check',3600,True),'formal':('scripts/train_dense_sequence.py','formal',72000,True)}
if __name__=='__main__':s.main()
