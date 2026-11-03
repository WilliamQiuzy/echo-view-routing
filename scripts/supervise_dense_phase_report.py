#!/usr/bin/env python3
"""Aggregate-only figures through the existing A57 lock, ledger and sole queue."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import supervise_annotation_coverage as queue
queue.OUT=ROOT/'runs/a57_dense_sequence_v1'
queue.COMMANDS={'phase_plots':('scripts/plot_dense_phase.py','run',3600,False)}
if __name__=='__main__':queue.main()
