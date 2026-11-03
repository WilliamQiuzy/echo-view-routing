#!/usr/bin/env python3
"""One cached-only aggregate package job, after all primary comparisons are sealed."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.structured_common import verify_freeze
from scripts.export_structured_curves import main as curves
from scripts.plot_structured_evidence import main as plot
if __name__=='__main__':verify_freeze('package_freeze.json');curves('export');plot()
