#!/usr/bin/env python3
"""Future stages reuse the immutable v2 queue with corrected prior context costs."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import supervise_structured_v2 as supervisor
from scripts.structured_common import RUN,read
from echo_routing.temporal.structured_context import classify_context_records

original=supervisor.remaining_limits
def corrected(budget,rows,now,gpu,closing=False):
    guard=read(RUN/'context_guard_receipt.json') if (RUN/'context_guard_receipt.json').exists() else None
    return original(budget,classify_context_records(rows,guard),now,gpu,closing)
supervisor.remaining_limits=corrected
if __name__=='__main__':supervisor.main()
