#!/usr/bin/env python3
"""Cache-only application of both preregistered source-advancement questions."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.temporal.direct_view_transfer import transfer_qualification

if __name__ == "__main__":
    pub = ROOT / "docs/results/2026-11-01_direct_view_evidence/pilot"
    path = pub / "independent_source_final.json"
    audited = json.loads(path.read_text())
    assert audited["passed"] and audited["cache_only_no_new_model_forwards"]
    result = transfer_qualification(audited["source_metrics"], audited["source_advancement_gate"]["passed"])
    result["independent_source_audit_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["new_model_forwards"] = 0
    result["rules_precede_source_readout"] = "A51-P original source rule and A51-T separate absolute source-classifier rule both fixed before the first source heldout model forward."
    (pub / "transfer_qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
