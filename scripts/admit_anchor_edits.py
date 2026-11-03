#!/usr/bin/env python3
"""A59 complete source-cache/matrix forecast from real frozen FIT checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.anchor_common import RUN, PUB, read, write, sha, verify_freeze
from echo_routing.temporal.anchor_budget import remaining_components, storage_forecast
from echo_routing.temporal.anchor_resources import totals, forecast
from echo_routing.temporal.structured_guard import storage_bytes


def main():
    freeze = verify_freeze('cache_admission_freeze.json')
    metadata = read(PUB / 'input_manifest.json')
    plan = read(PUB / 'profile_input_identity.json')
    gpu, cpu = read(PUB / 'FIT_GPU_checks.json'), read(PUB / 'FIT_CPU_checks.json')
    if not all(x['passed'] for x in [metadata, plan, gpu, cpu]) or cpu['formal_optimizer_updates'] != 0:
        raise ValueError('complete real FIT source/CPU engineering checks required')
    if (RUN / 'models').exists():
        raise ValueError('cache route must be admitted before formal head outcomes')
    ledger = RUN / 'processes.jsonl'
    import json
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    external = read(PUB / 'local_cpu_receipts.json')
    rows += external['receipts']
    for path, expected in external['file_pins'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('measured local test receipt pin changed')
    spent = totals(rows)
    remaining = remaining_components(metadata['roles'], plan['FIT_inner_physical_workload'], gpu, cpu)
    storage = storage_forecast(metadata['roles'], storage_bytes([RUN, PUB]))
    # Current admission's final actual receipt will be added by the supervisor;
    # this explicit remaining margin covers its in-flight CPU/IO, not spent cost.
    remaining['remaining_CPU_components_seconds']['inflight_admission_and_transition_margin'] = 60.
    remaining['remaining_CPU_seconds'] += 60.
    remaining['remaining_serial_wall_seconds'] += 60.
    result = forecast(spent, remaining['remaining_GPU_seconds'], remaining['remaining_CPU_seconds'],
                      storage['incremental_tracked_path_peak_upper_bytes'], time.time(),
                      remaining['remaining_serial_wall_seconds'])
    value = {**result, 'remaining': remaining, 'storage': storage,
             'closed_actual_totals_at_snapshot': {k: v for k, v in spent.items() if k != 'receipt_records'},
             'closed_receipt_pins': {str(p.relative_to(ROOT)): sha(p) for p in [ledger, PUB / 'local_cpu_receipts.json']},
             'source_input_manifest_sha256': sha(PUB / 'input_manifest.json'),
             'GPU_check_sha256': sha(PUB / 'FIT_GPU_checks.json'), 'CPU_check_sha256': sha(PUB / 'FIT_CPU_checks.json'),
             'admission_freeze_git': freeze['git_commit'],
             'source_cache_route': 'freshoriginal5A08FITinner;qualifiedJEPA_native_reuse;truejoinednewbanks;oldouterreadonlyreuse',
             'formal_updates_at_cache_admission': 0,
             'engineering_updates_not_formal': 84, 'engineering_mutable_models_including_failed': 16,
             'wholeFIT_fixed_plan_graph_required_before_formal': True,
             'all4models_6000_12inner_70rows_96comparisons_oracle_closure_forecasted': True,
             'scope': 'prospective forecast; actual final receipts replace estimates; neither cached-head latency nor shared-cache charge is full-system inference cost',
             'epoch': time.time()}
    write(PUB / 'cache_admission.json', value)
    if not result['all_caps_fit']:
        raise ValueError('complete remaining A59 workflow does not fit unchanged original caps')
    print({k: v for k, v in result.items()}, flush=True)


if __name__ == '__main__':
    if sys.argv[1:] != ['cache']:
        raise ValueError('explicit cache admission stage required')
    main()
