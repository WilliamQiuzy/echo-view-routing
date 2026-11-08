"""Literal prospective FIT v2 contract; the unexecuted original freeze survives.

No execution/IO mutation here. Full fresh launch costs, OS and GPU ownership are
still caller/guardian duties, beyond the immutable admission and Git bindings.
"""
import re
from pathlib import PurePosixPath

from scripts.selected_risk_common import *
from scripts.selected_risk_shared_contracts import required as old_required
from echo_routing.temporal.selected_risk_contracts import source_closure
from echo_routing.temporal.selected_risk_storage_engineering import expand_proof

STAGE = 'source_FIT_v2'
ENTRY = 'scripts/a63_cache_selected_risk_FIT_v2.py'
FREEZER = 'scripts/a63_freeze_selected_risk_FIT_v2.py'
GUARDIAN = 'scripts/a63_supervise_selected_risk_FIT_v2.py'
VERIFIER = 'scripts/a63_selected_risk_FIT_v2_contract.py'
ADMISSION_GIT = 'f558e0d1b0137d534446fdcc261623ca3234b1e4'


def required(stage=STAGE):
    if stage != STAGE:
        raise ValueError('only literal source_FIT_v2 may use this contract')
    inputs = [str((PUB/n).relative_to(ROOT)) for n in ('sources_admission.json', 'sources_admission_v3.json')]
    return (old_required('source_FIT') | set(HISTORICAL_PATHS) |
            set(source_closure(ROOT, [ENTRY, FREEZER, GUARDIAN, VERIFIER,
                                      'scripts/cache_selected_risk.py', *inputs])))


def verify_admission():
    """Require the actual v3 result, not a stale locally available false result."""
    try:
        a = PUB/'sources_admission.json'; b = PUB/'sources_admission_v3.json'
        if a.read_bytes() != b.read_bytes():
            raise ValueError('canonical and immutable actual v3 admission bytes differ')
        value = read(a)
        if (value['all_caps_fit'] is not True or value['stage'] != 'sources' or
                value['source_phase'] != 'FIT' or type(value['version']) is not int or value['version'] != 3 or
                value['admission_freeze_git'] != ADMISSION_GIT or
                value['input_manifest_sha256'] != sha(PUB/'input_manifest.json') or
                type(value['deadline_epoch']) is not int or value['deadline_epoch'] != 1793897040):
            raise ValueError('exact passed actual v3 FIT/input/window admission required')
        return value
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError('actual v3 admission must be mirrored before preparing/executing FIT v2') from exc


def _source_path(path):
    if (not isinstance(path, str) or not path or str(PurePosixPath(path)) != path or
            PurePosixPath(path).is_absolute() or any(p in ('.', '..') for p in path.split('/'))):
        raise ValueError('canonical within-project source/input path required')
    target = ROOT/path
    for p in (target, *target.parents):
        if p == ROOT:
            break
        if p.is_symlink():
            raise ValueError('source/input ancestor and leaf symlinks forbidden')
    return target


def verify_shared_freeze(stage=STAGE):
    if stage != STAGE:
        raise ValueError('old source_FIT is not an alias for its new version')
    try:
        if (str(RUN.relative_to(ROOT)) != 'runs/a63_selected_action_risk_v1' or
                str(PUB.relative_to(ROOT)) != 'docs/results/2026-11-05_selected_action_risk'):
            raise ValueError('literal new A63 namespace required')
        old = verify_freeze('metadata')
        c = read(RUN/'contracts'/(STAGE+'.json')); objects = c['objects']
        refs = read(verify_pin(read(CFG)['source_refs']))
        if (c['stage'] != STAGE or c.get('proof_encoding') != 'A63_shared_Git_objects_v1' or
                c['shared_object_pool'] != old['actual_Git_object_proof'] or
                not required(stage) <= set(objects) or
                set(c['historical_frozen_imports']) != set(HISTORICAL_PATHS) or
                type(c['fixed_T0_epoch']) is not int or c['fixed_T0_epoch'] != 1793853840 or
                type(c['fixed_deadline_epoch']) is not int or c['fixed_deadline_epoch'] != 1793897040):
            raise ValueError('complete prospective source/pool/8 historical/window bindings required')
        validate_frozen_prior(objects)
        for path, expected_sha in objects.items():
            if not isinstance(expected_sha, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha) or sha(_source_path(path)) != expected_sha:
                raise ValueError('current versioned source/input SHA differs: '+str(path))
        expected = dict(objects)
        for path, value in c['historical_frozen_imports'].items():
            if (expected.pop(path) != value['sha256'] or refs['historical_frozen_imports'].get(path) !=
                    {k: value[k] for k in ('git_commit', 'blob_oid', 'sha256')}):
                raise ValueError('all eight historical overlays must preserve approved identities')
            verify_Git(read(verify_pin(value['proof'])), value['git_commit'], {path: value['sha256']})
        packed = read(verify_pin(c['actual_Git_object_proof']))
        pool = read(verify_pin(c['shared_object_pool']))
        verify_Git(expand_proof(packed, pool), c['git_commit'], expected)
        verify_admission()
        return c
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError('complete new FIT v2 source/admission/actual Git contract required') from exc
