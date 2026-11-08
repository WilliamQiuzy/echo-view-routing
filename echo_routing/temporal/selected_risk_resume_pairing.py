"""Qualified A63 metric/count production from already decoded private axes.

No model forward, fitting, cache adoption or actual-Git authorization occurs
here. The caller verifies those barriers and supplies independently pinned
expected evidence. Never backfill expected evidence from incoming metrics.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np

from echo_routing.temporal import selected_risk_analysis as analysis


def _pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in 'biuf':
            raise ValueError('numeric typed private count arrays only')
        if not np.isfinite(value).all():
            raise ValueError('finite private count arrays required')
        return dict(array_dtype=value.dtype.str, shape=list(value.shape),
                    bytes_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('named private count mappings required')
        return {k: _pack(v) for k, v in value.items()}
    if isinstance(value, set):
        return {'set': sorted(_pack(x) for x in value)}
    if isinstance(value, (list, tuple)):
        return [_pack(x) for x in value]
    return value


def _digest(value):
    return analysis._digest(_pack(value))


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = ('temporal/selected_risk_resume_pairing.py', 'temporal/event_risk_metrics.py',
             'temporal/structured_uncertainty.py', 'temporal/safety_metrics.py',
             'temporal/proposal_metrics.py')
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def _point_and_gross(system, row):
    try:
        session = np.asarray(system['session_confusion'])
        if (session.shape != (len(system['session_order']), 5, 5) or
                session.dtype.kind not in 'iu' or (session < 0).any() or
                session.sum(0).tolist() != row['native_confusion']):
            raise ValueError('paired native count point differs from qualified row')
        streams = system['streams']['multi']
        stats = np.stack([r[1] for r in streams])
        if stats.ndim != 2 or stats.shape[1] != 18 or stats.dtype.kind not in 'iu' or (stats < 0).any():
            raise ValueError('complete nonnegative paired multi statistics required')
        matched, predicted, truth = map(int, stats[:, :3].sum(0))
        if not predicted+truth or matched > min(predicted, truth):
            raise ValueError('defined canonical matched/predicted/true counts required')
        if 2*matched/(predicted+truth) != row['multi_f1']:
            raise ValueError('paired count point differs from exact qualified pooled F1')
        for kind in ('native', 'multi', 'pair'):
            gross = (np.asarray(system['session_gross']) if kind == 'native' else
                     np.stack([r[1] for r in system['stream_gross'][kind]]))
            if gross.ndim != 2 or gross.shape[1] != 2 or gross.dtype.kind not in 'iu' or (gross < 0).any():
                raise ValueError('complete integer paired harm/repair counts required')
            if gross.sum(0).tolist() != [row['H'][kind], row['R'][kind]]:
                raise ValueError('paired harm/repair point differs from qualified row')
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('complete qualified private paired system required') from exc


def evaluate_and_prepare(sequences, baselines, values, table, half, seed, policy,
                         input_identity, model_identity, fit_f1, *, expected_provenance):
    """One historical metrics pass followed immediately by the paired backend.

    Rich metrics/tables are produced internally; no opaque F1/table bundle is
    accepted. These hashes bind artifacts but cannot prove model execution or
    clinical correctness. All private unit IDs and count arrays stay server-side.
    """
    actual = analysis.endpoint_provenance(sequences, baselines, values, table, half, seed,
                                          policy, input_identity, model_identity, fit_f1)
    if _digest(actual) != _digest(expected_provenance):
        raise ValueError('independently expected source/model/prediction provenance differs')
    rich, summary, tables = analysis.qualified_evaluate_endpoint(
        sequences, baselines, values, table, half, seed, policy, input_identity, model_identity, fit_f1)
    row = analysis.normalize_endpoint(table, half, seed, policy, input_identity, rich, summary,
                                       expected_provenance=expected_provenance)
    ids = [q['id'] for q in sequences]; bases = analysis._bank(baselines, ids)
    decoded = [None]*len(ids) if policy == 'Base' else analysis._bank(values, ids)
    predictions = {q['id']: np.asarray(b if policy == 'Base' else v[policy]['prediction']).copy()
                   for q, b, v in zip(sequences, bases, decoded)}
    base_bank = {q['id']: np.asarray(b).copy() for q, b in zip(sequences, bases)}
    from echo_routing.temporal.event_risk_metrics import prepare_system
    system = prepare_system(predictions, base_bank, sequences, tables)
    binding = dict(version=1, endpoint_provenance=copy.deepcopy(actual),
                   normalized_row_sha256=_digest(row), system_sha256=_digest(system),
                   pair_source_identity=_source_identity())
    bundle = dict(system=system, binding=binding, rich_metrics=copy.deepcopy(rich),
                  safety_summary=copy.deepcopy(summary))
    validate_pair_bundle(bundle, row, expected_provenance)
    return dict(row=row, rich_metrics=rich, safety_summary=summary, pair_bundle=bundle)


def validate_pair_bundle(bundle, row, expected_provenance):
    """Reject a swapped row, payload, scope, model or expected-evidence map."""
    try:
        keys = {'version', 'scope', 'input_identity', 'intrinsic_identity', 'model_identity', 'evaluator_identity'}
        if not isinstance(expected_provenance, dict) or set(expected_provenance) != keys:
            raise ValueError('independent expected provenance required')
        analysis._models(expected_provenance['model_identity'])
        spec = expected_provenance['scope']
        canonical_row = analysis.normalize_endpoint(spec['table'], spec['half'], spec['seed'], spec['policy'],
            expected_provenance['input_identity'], bundle['rich_metrics'], bundle['safety_summary'],
            expected_provenance=expected_provenance)
        if _digest(canonical_row) != _digest(row):
            raise ValueError('incoming flat row differs from producer-qualified canonical row')
        b = bundle['binding']
        endpoint = row['endpoint_binding']
        identity = {k: endpoint[k] for k in expected_provenance}
        if _digest(identity) != _digest(expected_provenance):
            raise ValueError('row differs from independently expected provenance')
        required = dict(version=1, endpoint_provenance=copy.deepcopy(expected_provenance),
                        normalized_row_sha256=_digest(row), system_sha256=_digest(bundle['system']),
                        pair_source_identity=_source_identity())
        if _digest(b) != _digest(required):
            raise ValueError('qualified pair payload/row/source provenance binding differs')
        # Scope comes from the frozen plan, not a free-form incoming label.
        canonical = analysis._scope(spec['table'], spec['half'], spec['seed'], spec['policy'])
        if any(not analysis.endpoint._same(row.get(k), v) for k, v in canonical.items()):
            raise ValueError('qualified row endpoint differs from frozen plan')
        analysis.endpoint._validate_row(row)
        _point_and_gross(bundle['system'], row)
        return True
    except (KeyError, TypeError) as exc:
        raise ValueError('complete independently bound paired artifact required') from exc
