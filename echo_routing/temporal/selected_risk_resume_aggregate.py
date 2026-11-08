"""Complete A63 paired matrix on qualified frozen private count bundles only.

No prediction, head forward, support solving or threshold selection occurs here.
Validate every artifact before generating any of the 140 paired5000 intervals.
"""
import copy

from echo_routing.temporal import selected_risk_metrics as endpoint
from echo_routing.temporal.selected_risk_resume_pairing import validate_pair_bundle


def aggregate(outer_rows, bundles_by_key, expected_provenance_by_key):
    """Compare fixed whole units; expected pins come from independent artifacts.

    Keys are (table, half, system). Never construct expected provenance from the
    incoming bundle header. The original readout Git seal and private artifact
    verification belong to the execution caller, not this pure backend.
    """
    plan = endpoint.readout_plan()
    rows = endpoint._matrix(outer_rows, plan['rows'], ('table', 'half', 'seed', 'policy'))
    expected_keys = {(r['table'], r['half'], r['system']) for r in rows.values()}
    if (not isinstance(bundles_by_key, dict) or set(bundles_by_key) != expected_keys or
            not isinstance(expected_provenance_by_key, dict) or set(expected_provenance_by_key) != expected_keys):
        raise ValueError('exact238 bundles and independent expected evidence required')
    for row in rows.values():
        key = row['table'], row['half'], row['system']
        validate_pair_bundle(bundles_by_key[key], row, expected_provenance_by_key[key])
    # All controls must share the full underlying labels, refs, joins and Base,
    # not merely a caller-supplied role name or external manifest checksum.
    for table in endpoint.TABLES:
        for half in endpoint.HALVES:
            base = rows[(table, half, None, 'Base')]
            for seed in endpoint.SEEDS:
                for policy in endpoint.POLICIES:
                    row = rows[(table, half, seed, policy)]
                    if (row['input_identity'] != base['input_identity'] or any(
                            row['intrinsic_identity'][k] != base['intrinsic_identity'][k]
                            for k in ('data_sha256', 'base_sha256', 'fit_f1'))):
                        raise ValueError('matched policies must retain identical complete original input/Base')
    from echo_routing.temporal.event_risk_metrics import paired
    comparisons = []
    for spec in plan['comparisons']:
        left = rows[(spec['table'], spec['half'], spec['seed'], spec['left_policy'])]
        right = rows[(spec['table'], spec['half'], None if spec['right_policy'] == 'Base' else spec['seed'],
                      spec['right_policy'])]
        a = bundles_by_key[(spec['table'], spec['half'], spec['left'])]['system']
        b = bundles_by_key[(spec['table'], spec['half'], spec['right'])]['system']
        result = paired(a, b, n_boot=5000, seed=2026110461, primary=spec['primary_family'])
        result.setdefault('delta99_375', {})
        comparison = dict(copy.deepcopy(spec), **result, input_identity=left['input_identity'],
                          observed_delta={endpoint.METRIC: left['multi_f1']-right['multi_f1']})
        endpoint._validate_ci(comparison)
        comparisons.append(comparison)
    verdict = endpoint.acceptance(outer_rows, comparisons)
    return dict(comparisons=comparisons, acceptance=verdict,
                counts=dict(outer_rows=len(rows), paired_comparisons=len(comparisons),
                            primary_comparisons=sum(c['primary_family'] for c in comparisons)),
                independence='conditional shared-source fixed banks; sessions are not confirmed patients',
                threshold_seed_checkpoint_selection=False)
