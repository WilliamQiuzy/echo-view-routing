"""A63 frozen endpoint plan, paired intervals and bounded scientific verdicts.

Pure public-aggregate operations: this module never fits, decodes, reads labels,
solves support, or changes the deployment top. The caller must independently
verify the private inputs and supply the complete normalized readout matrix.
"""
import re
from numbers import Real

import numpy as np

SEEDS = (2026110461, 2026110462)
TABLES = ('main', 'a58_stress', 'a59_confirmation', 'a60_confirmation',
          'a61_confirmation', 'a62_confirmation', 'a63_confirmation')
HALVES = ('selection', 'calibration')
POLICIES = ('E', 'N', 'L0', 'M0', 'R1', 'R2', 'C0', 'C1')
BASE = 'A08_s0_original_B3'
CONTRASTS = (('C1', 'C0'), ('C0', 'Base'), ('C1', 'Base'),
             ('C0', 'R2'), ('C1', 'R2'))
KINDS = ('native', 'multi', 'pair')
PRIMARY_QUANTILES = (.003125, .996875)
METRIC = 'segment_f1_50_multi'


def system(seed, policy):
    if isinstance(seed, (bool, np.bool_)) or seed not in SEEDS or policy not in POLICIES:
        raise ValueError('fixed A63 seed and declared policy required')
    return f'seed{seed}_{policy}_fixed6000'


def readout_plan():
    """Return a fresh complete plan; no observed outcomes enter this function."""
    rows, inner, comparisons, support, native, base_native, regression = ([] for _ in range(7))
    for table in TABLES:
        for half in HALVES:
            base = dict(table=table, half=half, seed=None, policy='Base',
                        system=BASE, role='reference')
            rows.append(base)
            if table == 'main':
                base_native.append(dict(base))
            for seed in SEEDS:
                for policy in POLICIES:
                    row = dict(table=table, half=half, seed=seed, policy=policy,
                               system=system(seed, policy), role='policy', step=6000)
                    rows.append(row)
                    if table == 'main':
                        native.append(dict(row))
                for left, right in CONTRASTS:
                    primary = table == 'a63_confirmation' and left == 'C1' and right in ('C0', 'Base')
                    comparisons.append(dict(
                        table=table, half=half, seed=seed,
                        left_policy=left, right_policy=right,
                        left=system(seed, left), right=BASE if right == 'Base' else system(seed, right),
                        contrast=left + '-' + right, bootstrap_replicates=5000,
                        primary_family=primary,
                        primary_quantiles=list(PRIMARY_QUANTILES) if primary else None))
                for safety in ('event-safe', 'acceptance-safe'):
                    support.append(dict(table=table, half=half, seed=seed, safety=safety,
                                        domain='noop_or_original_N_admitted_SL_top',
                                        readonly_A62=table != 'a63_confirmation',
                                        read_after_outer_seal=True))
    for population in ('CAL', 'AUDIT'):
        inner.append(dict(population=population, seed=None, policy='Base', system=BASE, role='reference'))
        for seed in SEEDS:
            for policy in POLICIES:
                inner.append(dict(population=population, seed=seed, policy=policy,
                                  system=system(seed, policy), role='policy', step=6000))
    for axis in ('Case_A', 'Case_B'):
        for seed in SEEDS:
            for policy in POLICIES:
                mode = ('new_C0_C1' if policy in ('C0', 'C1') else
                        'readonly_derived_frozen_C6' if policy in ('E', 'N') else 'readonly_A62_literal')
                regression.append(dict(axis=axis, seed=seed, policy=policy, system=system(seed, policy),
                                       evidence_mode=mode, development_regression_not_independent_holdout=True))
    return dict(tables=list(TABLES), rows=rows, inner_rows=inner, comparisons=comparisons,
                support_cells=support, native_cells=native, base_native_references=base_native,
                regression_records=regression, outer_metric_rows=238, inner_metric_rows=34,
                paired_comparisons=140, bootstrap_replicates=5000, primary_comparisons=8,
                thresholds=4, auxiliary_models=8, auxiliary_updates=48000,
                auxiliary_fullstates=32, new_event_models=4, formal_updates=24000,
                fullstate_records=16, total_updates=72000,
                new_source_streams=3240, new_source_payloads=6480,
                stress_native_pairs_reuse_main=True, checkpoint_or_seed_selection=False,
                primary_system='C1', independent_clinical_safety_claim=False)


def paired_intervals(samples, primary=False):
    """Use one exact set of 5000 paired draws for both confidence levels.

    Preserve the historical conditional-on-finite convention and disclose every
    undefined replicate. This convention is not a claim that missingness is
    ignorable. No extra or replacement draws may be generated here.
    """
    if not isinstance(primary, bool) or not isinstance(samples, dict) or not samples:
        raise ValueError('named paired samples and explicit primary boolean required')
    out = dict(delta95={}, delta99_375={}, undefined_bootstrap_replicates={},
               bootstrap_replicates=5000,
               undefined_ratios='null if all undefined; otherwise conditional on finite paired replicates with count disclosed')
    for key, value in samples.items():
        if not isinstance(key, str) or not key:
            raise ValueError('nonempty metric names required')
        try:
            v = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('numeric complete paired draws required') from exc
        if v.shape != (5000,) or np.isinf(v).any():
            raise ValueError('same exact5000 samples, NaN only for undefined ratio required')
        finite = np.isfinite(v)
        out['undefined_bootstrap_replicates'][key] = int((~finite).sum())
        out['delta95'][key] = np.quantile(v[finite], [.025, .975]).tolist() if finite.any() else None
        if primary:
            out['delta99_375'][key] = np.quantile(v[finite], PRIMARY_QUANTILES).tolist() if finite.any() else None
    return out


def _count(value, signed=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or (not signed and value < 0):
        raise ValueError('exact integer counts required; booleans are not counts')
    return int(value)


def _finite(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError('finite real scalar required')
    return float(value)


def _same(actual, expected):
    # Type-sensitive identity prevents True from substituting for count/flag 1.
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, int):
        return not isinstance(actual, (bool, np.bool_)) and isinstance(actual, (int, np.integer)) and actual == expected
    return actual == expected


def _matrix(records, specs, keys):
    if not isinstance(records, (list, tuple)) or len(records) != len(specs):
        raise ValueError('complete exact matrix required')
    expected = {tuple(s[k] for k in keys): s for s in specs}
    result = {}
    for row in records:
        if not isinstance(row, dict) or any(k not in row for k in keys):
            raise ValueError('explicit matrix identity required')
        try:
            key = tuple(row[k] for k in keys)
            spec = expected.get(key)
        except TypeError as exc:
            raise ValueError('scalar matrix identity required') from exc
        if spec is None or key in result or any(k not in row or not _same(row[k], v) for k, v in spec.items()):
            raise ValueError('missing, duplicate, or changed frozen endpoint identity')
        result[key] = row
    if set(result) != set(expected):
        raise ValueError('incomplete endpoint matrix')
    return result


def _validate_row(row):
    identity = row.get('input_identity')
    if not isinstance(identity, str) or re.fullmatch(r'[0-9a-f]{64}', identity) is None:
        raise ValueError('data-only immutable input SHA256 required')
    try:
        confusion = row['native_confusion']
        if len(confusion) != 5 or any(len(line) != 5 for line in confusion):
            raise ValueError('five-class confusion required')
        cells = [_count(x) for line in confusion for x in line]
        if sum(cells) == 0:
            raise ValueError('nonempty native reference required')
        _count(row['native_vote_changes'])
        for name in ('H', 'R', 'dFP16', 'short16_lost'):
            if not isinstance(row[name], dict) or set(row[name]) != set(KINDS):
                raise ValueError('all three kinds must have explicit effects')
            for value in row[name].values():
                _count(value, signed=name == 'dFP16')
        if len(row['class_H']) != 5 or sum(_count(v) for v in row['class_H']) != sum(row['H'].values()):
            raise ValueError('five class gross-harm counts must reconcile with kind H')
        _count(row['JEPA_short16_recovered'])
        if not 0 <= _finite(row['multi_f1']) <= 1:
            raise ValueError('multi F1 outside unit interval')
        if row['policy'] == 'Base' and (row['native_vote_changes'] or row['JEPA_short16_recovered'] or
                                       any(any(row[k].values()) for k in ('H', 'R', 'dFP16', 'short16_lost'))):
            raise ValueError('Base-relative effects must be zero')
    except (KeyError, TypeError) as exc:
        raise ValueError('complete normalized scientific row required') from exc


def _validate_ci(row):
    try:
        undefined = row['undefined_bootstrap_replicates']
        if not isinstance(undefined, dict) or METRIC not in undefined:
            raise ValueError('undefined replicate count required')
        for count in undefined.values():
            if _count(count) > 5000:
                raise ValueError('undefined count exceeds exact draws')
        for field in ('delta95', 'delta99_375'):
            if field == 'delta99_375' and not row['primary_family']:
                if row.get(field, {}) != {}:
                    raise ValueError('undeclared family interval')
                continue
            mapping = row[field]
            if not isinstance(mapping, dict) or METRIC not in mapping or set(mapping) != set(undefined):
                raise ValueError('complete paired interval/count metric identity required')
            for metric, interval in mapping.items():
                if interval is None:
                    if undefined[metric] != 5000:
                        raise ValueError('null interval requires all draws undefined')
                elif (not isinstance(interval, (list, tuple)) or len(interval) != 2 or
                      undefined[metric] == 5000 or _finite(interval[0]) > _finite(interval[1])):
                    raise ValueError('ordered finite interval consistent with missing counts required')
                elif metric == METRIC and not (-1 <= interval[0] <= interval[1] <= 1):
                    raise ValueError('F1 delta interval must remain in its mathematical range [-1,1]')
        if row['primary_family']:
            for metric, wider in row['delta99_375'].items():
                narrower = row['delta95'][metric]
                if wider is not None and not (wider[0] <= narrower[0] <= narrower[1] <= wider[1]):
                    raise ValueError('same draws require nested primary and 95 percent intervals')
    except (KeyError, TypeError) as exc:
        raise ValueError('complete paired interval schema required') from exc


def acceptance(normalized_rows, comparisons):
    """Validate completeness, then keep three prespecified verdicts separate.

    A schema/input mismatch raises ValueError: it is an incomplete audit, not a
    scientific negative. Legitimate all-undefined intervals fail significance.
    Gross harm, short loss and seed-specific retention cannot be netted away.
    """
    plan = readout_plan()
    rows = _matrix(normalized_rows, plan['rows'], ('table', 'half', 'seed', 'policy'))
    paired = _matrix(comparisons, plan['comparisons'], ('table', 'half', 'seed', 'left_policy', 'right_policy'))
    for row in rows.values():
        _validate_row(row)
    for row in paired.values():
        _validate_ci(row)
        left = rows[(row['table'], row['half'], row['seed'], row['left_policy'])]
        right = rows[(row['table'], row['half'], None if row['right_policy'] == 'Base' else row['seed'], row['right_policy'])]
        delta = row.get('observed_delta')
        if (row.get('input_identity') != left['input_identity'] or
                not isinstance(delta, dict) or METRIC not in delta or
                not np.isclose(_finite(delta[METRIC]), left['multi_f1'] - right['multi_f1'], rtol=1e-12, atol=1e-14)):
            raise ValueError('paired input and observed left-minus-right point must bind to the exact normalized rows')
    for table in TABLES:
        for half in HALVES:
            base = rows[(table, half, None, 'Base')]
            for seed in SEEDS:
                for policy in POLICIES:
                    row = rows[(table, half, seed, policy)]
                    if row['input_identity'] != base['input_identity']:
                        raise ValueError('policy/reference data-only input identities differ')
                    if [sum(r) for r in row['native_confusion']] != [sum(r) for r in base['native_confusion']]:
                        raise ValueError('native class denominators differ from Base')
    failures = []

    def require(ok, row, criterion, actual, required):
        if not ok:
            failures.append(dict(table=row['table'], half=row['half'], seed=row['seed'],
                                 criterion=criterion, actual=actual, required=required))
        return bool(ok)

    hard = True
    for table in TABLES:
        for half in HALVES:
            base = rows[(table, half, None, 'Base')]
            for seed in SEEDS:
                row = rows[(table, half, seed, 'C1')]
                hard &= require(row['native_vote_changes'] == 0 and row['native_confusion'] == base['native_confusion'],
                                row, 'native_vote_preservation',
                                dict(changed=row['native_vote_changes'], confusion=row['native_confusion']),
                                'zero changed clip votes and exact Base five-class confusion')
                for kind in KINDS:
                    hard &= require(row['H'][kind] == 0, row, 'gross_H_' + kind, row['H'][kind], 0)
                    hard &= require(row['dFP16'][kind] <= 0, row, 'dFP16_' + kind, row['dFP16'][kind], '<=0')
                    hard &= require(row['short16_lost'][kind] == 0, row, 'short16_lost_' + kind, row['short16_lost'][kind], 0)
                hard &= require(not any(row['class_H']), row, 'class_gross_H', row['class_H'], [0] * 5)
    useful, increment = True, True
    retain = {('selection', SEEDS[0]): 6, ('selection', SEEDS[1]): 5,
              ('calibration', SEEDS[0]): 4, ('calibration', SEEDS[1]): 3}
    for (half, seed), minimum in retain.items():
        old = rows[('a61_confirmation', half, seed, 'C1')]
        useful &= require(old['JEPA_short16_recovered'] >= minimum, old, 'retain_A61_JEPA_short16',
                          old['JEPA_short16_recovered'], '>=' + str(minimum))
        row = rows[('a63_confirmation', half, seed, 'C1')]
        useful &= require(row['JEPA_short16_recovered'] >= 1, row, 'new_JEPA_short16', row['JEPA_short16_recovered'], '>=1')
        for right in ('Base', 'C0'):
            comparison = paired[('a63_confirmation', half, seed, 'C1', right)]
            ci = comparison['delta99_375'][METRIC]
            reference = rows[('a63_confirmation', half, None if right == 'Base' else seed, right)]
            # The incoming point is only a checked report field. Never use its
            # comparison tolerance to turn an exact canonical tie into gain.
            observed = row['multi_f1'] - reference['multi_f1']
            passed = require(observed > 0 and ci is not None and ci[0] > 0, row,
                             'new_vs_Base_CI' if right == 'Base' else 'sampling_increment_CI',
                             dict(observed_left_minus_right=observed, CI=ci),
                             'positive observed left-minus-right and 99.375 percent paired multi F1 CI lower >0')
            if right == 'Base':
                useful &= passed
            else:
                increment &= passed
    return dict(hard_protection_pass=bool(hard), joint_objective_pass=bool(hard and useful),
                H63_claim_supported=bool(hard and useful and increment), C1_cells=28,
                failures=failures, independent_clinical_safety_claim=False,
                interpretation='bounded fixed-bank sampling package contrast; no clinical or whole-vision OOF safety guarantee')
