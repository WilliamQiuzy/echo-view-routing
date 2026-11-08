"""Aggregate frozen predictions on their original axes, without selecting again.

Reference-assisted effects are diagnostics. No forwards, fitting, oracle actions,
patient independence or scientific success is established by these functions.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
from collections.abc import Mapping
from numbers import Real
from types import SimpleNamespace

import numpy as np

from echo_routing.evaluate.failure_units import match_segments
from echo_routing.temporal.anchor_edits import aligned, majority
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.temporal import selected_risk_metrics as endpoint

KINDS = ('native', 'multi', 'pair')
SCALARS = ('H', 'R', 'dE', 'dFP16', 'short16_lost', 'short16_recovered',
           'changed_positions', 'changed_sequences', 'wrong_to_other_wrong',
           'reference_positions', 'axis_counts')
SIGNED = ('dE', 'dFP16')


def _digest(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError('finite JSON-compatible provenance payload required') from exc
    return hashlib.sha256(raw).hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('explicit immutable SHA256 identity required')
    return value


def _scope(table, half, seed, policy):
    rows = [r for r in endpoint.readout_plan()['rows']
            if r['table'] == table and r['half'] == half and r['policy'] == policy
            and endpoint._same(seed, r['seed'])]
    if len(rows) != 1:
        raise ValueError('exact frozen table/half/seed/policy endpoint required')
    return rows[0]


def _models(identity):
    required = {'producer_manifest_sha256', 'head_seal_sha256', 'policy_manifest_sha256', 'model_roles'}
    if not isinstance(identity, dict) or set(identity) != required:
        raise ValueError('complete producer/head/policy seals and exact model-role SHA map required')
    for k in required - {'model_roles'}:
        _sha(identity[k])
    roles = identity['model_roles']
    if not isinstance(roles, dict) or not roles or any(not isinstance(k, str) or not k for k in roles):
        raise ValueError('nonempty exact model-role SHA map required')
    for value in roles.values():
        _sha(value)


def _intrinsic(sequences, bases, decoded, policy, fit_f1):
    """Private references are hashed in full and never emitted as public rows."""
    data, base_rows, predictions = [], [], []
    for q, base, value in zip(sequences, bases, decoded):
        refs = q['refs']; joins = q['joins']; truth = np.asarray(q['labels'])
        if not isinstance(refs, (list, tuple)) or len(refs) != len(truth):
            raise ValueError('complete original source/frame/gamma references required')
        clean_refs = []
        for ref in refs:
            if (not isinstance(ref, (tuple, list)) or len(ref) != 3 or
                    not isinstance(ref[0], str) or not ref[0] or
                    isinstance(ref[1], (bool, np.bool_)) or not isinstance(ref[1], (int, np.integer)) or ref[1] < 0 or
                    isinstance(ref[2], (bool, np.bool_)) or not isinstance(ref[2], Real) or
                    not np.isfinite(ref[2]) or ref[2] <= 0):
                raise ValueError('typed source/frame/gamma provenance required')
            clean_refs.append([ref[0], int(ref[1]), float(ref[2])])
        if not isinstance(joins, (list, tuple, np.ndarray)):
            raise ValueError('original joins required')
        clean_joins = []
        for j in joins:
            if isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) or not 0 < j < len(truth):
                raise ValueError('original whole-axis internal integer joins required')
            clean_joins.append(int(j))
        if clean_joins != sorted(set(clean_joins)) or q['kind'] == 'native' and clean_joins:
            raise ValueError('unique ordered original joins required')
        data.append(dict(id=q['id'], kind=q['kind'], labels=truth.tolist(),
                         refs=clean_refs, joins=clean_joins))
        base_rows.append(dict(id=q['id'], labels=np.asarray(base).tolist()))
        actions, index, pred = [], None, base
        if policy != 'Base':
            for a in value['actions']:
                apply(base, a)  # Validate every action, including rejected candidates.
                if not all(isinstance(x, (bool, np.bool_)) for x in (a.old, a.jepa)):
                    raise ValueError('original boolean action provenance required')
                actions.append(dict(start=int(a.start), end=int(a.end), replacement=list(a.replacement),
                                    old=bool(a.old), jepa=bool(a.jepa)))
            index = value[policy]['accepted_index']; pred = value[policy]['prediction']
        predictions.append(dict(id=q['id'], ordered_actions=actions,
                                accepted_index=None if index is None else int(index),
                                prediction=np.asarray(pred).tolist()))
    return dict(data_sha256=_digest(data), base_sha256=_digest(base_rows),
                prediction_sha256=_digest(dict(policy=policy, axes=predictions)),
                fit_f1=float(fit_f1), policy=policy)


def _evaluator_identity():
    root = Path(__file__).resolve().parents[1]
    paths = ('temporal/selected_risk_analysis.py', 'temporal/event_risk_metrics.py',
             'temporal/selected_risk_metrics.py', 'temporal/proposal_actions.py',
             'temporal/safety_metrics.py', 'temporal/proposal_metrics.py',
             'temporal/structured_metrics.py', 'temporal/dense_runtime.py',
             'temporal/dense_analysis.py', 'temporal/anchor_edits.py',
             'temporal/segments.py', 'evaluate/dev_metrics.py',
             'evaluate/failure_units.py', 'evaluate/metrics_segmentation.py')
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def endpoint_provenance(sequences, baselines, values, table, half, seed, policy,
                        input_identity, model_identity, fit_f1):
    """Construct expected evidence from separately verified input/model artifacts.

    Callers must load these arguments from pinned private sources and the sealed
    model-role mapping. Never construct expected evidence from incoming metrics.
    This pure API cannot prove that a declared model produced its predictions;
    actual-Git/cache verification and independent private recomputation remain
    mandatory. No patient or clinical validity follows from matching hashes.
    """
    scope = _scope(table, half, seed, policy); _sha(input_identity)
    _models(model_identity)
    ids = [q['id'] for q in sequences]; bases = _bank(baselines, ids)
    decoded = [None]*len(ids) if policy == 'Base' else _bank(values, ids)
    return dict(version=1, scope=scope, input_identity=input_identity,
                intrinsic_identity=_intrinsic(sequences, bases, decoded, policy, fit_f1),
                model_identity=copy.deepcopy(model_identity), evaluator_identity=_evaluator_identity())


def qualified_evaluate_endpoint(sequences, baselines, values, table, half, seed, policy,
                                input_identity, model_identity, fit_f1):
    """Produce metrics and provenance together using the unchanged task backend.

    This is a CPU metrics pass on already decoded predictions, with no NN. The
    caller must verify the input/model manifest pins and actual-Git barrier;
    hashes here detect mismatched artifacts, not forged or clinically valid data.
    No arbitrary caller-supplied F1 is qualified by this function.
    """
    _scope(table, half, seed, policy); _sha(input_identity); _models(model_identity)
    s = safety_summary(sequences, baselines, values, policy, fit_f1)
    ids = [q['id'] for q in sequences]; bases = _bank(baselines, ids)
    decoded = [None]*len(ids) if policy == 'Base' else _bank(values, ids)
    predictions = {q['id']: np.asarray(b if policy == 'Base' else v[policy]['prediction']).copy()
                   for q, b, v in zip(sequences, bases, decoded)}
    base_bank = {q['id']: np.asarray(b).copy() for q, b in zip(sequences, bases)}
    # Build the metric context from the exact hashed axes, not an alien eval_set.
    from echo_routing.temporal.dense_runtime import metric_stream
    from echo_routing.temporal.event_risk_metrics import evaluate
    eval_set = SimpleNamespace(
        natives=[SimpleNamespace(seq_id=q['id'], labels=q['labels']) for q in sequences if q['kind'] == 'native'],
        multi=[metric_stream(q) for q in sequences if q['kind'] == 'multi'],
        pairs=[metric_stream(q) for q in sequences if q['kind'] in ('pair', 'pairs')])
    rich, tables = evaluate(predictions, dict(sequences=sequences, eval_set=eval_set), base_bank)
    binding = endpoint_provenance(sequences, baselines, values, table, half, seed, policy,
                                  input_identity, model_identity, fit_f1)
    binding.update(rich_metrics_sha256=_digest(rich), safety_summary_sha256=_digest(s))
    rich['endpoint_binding'] = copy.deepcopy(binding)
    s['endpoint_binding'] = copy.deepcopy(binding)
    return rich, s, tables


def _verify_binding(table, half, seed, policy, input_identity, rich, summary, expected_provenance):
    try:
        required = {'version', 'scope', 'input_identity', 'intrinsic_identity',
                    'model_identity', 'evaluator_identity'}
        if not isinstance(expected_provenance, dict) or set(expected_provenance) != required:
            raise ValueError('independently pinned expected endpoint provenance required')
        binding = rich['endpoint_binding']
        _models(expected_provenance['model_identity'])
        if _digest({k: binding[k] for k in required}) != _digest(expected_provenance):
            raise ValueError('incoming provenance differs from independently expected input/model/prediction pins')
        if _digest(binding) != _digest(summary['endpoint_binding']):
            raise ValueError('rich/summary endpoint provenance binding differs')
        expected = dict(version=1, scope=_scope(table, half, seed, policy),
                        input_identity=_sha(input_identity), intrinsic_identity=summary['intrinsic_identity'],
                        model_identity=binding['model_identity'], evaluator_identity=_evaluator_identity(),
                        rich_metrics_sha256=_digest({k: v for k, v in rich.items() if k != 'endpoint_binding'}),
                        safety_summary_sha256=_digest({k: v for k, v in summary.items() if k != 'endpoint_binding'}))
        if _digest(binding) != _digest(expected):
            raise ValueError('scope/input/prediction/evaluator/payload provenance binding differs')
    except (KeyError, TypeError) as exc:
        raise ValueError('qualified producer endpoint provenance binding required') from exc


def _bank(bank, ids):
    if isinstance(bank, Mapping):
        if set(bank) != set(ids):
            raise ValueError('exact original-axis bank identity required')
        return [bank[k] for k in ids]
    if not isinstance(bank, (list, tuple)) or len(bank) != len(ids):
        raise ValueError('complete aligned original-axis bank required')
    return bank


def _short(truth, before, after):
    """Use the unchanged full-axis class-aware IoU50 one-to-one matcher."""
    true = labels_to_segments(truth); old = labels_to_segments(before); new = labels_to_segments(after)
    left = match_segments(old, true, .5); right = match_segments(new, true, .5)
    used_left, used_right = set(left.values()), set(right.values())
    fp_old = sum(s.length <= 16 and i not in used_left for i, s in enumerate(old))
    fp_new = sum(s.length <= 16 and i not in used_right for i, s in enumerate(new))
    lost = sum(true[i].length <= 16 for i in set(left)-set(right))
    recovered = sum(true[i].length <= 16 for i in set(right)-set(left))
    return int(fp_new-fp_old), int(lost), int(recovered)


def safety_summary(sequences, baselines, values, policy, fit_f1):
    """Count already-decoded predictions; reject altered/noop/action alignment.

    Each axis counts its original positions, even when source refs are reused.
    pairs is the sole kind alias. No source grouping or q.kind is rewritten.
    """
    if (policy not in (*endpoint.POLICIES, 'Base') or isinstance(fit_f1, (bool, np.bool_)) or
            not isinstance(fit_f1, Real) or not np.isfinite(fit_f1) or not 0 <= fit_f1 <= 1):
        raise ValueError('fixed declared policy and frozen finite FIT point required')
    if not isinstance(sequences, (list, tuple)) or not sequences:
        raise ValueError('nonempty original-axis sequence list required')
    try:
        ids = [q['id'] for q in sequences]
        if any(not isinstance(k, str) or not k for k in ids) or len(set(ids)) != len(ids):
            raise ValueError('unique nonempty original-axis identities required')
        bases = _bank(baselines, ids)
        if policy == 'Base':
            if values is not None:
                raise ValueError('Base uses baselines only, not an alternate policy value')
            decoded = [None]*len(ids)
        else:
            decoded = _bank(values, ids)
        out = {key: {kind: 0 for kind in KINDS} for key in SCALARS}
        out.update(H_by_true_class={k: [0]*5 for k in KINDS}, R_by_true_class={k: [0]*5 for k in KINDS},
                   class_H=[0]*5, native_confusion=[[0]*5 for _ in range(5)],
                   native_base_confusion=[[0]*5 for _ in range(5)], native_vote_changes=0,
                   position_confusion={k: [[0]*5 for _ in range(5)] for k in KINDS},
                   JEPA_short16_recovered=0, JEPA_short16_recovered_by_kind={k: 0 for k in KINDS},
                   accepted_actions=0, fit_f1=float(fit_f1),
                   reference_scope='original axis positions; repeated source refs are not new patients')
        for q, base, value in zip(sequences, bases, decoded):
            kind = 'pair' if q['kind'] == 'pairs' else q['kind']
            if kind not in KINDS:
                raise ValueError('only native, multi, pair/pairs kinds allowed')
            base, truth = aligned(base, q['labels']); action = None
            if kind == 'native' and not np.all(truth == truth[0]):
                raise ValueError('native clip reference must retain its single propagated class')
            if policy == 'Base':
                pred = base
            else:
                row = value[policy]; pred, _ = aligned(row['prediction'], truth); index = row['accepted_index']
                actions = value['actions']
                if not isinstance(actions, (list, tuple)):
                    raise ValueError('original ordered candidate list required')
                if index is None:
                    if not np.array_equal(pred, base):
                        raise ValueError('rejected/noop top must retain exact Base prediction')
                else:
                    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(actions):
                        raise ValueError('one scalar accepted candidate index or explicit None required')
                    action = actions[index]
                    if not np.array_equal(pred, apply(base, action)):
                        raise ValueError('prediction must equal exactly the accepted canonical action')
                    if not all(isinstance(x, (bool, np.bool_)) for x in (action.old, action.jepa)):
                        raise ValueError('original boolean provenance flags required')
                    out['accepted_actions'] += 1
            h = (base == truth) & (pred != truth); repair = (base != truth) & (pred == truth)
            changed = base != pred
            for name, n in [('H', h.sum()), ('R', repair.sum()), ('dE', h.sum()-repair.sum()),
                            ('changed_positions', changed.sum()), ('changed_sequences', changed.any()),
                            ('wrong_to_other_wrong', (changed & (base != truth) & (pred != truth)).sum()),
                            ('reference_positions', len(truth)), ('axis_counts', 1)]:
                out[name][kind] += int(n)
            for key, mask in [('H_by_true_class', h), ('R_by_true_class', repair)]:
                out[key][kind] = (np.asarray(out[key][kind])+np.bincount(truth[mask], minlength=5)).tolist()
            dfp, lost, recovered = _short(truth, base, pred)
            for key, n in [('dFP16', dfp), ('short16_lost', lost), ('short16_recovered', recovered)]:
                out[key][kind] += n
            if action is not None and action.jepa and not action.old:
                out['JEPA_short16_recovered_by_kind'][kind] += recovered
            matrix = np.asarray(out['position_confusion'][kind], np.int64)
            np.add.at(matrix, (truth, pred), 1); out['position_confusion'][kind] = matrix.tolist()
            if kind == 'native':
                label, old_vote, vote = int(truth[0]), majority(base), majority(pred)
                out['native_confusion'][label][vote] += 1
                out['native_base_confusion'][label][old_vote] += 1
                out['native_vote_changes'] += int(vote != old_vote)
        out['class_H'] = np.asarray(list(out['H_by_true_class'].values()), np.int64).sum(0).tolist()
        out['JEPA_short16_recovered'] = out['JEPA_short16_recovered_by_kind']['multi']
        out['intrinsic_identity'] = _intrinsic(sequences, bases, decoded, policy, fit_f1)
        _validate_summary(out)
        return out
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('complete aligned qualified prediction schema required') from exc


def _matrix(value):
    if not isinstance(value, (list, tuple)) or len(value) != 5 or any(not isinstance(r, (list, tuple)) or len(r) != 5 for r in value):
        raise ValueError('complete five-class count matrix required')
    return [[endpoint._count(v) for v in r] for r in value]


def _validate_summary(summary):
    try:
        for name in SCALARS:
            if not isinstance(summary[name], dict) or set(summary[name]) != set(KINDS):
                raise ValueError('explicit three-kind complete effect summary required')
            for n in summary[name].values():
                endpoint._count(n, signed=name in SIGNED)
        for kind in KINDS:
            if summary['dE'][kind] != summary['H'][kind]-summary['R'][kind]:
                raise ValueError('gross harm and repairs must reconcile with net errors')
            if (summary['changed_positions'][kind] != summary['H'][kind]+summary['R'][kind]+summary['wrong_to_other_wrong'][kind] or
                    summary['changed_positions'][kind] > summary['reference_positions'][kind] or
                    summary['changed_sequences'][kind] > summary['axis_counts'][kind]):
                raise ValueError('all changed positions must partition into harm, repair or wrong-to-other-wrong')
            for key, count in [('H_by_true_class', 'H'), ('R_by_true_class', 'R')]:
                v = summary[key][kind]
                if len(v) != 5 or sum(endpoint._count(x) for x in v) != summary[count][kind]:
                    raise ValueError('all five class gross counts must reconcile')
            position = _matrix(summary['position_confusion'][kind])
            if sum(map(sum, position)) != summary['reference_positions'][kind]:
                raise ValueError('whole-axis position confusion/reference size differs')
        class_h = [sum(summary['H_by_true_class'][k][i] for k in KINDS) for i in range(5)]
        if len(summary['class_H']) != 5 or [endpoint._count(v) for v in summary['class_H']] != class_h:
            raise ValueError('complete class H cannot omit kind harm')
        for key in ('native_confusion', 'native_base_confusion'):
            matrix = _matrix(summary[key])
            if sum(map(sum, matrix)) != summary['axis_counts']['native']:
                raise ValueError('native confusion must count original clips, not frames')
        if [sum(r) for r in summary['native_confusion']] != [sum(r) for r in summary['native_base_confusion']]:
            raise ValueError('native reference class denominators differ from Base')
        if endpoint._count(summary['native_vote_changes']) > summary['changed_sequences']['native']:
            raise ValueError('native vote-change count exceeds clips')
        j = summary['JEPA_short16_recovered_by_kind']
        if not isinstance(j, dict) or set(j) != set(KINDS):
            raise ValueError('complete by-kind JEPA-exclusive short recovery required')
        if any(endpoint._count(j[k]) > summary['short16_recovered'][k] for k in KINDS):
            raise ValueError('JEPA-exclusive short recovery cannot exceed matched kind recovery')
        if endpoint._count(summary['JEPA_short16_recovered']) != j['multi']:
            raise ValueError('canonical JEPA-exclusive short recovery is multi-only')
        if endpoint._count(summary['accepted_actions']) != sum(summary['changed_sequences'].values()):
            raise ValueError('one canonical nonempty accepted action per changed axis required')
    except (KeyError, TypeError) as exc:
        raise ValueError('complete aggregate scientific summary required') from exc


def normalize_endpoint(table, half, seed, policy, input_identity, rich_metrics, safety_summary,
                       *, expected_provenance=None):
    """Bind original qualified rich metrics to a complete prespecified outer row.

    segment_f1_50_multi is copied, never redefined from an alternative matcher.
    The caller retains the full rich record and private input/source audit.
    """
    try:
        _verify_binding(table, half, seed, policy, input_identity, rich_metrics, safety_summary, expected_provenance)
        specs = [r for r in endpoint.readout_plan()['rows']
                 if r['table'] == table and r['half'] == half and r['policy'] == policy and
                 endpoint._same(seed, r['seed'])]
        if len(specs) != 1:
            raise ValueError('exact frozen table/half/seed/policy endpoint required')
        s = safety_summary; _validate_summary(s)
        if _matrix(rich_metrics['native_confusion']) != s['native_confusion']:
            raise ValueError('qualified native confusion differs from decoded clip votes')
        gross = rich_metrics['gross_reference_position_effects']; positions = rich_metrics['decoded_position_confusion']
        if set(gross) != set(KINDS) or set(positions) != set(KINDS):
            raise ValueError('qualified rich whole-axis gross/position records required')
        for kind in KINDS:
            want = dict(H=s['H'][kind], R=s['R'][kind], dE=s['dE'][kind],
                        H_by_true_class=s['H_by_true_class'][kind], reference_positions=s['reference_positions'][kind],
                        R_by_true_class=s['R_by_true_class'][kind], changed_positions=s['changed_positions'][kind],
                        changed_sequences=s['changed_sequences'][kind], wrong_to_other_wrong=s['wrong_to_other_wrong'][kind])
            required = ('H', 'R', 'dE', 'H_by_true_class', 'reference_positions')
            for key, n in want.items():
                if key not in gross[kind]:
                    if key in required:
                        raise ValueError('qualified rich gross count missing')
                    continue
                actual = gross[kind][key]
                if isinstance(n, list):
                    if not isinstance(actual, (list, tuple)) or len(actual) != 5:
                        raise ValueError('rich gross class counts must retain all five classes')
                    actual = [endpoint._count(v) for v in actual]
                else:
                    actual = endpoint._count(actual, signed=key == 'dE')
                if actual != n:
                    raise ValueError('qualified rich gross effects differ from decoded whole axes')
            if _matrix(positions[kind]) != s['position_confusion'][kind]:
                raise ValueError('qualified rich position confusion differs from decoded whole axes')
        row = dict(copy.deepcopy(specs[0]), input_identity=input_identity,
                   intrinsic_identity=copy.deepcopy(s['intrinsic_identity']),
                   endpoint_binding=copy.deepcopy(rich_metrics['endpoint_binding']),
                   native_confusion=copy.deepcopy(s['native_confusion']), native_vote_changes=s['native_vote_changes'],
                   H=copy.deepcopy(s['H']), R=copy.deepcopy(s['R']), dFP16=copy.deepcopy(s['dFP16']),
                   short16_lost=copy.deepcopy(s['short16_lost']), class_H=copy.deepcopy(s['class_H']),
                   JEPA_short16_recovered=s['JEPA_short16_recovered'], multi_f1=rich_metrics['segment_f1_50_multi'])
        endpoint._validate_row(row)
        return row
    except (KeyError, TypeError) as exc:
        raise ValueError('complete qualified rich metrics and frozen endpoint schema required') from exc
