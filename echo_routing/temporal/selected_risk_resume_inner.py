"""Qualified CAL/AUDIT diagnostics on the original pair-absent inner backend.

This CPU-only metric adapter fits no threshold and performs no model forward.
Expected evidence must be built from independently pinned inputs/model roles;
artifact identity alone does not establish training or clinical correctness.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np

from echo_routing.temporal import selected_risk_analysis as analysis
from echo_routing.temporal.selected_risk_metrics import readout_plan, _same


def _scope(population, seed, policy):
    rows = [r for r in readout_plan()['inner_rows'] if r['population'] == population
            and r['policy'] == policy and _same(seed, r['seed'])]
    if len(rows) != 1:
        raise ValueError('exact CAL/AUDIT population, fixed seed and policy required')
    return rows[0]


def _sources():
    root = Path(__file__).resolve().parents[2]
    return {**analysis._evaluator_identity(), **{
        p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in (
            'echo_routing/temporal/selected_risk_resume_inner.py', 'scripts/calibrate_event_risk.py')}}


def inner_provenance(sequences, baselines, values, population, seed, policy,
                      input_identity, model_identity, fit_f1):
    """Expected mapping from independent source/decode and precalibration seal."""
    spec = _scope(population, seed, policy)
    analysis._sha(input_identity); analysis._models(model_identity)
    if (not isinstance(sequences, (list, tuple)) or
            {q['kind'] for q in sequences} != {'native', 'multi'}):
        raise ValueError('complete native/multi inner bank required; pair bank is absent')
    ids = [q['id'] for q in sequences]
    if any(not isinstance(k, str) or not k for k in ids) or len(set(ids)) != len(ids):
        raise ValueError('unique original inner axes required')
    bases = analysis._bank(baselines, ids)
    if policy == 'Base':
        if values is not None:
            raise ValueError('Base uses original baselines only')
        decoded = [None]*len(ids)
    else:
        decoded = analysis._bank(values, ids)
    return dict(version=1, scope=spec, input_identity=input_identity,
                intrinsic_identity=analysis._intrinsic(sequences, bases, decoded, policy, fit_f1),
                model_identity=copy.deepcopy(model_identity), evaluator_identity=_sources())


def _inner_metric(qs, pred, bases):
    # Exact statement projection of the qualified old inner_metric. Keep CLI
    # globals out of the new runtime; its original function stays read-only.
    from echo_routing.temporal.dense_runtime import metrics
    from echo_routing.temporal.dense_analysis import short_metrics
    from echo_routing.temporal.safety_metrics import gross_frames
    value=metrics(qs,pred)[0];value.update(short_metrics(qs,pred,16));g=gross_frames(qs,{q['id']:b for q,b in zip(qs,bases)},pred);cms={k:np.zeros((5,5),np.int64) for k in ['native','multi','pair']}
    for q in qs:np.add.at(cms[q['kind']],(q['labels'],pred[q['id']]),1)
    value.update(gross_reference_position_effects=g,decoded_position_confusion={k:v.tolist() for k,v in cms.items()},segment_f1_50_pairs=None,boundary_precision_pairs=None,boundary_recall_pairs=None,pair_reference_positions=0,inner_pair_bank_absent=True)
    for kind,r in g.items():value[kind+'_new_wrong_reference_positions']=r['H'];value[kind+'_repaired_reference_positions']=r['R']
    return value


def evaluate_inner(sequences, baselines, values, population, seed, policy,
                    input_identity, model_identity, fit_f1, *, expected_provenance):
    """Produce one of the34 prespecified records; never disguise inner as outer.

    head_seal_sha256 identifies the audited final-model/precalibration seal at
    this phase. Final model+calibration+endpoint actual-Git seal remains a
    separate prerequisite for later outer inference.
    """
    actual = inner_provenance(sequences, baselines, values, population, seed, policy,
                              input_identity, model_identity, fit_f1)
    if not isinstance(expected_provenance, dict) or analysis._digest(actual) != analysis._digest(expected_provenance):
        raise ValueError('independently expected inner data/model/prediction provenance differs')
    summary = analysis.safety_summary(sequences, baselines, values, policy, fit_f1)
    ids = [q['id'] for q in sequences]; bases = analysis._bank(baselines, ids)
    decoded = [None]*len(ids) if policy == 'Base' else analysis._bank(values, ids)
    predictions = {q['id']: np.asarray(b if policy == 'Base' else v[policy]['prediction']).copy()
                   for q, b, v in zip(sequences, bases, decoded)}
    rich = _inner_metric(sequences, predictions, bases)
    if rich['native_confusion'] != summary['native_confusion'] or rich['decoded_position_confusion'] != summary['position_confusion']:
        raise ValueError('inner rich native/position counts differ from decoded summary')
    for kind in analysis.KINDS:
        for name in ('H', 'R', 'dE'):
            if rich['gross_reference_position_effects'][kind][name] != summary[name][kind]:
                raise ValueError('inner gross counts differ from decoded summary')
    if (rich['inner_pair_bank_absent'] is not True or rich['pair_reference_positions'] != 0 or
            any(rich[k] is not None for k in ('segment_f1_50_pairs', 'boundary_precision_pairs', 'boundary_recall_pairs'))):
        raise ValueError('absent pair ratios must remain null, not invented zeros')
    binding = dict(copy.deepcopy(actual), rich_metrics_sha256=analysis._digest(rich),
                   safety_summary_sha256=analysis._digest(summary))
    record = dict(copy.deepcopy(actual['scope']), **copy.deepcopy(rich), input_identity=input_identity,
                  intrinsic_identity=copy.deepcopy(actual['intrinsic_identity']),
                  endpoint_binding=copy.deepcopy(binding))
    rich['endpoint_binding'] = copy.deepcopy(binding)
    summary['endpoint_binding'] = copy.deepcopy(binding)
    return dict(record=record, rich_metrics=rich, safety_summary=summary)
