"""Fabricated final-state/manifest fixtures only; no server, source, NN or SGD."""
import copy
import hashlib
import importlib
import os

import pytest

from echo_routing.temporal.selected_risk_budget import totals

def fixture():
    minimums = dict(D_danger30=30, D_sessions10=10, X_danger30=30, X_sessions10=10,
                    union60=60, union_sessions20=20, Hpositive40=40, Hge8_15=15,
                    pureFP8=8, pureFP_sessions5=5, A4C_A5C15=15, A4C_A5C_sessions6=6,
                    PLAX_PSAX8=8, PLAX_PSAX_sessions4=4, fullrun15=15, partial15=15,
                    D_safe100=100, X_safe100=100, D_safeJ30=30, X_safeJ30=30)
    seeds = {}
    for seed in (2026110461, 2026110462):
        checks = {k: dict(actual=v, minimum=v, passed=True) for k, v in minimums.items()}
        for k in ('max_p_D', 'max_p_X', 'max_p_mix'):
            checks[k] = dict(actual=.1, maximum=.2, passed=True)
        for k in ('pureFP8', 'pureFP_sessions5'):
            checks[k].update(actual=0, passed=False)
        member = dict(dangerous_signatures=30, responsible_sessions=10, safe_useful_signatures=100,
                      safe_J_signatures=30, safe_J_conflicts=0, same_draw_effect_conflicts=0)
        union = dict(dangerous_signatures=60, responsible_sessions=20, H_positive_signatures=40,
                     H_ge8_signatures=15, pure_FP_signatures=0, pure_FP_sessions=0,
                     A4C_A5C_signatures=15, A4C_A5C_sessions=6, PLAX_PSAX_signatures=8,
                     PLAX_PSAX_sessions=4, full_run_signatures=15, partial_run_signatures=15,
                     geometry_conflicts=0, replacement_class_signatures={str(i): 12 for i in range(5)})
        seeds[str(seed)] = dict(passed=False, checks=checks, failed_checks=['pureFP8', 'pureFP_sessions5'],
                                D=dict(member), X_only=dict(member), union=union, max_p_D=.1,
                                max_p_X=.1, max_p_mix=.1, native_D_danger_actions=0, native_X_danger_actions=0)
    return dict(passed=False, seeds=seeds, aux_optimizer_updates=48000,
                formal_optimizer_updates=0, source_isolation={'passed': True}), None


def module():
    return importlib.import_module('echo_routing.temporal.selected_risk_terminal')


def inputs():
    coverage, _ = fixture()
    audit = dict(passed=True, aux_optimizer_updates=48000, physical_models=8, fullstate_count=32)
    source = dict(teacher_payloads=6000, A08_windows=88731, JEPA_chunks=15150,
                  teacher_forwards=None, aux_optimizer_updates=48000,
                  formal_optimizer_updates=0, C0_C1_instances=0)
    receipts = [dict(receipt_id='prelaunch:1', stage='source-admission', measurement_host='fixture',
                     model_pid=3, started_epoch=1., finished_epoch=2., full_process_wall_seconds=1.,
                     measured_CPU_core_seconds=.5, GPU_job=False, CUDA_context_occupied=False,
                     observed_worker_exit_code=1, child_exit_code_directly_waited=True),
                dict(receipt_id='metadata:1', stage='metadata', measurement_host='fixture',
                     model_pid=4, started_epoch=3., finished_epoch=4., full_process_wall_seconds=1.,
                     measured_CPU_core_seconds=.75, GPU_job=False, CUDA_context_occupied=False,
                     observed_worker_exit_code=0, child_exit_code_directly_waited=True),
                dict(receipt_id='admin:1', stage='admin', measurement_host='fixture',
                     model_pid=None, started_epoch=5., finished_epoch=6., full_process_wall_seconds=1.,
                     measured_CPU_core_seconds=None, CPU_accounting_allowance_seconds=1.,
                     measurement_kind='administrative_partial_receipt', GPU_job=False,
                     CUDA_context_occupied=False, observed_worker_exit_code=None)]
    return coverage, audit, source, totals(receipts), receipts


def test_negative_retains_every_failed_check_untrained_stages_unknown_units_and_nonzero_OS():
    args = inputs(); state = module().terminal_state('coverage-negative', *args)
    assert state['mode'] == 'coverage-negative' and state['coverage_assessed']
    assert state['coverage_passed'] is False and not state['goal_met']
    assert not state['overall_complete'] and not state['C0_C1_sampling_hypothesis_tested']
    assert state['actual_counters']['teacher_forwards'] is None
    assert len(state['failed_checks']) == 4
    assert {x['seed'] for x in state['failed_checks']} == {2026110461, 2026110462}
    assert state['OS_summary']['nonzero_receipts'] == ['prelaunch:1']
    assert state['OS_summary']['unknown_receipts'] == ['admin:1']
    assert state['provided_worker_receipts_all_OS0'] is False
    assert set(state['publication_pending']) == {'independent_caller_OS_and_cost_tail', 'root_protected_main_integration'}
    assert state['not_run']['formal_optimizer_updates']['expected_if_admitted'] == 24000
    assert {k: state['not_run'][k]['expected_if_admitted'] for k in ['outer_rows', 'inner_rows', 'paired_comparisons', 'support_records', 'native_records', 'regression_records']} == dict(outer_rows=238, inner_rows=34, paired_comparisons=140, support_records=56, native_records=34, regression_records=32)
    assert all(x['status'] == 'not_run' and x['produced'] == 0 for x in state['not_run'].values())


@pytest.mark.parametrize('change', ['missing_seed', 'floor', 'aggregate', 'failure_list', 'gate', 'isolation', 'aux', 'formal', 'audit', 'source'])
def test_negative_rejects_incomplete_or_inconsistent_proof(change):
    c, a, s, cost, receipts = inputs()
    r = c['seeds']['2026110461']
    if change == 'missing_seed': del c['seeds']['2026110462']
    if change == 'floor': r['checks']['pureFP8']['minimum'] = 0
    if change == 'aggregate': r['union']['pure_FP_signatures'] = 4
    if change == 'failure_list': r['failed_checks'] = []
    if change == 'gate': c['passed'] = True
    if change == 'isolation': c['source_isolation']['passed'] = False
    if change == 'aux': c['aux_optimizer_updates'] = 47999
    if change == 'formal': s['formal_optimizer_updates'] = 1
    if change == 'audit': a['fullstate_count'] = 31
    if change == 'source': s['teacher_payloads'] = None
    with pytest.raises(ValueError): module().terminal_state('coverage-negative', c, a, s, cost, receipts)


def test_resource_stop_is_unassessed_not_a_coverage_or_hypothesis_failure():
    _, _, _, cost, receipts = inputs()
    state = module().terminal_state('resource-stopped', None, None, None, cost, receipts)
    assert not state['coverage_assessed'] and state['coverage_passed'] is None
    assert state['failed_checks'] == [] and not state['goal_met']
    assert not state['C0_C1_sampling_hypothesis_tested']
    assert all(v is None for v in state['actual_counters'].values())
    assert all(x['status'] == 'not_established' and x['produced'] is None for x in state['not_run'].values())
    report = module().render_negative_report(state)
    assert '未评估' in report and 'unknown' in report and '未被训练检验' in report
    assert '覆盖不通过' not in report


def test_resource_partial_auxiliary_counts_are_preserved_not_filled_to_plan():
    _, _, s, cost, receipts = inputs()
    s.update(aux_optimizer_updates=71, teacher_payloads=200)
    a = dict(passed=False, aux_optimizer_updates=71, physical_models=1, fullstate_count=0)
    st = module().terminal_state('resource-stopped', None, a, s, cost, receipts)
    assert st['actual_counters']['aux_optimizer_updates'] == 71 and not st['auxiliary_audit_complete']
    assert st['not_run']['formal_optimizer_updates']['produced'] == 0
    s['C0_C1_instances'] = 1
    with pytest.raises(ValueError): module().terminal_state('resource-stopped', None, a, s, cost, receipts)


def test_unsupported_mode_assessed_coverage_or_false_cost_OS_projection_rejected():
    c, a, s, cost, receipts = inputs()
    with pytest.raises(ValueError): module().terminal_state('passed', c, a, s, cost, receipts)
    with pytest.raises(ValueError): module().terminal_state('resource-stopped', c, a, s, cost, receipts)
    cost['failed_stages'] = 0
    with pytest.raises(ValueError): module().terminal_state('coverage-negative', c, a, s, cost, receipts)
    c, a, s, cost, receipts = inputs(); receipts[0]['observed_worker_exit_code'] = 0
    with pytest.raises(ValueError): module().terminal_state('coverage-negative', c, a, s, cost, receipts)


def test_pure_state_and_report_preserve_inputs_and_all_negative_values():
    args = inputs(); before = copy.deepcopy(args)
    state = module().terminal_state('coverage-negative', *args)
    report = module().render_negative_report(state)
    assert args == before
    assert all(k in report for k in ['pureFP8', 'pureFP_sessions5', '2026110461', '2026110462', '6000', '48000', 'unknown', '未被训练检验'])
    assert 'overall_complete=false' in report
    state['goal_met'] = True
    with pytest.raises(ValueError): module().render_negative_report(state)


@pytest.mark.parametrize('change', ['failed_checks', 'OS_summary', 'not_run'])
def test_report_cannot_drop_failures_or_rewrite_actual_status(change):
    state = module().terminal_state('coverage-negative', *inputs())
    if change == 'failed_checks': state['failed_checks'] = []
    if change == 'OS_summary': state['OS_summary']['nonzero_receipts'] = []
    if change == 'not_run': state['not_run']['outer_rows']['status'] = 'completed'
    with pytest.raises(ValueError): module().render_negative_report(state)


def test_exact_manifest_uses_only_explicit_list_excludes_self_and_does_not_write(tmp_path):
    rel = 'docs/results/2026-11-05_selected_action_risk/result.json'; p = tmp_path / rel
    p.parent.mkdir(parents=True); p.write_bytes(b'\x00test\xff')
    ignored = p.with_name('unlisted.json'); ignored.write_text('never auto glob')
    own = str(p.with_name('PUBLIC_MANIFEST.json').relative_to(tmp_path))
    before = sorted(str(q) for q in tmp_path.rglob('*'))
    result = module().exact_manifest(tmp_path, [own, rel], 'a'*40, own)
    assert result['files'] == [dict(path=rel, bytes=6, sha256=hashlib.sha256(p.read_bytes()).hexdigest())]
    assert result['file_count'] == 1 and result['total_bytes'] == 6
    assert result['self_excluded'] == own and result['source_commit'] == 'a'*40
    assert not result['Git_objects_independently_verified'] and not result['overall_complete']
    assert sorted(str(q) for q in tmp_path.rglob('*')) == before
    assert not (tmp_path / own).exists()


@pytest.mark.parametrize('bad', ['../outside.json', '/tmp/absolute.json', 'docs/results/2026-11-04_action_safety_ranking/x.json', 'echo_routing/temporal/dense_analysis.py', 'echo_routing/temporal/jepa21_source.py', 'web/index.html'])
def test_manifest_rejects_unauthorized_or_unnormalized_paths(tmp_path, bad):
    with pytest.raises(ValueError): module().exact_manifest(tmp_path, [bad], 'a'*40, 'docs/results/2026-11-05_selected_action_risk/PUBLIC_MANIFEST.json')


def test_manifest_rejects_duplicates_symlink_ancestor_and_wrong_explicit_pin(tmp_path):
    rel = 'docs/results/2026-11-05_selected_action_risk/x.json'; p = tmp_path / rel
    p.parent.mkdir(parents=True); p.write_text('actual')
    own = str(p.with_name('PUBLIC_MANIFEST.json').relative_to(tmp_path))
    with pytest.raises(ValueError): module().exact_manifest(tmp_path, [rel, rel], 'a'*40, own)
    with pytest.raises(ValueError): module().exact_manifest(tmp_path, [dict(path=rel, sha256='b'*64, bytes=6)], 'a'*40, own)
    p.unlink(); p.symlink_to(tmp_path / 'outside')
    with pytest.raises(ValueError): module().exact_manifest(tmp_path, [rel], 'a'*40, own)
    p.unlink(); p.parent.rmdir(); p.parent.symlink_to(tmp_path)
    with pytest.raises(ValueError): module().exact_manifest(tmp_path, [rel], 'a'*40, own)


@pytest.mark.parametrize('kind', ['leaf', 'ancestor'])
def test_excluded_self_path_also_rejects_symlinks_without_reading_its_bytes(tmp_path, kind):
    rel = 'docs/results/2026-11-05_selected_action_risk/payload.json'
    p = tmp_path/rel; p.parent.mkdir(parents=True); p.write_text('payload')
    outside = tmp_path/'outside'; outside.mkdir()
    own = p.parent/'PUBLIC_MANIFEST.json'
    if kind == 'leaf': own.symlink_to(outside/'not-read')
    else:
        link = p.parent/'alias'; link.symlink_to(outside, target_is_directory=True)
        own = link/'PUBLIC_MANIFEST.json'
    with pytest.raises(ValueError):
        module().exact_manifest(tmp_path, [rel], 'a'*40, str(own.relative_to(tmp_path)))


def test_manifest_opens_nonblocking_then_rejects_fifo_before_any_read(tmp_path, monkeypatch):
    m = module(); rel = 'docs/results/2026-11-05_selected_action_risk/fifo.json'
    p = tmp_path/rel; p.parent.mkdir(parents=True); os.mkfifo(p)
    own = str(p.with_name('PUBLIC_MANIFEST.json').relative_to(tmp_path)); original = os.open
    def checked_open(path, flags, *args, **kwargs):
        if path == 'fifo.json':
            assert flags & os.O_NONBLOCK, 'must not block opening a non-regular payload'
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(m.os, 'open', checked_open)
    with pytest.raises(ValueError): m.exact_manifest(tmp_path, [rel], 'a'*40, own)

