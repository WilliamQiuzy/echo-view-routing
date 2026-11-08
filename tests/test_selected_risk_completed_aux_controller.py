"""Pure/mock closed-aux storage control. All traces/states below are synthetic."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from echo_routing.temporal.selected_risk_resume_S import StageS, SEEDS
from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
from echo_routing.temporal.selected_risk_resume_contract import RUN

SOURCE = 'echo_routing/temporal/selected_risk_completed_aux_controller.py'
TRACE = 'retained_archive_plus_one_active_raw_and_atomic_peak'
RAW = [14017317, 9920837, 14019602, 9920992, 14384888, 9981081, 14392905, 9981204]


def api():
    path = Path(__file__).resolve().parents[1]/SOURCE
    spec = importlib.util.spec_from_file_location('private_completed_aux_controller', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class FakeContext:
    def __init__(self, root):
        self.root = root; self.owner = {'mock_owner': 'same'}; self.owner_sha = 'a'*64
        self.freeze = {'objects': {}}; self.values = {}; self.entered = []
    def put(self, path, value):
        raw = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + '\n').encode()
        target = self.root/path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        return {'path': path, 'sha256': hashlib.sha256(raw).hexdigest()}
    def raw(self, pin):
        assert set(pin) == {'path', 'sha256'}
        p = self.root/pin['path']
        if p.is_symlink() or not p.is_file(): raise ValueError('mock pinned ordinary file missing')
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin['sha256']: raise ValueError('mock current SHA differs')
        return raw
    def read(self, pin): return json.loads(self.raw(pin))
    def load_value(self, pin): self.raw(pin); return self.values[pin['path']]
    def enter(self, stage, freeze, descriptors, *, required):
        self.entered.append(set(required))
        for path in required:
            if path not in self.freeze['objects']: raise ValueError('source not in mock Git-qualified freeze')
            self.raw({'path': path, 'sha256': self.freeze['objects'][path]})
        return self.freeze, {key: {'payload': self.read(desc['payload_pin'])} for key, desc in descriptors.items()}


def case(tmp_path, monkeypatch):
    module = api(); ctx = FakeContext(tmp_path)
    import echo_routing.temporal.proposal_training as journal_module
    qualified = Path(journal_module.__file__).resolve().parents[2]
    old = {}
    for path in ['echo_routing/temporal/event_risk_model.py', 'echo_routing/temporal/selected_risk_resume_formal.py', 'echo_routing/temporal/proposal_training.py', 'echo_routing/temporal/selected_risk_resume_S.py']:
        pin = ctx.put(path, (qualified/path).read_bytes()); old[path] = pin['sha256']
    ctx.freeze['objects'].update(old)
    cfg = ctx.put('configs/experiments/selected_action_risk_v1.json', b'mock immutable config')
    ctx.freeze['objects'][cfg['path']] = cfg['sha256']
    for path, raw in [(SOURCE, Path(module.__file__).read_bytes()), ('echo_routing/temporal/new_extra_frozen_source.py', b'new controller-only closure')]:
        pin = ctx.put(path, raw); ctx.freeze['objects'][path] = pin['sha256']
    spec = {'stage': 'audit_aux', 'metadata': ctx.put('mock_metadata.json', {'input': 'fixed'}), 'freeze': {'path': 'mock_freeze.json', 'sha256': 'b'*64}, 'predecessors': {}, 'arguments': {'stage_CPU_upper_seconds': 500., 'stage_wall_upper_seconds': 700.}}
    def evidence(key, path, payload):
        pin = ctx.put(path, payload); spec['predecessors'][key] = {'payload_pin': pin}; return pin
    admit_pin = evidence('admit_aux', 'mock_admit_aux.json', {'passed': True})
    plans = {str(seed): {role: {'indices': np.zeros((6000,64), dtype=np.int64)} for role in ['utility','risk']} for seed in SEEDS}
    folds = []
    for held in [0,1]:
        data = {'normalizers': {'candidate': {'mean': np.zeros(1), 'std': np.ones(1)}}, 'F_fit': 1., 'scale': 1., 'targets': np.zeros(1)}
        pin = ctx.put(RUN+f'/aux/held{held}/training_inputs.private.npz', b'synthetic typed placeholder')
        ctx.values[pin['path']] = {'data': data, 'plans': plans}
        folds.append({'held_fold': held, 'data': pin, 'code_pins': dict(old)})
    evidence('prepare_aux', 'mock_prepare_aux.json', {'passed': True, 'folds': folds, 'input_manifest_sha256': spec['metadata']['sha256']})
    geometry = {'passed': True, 'roles': {name: {'geometry': {'C': n}} for name,n in zip(['F0_native','F1_native','F0_mixed','F1_mixed'], [156,109,3901,3945])}, 'aux_full_predraw': []}
    keys = [(held,seed,role) for held in [0,1] for seed in SEEDS for role in ['utility','risk']]
    geometry['aux_full_predraw'] = [{'held_fold': held, 'seed': seed, 'role': role, 'updates': 6000, 'trace_raw_byte_upper': raw} for (held,seed,role),raw in zip(keys,RAW)]
    evidence('observe_geometry', 'mock_geometry.json', geometry)
    stage_s = StageS(ctx, SimpleNamespace(metadata_pin=spec['metadata']), spec['freeze'], spec['predecessors'])
    records = []; proofs = []; state_total = 0; archive_total = 0
    for held,seed,role in keys:
        fold = folds[held]; data = ctx.values[fold['data']['path']]['data']; identity = stage_s._identity(fold, data, plans[str(seed)][role], ctx.freeze, seed, role)
        folder = RUN+f'/aux/seed{seed}/held{held}/{role}'
        states = [ctx.put(folder+'/'+name, (folder+name).encode()) for name in ['step2000.pt','step4000.pt','step6000.pt','latest.pt']]
        raw = b''.join((json.dumps({'u': u, 'identity': identity}, sort_keys=True)+'\n').encode() for u in range(6000))
        compressed = gzip.compress(raw, compresslevel=1, mtime=0)
        archive = ctx.put(folder+'/physical_updates.jsonl.gz', compressed)
        ar = {'schema': 'A63_future_complete_trace_gzip_v1', 'archive_name': 'physical_updates.jsonl.gz', 'archive_sha256': archive['sha256'], 'archive_bytes': len(compressed), 'uncompressed_sha256': hashlib.sha256(raw).hexdigest(), 'uncompressed_bytes': len(raw), 'rows': 6000, 'identity': identity, 'lossless_original_bytes_available': True, 'original_physical_update_count_unchanged': True}
        receipt = ctx.put(folder+'/physical_updates.jsonl.archive.json', ar)
        completion = ctx.put(folder+'/completion.json', {'passed': True, 'identity': identity, 'physical_updates': 6000, 'fullstate_count': 4, 'state_pins': states, 'raw_trace_sha256': ar['uncompressed_sha256']})
        records.append({'seed': seed, 'held_fold': held, 'role': role, 'reused_complete': False, 'final': states[2], 'completion': completion, 'archive_sha256': archive['sha256']})
        sizes = [{'pin': pin, 'bytes': len(ctx.raw(pin))} for pin in states]
        proofs.append({'seed': seed, 'held_fold': held, 'role': role, 'completion': completion, 'archive_receipt': receipt, 'archive': archive, 'archive_bytes': len(compressed), 'uncompressed_bytes': len(raw), 'uncompressed_sha256': ar['uncompressed_sha256'], 'rows': 6000, 'fullstate_pins_and_bytes': sizes, 'identity': identity, 'original_raw_retained': False})
        state_total += sum(r['bytes'] for r in sizes); archive_total += len(compressed)
    train = evidence('train_aux', 'mock_train_aux.json', {'passed': True, 'models': records, 'aux_optimizer_updates': 48000, 'formal_optimizer_updates': 0, 'physical_models': 8, 'fullstate_count': 32})
    stdout = ctx.put('mock_train.stdout', {'stage': 'train_aux', 'stage_returned_success': True, 'runtime_authorization_sha256': ctx.owner_sha, 'output': train})
    worker = {'stage': 'train_aux', 'model_pid': 1234, 'stage_exit_status': 0, 'observed_worker_exit_code': 0, 'inner_model_observed_OS_exit_code': 0, 'whole_child_accounting_complete': True, 'actual_child_directly_waited': True, 'unresolved_worker_uncertainty': False, 'runtime_authorization_sha256': ctx.owner_sha, 'log_pins': [stdout], 'measured_CPU_core_seconds': 8.}
    parent = ctx.put('mock_train_parent.json', worker)
    proof = {'schema': 'A63_closed_aux_storage_materialization_readonly_v1', 'passed': True, 'complete_train_manifest': train, 'actual_train_parent_OS_receipt': parent, 'records': proofs, 'physical_updates': 48000, 'fullstate_count': 32, 'archive_total_bytes': archive_total, 'state_total_bytes': state_total}
    proofpin = ctx.put('docs/results/2026-11-05_a63_resume_preparation/closed_aux_storage_readonly_v1.json', proof)
    spec['arguments']['completed_aux_storage_proof'] = proofpin
    components = {'current_namespace_plus_admin_upper': 836947753, 'teacher_payload_journal_atomic_upper': 4355616, 'retained_fullstates_atomic_upper': 12500000, '12_draw_arrays_plus_atomic_temp': 39936000, 'closing_reserve': 128000000, 'minimal_aux_plus_compact_shared_cache_phase_peak': 76628035, TRACE: 155692612, 'selected_typed_records_atomic_upper': 7944314, 'continuation_bound_rich_private_bundles_allowance': 48000000, 'FIT_final_risk_q_exposure_and_report_complete_bound': 15205868}
    work = {'storage_components': components, 'storage_upper_bytes': sum(components.values()), 'GPU_seconds': 0., 'CPU_seconds': 12345., 'remaining_wall_seconds': 19748., 'phases': {'unchanged_full_scope': 12345.}, 'formal_trace_structural_bound': {'per_head_raw_bytes': 8892000}, 'actual_aux_full_predraw_raw_bytes': sum(RAW), 'actual_predraw_all_trace_gzip_worst_plus_active_bytes': 146620869}
    base = {'all_caps_fit': False, 'remaining_complete_workload': work, 'already_spent': {'before': True}, 'stage_CPU_limit': 490., 'stage_wall_limit': 690., 'actual_storage_cap': 1300000000, 'evaluated_epoch': 99.}
    monkeypatch.setattr(RuntimeState, 'admission', lambda self: copy.deepcopy(base))
    instance = module.CompletedAuxRuntimeState(ctx, spec, {'path': 'mock_spec.json', 'sha256': 'f'*64})
    monkeypatch.setattr(instance, '_worker_rows', lambda: [worker])
    monkeypatch.setattr(instance, 'spent', lambda: {'fresh_actual_spent': True})
    import echo_routing.temporal.selected_risk_resume_budget as budget
    calls = []
    def forecast(*args, **kwargs): calls.append((args,kwargs)); return {'all_caps_fit': True, 'forecast_incremental_tracked_storage_upper_bytes': args[3]}
    monkeypatch.setattr(budget, 'forecast', forecast)
    monkeypatch.setattr(budget, 'remaining_limits', lambda *a,**k: {'remaining_CPU_seconds': 500., 'stage_wall_seconds': 700.})
    return SimpleNamespace(module=module, ctx=ctx, state=instance, base=base, work=work, spec=spec, worker=worker, proof=proof, proofpin=proofpin, records=records, calls=calls)


def test_exact_completed_aux_obligation_only_preserves_every_other_component(tmp_path, monkeypatch):
    c = case(tmp_path, monkeypatch); result = c.state.admission(); work = result['remaining_complete_workload']
    assert work['storage_components'][TRACE] == 53542859
    assert work['storage_components']['completed_aux_controller_compact_future_metadata_upper'] == 33554432
    assert c.work['storage_upper_bytes'] - work['storage_upper_bytes'] == 68595321
    assert {k:v for k,v in work['storage_components'].items() if k not in [TRACE, 'completed_aux_controller_compact_future_metadata_upper']} == {k:v for k,v in c.work['storage_components'].items() if k != TRACE}
    assert work['GPU_seconds'] == c.work['GPU_seconds']
    assert work['CPU_seconds'] == c.work['CPU_seconds'] + 300.
    assert work['remaining_wall_seconds'] == c.work['remaining_wall_seconds'] + 300.
    assert work['phases']['completed_aux_revalidation_remaining_CPU'] == 300.
    assert {k:v for k,v in work['phases'].items() if k != 'completed_aux_revalidation_remaining_CPU'} == c.work['phases']
    assert result['already_spent'] == {'fresh_actual_spent': True}
    assert result['stage_CPU_limit'] == 490. and result['stage_wall_limit'] == 690.
    assert result['actual_storage_cap'] == c.base['actual_storage_cap']
    assert c.calls[0][0][1:4] == (work['GPU_seconds'], work['CPU_seconds'], work['storage_upper_bytes'])
    assert c.calls[0][0][5] == work['remaining_wall_seconds']
    assert c.calls[0][0][3] == work['storage_upper_bytes'] and c.calls[0][0][6] == c.ctx.owner
    assert c.calls[0][0][4] > 99. and c.calls[0][1] == {'closing': False}
    assert SOURCE in set.union(*c.ctx.entered)
    # Already-stored gzip bytes were not deducted from currentdu or states.
    assert work['storage_components']['current_namespace_plus_admin_upper'] == 836947753
    assert c.proof['archive_total_bytes'] != 102149753


def test_no_train_or_receipt_only_closure_keeps_original_forecast(tmp_path, monkeypatch):
    c = case(tmp_path, monkeypatch)
    del c.spec['predecessors']['train_aux']; del c.spec['arguments']['completed_aux_storage_proof']
    assert c.state.admission() == c.base and not c.calls
    # No geometry/trace component or proof is required for bounded closure.
    c.spec['predecessors'] = {}; c.base['remaining_complete_workload'] = {'scope': 'bounded receipt-only closure'}
    for stage in ['close_negative','close_full']:
        c.state.stage = stage; c.spec['stage'] = stage
        assert c.state.admission() == c.base and not c.calls


def test_bad_OS_missing_witness_corrupt_archive_pending_and_source_pin_fail_closed(tmp_path, monkeypatch):
    c = case(tmp_path, monkeypatch)
    c.worker['inner_model_observed_OS_exit_code'] = 7
    with pytest.raises(ValueError): c.state.admission()
    c.worker['inner_model_observed_OS_exit_code'] = 0
    proofpin = c.spec['arguments'].pop('completed_aux_storage_proof')
    with pytest.raises(ValueError): c.state.admission()
    c.spec['arguments']['completed_aux_storage_proof'] = proofpin
    first = c.proof['records'][0]; archive = c.ctx.root/first['archive']['path']; before = archive.read_bytes(); archive.write_bytes(before+b'corrupt')
    with pytest.raises(ValueError): c.state.admission()
    archive.write_bytes(before)
    pending = archive.parent/'pending.json'; pending.write_text('{}')
    with pytest.raises(ValueError): c.state.admission()
    pending.unlink()
    old = c.ctx.freeze['objects'].pop(SOURCE)
    with pytest.raises(ValueError): c.state.admission()
    c.ctx.freeze['objects'][SOURCE] = old
    # Original actual-Git manifest, not a self-consistent new archive, anchors completion.
    completion = c.ctx.root/c.records[0]['completion']['path']; completion.write_text('{}')
    with pytest.raises(ValueError): c.state.admission()
