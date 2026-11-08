"""Read-only completed-aux trace obligation refinement; all other runtime is inherited."""
import copy
from pathlib import Path
import time
from types import SimpleNamespace

from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState

SOURCE = 'echo_routing/temporal/selected_risk_completed_aux_controller.py'
_TRACE = 'retained_archive_plus_one_active_raw_and_atomic_peak'


class CompletedAuxRuntimeState(RuntimeState):
    def admission(self):
        original = super().admission()
        # The new controller must itself be in the independently verified current
        # actual-Git freeze. A caller boolean cannot opt into an unchecked source.
        freeze, evidence = self.ctx.enter(self.stage, self.spec['freeze'],
                                         self.spec['predecessors'], required={SOURCE})
        self.ctx.raw({'path': SOURCE, 'sha256': freeze['objects'][SOURCE]})
        if self.stage in ['close_negative', 'close_full'] or 'train_aux' not in evidence:
            return original

        from echo_routing.temporal.selected_risk_resume_estimate import _closed_storage_stage, formal_trace_projection
        from echo_routing.temporal.selected_risk_resume_S import StageS, SEEDS, SOURCE as STAGE_S_SOURCE, completed_head
        from echo_routing.temporal.selected_risk_resume_contract import RUN
        from echo_routing.temporal.selected_risk_resume_budget import forecast, remaining_limits

        verified = {key: value['payload'] for key, value in evidence.items()}
        if not {'train_aux', 'prepare_aux', 'admit_aux', 'observe_geometry'} <= set(verified):
            raise ValueError('complete original auxiliary preparation/admission/geometry evidence required')
        proof_pin = self.spec['arguments'].get('completed_aux_storage_proof')
        if proof_pin is None:
            raise ValueError('explicit externally pinned closed auxiliary storage proof required')
        proof = self.ctx.read(proof_pin)
        train = verified['train_aux']
        if (proof.get('schema') != 'A63_closed_aux_storage_materialization_readonly_v1'
                or proof.get('passed') is not True
                or proof.get('complete_train_manifest') != self.spec['predecessors']['train_aux']['payload_pin']
                or (proof.get('physical_updates'), proof.get('fullstate_count')) != (48000, 32)
                or train.get('passed') is not True
                or (train.get('aux_optimizer_updates'), train.get('formal_optimizer_updates'),
                    train.get('physical_models'), train.get('fullstate_count')) != (48000, 0, 8, 32)):
            raise ValueError('complete original eight-head/48000/32-state manifest/proof required')
        parent = _closed_storage_stage(self.ctx, self.spec, verified, self._worker_rows(), 'train_aux')
        if self.ctx.read(proof['actual_train_parent_OS_receipt']) != parent:
            raise ValueError('closed auxiliary proof must bind the exact accounted successful parent')

        def key(row):
            if type(row.get('held_fold')) is not int or type(row.get('seed')) is not int:
                raise ValueError('exact original auxiliary seed/fold types required')
            return row['held_fold'], row['seed'], row['role']

        expected = {(held, seed, role) for held in [0, 1] for seed in SEEDS for role in ['utility', 'risk']}
        records = train['models']; reported = proof['records']
        geometry = verified['observe_geometry']; predraw = geometry['aux_full_predraw']
        if any(len(rows) != 8 or {key(row) for row in rows} != expected for rows in [records, reported, predraw]):
            raise ValueError('all and only eight distinct original auxiliary heads required')
        aux_raw = {}
        for row in predraw:
            n = row['trace_raw_byte_upper']
            if type(n) is not int or n <= 0 or type(row.get('updates')) is not int or row['updates'] != 6000:
                raise ValueError('complete original positive full6000 predraw byte bounds required')
            aux_raw[key(row)] = n
        by_key = {key(row): row for row in records}
        reports = {key(row): row for row in reported}
        prepared = verified['prepare_aux']; folds = prepared['folds']
        if (len(folds) != 2 or any(type(f.get('held_fold')) is not int for f in folds)
                or {f['held_fold'] for f in folds} != {0, 1}
                or prepared.get('input_manifest_sha256') != self.spec['metadata']['sha256']):
            raise ValueError('both original independently pinned complementary input folds required')
        old_code = folds[0]['code_pins']
        if not old_code or STAGE_S_SOURCE not in old_code or any(not path.endswith('.py') for path in old_code):
            raise ValueError('complete original preparation code pins required')
        for fold in folds:
            if fold['code_pins'] != old_code:
                raise ValueError('original two preparation source closures differ')
        for path, sha256 in old_code.items():
            if freeze['objects'].get(path) != sha256:
                raise ValueError('original trained source changed in current actual-Git freeze')
            self.ctx.raw({'path': path, 'sha256': sha256})
        # Identity uses ONLY the original prepared code pins and original
        # admit_aux descriptor. New freeze/controller sources never enter it.
        stage_s = StageS(self.ctx, SimpleNamespace(metadata_pin=self.spec['metadata']),
                         self.spec['freeze'], self.spec['predecessors'])

        def normalize(pin):
            p = Path(pin['path'])
            if p.is_absolute():
                try:
                    p = p.relative_to(self.ctx.root)
                except ValueError as error:
                    raise ValueError('fullstate pin outside the original project') from error
            return {'path': p.as_posix(), 'sha256': pin['sha256']}

        def verify_pin(pin):
            return self.ctx.raw(normalize(pin))

        archive_bytes = state_bytes = 0
        for fold in folds:
            value = self.ctx.load_value(fold['data']); data, plans = value['data'], value['plans']
            for seed in SEEDS:
                for role in ['utility', 'risk']:
                    k = fold['held_fold'], seed, role; rec, rep = by_key[k], reports[k]
                    identity = stage_s._identity(fold, data, plans[str(seed)][role], freeze, seed, role)
                    relative = RUN + f'/aux/seed{seed}/held{fold["held_fold"]}/{role}'
                    folder = self.ctx.root / relative
                    if (rec.get('reused_complete') is not False
                            or rec['completion']['path'] != relative + '/completion.json'
                            or rep['completion'] != rec['completion']
                            or rep['archive_receipt']['path'] != relative + '/physical_updates.jsonl.archive.json'
                            or rep['archive'] != {'path': relative + '/physical_updates.jsonl.gz', 'sha256': rec['archive_sha256']}):
                        raise ValueError('original train manifest must externally pin every completion and archive')
                    completion = self.ctx.read(rec['completion'])
                    ar = self.ctx.read(rep['archive_receipt']); compressed = self.ctx.raw(rep['archive'])
                    if (ar.get('schema') != 'A63_future_complete_trace_gzip_v1'
                            or ar.get('archive_sha256') != rec['archive_sha256']
                            or ar.get('identity') != identity or rep.get('identity') != identity
                            or ar.get('uncompressed_sha256') != completion.get('raw_trace_sha256')
                            or rep.get('uncompressed_sha256') != ar.get('uncompressed_sha256')
                            or ar.get('rows') != 6000 or rep.get('rows') != 6000
                            or type(ar.get('uncompressed_bytes')) is not int
                            or not 0 < ar['uncompressed_bytes'] <= aux_raw[k]
                            or rep.get('uncompressed_bytes') != ar['uncompressed_bytes']
                            or ar.get('archive_bytes') != len(compressed)
                            or rep.get('archive_bytes') != len(compressed)
                            or rep.get('original_raw_retained') is not False
                            or (folder / 'physical_updates.jsonl').exists()
                            or (folder / 'physical_updates.jsonl').is_symlink()):
                        raise ValueError('original audited complete archive identity/bytes differ')
                    expected_states = {relative + '/' + name for name in ['step2000.pt', 'step4000.pt', 'step6000.pt', 'latest.pt']}
                    states = [normalize(pin) for pin in completion['state_pins']]
                    sizes = rep['fullstate_pins_and_bytes']
                    if (len(states) != 4 or {pin['path'] for pin in states} != expected_states
                            or len(sizes) != 4 or {r['pin']['path'] for r in sizes} != expected_states
                            or {r['pin']['path']: r['pin']['sha256'] for r in sizes} != {p['path']: p['sha256'] for p in states}
                            or normalize(rec['final']) not in states
                            or normalize(rec['final'])['path'] != relative + '/step6000.pt'):
                        raise ValueError('all original four fullstate SHA pins per head required')
                    for row in sizes:
                        raw = verify_pin(row['pin'])
                        if type(row['bytes']) is not int or row['bytes'] != len(raw):
                            raise ValueError('independent fullstate byte proof differs')
                        state_bytes += len(raw)
                    done = completed_head(folder, identity, verify_pin)
                    if done is None or done != completion:
                        raise ValueError('complete original full6000 trace/audit and no pending required')
                    archive_bytes += len(compressed)
        if proof.get('archive_total_bytes') != archive_bytes or proof.get('state_total_bytes') != state_bytes:
            raise ValueError('independent archive/fullstate totals differ')

        work = copy.deepcopy(original['remaining_complete_workload'])
        formal = formal_trace_projection(self.ctx, geometry)['per_head_raw_bytes']
        if work['formal_trace_structural_bound']['per_head_raw_bytes'] != formal:
            raise ValueError('original structural formal trace projection differs')
        def gzip_worst(n):
            return n + 5 * ((n + 16382) // 16383) + 64
        aux_gzip = sum(gzip_worst(n) for n in aux_raw.values())
        formal_gzip = 4 * gzip_worst(formal)
        old_active = max([*aux_raw.values(), formal])
        hard_full = aux_gzip + formal_gzip + old_active
        before = work['storage_components'][_TRACE]
        if (work['actual_aux_full_predraw_raw_bytes'] != sum(aux_raw.values())
                or work['actual_predraw_all_trace_gzip_worst_plus_active_bytes'] != hard_full
                or type(before) is not int or before < hard_full):
            raise ValueError('original complete trace obligation cannot be reconstructed')
        slack = before - hard_full
        after = formal_gzip + formal + slack
        credit = before - after
        if not 0 <= credit <= before <= work['storage_upper_bytes']:
            raise ValueError('nonnegative future-only auxiliary trace removal required')
        work['storage_components'][_TRACE] = after
        work['storage_upper_bytes'] -= credit
        metadata_key = 'completed_aux_controller_compact_future_metadata_upper'
        if metadata_key in work['storage_components']:
            raise ValueError('completed-aux future metadata room already allocated')
        # Root-approved nonnegative extra room for <=128KiB compact snapshots
        # over all33 stages: guardian/caller x current/Git, plus headers/atomic.
        # This is an independent32MiB addition; original closing stays.
        work['storage_components'][metadata_key] = 33_554_432
        work['storage_upper_bytes'] += 33_554_432
        phase = 'completed_aux_revalidation_remaining_CPU'
        if phase in work['phases']:
            raise ValueError('future auxiliary revalidation phase already allocated')
        # Nonnegative engineering estimate: measured read-only3.673203s x33
        # stages x2 uncertainty =242.431398s, plus room to300. Not actual spend.
        work['phases'][phase] = 300.
        work['CPU_seconds'] += 300.
        work['remaining_wall_seconds'] += 300.
        work['completed_aux_future_trace_refinement'] = {
            'proof_pin': proof_pin, 'original_trace_component_bytes': before,
            'completed_aux_gzip_worst_bytes': aux_gzip,
            'original_active_raw_bytes': old_active, 'formal_active_raw_bytes': formal,
            'retained_four_formal_gzip_worst_bytes': formal_gzip,
            'original_unallocated_slack_bytes': slack,
            'future_trace_component_bytes': after, 'future_obligation_removed_bytes': credit,
            'current_aux_archive_bytes_not_subtracted': archive_bytes,
            'current_fullstate_bytes_not_subtracted': state_bytes,
            'all_original_other_storage_components_and_CPU_phases_unchanged': True,
            'additional_compact_future_metadata_upper_bytes': 33_554_432,
            'additional_future_revalidation_CPU_seconds': 300.,
            'additional_future_revalidation_wall_seconds': 300.,
        }
        # Both super() and this read-only verification execute inside the same
        # guardian whole-process CPU/wall meter; no verification cost is refunded.
        spent = self.spent(); now = time.time()
        full = forecast(spent, work['GPU_seconds'], work['CPU_seconds'], work['storage_upper_bytes'],
                        now, work['remaining_wall_seconds'], self.owner, closing=False)
        limits = remaining_limits(spent, now, self.owner,
                                  potential_GPU_context=self.stage in ['source_FIT', 'source_CONFIRM'], closing=False)
        args = self.spec['arguments']
        return {**original, **full, 'remaining_complete_workload': work, 'already_spent': spent,
                'stage_CPU_limit': min(float(args['stage_CPU_upper_seconds']), limits['remaining_CPU_seconds'] - 10.),
                'stage_wall_limit': min(float(args['stage_wall_upper_seconds']), limits['stage_wall_seconds'] - 10.),
                'evaluated_epoch': now}
