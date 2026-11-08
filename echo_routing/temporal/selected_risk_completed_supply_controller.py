"""Future-only compact supply storage refinement; no scientific seal or NN path."""
import copy
from pathlib import Path
import time

from echo_routing.temporal.selected_risk_completed_aux_controller import CompletedAuxRuntimeState

SOURCE = 'echo_routing/temporal/selected_risk_completed_supply_controller.py'
_COMPONENT = 'minimal_aux_plus_compact_shared_cache_phase_peak'
_ROLES = ('F0_native', 'F0_mixed', 'F1_native', 'F1_mixed')
_LAYOUT_PINS = {
    'echo_routing/temporal/selected_risk_compact_cache.py': 'b4f847722fc432ab718ebdb5855b11da96f35fc80d64403189a6f88ef736c907',
    'echo_routing/temporal/selected_risk_cache.py': '19692700c31b1f2eaf36a1948429948091f400923df4703bec1923a916f68600',
    'echo_routing/temporal/selected_risk_refined_forecast.py': 'c931b90963cba41096e24444eff270f968b7a2f6f988928643e4dfe87009c609',
    'echo_routing/temporal/selected_risk_resume_supply.py': '6fcea37cca68c758c198691b19507d9e302f87bd91ef793daf4b2ee4f01528c2',
    'echo_routing/temporal/selected_risk_resume_seals.py': 'cfc30e61bfb75fa3c1bc965102dc60f4d6f877a3b7f5edf107b6247116159366',
}
_OLD_EXTRA_PINS = {
    'echo_routing/temporal/selected_risk_completed_aux_controller.py': '825b45d92deff842275507dfd81f27f1abcf9015164a2f27191721ea5c11816d',
    'scripts/a63_prepare_completed_aux_selected_risk.py': 'ed791f20fd1e286f8c7ad773374227abe73c1576cb729e14fdd4a1ad5134979d',
    'scripts/a63_resume_completed_aux_selected_risk.py': '7e4d48af52343377a62fa3adc5cdbd3bde4a96a1b8014aec1f5c3cf1a5ed379d',
    'scripts/a63_wait_completed_aux_selected_risk.py': '0f89201b2bcdc34a7341e440b50ac513c9e09618d45c275ac7a5d321f03ab383',
}


class CompletedSupplyRuntimeState(CompletedAuxRuntimeState):
    def admission(self):
        original = super().admission()
        required = {SOURCE} | set(_LAYOUT_PINS) | set(_OLD_EXTRA_PINS)
        freeze, evidence = self.ctx.enter(self.stage, self.spec['freeze'],
                                         self.spec['predecessors'], required=required)
        for path, sha256 in {**_LAYOUT_PINS, **_OLD_EXTRA_PINS}.items():
            if freeze['objects'].get(path) != sha256:
                raise ValueError('original writer/cache/controller source changed')
            self.ctx.raw({'path': path, 'sha256': sha256})
        self.ctx.raw({'path': SOURCE, 'sha256': freeze['objects'][SOURCE]})
        if self.stage in ['close_negative', 'close_full']:
            return original
        work = copy.deepcopy(original['remaining_complete_workload'])
        materialized = work.get('verified_future_storage_obligations_completed', [])
        if not isinstance(materialized, list) or len(materialized) != len(set(materialized)):
            raise ValueError('exact verified materialization stage set required')
        if 'prepare_aux' not in materialized:
            return original

        from echo_routing.temporal.selected_risk_resume_estimate import _closed_storage_stage
        from echo_routing.temporal.selected_risk_resume_contract import RUN, PUB, digest
        from echo_routing.temporal.selected_risk_resume_inputs import ReadOnlyInputs
        from echo_routing.temporal.selected_risk_compact_cache import CompactSupplyCache, payload_upper
        from echo_routing.temporal.selected_risk_resume_supply import Supply, POST_NATIVE_PINS
        from echo_routing.temporal.selected_risk_resume_S import SEEDS
        from echo_routing.temporal.selected_risk_resume_budget import forecast, remaining_limits
        from echo_routing.temporal.safety_git import verify

        verified = {key: value['payload'] for key, value in evidence.items()}
        if not {'prepare_aux', 'observe_geometry'} <= set(verified):
            raise ValueError('original prepared/actual geometry descriptors required')
        workers = self._worker_rows()
        for stage in ['prepare_aux', 'seal_selectors', 'prepare_formal']:
            if stage in materialized:
                if stage not in verified:
                    raise ValueError('materialized stage lacks actual evidence: ' + stage)
                _closed_storage_stage(self.ctx, self.spec, verified, workers, stage)
        if 'prepare_formal' in materialized and 'seal_selectors' not in materialized:
            raise ValueError('formal materialization requires actual selector seal')
        _closed_storage_stage(self.ctx, self.spec, verified, workers, 'observe_geometry')
        prepared = verified['prepare_aux']; geometry = verified['observe_geometry']
        folds = prepared['folds']
        if (len(folds) != 2 or {f['held_fold'] for f in folds} != {0, 1}
                or prepared.get('input_manifest_sha256') != self.spec['metadata']['sha256']
                or geometry.get('passed') is not True or geometry.get('complete') is not True
                or geometry.get('input_manifest_sha256') != self.spec['metadata']['sha256']
                or geometry.get('producer_FIT_payload_sha256') != self.spec['FIT_producer']['sha256']
                or prepared.get('producer_FIT_sha256') != geometry['producer_FIT_payload_sha256']
                or prepared.get('actual_geometry_sha256') != digest(geometry)):
            raise ValueError('original complete geometry/teacher/input bindings differ')
        old_code = folds[0]['code_pins']
        if not old_code or any(f['code_pins'] != old_code for f in folds):
            raise ValueError('original complete two-fold code closure required')
        for path, sha256 in old_code.items():
            if freeze['objects'].get(path) != sha256:
                raise ValueError('original prepared source changed in current actual-Git freeze')
            self.ctx.raw({'path': path, 'sha256': sha256})
        if any(old_code.get(path) != sha256 for path, sha256 in _LAYOUT_PINS.items()):
            raise ValueError('layout proof must use the original prepared writer/cache sources')
        if set(geometry['roles']) != set(_ROLES):
            raise ValueError('complete exact four original FIT roles required')
        observed = {}
        for role in _ROLES:
            g = geometry['roles'][role]['geometry']; n, c = g['N'], g['C']
            if any(type(v) is not int or v < 0 for v in [n, c]) or n == 0 and c != 0:
                raise ValueError('nonnegative actual axis/candidate geometry required')
            observed[role] = n, c
        header = payload_upper(0, reference=False)
        if header != 4096 or payload_upper(0, reference=True) != 4096:
            raise ValueError('original4096 payload header must remain unchanged')
        # Two seed payloads: first has217 float64 scalars/candidate, reference
        # seed has12; BOTH retain4096 bytes for EVERY axis, including empty G.
        actual_joint = sum(payload_upper(c, reference=False) - header
                           + payload_upper(c, reference=True) - header + 2 * header * n
                           for n, c in observed.values())
        layout = work['feature_layout_components']; old_joint = layout['complete_two_seed_compact_cache']
        formal_targets = layout['formal_targets_only_no_persistent205']
        if any(type(v) is not int or v < 0 for v in [old_joint, formal_targets]) or actual_joint > old_joint:
            raise ValueError('actual complete cache layout exceeds original future bound')
        selector_done = 'seal_selectors' in materialized
        formal_done = 'prepare_formal' in materialized
        target_future = 0 if formal_done else formal_targets
        before = work['storage_components'][_COMPONENT]
        if before != (0 if selector_done else old_joint) + target_future:
            raise ValueError('original materialized feature/cache component does not reconcile')
        future_joint = 0 if selector_done else actual_joint
        held_verified = False

        if 'held_score' in verified:
            parent = _closed_storage_stage(self.ctx, self.spec, verified, workers, 'held_score')
            intent_pin = parent.get('caller_intent')
            if not isinstance(intent_pin, dict):
                raise ValueError('canonical closed held caller intent witness required')
            intent = self.ctx.read(intent_pin); origin_pin = intent['runtime_spec']
            if (intent.get('stage') != 'held_score'
                    or intent.get('runtime_authorization_sha256') != self.ctx.owner_sha
                    or origin_pin.get('sha256') != parent.get('runtime_spec_sha256')
                    or (intent.get('caller_pid'), intent.get('caller_start_ticks'), intent.get('boot_id'), intent.get('caller_nonce'))
                    != (parent.get('caller_pid'), parent.get('caller_start_ticks'), parent.get('caller_boot_id'), parent.get('caller_nonce'))):
                raise ValueError('original closed held caller/spec identity differs')
            origin_spec = self.ctx.read(origin_pin); origin = self.ctx.read(origin_spec['freeze'])
            fields = {'schema', 'stage', 'git_commit', 'objects', 'historical_frozen_imports',
                      'actual_Git_object_proof', 'runtime_authorization_sha256', 'runtime_spec_semantics_sha256'}
            if (set(origin) not in [fields, fields | {'proof_encoding', 'shared_object_pool'}]
                    or origin_spec.get('stage') != 'held_score' or origin.get('stage') != 'held_score'
                    or origin.get('schema') != 'A63_resume_stage_freeze_v1'
                    or origin.get('runtime_authorization_sha256') != self.ctx.owner_sha
                    or origin.get('runtime_spec_semantics_sha256') != digest({k: v for k, v in origin_spec.items() if k != 'freeze'})
                    or origin_spec.get('metadata') != self.spec['metadata']
                    or origin_spec.get('FIT_producer') != self.spec['FIT_producer']):
                raise ValueError('original actual held-score freeze/semantics/input witness differs')
            current = dict(origin['objects']); origin_py = {p: h for p, h in current.items() if p.endswith('.py')}
            if not origin_py or any(origin_py.get(p) != h for p, h in {**old_code, **_OLD_EXTRA_PINS}.items()):
                raise ValueError('original held stage omitted prepared sources/old extras')
            for path, sha256 in current.items():
                self.ctx.raw({'path': path, 'sha256': sha256})
                if path.endswith('.py') and freeze['objects'].get(path) != sha256:
                    raise ValueError('original held source changed; new extra sources must stay separate')
            for pin in [self.ctx.owner_pin, self.ctx.proposal_pin, self.ctx.baseline_pin]:
                if current.get(pin['path']) != pin['sha256']:
                    raise ValueError('original held stage authority differs')
            historical = self.ctx._historical_expected()
            if set(origin['historical_frozen_imports']) != set(historical):
                raise ValueError('complete literal historical table required in original held freeze')
            for path, row in origin['historical_frozen_imports'].items():
                expected = historical[path]
                if {k: row[k] for k in ['git_commit', 'blob_oid', 'sha256']} != expected or current.pop(path, None) != row['sha256']:
                    raise ValueError('original held literal historical source differs')
                verify(self.ctx.contract_proof(origin, row['proof']), row['git_commit'], {path: row['sha256']})
            verify(self.ctx.contract_proof(origin, origin['actual_Git_object_proof']), origin['git_commit'], current)
            if 'seal_aux' not in verified or origin_spec['predecessors']['seal_aux']['payload_pin'] != self.spec['predecessors']['seal_aux']['payload_pin']:
                raise ValueError('original held auxiliary core seal pin must be retained')
            inputs = ReadOnlyInputs(self.ctx, self.spec['metadata'], FIT_producer_pin=self.spec['FIT_producer'], CONFIRM_producer_pin=self.spec['CONFIRM_producer'])
            supply = Supply(self.ctx, inputs, origin_spec['freeze'], origin_spec['predecessors'])
            values, pins = supply._aux(verified['seal_aux'])
            manifest = verified['held_score']; records = manifest['records']
            expected_keys = {(seed, role) for seed in SEEDS for role in _ROLES}
            if (self.spec['predecessors']['held_score']['payload_pin']['path'] != PUB + '/held_selector_cache_manifest.json'
                    or manifest.get('passed') is not True or manifest.get('complete') is not True
                    or manifest.get('effect_lookup_before_seal') is not False
                    or manifest.get('aux_core_pins') != pins or len(records) != 8
                    or {(r['seed'], r['role']) for r in records} != expected_keys
                    or manifest.get('native_D_C6_frozen_pins') != {str(seed): pin for seed, pin in POST_NATIVE_PINS.items()}):
                raise ValueError('complete original label-free eight-cache/three-core bindings required')
            refs = {}
            for seed in SEEDS:
                for role in _ROLES:
                    rec = next(r for r in records if (r['seed'], r['role']) == (seed, role))
                    held = int(role[1]); relative = RUN + f'/held/seed{seed}/{role}'
                    identity = digest({'stage_code': origin_py, 'seed': seed, 'held_fold': held, 'role': role,
                                       'metadata': inputs.metadata_pin['sha256'], 'aux_core_pins': pins,
                                       'native_frozen_C6_pin': POST_NATIVE_PINS[seed] if role.endswith('_native') else None})
                    folder = self.ctx.root / relative
                    if (rec.get('path') != relative or rec.get('identity') != identity
                            or rec.get('feature_reference_seed') != (None if seed == SEEDS[0] else SEEDS[0])
                            or folder.is_symlink() or (folder / 'pending.json').exists() or (folder / 'pending.json').is_symlink()):
                        raise ValueError('original ordered cache identity/reference or pending prefix differs')
                    qs, aa, jj, bb = inputs.bank(role, purpose='selector')
                    if len(qs) != observed[role][0] or type(rec.get('axes')) is not int or rec['axes'] != len(qs):
                        raise ValueError('original complete cache axis geometry differs')
                    cache = CompactSupplyCache(folder, identity, bb, [p.argmax(1) for p in aa], [p.argmax(1) for p in jj], readonly=True, feature_reference=refs.get(role))
                    journal_pin = {'path': str(cache.journal.relative_to(self.ctx.root)), 'sha256': rec['journal_sha256']}
                    self.ctx.raw(journal_pin)
                    if (cache.cursor != len(qs) or rec.get('input_sha256') != inputs.manifest['roles'][role]['sha256']
                            or sum(len(cache._actions(i)) for i in range(cache.cursor)) != observed[role][1]):
                        raise ValueError('complete eager original cache journal/payload/action/input proof differs')
                    if seed == SEEDS[0]:
                        refs[role] = cache
            held_verified = True
            # Budget materialization ONLY: no selector_identity/effect lookup or
            # scientific seal is produced. Existing actual files remain in du.
            future_joint = 0

        after = future_joint + target_future; credit = before - after
        if not 0 <= credit <= before <= work['storage_upper_bytes']:
            raise ValueError('nonnegative future-only complete cache removal required')
        work['storage_components'][_COMPONENT] = after
        work['storage_upper_bytes'] -= credit
        layout['complete_two_seed_compact_cache'] = actual_joint
        phase = 'completed_supply_cache_revalidation_remaining_CPU'
        if phase in work['phases']:
            raise ValueError('future cache verification room already allocated')
        work['phases'][phase] = 3000.
        work['CPU_seconds'] += 3000.; work['remaining_wall_seconds'] += 3000.
        work['completed_supply_future_storage_refinement'] = {
            'geometry_pin': self.spec['predecessors']['observe_geometry']['payload_pin'],
            'original_two_seed_cache_future_bytes': old_joint,
            'actual_geometry_two_seed_cache_upper_bytes': actual_joint,
            'formal_target_bound_unchanged_bytes': formal_targets,
            'component_before_bytes': before, 'component_after_bytes': after,
            'future_storage_obligation_removed_bytes': credit,
            'actual_closed_held_score_readonly_materialization_verified': held_verified,
            'actual_selector_seal_materialized_already': selector_done,
            'actual_formal_inputs_materialized_already': formal_done,
            'currentdu_and_original_other_components_and_phases_unchanged': True,
            'additional_future_cache_verification_CPU_seconds': 3000.,
            'additional_future_cache_verification_wall_seconds': 3000.,
            'selector_identity_or_effect_lookup_written': False,
        }
        # All verification and the prior super call remain inside the original
        # whole guardian/caller meter; the3000 is future engineering room only.
        spent = self.spent(); now = time.time()
        full = forecast(spent, work['GPU_seconds'], work['CPU_seconds'], work['storage_upper_bytes'], now, work['remaining_wall_seconds'], self.owner, closing=False)
        limits = remaining_limits(spent, now, self.owner, potential_GPU_context=self.stage in ['source_FIT', 'source_CONFIRM'], closing=False)
        args = self.spec['arguments']
        return {**original, **full, 'remaining_complete_workload': work, 'already_spent': spent,
                'stage_CPU_limit': min(float(args['stage_CPU_upper_seconds']), limits['remaining_CPU_seconds'] - 10.),
                'stage_wall_limit': min(float(args['stage_wall_upper_seconds']), limits['stage_wall_seconds'] - 10.),
                'evaluated_epoch': now}
