"""Future-only measured cache engineering forecast; scientific runtime is inherited."""
import ast
import copy
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

from echo_routing.temporal.selected_risk_completed_supply_controller import CompletedSupplyRuntimeState

SOURCE='echo_routing/temporal/selected_risk_measured_cache_controller.py'
_OLD8_PINS={'echo_routing/temporal/selected_risk_completed_aux_controller.py': '825b45d92deff842275507dfd81f27f1abcf9015164a2f27191721ea5c11816d', 'echo_routing/temporal/selected_risk_completed_supply_controller.py': '280c8785ca287da14e439ec263ae9e3130219b0c7a3e393658147916f968d25c', 'scripts/a63_prepare_completed_aux_selected_risk.py': 'ed791f20fd1e286f8c7ad773374227abe73c1576cb729e14fdd4a1ad5134979d', 'scripts/a63_prepare_completed_supply_selected_risk.py': 'f2f0f26c7fdbda11e8852abb806976f02e873e6dd4a76fec2661dbae041438d1', 'scripts/a63_resume_completed_aux_selected_risk.py': '7e4d48af52343377a62fa3adc5cdbd3bde4a96a1b8014aec1f5c3cf1a5ed379d', 'scripts/a63_resume_completed_supply_selected_risk.py': '5a6bfef8182d0ec1a335b993eac50a146056a116cf75a85e78ba0cb28049e7bb', 'scripts/a63_wait_completed_aux_selected_risk.py': '0f89201b2bcdc34a7341e440b50ac513c9e09618d45c275ac7a5d321f03ab383', 'scripts/a63_wait_completed_supply_selected_risk.py': 'e9d8e7dfad6bd91152dfbe00e7e88e9c0e34991c5effbb95453eda62332806ac'}
_FORMULA_PINS={'echo_routing/temporal/selected_risk_cost_units.py': 'd28a95223f2b79dcff11c3820e4ffc7b7a7c6133d555f967923ab8318dbe917c', 'echo_routing/temporal/selected_risk_forecast.py': 'c7ae30f98f70326cff2098d73a157e52013d01370e0e62458bea718f55a5b66e', 'echo_routing/temporal/selected_risk_refined_forecast.py': 'c931b90963cba41096e24444eff270f968b7a2f6f988928643e4dfe87009c609', 'echo_routing/temporal/selected_risk_resume_contract.py': '07e106974dc723f6ab2cc3eaa7cf7638d6955f640f8e29a9b31f0578e70119e1', 'echo_routing/temporal/selected_risk_resume_guardian.py': 'e227a057710dcf6d369010070a7e748d76d92be2f30a18e7519a4a8371da24f0'}
_BASE_ROUTE='scripts/a63_resume_completed_supply_selected_risk.py'
_NEW_ROUTE='scripts/a63_resume_measured_cache_selected_risk.py'
_PROBE_SOURCES=('scripts/a63_probe_completed_supply_selected_risk_cost.py',
                'scripts/a63_wait_completed_supply_selected_risk_cost.py')
_EXECUTION_SOURCES=tuple(_OLD8_PINS)+(SOURCE,_NEW_ROUTE,
    'scripts/a63_wait_measured_cache_selected_risk.py',
    'scripts/a63_prepare_measured_cache_selected_risk.py')+_PROBE_SOURCES
_PROFILE_PIN={'path':'docs/results/2026-11-05_selected_action_risk/cost_profile.json',
              'sha256':'797a7455578477318fff6b8663745e35889edea642c68ccf51730564845c7e37'}
_BINDINGS=('probe_spec_pin','source_runtime_spec_pin','source_Git_contract_pin',
           'input_manifest_pin','geometry_pin','core_pins','source_pins')
_PURPOSE='cost_only_synthetic_nonadoptable'
_ROOM='completed_supply_cache_revalidation_remaining_CPU'


def _need(condition,message):
    if not condition:raise ValueError(message)


def _number(value,*,positive=False):
    _need(not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)
          and (value>0 if positive else value>=0),'finite nonnegative/positive measured number required')
    return float(value)


def _one_output(context,pin,key):
    from echo_routing.temporal.selected_risk_resume_context import strict_json
    lines=context.raw(pin).splitlines()
    _need(len(lines)==1,'exact single successful raw stdout JSON line required')
    value=strict_json(lines[0])
    _need(isinstance(value,dict) and set(value)=={'passed',key} and value['passed'] is True,
          'raw successful stdout must expose exact externally pinned artifact')
    return value[key]


def _canonical_relative(value):
    _need(isinstance(value,str) and str(PurePosixPath(value))==value
          and not PurePosixPath(value).is_absolute() and value
          and not any(p in ['', '.', '..'] for p in value.split('/')),
          'canonical owned relative path required')
    return value


def _normalized_route(raw):
    tree=ast.parse(raw)
    tree.body=[n for i,n in enumerate(tree.body)
               if not (i==0 and isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))
               and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXTRA_SOURCES' for t in n.targets))]
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):node.value=node.value.replace('completed_supply','measured_cache')
            return node
        def visit_Name(self,node):
            if node.id=='CompletedSupplyRuntimeState':node.id='MeasuredCacheRuntimeState'
            return node
        def visit_ImportFrom(self,node):
            if node.module:node.module=node.module.replace('completed_supply','measured_cache')
            for name in node.names:
                if name.name=='CompletedSupplyRuntimeState':name.name='MeasuredCacheRuntimeState'
            return node
    return ast.dump(Normalize().visit(tree),include_attributes=False)


def _call_quantity(context,spec,verified,workers,freeze):
    from echo_routing.temporal.selected_risk_resume_contract import stage_graph
    from echo_routing.temporal.selected_risk_resume_estimate import _closed_storage_stage
    base=context.raw({'path':_BASE_ROUTE,'sha256':_OLD8_PINS[_BASE_ROUTE]})
    current=context.raw({'path':_NEW_ROUTE,'sha256':freeze['objects'][_NEW_ROUTE]})
    _need(_normalized_route(base)==_normalized_route(current),'reviewed guardian/admission-child routing AST differs')
    extra=next((ast.literal_eval(n.value) for n in ast.parse(current).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='EXTRA_SOURCES' for t in n.targets)),None)
    _need(extra is not None and len(extra)==14 and set(extra)==set(_EXECUTION_SOURCES),
          'all old8/new4/probe2 execution sources must stay mandatory')
    guardian=context.raw({'path':'echo_routing/temporal/selected_risk_resume_guardian.py',
                          'sha256':_FORMULA_PINS['echo_routing/temporal/selected_risk_resume_guardian.py']})
    execute=next(n for n in ast.parse(guardian).body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    _need(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
              and isinstance(n.func.value,ast.Name) and n.func.value.id=='state'
              and n.func.attr=='admission' for n in ast.walk(execute))==1,
          'one verified guardian admission call per nonclosing stage required')
    graph=stage_graph();_need(len(graph)==33,'original frozen33-stage DAG required')
    done=set()
    for stage in verified:
        if stage not in graph or stage in ['source_FIT','close_negative','close_full']:continue
        _closed_storage_stage(context,spec,verified,workers,stage);done.add(stage)
    if 'source_FIT_adoption' in done:
        # The original failed source remains OS1/charged; adoption disposes its
        # completed payload, never inventing a successful source OS or replay.
        done.add('source_FIT')
    elif 'source_FIT' in verified:
        _closed_storage_stage(context,spec,verified,workers,'source_FIT');done.add('source_FIT')
    remaining=[stage for stage in graph if stage not in done and stage!='close_negative']
    _need(spec['stage'] in remaining,'current scientific stage is already disposed')
    guardians=[stage for stage in remaining if stage!='close_full']
    admissions=[stage for stage in guardians if stage.startswith('admit_')]
    return len(guardians)+len(admissions),guardians,admissions


def _probe(context,spec,workers,freeze,geometry):
    from echo_routing.temporal.selected_risk_resume_runtime import bootstrap
    from echo_routing.temporal.selected_risk_resume_contract import RUN,digest
    from echo_routing.temporal.selected_risk_contracts import source_closure
    from echo_routing.temporal.selected_risk_resume_S import SEEDS
    from echo_routing.temporal.safety_git import verify
    args=spec['arguments'];pp=args.get('measured_cache_probe');parent_pin=args.get('measured_cache_probe_parent')
    _need(pp is not None and parent_pin is not None,'explicit complete probe and actual parent pins required')
    payload=context.read(pp);parent=context.read(parent_pin)
    _need(payload.get('schema')=='A63_full_cache_verifier_cost_probe_v1' and payload.get('passed') is True
          and payload.get('runtime_authorization_sha256')==context.owner_sha,'qualified same-owner full-cache probe required')
    _need(parent.get('schema')=='A63_cost_probe_outer_wait4_receipt_v1'
          and parent.get('runtime_authorization_sha256')==context.owner_sha
          and parent.get('actual_child_OS_exit_code')==0 and type(parent.get('actual_child_OS_exit_code')) is int
          and parent.get('observed_worker_exit_code')==0 and type(parent.get('observed_worker_exit_code')) is int
          and parent.get('stage_exit_status')==0 and type(parent.get('stage_exit_status')) is int
          and parent.get('actual_child_directly_waited') is True and parent.get('whole_child_accounting_complete') is True
          and parent.get('SELF_OS_exit_code') is None and any(row==parent for row in workers),
          'actual successful cost-parent direct wait and canonical full accounting required; SELF null is not OS0')
    observation=parent.get('cuda_context_observation')
    _need(all(parent.get(k) is False for k in ['GPU_job','CUDA_context_occupied','unresolved_worker_uncertainty'])
          and isinstance(observation,dict) and observation.get('known') is True and observation.get('initialized') is False,
          'actual parent must explicitly prove known no-GPU/no-CUDA/no-uncertainty context')
    _need(_one_output(context,parent['successful_stdout_pin'],'payload_pin')==pp and parent.get('payload_pin')==pp,
          'actual cost-parent raw successful stdout must name this exact payload')
    probe_spec=context.read(payload['probe_spec_pin']);contract=context.read(payload['source_Git_contract_pin'])
    _need(probe_spec.get('schema')=='A63_full_cache_verifier_cost_probe_spec_v1'
          and probe_spec.get('purpose')==_PURPOSE and probe_spec.get('runtime_authorization_sha256')==context.owner_sha
          and probe_spec.get('limits')=={'CPU_seconds':600,'wall_seconds':600,'temp_bytes':80000000}
          and all(type(v) is int for v in probe_spec['limits'].values()),'exact independently bounded cost-only600/600/80MB spec required')
    for key in ['probe_spec_pin','source_runtime_spec_pin','source_Git_contract_pin']:
        _need(parent.get(key)==payload[key],'actual parent/probe input pins differ')
    for key in ['source_runtime_spec_pin','source_Git_contract_pin','input_manifest_pin','geometry_pin','core_pins']:
        _need(probe_spec.get(key)==payload[key],'actual probe spec/payload bindings differ')
    _need(payload['input_manifest_pin']==spec['metadata']
          and payload['geometry_pin']==spec['predecessors']['observe_geometry']['payload_pin'],
          'measured input/actual geometry must equal the future workload')
    source_ctx,source_spec=bootstrap(context.root,payload['source_runtime_spec_pin'],context.owner_sha)
    _need(source_ctx.owner_sha==context.owner_sha and source_spec['metadata']==spec['metadata']
          and source_spec['FIT_producer']==spec['FIT_producer'],'original readonly runtime source inputs differ')
    oldfreeze=source_ctx.verify_stage(source_spec['stage'],source_spec['freeze'],required=set(_OLD8_PINS)|set(_FORMULA_PINS))
    _need(contract.get('schema')=='A63_full_cache_verifier_cost_probe_sources_v1'
          and contract.get('runtime_authorization_sha256')==context.owner_sha
          and contract.get('readonly_source_runtime_spec_pin')==payload['source_runtime_spec_pin']
          and contract.get('readonly_source_freeze_pin')==source_spec['freeze']
          and contract.get('probe_spec_semantics_sha256')==digest({k:v for k,v in probe_spec.items() if k!='source_Git_contract_pin'})
          and contract.get('historical_frozen_imports')=={},'separate exact original and new source-proof scopes required')
    old=dict(oldfreeze['objects']);new=contract['objects']
    _need(isinstance(new,dict) and not set(old)&set(new),'new tiny source proof may not replace old frozen objects')
    for path,sha256 in {**old,**new}.items():
        context.raw({'path':path,'sha256':sha256})
        if path.endswith('.py'):
            _need(freeze['objects'].get(path)==sha256,'measured source differs from current execution freeze')
    verify(context.contract_proof(contract,contract['actual_Git_object_proof']),contract['git_commit'],new)
    combined={**old,**new}
    _need(contract.get('combined_source_objects_sha256')==digest(combined)
          and set(source_closure(context.root,_PROBE_SOURCES))<=set(combined),
          'both actual-Git scopes must cover complete current probe import closure')
    sources=payload['source_pins']
    _need(isinstance(sources,list) and all(isinstance(p,dict) and set(p)=={'path','sha256'} for p in sources)
          and len(sources)==len({p['path'] for p in sources})
          and {p['path']:p['sha256'] for p in sources}=={p:h for p,h in combined.items() if p.endswith('.py')},
          'complete exact old/new Python source pins required')
    _need(payload['core_pins']=={k:source_spec['arguments']['core'][k] for k in ['models_manifest','audit','training_inputs']},
          'actual probe must use original independently pinned auxiliary core')
    for pin in [payload['input_manifest_pin'],payload['geometry_pin'],*payload['core_pins'].values()]:context.raw(pin)
    roles=geometry['roles'];pool={'purpose':_PURPOSE,'axes':sum(v['geometry']['N'] for v in roles.values()),
          'admitted_G':sum(v['geometry']['C'] for v in roles.values()),
          'ALL_G':sum(v['geometry'].get('all_C',v['geometry']['C']) for v in roles.values()),
          'seeds':list(SEEDS),'cache_records':8}
    _need((pool['axes'],pool['admitted_G'],pool['ALL_G'])==(5736,8111,8522)
          and payload.get('synthetic_pool')==pool,'complete fixed actual geometry/cost-only cache scope required')
    bindings={key:payload[key] for key in _BINDINGS}
    def flags(value):
        _need(all(type(value.get(k)) is int and value[k]==0 for k in ['NN_calls','SGD_updates','model_instances'])
              and value.get('GT_or_head_effect_used_for_cost') is False
              and isinstance(value.get('metadata_label_parse_exception'),str) and bool(value['metadata_label_parse_exception']),
              'cost-only zero-NN/model/SGD/noGT assertions and parser disclosure required')
    flags(payload)
    def process(entry,prepare):
        raw={k:v for k,v in entry.items() if k!='raw_receipt_pin'}
        _need(context.read(entry['raw_receipt_pin'])==raw,'actual wait4 process record differs')
        _need(all(raw.get(k) is False for k in ['GPU_context_unknown','CUDA_context_observed','unresolved_worker_uncertainty']),
              'each prepare/verifier process must explicitly exclude GPU/CUDA/termination uncertainty')
        _need(type(raw.get('pid')) is int and raw['pid']>0 and type(raw.get('start_ticks')) is int and raw['start_ticks']>0
              and raw.get('boot_id')==parent.get('boot_id')==source_spec['expected_boot_id']==spec['expected_boot_id']
              and type(raw.get('actual_OS_exit_code')) is int and raw['actual_OS_exit_code']==0
              and raw.get('full_accounting_complete') is True and raw.get('actual_directly_waited') is True,
              'fresh full-process CPU and actual directly waited OS0 required')
        cpu=_number(raw['CPU_core_seconds'],positive=True);measured_wall=_number(raw['wall_seconds'],positive=True)
        context.raw(raw['raw_stderr_pin'])
        summary=context.read(_one_output(context,raw['raw_stdout_pin'],'output_pin'))
        _need(summary.get('schema')==('A63_cost_only_cache_prepare_summary_v1' if prepare else 'A63_cost_only_cache_verifier_summary_v1')
              and summary.get('passed') is True and summary.get('complete') is True
              and summary.get('bindings')==bindings and summary.get('synthetic_pool')==pool
              and summary.get('cache_manifest_pin')==payload['cache_manifest_pin'],
              'complete measured output must bind the exact inputs/core/geometry/Git and cache')
        flags(summary)
        if not prepare:
            _need(type(summary.get('verified_payloads')) is int and summary['verified_payloads']==2*pool['axes']
                  and all(summary.get(k) is True for k in ['read_only_cache_payloads_verified','canonical_actions_verified',
                          'reference_raw_features_bitwise_verified','no_pending']),
                  'all axes/payload SHA/action/reference/readonly verifier scope required')
        return cpu,(raw['pid'],raw['start_ticks'],raw['boot_id']),summary,measured_wall
    three=payload['three_fresh_verifier_processes']
    _need(isinstance(three,list) and len(three)==3,'exactly three independent fresh verifier processes required')
    prepare_cpu,prepare_id,prepared,_=process(payload['prepare_process'],True)
    _need(prepared.get('NPZ_STORED') is True,'prepare must verify every actual stored NPZ format')
    measured=[process(row,False) for row in three]
    _need(len({r[1] for r in measured})==3 and prepare_id not in {r[1] for r in measured},'fresh process identities must be distinct')
    _need(parent.get('prepare_process_raw_receipt_pin')==payload['prepare_process']['raw_receipt_pin']
          and parent.get('three_fresh_verifier_raw_receipt_pins')==[r['raw_receipt_pin'] for r in three],
          'outer actual accounting must cover the exact four measured child processes')
    total_cpu=_number(parent['measured_CPU_core_seconds'],positive=True)
    wall=_number(parent['full_process_wall_seconds'],positive=True)
    _need(total_cpu+_number(parent['caller_exit_CPU_allowance_seconds'])<=600
          and total_cpu+1e-6>=prepare_cpu+sum(r[0] for r in measured) and wall<=600,
          'actual cost attempt CPU/wall caps or nested accounting differ')
    _need(type(parent.get('sampled_temp_peak_bytes')) is int and 0<=parent['sampled_temp_peak_bytes']<=80000000,
          'actual sampled temporary scope must fit independent cost-only allowance')
    prefix=_canonical_relative(probe_spec['output_prefix']);owned=prefix+'/payloads'
    _need(prefix.startswith(RUN+'/cost_only_synthetic_cache_probe/')
          and '/' not in prefix[len(RUN+'/cost_only_synthetic_cache_probe/'):]
          and probe_spec.get('nonce')==PurePosixPath(prefix).name,'dedicated owned cost-only namespace required')
    cleanup=payload['cleanup'];deleted_pin=cleanup['deleted_file_manifest_pin']
    _need(cleanup.get('owned_payload_root')==owned and cleanup.get('namespace')==_PURPOSE
          and cleanup.get('fully_removed') is True and cleanup.get('owned_only') is True
          and type(cleanup.get('after_regular_bytes')) is int and cleanup['after_regular_bytes']==0
          and not (context.root/owned).exists() and not (context.root/owned).is_symlink(),
          'owned-only physical payload cleanup must be complete, never a science cache')
    files=context.read(deleted_pin)
    _need(isinstance(files,list) and len(files)>0,'original deleted-file manifest required')
    seen=set();byte_count=0;npz=0
    for row in files:
        _need(isinstance(row,dict) and set(row)=={'path','sha256','bytes'},'exact deleted-file metadata schema required')
        path=_canonical_relative(row['path'])
        _need(path not in seen and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None
              and type(row['bytes']) is int and row['bytes']>=0
              and PurePosixPath(path).name!='pending.json' and not path.endswith('.tmp'),'unambiguous owned original regular-file inventory required')
        seen.add(path);byte_count+=row['bytes'];npz+=int(path.endswith('.npz'))
    _need(npz==2*pool['axes'] and byte_count==cleanup.get('before_regular_bytes')==prepared.get('payload_regular_bytes')
          and prepared.get('deleted_file_manifest_pin')==deleted_pin
          and type(prepared.get('structural_peak_upper_bytes')) is int
          and byte_count<=prepared['structural_peak_upper_bytes']<=80000000
          and byte_count<=parent['sampled_temp_peak_bytes'],
          'all original cache payloads/bytes/atomic bounds and deletion witnesses must reconcile')
    manifest=context.read(payload['cache_manifest_pin']);records=manifest['records']
    _need(manifest.get('schema')=='A63_cost_only_cache_manifest_v1' and manifest.get('purpose')==_PURPOSE
          and len(records)==8 and {(r['seed'],r['role']) for r in records}=={(s,r) for s in SEEDS for r in roles},
          'retained metadata must cover exactly eight synthetic caches')
    for row in records:
        _need(row.get('axes')==roles[row['role']]['geometry']['N'] and row.get('admitted_G')==roles[row['role']]['geometry']['C']
              and row.get('relative_path')==f'seed{row["seed"]}/{row["role"]}'
              and _canonical_relative(owned+'/'+row['relative_path']).startswith(owned+'/')
              and row.get('feature_reference_seed')==(None if row['seed']==SEEDS[0] else SEEDS[0]),
              'all role axes/candidates and original first-seed reference scope required')
    return max(r[0] for r in measured),max(r[3] for r in measured),[r[0] for r in measured],[r[3] for r in measured],pp,parent_pin


class MeasuredCacheRuntimeState(CompletedSupplyRuntimeState):
    def admission(self):
        original=super().admission()
        freeze,evidence=self.ctx.enter(self.stage,self.spec['freeze'],self.spec['predecessors'],
                                       required=set(_EXECUTION_SOURCES)|set(_FORMULA_PINS))
        for path,sha256 in {**_OLD8_PINS,**_FORMULA_PINS}.items():
            _need(freeze['objects'].get(path)==sha256,'original source/formula/routing changed')
            self.ctx.raw({'path':path,'sha256':sha256})
        if self.stage in ['close_negative','close_full']:return original
        from echo_routing.temporal.selected_risk_resume_contract import digest
        from echo_routing.temporal.selected_risk_resume_estimate import _closed_storage_stage
        from echo_routing.temporal.selected_risk_cost_units import project_coefficients
        from echo_routing.temporal.selected_risk_forecast import add,backend,TERMS
        from echo_routing.temporal.selected_risk_resume_budget import forecast,remaining_limits
        verified={key:value['payload'] for key,value in evidence.items()};workers=self._worker_rows()
        _need({'observe_geometry','prepare_aux','train_aux','audit_aux'}<=set(verified),'complete original auxiliary prefix required')
        _closed_storage_stage(self.ctx,self.spec,verified,workers,'observe_geometry')
        geometry=verified['observe_geometry']
        _need(geometry.get('complete') is True and geometry.get('input_manifest_sha256')==self.spec['metadata']['sha256']
              and geometry.get('producer_FIT_payload_sha256')==self.spec['FIT_producer']['sha256']
              and set(geometry['roles'])=={'F0_native','F1_native','F0_mixed','F1_mixed'},'exact actual four-role FIT geometry required')
        worst,worst_wall,raw_CPUs,raw_walls,probe_pin,parent_pin=_probe(self.ctx,self.spec,workers,freeze,geometry)
        count,guardians,children=_call_quantity(self.ctx,self.spec,verified,workers,freeze)
        work=copy.deepcopy(original['remaining_complete_workload']);phases=work['phases']
        profile=self.ctx.read({'path':self.ctx.closed_publication_path(_PROFILE_PIN['path']),'sha256':_PROFILE_PIN['sha256']})
        coef,proof=project_coefficients(profile)
        _need(work['actual_four_profile_projection_proof']['new_coefficients']==coef,'original four-profile coefficient assignment changed')
        oldfit=add([profile['geometry']['native'],*[work['projected_geometry'][r] for r in ['F0_mixed','F1_mixed']]])
        actual={term:sum(_number(row['geometry'].get('all_'+term,row['geometry'].get(term,0))) for row in geometry['roles'].values()) for term in TERMS}
        _need(all(term in row['geometry'] or 'all_'+term in row['geometry'] for row in geometry['roles'].values() for term in TERMS),
              'all nine actual whole/pre-G geometry terms required')
        old_backend=backend(oldfit,coef);actual_backend=backend(actual,coef);changes={};delta=0.
        for key,multiple,done in [('complement_prepare_and_independent_recount',4,None),
                                  ('D_X_held_score_all_features_2seeds_and_cache_IO',8,'seal_selectors'),
                                  ('selected_effects_signatures_coverage_recount',4,'coverage')]:
            before=_number(phases[key]);expected=multiple*old_backend
            if before==0 and done is not None:
                _need(done in verified,'zeroed future phase lacks completed-stage evidence')
                _closed_storage_stage(self.ctx,self.spec,verified,workers,done);after=0.
            else:
                _need(math.isclose(before,expected,rel_tol=1e-12,abs_tol=1e-8),'original projected FIT phase does not reconcile: '+key)
                after=multiple*actual_backend
            phases[key]=after;delta+=after-before;changes[key]={'before_seconds':before,'after_seconds':after,'same_passes':multiple}
        _need(phases.get(_ROOM)==3000.,'original new cache engineering room must be exactly3000 before replacement')
        measured_room=2*worst*count+60. # No clipping, even if the actual measurement is slower.
        measured_wall_room=2*max(worst,worst_wall)*count+60.
        phases[_ROOM]=measured_room
        work['CPU_seconds']+=delta+measured_room-3000.
        work['remaining_wall_seconds']+=delta+measured_wall_room-3000.
        # New explicit scope: retire only the32 already current auxiliary states.
        # The original formula's250000/unit,16formal and2atomic units do not fall.
        proofpin=self.spec['arguments'].get('completed_aux_storage_proof')
        _need(proofpin is not None and work['completed_aux_future_trace_refinement'].get('proof_pin')==proofpin,
              'original825 independently verified auxiliary storage witness required')
        aux=self.ctx.read(proofpin);train=verified['train_aux'];model_rows=train['models']
        _closed_storage_stage(self.ctx,self.spec,verified,workers,'train_aux')
        _need(aux.get('complete_train_manifest')==self.spec['predecessors']['train_aux']['payload_pin']
              and aux.get('passed') is True and (aux.get('physical_updates'),aux.get('fullstate_count'))==(48000,32)
              and len(aux['records'])==8 and len(model_rows)==8,'original actual complete32state auxiliary descriptor required')
        state_paths=set()
        for row in aux['records']:
            matches=[m for m in model_rows if (m['seed'],m['held_fold'],m['role'])==(row['seed'],row['held_fold'],row['role'])]
            _need(len(matches)==1 and matches[0]['completion']==row['completion'],'original train completion pin differs')
            completion=self.ctx.read(row['completion']);pins=completion['state_pins']
            _need(completion.get('identity')==row['identity'] and completion.get('passed') is True
                  and (completion.get('physical_updates'),completion.get('fullstate_count'))==(6000,4)
                  and len(pins)==4 and len(row['fullstate_pins_and_bytes'])==4,'original complete per-head state identity differs')
            def relative(pin):
                path=Path(pin['path']);path=path.relative_to(self.ctx.root) if path.is_absolute() else path
                return {'path':path.as_posix(),'sha256':pin['sha256']}
            originals={p['path']:p['sha256'] for p in map(relative,pins)}
            _need(len(originals)==4,'distinct original state pins required')
            for record in row['fullstate_pins_and_bytes']:
                pin=relative(record['pin']);_need(originals.get(pin['path'])==pin['sha256'] and pin['path'] not in state_paths,
                                                 'all and only original32distinct fullstate pins required')
                raw=self.ctx.raw(pin);_need(type(record['bytes']) is int and len(raw)==record['bytes'],'actual state size differs')
                state_paths.add(pin['path'])
        _need(len(state_paths)==32,'all32actual fullstate pins required')
        statekey='retained_fullstates_atomic_upper';_need(work['storage_components'][statekey]==48*250000+2*250000,'original48state+2atomic formula changed')
        work['storage_components'][statekey]-=32*250000;work['storage_upper_bytes']-=32*250000
        work['measured_cache_future_cost_refinement']={
            'probe_pin':probe_pin,'actual_parent_pin':parent_pin,'worst_full_verifier_CPU_seconds':worst,
            'worst_full_verifier_wall_seconds':worst_wall,
            'raw_three_verifier_CPU_seconds':raw_CPUs,'raw_three_verifier_wall_seconds':raw_walls,
            'remaining_guardian_calls':len(guardians),'remaining_admission_child_calls':len(children),
            'remaining_full_verifier_calls':count,'quantity_source':'frozen33DAG + actualclosedprefix + normalized full routing AST + one guardian admission call',
            'unchanged_uncertainty_factor':2,'startup_room_seconds':60.,'measured_future_CPU_room_seconds':measured_room,
            'measured_future_wall_room_seconds':measured_wall_room,
            'original_cache_CPU_room_seconds':3000.,'original_cache_wall_room_seconds':3000.,
            'same_FIT_phase_CPU_and_wall_delta_seconds':delta,
            'cache_CPU_room_delta_seconds':measured_room-3000.,
            'cache_wall_room_delta_seconds':measured_wall_room-3000.,
            'complete_remaining_CPU_delta_seconds':delta+measured_room-3000.,
            'complete_remaining_wall_delta_seconds':delta+measured_wall_room-3000.,
            'original_projected_FIT_backend_seconds':old_backend,'actual_ALL_FIT_backend_seconds':actual_backend,
            'same_coefficients_and_passes_phase_changes':changes,'future_aux_state_units_retired':32,
            'unchanged_state_unit_bytes':250000,'unchanged_future_formal_states':16,'unchanged_atomic_state_units':2,
            'currentdu_actual_state_bytes_not_subtracted':True,'unknown_CONFIRM_outer_floors_metadata300_and_reserves_unchanged':True,
            'measurement_is_engineering_forecast_not_actual_spend_or_guarantee':True}
        spent=self.spent();now=time.time()
        full=forecast(spent,work['GPU_seconds'],work['CPU_seconds'],work['storage_upper_bytes'],now,work['remaining_wall_seconds'],self.owner,closing=False)
        limits=remaining_limits(spent,now,self.owner,potential_GPU_context=self.stage in ['source_FIT','source_CONFIRM'],closing=False)
        args=self.spec['arguments']
        return {**original,**full,'remaining_complete_workload':work,'already_spent':spent,
                'stage_CPU_limit':min(float(args['stage_CPU_upper_seconds']),limits['remaining_CPU_seconds']-10.),
                'stage_wall_limit':min(float(args['stage_wall_upper_seconds']),limits['stage_wall_seconds']-10.),'evaluated_epoch':now}
