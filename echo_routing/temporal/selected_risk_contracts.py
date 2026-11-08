"""A63 mandatory stages and exact local source import closure."""
import ast
from pathlib import Path
PUB='docs/results/2026-11-05_selected_action_risk/'
SPECS={'sources_admission': (['admit_selected_risk'], ['input_manifest.json','cost_profile.json']), 'confirm_admission': (['admit_selected_risk'], ['input_manifest.json','cost_profile.json','coverage.json','aux_model_audit.json']), 'formal_admission': (['admit_selected_risk'], ['input_manifest.json','cost_profile.json','coverage.json','training_inputs.json','aux_model_audit.json']), 'outer_admission': (['admit_selected_risk'], ['input_manifest.json','cost_profile.json','coverage.json','models_manifest.json','calibration.json']), 'cost_GPU_probe': (['probe_selected_risk_source'], ['input_manifest.json','GPU_probe_admission.json']), 'metadata': (['prepare_selected_risk'], []), 'cost_probe': (['probe_selected_risk'], ['input_manifest.json']), 'source_FIT': (['cache_selected_risk'], ['input_manifest.json', 'sources_admission.json', 'cost_profile.json']), 'prepare_aux': (['prepare_selected_risk_aux'], ['input_manifest.json', 'producer_FIT_manifest.json']), 'aux_train': (['train_selected_risk_aux'], ['aux_training_inputs.json', 'sources_admission.json']), 'aux_audit': (['audit_selected_risk'], ['aux_training_inputs.json', 'aux_completion.json', 'aux_train_identity.json']), 'held_score': (['score_selected_risk_supply'], ['aux_model_audit.json', 'aux_models_manifest.json', 'aux_identity.json', 'aux_training_inputs.json']), 'coverage': (['coverage_selected_risk'], ['selected_scores_manifest.json', 'aux_models_manifest.json', 'input_manifest.json']), 'coverage_terminal': (['close_selected_risk', 'wait_selected_risk_collector'], ['coverage.json', 'aux_model_audit.json', 'aux_models_manifest.json', 'input_manifest.json', 'independent_scientific_review.json']), 'source_CONFIRM': (['cache_selected_risk'], ['coverage.json', 'input_manifest.json', 'confirm_admission.json', 'aux_model_audit.json', 'aux_models_manifest.json', 'aux_training_inputs.json', 'aux_identity.json']), 'formal_prepare': (['prepare_selected_risk_formal'], ['coverage.json', 'selected_scores_manifest.json', 'input_manifest.json']), 'formal': (['train_selected_risk'], ['coverage.json', 'training_inputs.json', 'formal_admission.json']), 'model_audit': (['audit_selected_risk'], ['formal_completion.json', 'training_inputs.json', 'formal_identity.json']), 'calibration': (['calibrate_selected_risk'], ['model_audit.json', 'training_inputs.json', 'input_manifest.json']), 'final_audit': (['audit_selected_risk'], ['model_audit.json', 'calibration.json', 'inner_metrics.json', 'readout_plan.json']), 'outer': (['readout_selected_risk'], ['models_manifest.json', 'calibration.json', 'readout_plan.json', 'outer_identity.json', 'outer_admission.json', 'producer_CONFIRM_manifest.json']), 'aggregate': (['aggregate_selected_risk'], ['outer_metrics.json', 'readout_plan.json', 'models_manifest.json', 'calibration.json']), 'support': (['support_selected_risk'], ['summary.json', 'outer_metrics.json', 'models_manifest.json', 'calibration.json', 'readout_plan.json']), 'forensics': (['forensics_selected_risk'], ['summary.json', 'outer_metrics.json', 'models_manifest.json', 'calibration.json', 'readout_plan.json']), 'package': (['package_selected_risk'], ['summary.json', 'outer_metrics.json', 'inner_metrics.json', 'models_manifest.json', 'calibration.json', 'support_capacity.json', 'native_attribution.json', 'regression_summary.json', 'model_audit.json', 'acceptance.json']), 'full_terminal': (['close_selected_risk', 'wait_selected_risk_collector'], ['summary.json', 'outer_metrics.json', 'models_manifest.json', 'calibration.json', 'readout_plan.json', 'support_capacity.json', 'native_attribution.json', 'regression_summary.json', 'package_receipt.json', 'FINDINGS.md', 'README.md', 'inference_costs.json', 'figure_receipt.json', 'independent_scientific_review.json', 'root_figure_visual_QA.json']), 'incomplete_terminal': (['close_selected_risk', 'wait_selected_risk_collector'], ['local_cpu_receipts.json'])}

def stage_spec(stage):
    if stage not in SPECS:raise ValueError('unrecognized A63 stage; no arbitrary/empty source freeze')
    entries,inputs=SPECS[stage]
    return ['scripts/'+p+'.py' for p in entries],[PUB+p for p in inputs]

def source_closure(root,paths):
    ROOT=Path(root)
    from importlib.util import resolve_name
    pending=list(paths);seen=set()
    def module(name):
        if not name.startswith(('echo_routing','scripts')):return
        if name.startswith('scripts.') and 'selected_risk' not in name.split('.')[-1]:raise ValueError('old writing CLI import is forbidden: '+name)
        parts=Path(*name.split('.'));candidates=[parts.with_suffix('.py'),parts/'__init__.py']
        for q in candidates:
            if (ROOT/q).is_file():pending.append(str(q))
        for parent in [parts,*parts.parents]:
            init=parent/'__init__.py'
            if str(parent)!='.' and (ROOT/init).is_file():pending.append(str(init))
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path);p=ROOT/path
        if p.suffix!='.py':continue
        package='.'.join(Path(path).parent.parts)
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.Import):
                for alias in node.names:module(alias.name)
            elif isinstance(node,ast.ImportFrom):
                base=resolve_name('.'*node.level+(node.module or ''),package) if node.level else node.module
                if not base:continue
                module(base)
                for alias in node.names:
                    if alias.name!='*':
                        name=base+'.'+alias.name;candidate=Path(*name.split('.'))
                        if (ROOT/candidate.with_suffix('.py')).is_file() or (ROOT/candidate/'__init__.py').is_file():module(name)
    return sorted(seen)



def validate_coverage(root,stage,objects,base_paths):
    entries,inputs=stage_spec(stage);required=set(source_closure(root,entries+inputs+list(base_paths)))
    if not required<=set(objects):raise ValueError('mandatory stage source/input/import coverage absent: '+str(sorted(required-set(objects))))
    return True
