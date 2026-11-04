"""Exact stage entry/input coverage and local AST import closure; no execution."""
import ast,re
from pathlib import Path
PUB='docs/results/2026-11-04_anchor_context_risk/'
CONFIG='configs/experiments/anchor_context_risk_v1.json'
SPECS={
 'incomplete_terminal':(['close_event_risk','wait_event_risk_collector'],['local_cpu_receipts.json']),
 'metadata':(['prepare_event_risk'],['prior_science_binding.json']),
 'cost':(['probe_event_risk_cost'],['input_manifest.json']),
 'producer':(['cache_event_risk'],['input_manifest.json','sources_admission.json','cost_profile.json']),
 'FIT_prepare':(['prepare_event_risk'],['input_manifest.json','producer_manifest.json']),
 'FIT_check':(['check_event_risk'],['training_inputs.json']),
 'formal':(['train_event_risk'],['training_inputs.json','formal_admission.json','FIT_checks.json']),
 'model_audit':(['audit_event_risk'],['formal_completion.json','training_inputs.json','formal_identity.json']),
 'calibration':(['calibrate_event_risk'],['model_audit.json','training_inputs.json','input_manifest.json']),
 'final_audit':(['audit_event_risk'],['model_audit.json','calibration.json','inner_metrics.json','readout_plan.json']),
 'outer':(['cache_event_risk'],['models_manifest.json','calibration.json','readout_plan.json','outer_identity.json','outer_admission.json','producer_manifest.json']),
 'readout':(['readout_event_risk'],['models_manifest.json','calibration.json','readout_plan.json','outer_identity.json','source_adoption.json','input_manifest.json']),
 'aggregate':(['aggregate_event_risk'],['outer_metrics.json','readout_plan.json','models_manifest.json','calibration.json']),
 'oracle':(['oracle_event_risk'],['summary.json','outer_metrics.json','models_manifest.json','calibration.json','readout_plan.json']),
 'forensics':(['forensics_event_risk'],['prior_context_binding.json','summary.json','outer_metrics.json','models_manifest.json','calibration.json','readout_plan.json']),
 'package':(['package_event_risk'],['summary.json','outer_metrics.json','inner_metrics.json','models_manifest.json','calibration.json','action_space_diagnostic.json','native_attribution.json','regression_summary.json','model_audit.json','acceptance.json']),
 'terminal':(['close_event_risk','wait_event_risk_collector'],['summary.json','outer_metrics.json','models_manifest.json','calibration.json','readout_plan.json','action_space_diagnostic.json','native_attribution.json','regression_summary.json','package_receipt.json','FINDINGS.md','README.md','inference_costs.json','figure_receipt.json','independent_scientific_review.json','root_figure_visual_QA.json']),
}


def stage_spec(name):
 if re.fullmatch(r'(sources|formal|outer)_admission_v[1-9][0-9]*_freeze\.json',name):
  stage=name.split('_')[0];inputs=['input_manifest.json','cost_profile.json']
  if stage!='sources':inputs+=['training_inputs.json','FIT_checks.json','producer_manifest.json']
  if stage=='outer':inputs+=['models_manifest.json','calibration.json']
  return ['scripts/admit_event_risk.py'],[PUB+p for p in inputs]
 if not name.endswith('_freeze.json') or name[:-12] not in SPECS:raise ValueError('unrecognized authorized stage contract')
 entries,inputs=SPECS[name[:-12]]
 return ['scripts/'+p+'.py' for p in entries],[PUB+p for p in inputs]


def source_closure(root,paths):
    ROOT=Path(root)
    from importlib.util import resolve_name
    pending=list(paths);seen=set()
    def module(name):
        if not name.startswith(('echo_routing','scripts')):return
        if name.startswith('scripts.') and 'event_risk' not in name.split('.')[-1]:raise ValueError('old writing CLI import is forbidden: '+name)
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



def validate_coverage(root,name,objects,base_paths):
 entries,inputs=stage_spec(name);required=set(source_closure(root,entries+inputs+list(base_paths)))
 if not required<=set(objects):raise ValueError('incomplete stage source/input/import closure: '+str(sorted(required-set(objects))))
 return True
