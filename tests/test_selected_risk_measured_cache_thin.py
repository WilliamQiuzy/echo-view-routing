"""One admin-only AST/default-dry fixture. No real freeze/queue/guardian execution."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASES={
 'prepare':'f2f0f26c7fdbda11e8852abb806976f02e873e6dd4a76fec2661dbae041438d1',
 'resume':'5a6bfef8182d0ec1a335b993eac50a146056a116cf75a85e78ba0cb28049e7bb',
 'wait':'e9d8e7dfad6bd91152dfbe00e7e88e9c0e34991c5effbb95453eda62332806ac',
}
EXTRA=(
 'echo_routing/temporal/selected_risk_completed_aux_controller.py',
 'scripts/a63_resume_completed_aux_selected_risk.py',
 'scripts/a63_wait_completed_aux_selected_risk.py',
 'scripts/a63_prepare_completed_aux_selected_risk.py',
 'echo_routing/temporal/selected_risk_completed_supply_controller.py',
 'scripts/a63_resume_completed_supply_selected_risk.py',
 'scripts/a63_wait_completed_supply_selected_risk.py',
 'scripts/a63_prepare_completed_supply_selected_risk.py',
 'echo_routing/temporal/selected_risk_measured_cache_controller.py',
 'scripts/a63_resume_measured_cache_selected_risk.py',
 'scripts/a63_wait_measured_cache_selected_risk.py',
 'scripts/a63_prepare_measured_cache_selected_risk.py',
 'scripts/a63_probe_completed_supply_selected_risk_cost.py',
 'scripts/a63_wait_completed_supply_selected_risk_cost.py',
)


def normalized(raw):
    tree=ast.parse(raw)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXTRA_SOURCES' for t in n.targets))]
    class Rename(ast.NodeTransformer):
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
    return ast.dump(Rename().visit(tree),include_attributes=False)


def test_three_thins_only_standard_rename_extra14_and_default_dry(tmp_path):
    root=Path(__file__).resolve().parents[1]
    qualified=Path(os.environ.get('A63_THIN_BASELINE_ROOT',root))
    dryroot=tmp_path/'isolated_default_dry';scripts=dryroot/'scripts';scripts.mkdir(parents=True)
    for kind,sha in BASES.items():
        before=(qualified/f'scripts/a63_{kind}_completed_supply_selected_risk.py').read_bytes()
        after=(root/f'scripts/a63_{kind}_measured_cache_selected_risk.py').read_bytes()
        assert hashlib.sha256(before).hexdigest()==sha
        assert normalized(before)==normalized(after)
        (scripts/f'a63_{kind}_measured_cache_selected_risk.py').write_bytes(after)
    entry=ast.parse((scripts/'a63_resume_measured_cache_selected_risk.py').read_bytes())
    extras=next(ast.literal_eval(n.value) for n in entry.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXTRA_SOURCES' for t in n.targets))
    assert extras==EXTRA and len(extras)==len(set(extras))==14
    env={**os.environ,'PYTHONPATH':str(dryroot)+os.pathsep+str(qualified),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    before={p.relative_to(dryroot):p.read_bytes() for p in dryroot.rglob('*') if p.is_file()}
    for kind in BASES:
        cmd=[sys.executable,str(scripts/f'a63_{kind}_measured_cache_selected_risk.py')]
        if kind!='prepare':cmd+=['--stage','seal_aux']
        result=subprocess.run(cmd,capture_output=True,text=True,env=env,cwd=dryroot,check=True)
        assert result.returncode==0
        value=json.loads(result.stdout);assert value['preparation_only'] is True
        if kind=='prepare':assert value['new_process_or_queue_or_model'] is False and value['extra_source_paths']==list(EXTRA)
        else:assert value['execution_authorized'] is False and value['new_child_created'] is False
    assert {p.relative_to(dryroot):p.read_bytes() for p in dryroot.rglob('*') if p.is_file()}==before
    assert not (dryroot/'runs').exists() and not (dryroot/'docs').exists()
