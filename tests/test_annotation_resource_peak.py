"""Real meter expressions must preserve peak across atomic rename and repeated snapshots."""
import ast
from pathlib import Path

def test_completed_write_peak_is_monotone_without_recursive_double_charge():
 tree=ast.parse((Path(__file__).resolve().parents[1]/"scripts/meter_annotation_coverage.py").read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"observed_peak","peak"} for t in n.targets)]
 assert len(nodes)==2
 code=compile(ast.Module(body=nodes,type_ignores=[]),"actual_peak_expressions","exec")
 scope={"q":{"own_namespace_bytes":1600,"largest_completed_checkpoint_bytes":400},"previous":{"observed_namespace_peak_bytes":1700,"new_phase_peak_upper_bytes":1800},"c":{}}
 exec(code,scope)
 assert scope["observed_peak"]==1700 and scope["peak"]==10002100
 scope["previous"]={"observed_namespace_peak_bytes":scope["observed_peak"],"new_phase_peak_upper_bytes":scope["peak"]}
 scope["q"]["own_namespace_bytes"]=1500
 exec(code,scope)
 assert scope["observed_peak"]==1700 and scope["peak"]==10002100
