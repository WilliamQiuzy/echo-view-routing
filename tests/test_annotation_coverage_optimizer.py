"""Integration guard for inherited optimizer persistence schema and final schedule."""
import math
import torch
from scripts import train_session_correction as t31

def test_inherited_optimizer_group_schema_and_final_schedule():
 names = ["encoder_trunk", "marker_stem", "projection", "scalar_head"]
 rates = [1e-5, 1e-4, 1e-4, 1e-3]
 params = [torch.nn.Parameter(torch.ones(1)) for _ in names]
 opt = t31.opt_of(list(zip(names, [[p] for p in params], rates)))
 t31.sched(opt, 3599, {"updates": 3600, "warmup": 360})
 state = opt.state_dict()
 for group, name, rate in zip(state["param_groups"], names, rates):
  assert group["name"] == name and group["base"] == rate
  expected = rate * .5 * (1 + math.cos(math.pi * (3599 - 360) / (3600 - 360)))
  assert abs(group["lr"] - expected) < 1e-16
  assert group["weight_decay"] == 1e-4 and group["betas"] == (.9, .999) and group["eps"] == 1e-8


def test_code_pins_exclude_synthetic_torch_module_filenames():
 # Execute the actual dependency collector without optional video-decoder imports.
 import ast, hashlib, sys, types
 from pathlib import Path
 root = Path(__file__).resolve().parents[1]
 tree = ast.parse((root / "scripts/train_annotation_coverage.py").read_text())
 fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pins")
 scope = {"Path": Path, "ROOT": root, "sys": sys,
          "CARD": "docs/superpowers/plans/2026-11-01-five-hour-evidence-iteration.md",
          "ADDENDUM": "docs/superpowers/plans/2026-11-02-a55-source-package-clarification.md",
          "sha": lambda p: hashlib.sha256(p.read_bytes()).hexdigest()}
 exec(compile(ast.Module(body=[fn], type_ignores=[]), "actual_source_pins", "exec"), scope)
 fake = types.ModuleType("a55_synthetic_schema_test")
 fake.__file__ = str(root / "_classes.py")
 sys.modules[fake.__name__] = fake
 try:
  result = scope["pins"]()
  assert result and all((root / p).is_file() for p in result)
  assert "_classes.py" not in result
 finally:
  del sys.modules[fake.__name__]
