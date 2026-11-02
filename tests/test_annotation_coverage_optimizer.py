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
