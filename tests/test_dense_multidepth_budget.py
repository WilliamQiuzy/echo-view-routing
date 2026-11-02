import importlib

def test_component_storage_counts_joint_full_frozen_head_and_one_atomic_coexistence():
 a=importlib.import_module('echo_routing.temporal.multidepth_budget');m={'full_model_bytes':1000,'head_only_bytes':10,'atomic_full_checkpoint_bytes':3000,'frozen_latest_optimizer_bytes':1200};c=a.storage_components(2,m)
 assert c['joint_full_milestones']==24000 and c['frozen_head_only_milestones']==240 and c['all_joint_latest_optimizers']==6000 and c['all_frozen_latest_optimizers']==2400 and c['one_atomic_optimizer_coexistence']==3000
 assert c['new_source_weight_copy']==0
 one=a.storage_components(1,m)
 assert one['joint_full_milestones']==12000 and one['one_atomic_optimizer_coexistence']==3000

def test_tier_selection_uses_cost_only_and_preserves_closure_reserve():
 a=importlib.import_module('echo_routing.temporal.multidepth_budget')
 assert a.select_tier([{ 'seed_count':2,'GPU_total_including_reserve':23,'CPU_total':40,'peak_bytes':32e9,'wall_hours':36},{'seed_count':1,'GPU_total_including_reserve':18,'CPU_total':30,'peak_bytes':28e9,'wall_hours':30}])==2
 assert a.select_tier([{'seed_count':2,'GPU_total_including_reserve':25,'CPU_total':40,'peak_bytes':32e9,'wall_hours':36},{'seed_count':1,'GPU_total_including_reserve':18,'CPU_total':30,'peak_bytes':28e9,'wall_hours':30}])==1
 assert a.select_tier([{'seed_count':1,'GPU_total_including_reserve':18,'CPU_total':73,'peak_bytes':28e9,'wall_hours':30}])==0
