import json,hashlib,importlib
from pathlib import Path


def test_physical_production_attempts_are_separate_from_readonly_adoption():
    from echo_routing.temporal.safety_package import source_costs
    record={'source':'a08','role':'fresh','complete':True,'committed_sequences':1,'new_neural_units':3};prod={'stage':'producer_precache','attempt':1,'receipt_id':'producer_precache:1','GPU_job':True,'CUDA_context_occupied':True,'full_process_wall_seconds':10.,'measured_CPU_core_seconds':8.,'observed_worker_exit_code':0,'child_exit_code_directly_waited':True};adopt={**prod,'stage':'cache_outer','receipt_id':'cache_outer:1','GPU_job':False,'CUDA_context_occupied':False,'full_process_wall_seconds':3.,'measured_CPU_core_seconds':2.}
    value=source_costs([record],{('a08','fresh'):[3]},[prod,adopt],production_stage='producer_precache')
    assert value['source_cache_GPU_jobwall_seconds']==10 and value['source_cache_CPU_seconds']==8
    assert value['read_only_adoption_CPU_seconds']==2 and value['read_only_adoption_new_source_forward_calls']==0
    assert value['actual_total_source_model_forward_calls']['a08']==1


def test_complete_source_binding_uses_adopted_immutable_path(tmp_path,monkeypatch):
    m=importlib.import_module('scripts.readout_action_safety');monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'RUN',tmp_path/'runs/a61_action_safety_ranking_v1');pub=tmp_path/'pub';pub.mkdir();monkeypatch.setattr(m,'PUB',pub);inputs={'roles':{},'source_cache_roles':{}};banks={};records=[]
    for table in m.TABLES:
        for half in m.HALVES:
            role=m.role_for(table,half);inputs['roles'][role]={'sha256':role};ids={}
            for source in ['a08','jepa']:
                fresh=table=='a61_confirmation';path=('runs/a61_action_safety_ranking_v1/producer_precache_v1' if fresh else 'closed_readonly')+'/'+source+'/'+role;folder=tmp_path/path;folder.mkdir(parents=True);raw=(source+role).encode();(folder/'committed.jsonl').write_bytes(raw);row={'source':source,'role':role,'path':path,'identity':source+role,'input_sha256':role,'journal_sha256':hashlib.sha256(raw).hexdigest(),'complete':True,'committed_sequences':1};ids[source]=row['identity']
                if fresh:records.append(row)
                else:inputs['source_cache_roles'].setdefault(role,{})[source]=row
            banks[role]=([{}],None,None,None,ids)
    (pub/'input_manifest.json').write_text(json.dumps(inputs));m.source_banks({'records':records,'read_only_early_producer_cache_adopted':True},banks)
