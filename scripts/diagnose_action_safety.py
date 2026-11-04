#!/usr/bin/env python3
"""A61 main-native56 cached full-axis attribution; no new model forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scripts.safety_common import *
from scripts.readout_action_safety import bank,current_cache,historical_cache
from echo_routing.temporal.safety_native import diagnose
from echo_routing.temporal.safety_readout_plan import plan as fixed_plan,HALVES,BASE


def main():
    freeze=verify_freeze('native_freeze.json');models=read(PUB/'models_manifest.json');out=read(PUB/'outer_metrics.json');summary=read(PUB/'summary.json');inputs=read(PUB/'training_inputs.json');plan=read(PUB/'readout_plan.json')
    if plan!=fixed_plan() or not summary['all314_metrics_independently_recounted'] or summary['outer_metrics_sha256']!=sha(PUB/'outer_metrics.json'):raise ValueError('complete frozen314 independently recounted rows required')
    predictions=np.load(verify_pin(out['private_predictions']),allow_pickle=True).item();rows={(r['half'],r['system']):r for r in out['rows'] if r['table']=='main'};banks={h:bank(h) for h in HALVES};risk={r['seed']:r for r in models['models'] if r['role']=='risk'};utility={(r['seed'],r['arm']):r for r in models['models'] if r['role']=='utility'};taus={(r['seed'],r['arm']):r['tau'] for r in models['thresholds']};oldmodels=read(verify_pin(read(CFG)['prior_A60']['models_manifest.json']));legacy={r['seed']:r for r in oldmodels['models'] if r['arm']=='J'};cells=[];cache_values={}
    write(RUN/'runtimecoordination.json',{'phase':'cached56_native_gross_gate_attribution','model_pid':os.getpid(),'new_model_forwards':0,'epoch':time.time(),'next_action':'finish56_maincubes_thenfigures_actualresources_exit_publicpackage'})
    for item in plan['native_cells']:
        half,name=item['half'],item['system'];q,a,j,b,_=banks[half];old=item['role']=='A60_historical';model=legacy[item['old_seed']] if old else utility[(item['seed'],item['arm'])];weight=model['final_sha256'] if old else model['sha256'];key=(old,weight,half)
        if key not in cache_values:
            cache=historical_cache(model,half,banks[half],readonly=True) if old else current_cache(model,risk[item['seed']],taus[(item['seed'],item['arm'])],half,banks[half],readonly=True)
            cache_values[key]=[cache.load(i) for i in range(len(q))]
        values=cache_values[key];indices=[i for i,z in enumerate(q) if z['kind']=='native'];native=[q[i] for i in indices];bases=[b[i] for i in indices];selected=[values[i] for i in indices]
        for z,base,v in zip(native,bases,selected):
            if not np.array_equal(base,predictions['main'][half][BASE][z['id']]) or not np.array_equal(v[item['policy']]['prediction'],predictions['main'][half][name][z['id']]):raise ValueError('native original Base/policy axes disagree with sealed predictions')
        value=diagnose(native,bases,selected,item['policy'],model['arm'],inputs['F_fit'],legacy=old);metric=rows[(half,name)];baseline=rows[(half,BASE)];cube=np.asarray(value['true_Base_prediction_joint5cube']);system_cm=cube.sum(1);base_cm=cube.sum(2)
        if value['new_frame_errors']!=metric['native_frame_errors'] or value['Base_frame_errors']!=baseline['native_frame_errors'] or value['frame_error_delta']!=metric['native_frame_errors']-baseline['native_frame_errors'] or not np.array_equal(system_cm,metric['decoded_position_confusion']['native']) or not np.array_equal(base_cm,baseline['decoded_position_confusion']['native']):raise ValueError('native cube does not reconstruct sealed full position confusion/error counts')
        if not old and (value['Base_correct_to_wrong']!=metric['native_new_wrong_reference_positions'] or value['Base_wrong_to_correct']!=metric['native_repaired_reference_positions']):raise ValueError('native gross counts differ from independently recounted314 rows')
        cells.append({**item,**value,'utility_weight_sha256':weight,'same_native_outputs_reused_in_all_four_other_tables':True})
    if len(cells)!=56 or len(cache_values)!=20:raise ValueError('exact56 main native cells/20joint historical+new bindings required')
    write(PUB/'native_attribution.json',{'passed':True,'cells':cells,'cell_count':56,'new_source_forwards':0,'new_optimizer_updates':0,'source_outer_metrics_sha256':sha(PUB/'outer_metrics.json'),'source_summary_sha256':sha(PUB/'summary.json'),'private_predictions_sha256':out['private_predictions']['sha256'],'all56_cube_marginals_and_gross_counts_reproduce_sealed_metrics':True,'native_only_not_repeated_stress_observations':True,'no_ids_percase_arrays_or_weights_exported':True,'reference_scope':'propagated cine family labels;notexpertframevisibility/clinicalharm','class_order':read(CFG)['channel_mapping']['classes'],'freeze_git':freeze['git_commit'],'epoch':time.time()})
    pd.json_normalize([{k:v for k,v in r.items() if k not in ['accepted_groups','true_Base_prediction_joint5cube']} for r in cells]).to_csv(PUB/'native_attribution.csv',index=False)
    print({'native_cells':56,'new_model_forwards':0,'new_optimizer_updates':0},flush=True)
if __name__=='__main__':main()
