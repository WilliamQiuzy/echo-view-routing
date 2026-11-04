#!/usr/bin/env python3
"""A61 frozen314 complete-task rows: same-choice ENC and immutable A60 controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.safety_common import *
from scripts.cache_action_safety import reader,original_a08
from echo_routing.temporal.safety_barrier import validate as validate_preouter
from echo_routing.temporal.safety_tasks import compose as compose_task_values,align as align_task_values
from echo_routing.temporal.safety_training import Utility,Risk,optimizer,restore
from echo_routing.temporal.safety_runtime import infer
from echo_routing.temporal.safety_policy_cache import SafetyCache
from echo_routing.temporal.safety_metrics import evaluate,action_summary
from echo_routing.temporal.safety_readout_plan import plan as fixed_plan,system,historical,TABLES,HALVES,BASE,SEEDS
from echo_routing.temporal.proposal_training import ProposalRegressor,optimizer as old_optimizer,restore as old_restore
from echo_routing.temporal.proposal_runtime import infer as old_infer
from echo_routing.temporal.proposal_policy_cache import PolicyCache as OldPolicyCache
from echo_routing.temporal.proposal_metrics import action_summary as old_action_summary
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.structured_metrics import stress_context,compose_stress_predictions,verify_reference
from echo_routing.temporal.baselines.registry import decode_with_prob


def role_for(table,half):
    return half if table=='main' else {'a58_stress':'old_stress_','a59_confirmation':'confirmation_','a60_confirmation':'a60_confirmation_','a61_confirmation':'a61_confirmation_'}[table]+half


def contexts(cfg):
    path=ROOT/cfg['data']['old_outer_metadata']
    if sha(path)!=cfg['data']['old_outer_metadata_sha256']:raise ValueError('fixed original task context changed')
    main=np.load(path,allow_pickle=True).item()
    return {t:{h:main[h] if t=='main' else stress_context(main[h],sequences(role_for(t,h))) for h in HALVES} for t in TABLES}


def bank(role):
    qs=sequences(role);caches={s:reader(s,role,readonly=True) for s in ['a08','jepa']};values={s:[c.load(i) for i in range(len(qs))] for s,c in caches.items()};params=read(CFG)['source']['a08']['original_B3']
    b=[decode_with_prob('b3',p,{'decode':{'b3':params}})[0] for p in values['a08']]
    return qs,values['a08'],values['jepa'],b,{s:c.identity for s,c in caches.items()}


def source_banks(outer,banks):
    inputs=read(PUB/'input_manifest.json');fresh={(r['source'],r['role']):r for r in outer['records']}
    if len(outer['records'])!=4 or set(fresh)!={(s,'a61_confirmation_'+h) for s in ['a08','jepa'] for h in HALVES}:raise ValueError('exact4 new source-role cells required')
    count=0
    for role,(q,a,j,b,ids) in banks.items():
        for s in ['a08','jepa']:
            old=inputs['source_cache_roles'].get(role,{}).get(s);row=old or fresh[(s,role)]
            if old:folder=ROOT/old['path']
            elif outer.get('read_only_early_producer_cache_adopted'):
                from echo_routing.temporal.safety_precache import folder as expected_precache_folder
                if row['path']!=expected_precache_folder(s,role):raise ValueError('adopted cache must retain its original independent producer path')
                folder=ROOT/row['path']
            else:folder=RUN/'cache'/s/role
            if not row['complete'] or row['identity']!=ids[s] or row['journal_sha256']!=sha(folder/'committed.jsonl') or row['input_sha256']!=inputs['roles'][role]['sha256'] or row['committed_sequences']!=len(q):raise ValueError('source/input/order/gamma/completeaxis cache binding changed')
            count+=1
    if count!=20:raise ValueError('complete20 outer source-role cells required')


def current_cache(row,riskrow,tau,role,values,readonly=False):
    qs,a,j,b,ids=values
    identity=digest({'utility_SHA':row['sha256'],'risk_SHA':riskrow['sha256'],'head_identity':row['identity'],'tau':tau,'role':role,'sources':ids,'input':read(PUB/'input_manifest.json')['roles'][role]['sha256'],'readout_freeze':verify_freeze('readout_freeze.json')['objects']})
    return SafetyCache(RUN/'policy_readout'/row['sha256']/role,identity,b,[p.argmax(1) for p in a],[p.argmax(1) for p in j],tau,readonly=readonly)


def historical_cache(row,role,values,readonly=False):
    qs,a,j,b,ids=values
    if not role.startswith('a61_confirmation_'):
        old=read(ROOT/'docs/results/2026-11-03_jepa_proposals/outer_metrics.json')
        if old['models_manifest_sha256']!=read(CFG)['prior_A60']['models_manifest.json']['sha256']:raise ValueError('historical A60 published model identity changed')
        matches=[r for r in old['rows'] if r['system']==f"seed{row['seed']}_J-E_fixed6000" and role_for(r['table'],r['half'])==role]
        if len(matches)!=1:raise ValueError('unique historical original policy/source row required')
        info=matches[0]['policy_cache'];folder=ROOT/info['path']
        if info['source_cache_identities']!=ids or info['sequences']!=len(qs) or sha(folder/'committed.jsonl')!=info['journal_sha256']:raise ValueError('A60 original score journal/source identity changed')
        return OldPolicyCache(folder,info['identity'],b,[p.argmax(1) for p in a],[p.argmax(1) for p in j],readonly=True)
    identity=digest({'original_A60_SHA':row['final_sha256'],'original_head_identity':row['identity'],'policy_semantics':'original_A60_risk_filter_then_best_not_A61_topfirst','role':role,'source':ids,'input':read(PUB/'input_manifest.json')['roles'][role]['sha256'],'readout_freeze':verify_freeze('readout_freeze.json')['objects']})
    return OldPolicyCache(RUN/'historical_policy_readout'/row['final_sha256']/role,identity,b,[p.argmax(1) for p in a],[p.argmax(1) for p in j],readonly=readonly)


def full_values(table,half,values,main_values,banks,context):
    qs,a,j,b,_=banks[role_for(table,half)]
    if table=='main':return align_task_values(qs,(b,values),context['sequences'])
    mq,ma,mj,mb,_=banks[half]
    return compose_task_values(mq,(mb,main_values),qs,(b,values),context['sequences'])


def legacy_full_values(table,half,values,main_values,banks,context):
    qs,a,j,b,_=banks[role_for(table,half)]
    if table=='main':return align_task_values(qs,(a,j,b,values),context['sequences'])
    mq,ma,mj,mb,_=banks[half]
    return compose_task_values(mq,(ma,mj,mb,main_values),qs,(a,j,b,values),context['sequences'])


def main():
    freeze=verify_freeze('readout_freeze.json');verify_freeze('outer_freeze.json');models=read(PUB/'models_manifest.json');seal=read(PUB/'outer_identity.json');plan=read(PUB/'readout_plan.json');cfg=read(CFG);inputs=read(PUB/'training_inputs.json')
    validate_preouter(models,seal,sha(PUB/'models_manifest.json'),sha(PUB/'calibration.json'),sha(PUB/'readout_plan.json'),read(verify_pin(seal['actual_Git_object_proof'])))
    if plan!=fixed_plan() or (PUB/'outer_metrics.json').exists():raise ValueError('exact314/240/60/56 plan and fresh fixedreadout required')
    for row in models['models']:verify_pin({'path':row['path'],'sha256':row['sha256']})
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ctx=contexts(cfg);banks={role_for(t,h):bank(role_for(t,h)) for t in TABLES for h in HALVES};source_banks(read(PUB/'cache_outer_manifest.json'),banks)
    predictions={t:{h:{} for h in HALVES} for t in TABLES};rows=[];journals=[];history=[];main_outputs={};norms=read(verify_pin(inputs['normalizers']));scale=inputs['utility_scale']
    original=read(ROOT/'docs/results/2026-11-02_dense_target_sequence/A08_reference_metrics.json')['systems']
    def publish(table,half,name,pred,info):
        baseline=pred if name==BASE else predictions[table][half][BASE]
        metric,_=evaluate(pred,ctx[table][half],baseline);predictions[table][half][name]=pred;rows.append({'table':table,'half':half,'system':name,**info,**metric})
        write(PUB/'completed_outer_rows.json',{'rows':rows,'metric_rows':len(rows),'all10models24inner8taus_actualGit_frozen_before_outer':True,'provisional_until_independent_original_axis_aggregate':True,'readout_freeze_git':freeze['git_commit']})
        print({'row':name,'table':table,'half':half,'native_macro_f1':metric['native_macro_f1'],'segment_f1_50_multi':metric['segment_f1_50_multi'],'native_H':metric['native_new_wrong_reference_positions'],'short16_FN':metric['short_true_unmatched_16samples'],'short16_FP':metric['short_spurious_segments_16samples']},flush=True)
    for half in HALVES:
        qs,a,j,b,_=banks[half]
        for seed in [0,1,2]:
            probability,params=original_a08(half,qs,seed,cfg);name=f'A08_s{seed}_original_B3';pred={q['id']:decode_with_prob('b3',p,{'decode':{'b3':params}})[0].astype(np.int8) for q,p in zip(qs,probability)}
            publish('main',half,name,pred,{'role':'reference','original_A08_cache_parameters':True});verify_reference(rows[-1],original[half][f'A08_original_B3_s{seed}'])
            if seed==0 and any(not np.array_equal(pred[q['id']],base) for q,base in zip(qs,b)):raise ValueError('fixedBase original cache mismatch')
        for name,prob in [('A08_s0_original_raw',a),('JEPA_core_fixed6000_raw',j)]:publish('main',half,name,{q['id']:p.argmax(1).astype(np.int8) for q,p in zip(qs,prob)},{'role':'reference'})
        for table in TABLES[1:]:
            sq,sa,sj,sb,_=banks[role_for(table,half)]
            for name,values in [(BASE,sb),('A08_s0_original_raw',[p.argmax(1) for p in sa]),('JEPA_core_fixed6000_raw',[p.argmax(1) for p in sj])]:publish(table,half,name,compose_stress_predictions(predictions['main'][half][name],{q['id']:np.asarray(v,np.int8) for q,v in zip(sq,values)},qs,sq),{'role':'reference','native_pair_predictions_reused_identically':True})
    # Fixed A60 controls preserve original risk-filter-then-best selection.
    oldmodels=read(verify_pin(cfg['prior_A60']['models_manifest.json']))
    legacy=[r for r in oldmodels['models'] if r['arm']=='J']
    if {r['seed'] for r in legacy}!={2026110360,2026110361} or len(legacy)!=2:raise ValueError('exacttwo historical J models required')
    for row in legacy:
        oldseed=row['seed'];repseed=SEEDS[oldseed-2026110360];norm=read(verify_pin(row['normalizer']));scales=np.asarray(read(verify_pin(row['target_scales']))['scales']);path=ROOT/row['final_path']
        if sha(path)!=row['final_sha256']:raise ValueError('historical fixedweight changed')
        oldmodel=ProposalRegressor('J',oldseed)
        if old_restore(oldmodel,old_optimizer(oldmodel),torch.load(path,weights_only=True,map_location='cpu'),row['identity'],norm,scales)!=6000:raise ValueError('historical finalcursor changed')
        for table in TABLES:
            for half in HALVES:
                role=role_for(table,half);qs,a,j,b,ids=banks[role];cache=historical_cache(row,role,banks[role])
                if role.startswith('a61_confirmation_'):
                    for i in range(cache.cursor,len(qs)):cache.run_one(i,lambda i=i:old_infer(oldmodel,norm,scales,a[i],j[i],b[i],'J'))
                values=[cache.load(i) for i in range(len(qs))]
                if table=='main':main_outputs[('A60',oldseed,half)]=values
                info={'path':str(cache.path.relative_to(ROOT)),'identity':cache.identity,'journal_sha256':sha(cache.journal),'sequences':len(qs),'source_cache_identities':ids,'reused_closed_A60_readonly_cache':not role.startswith('a61_confirmation_'),'new_physical_MLP_forwards':sum(bool(v['actions']) for v in values) if role.startswith('a61_confirmation_') else 0};history.append(info)
                for policy in ['E','R']:
                    name=historical(repseed,policy);pred={q['id']:v[policy]['prediction'] for q,v in zip(qs,values)};source_stats=old_action_summary(qs,a,j,b,values,policy,inputs['F_fit']);fullstats=source_stats
                    if table!='main':
                        group=legacy_full_values(table,half,values,main_outputs[('A60',oldseed,half)],banks,ctx[table][half]);fullstats=old_action_summary(*group,policy,inputs['F_fit']);pred=compose_stress_predictions(predictions['main'][half][name],pred,banks[half][0],qs)
                    publish(table,half,name,pred,{'role':'A60_historical','old_seed':oldseed,'replicate_seed':repseed,'policy':policy,'step':6000,'weight_sha256':row['final_sha256'],'policy_cache':info,'original_A60_selection_semantics':True,'not_equal_training_seed_comparison':True,'action_statistics_source_role':source_stats,'action_statistics_full_task':fullstats,'source_role':role,'all_sequences_G_majority_unchanged':True})
    riskrows={r['seed']:r for r in models['models'] if r['role']=='risk'};riskheads={}
    for seed,row in riskrows.items():
        path=verify_pin({'path':row['path'],'sha256':row['sha256']});head=Risk(seed)
        if restore(head,optimizer(head),torch.load(path,weights_only=True,map_location='cpu'),row['identity'],norms,scale)!=6000:raise ValueError('finalrisk cursor changed')
        riskheads[seed]=head.eval()
    thresholds={(r['seed'],r['arm']):r['tau'] for r in models['thresholds']}
    utilityrows=[r for r in models['models'] if r['role']=='utility']
    for row in utilityrows:
        seed,arm=row['seed'],row['arm'];path=verify_pin({'path':row['path'],'sha256':row['sha256']});head=Utility(arm,seed)
        if restore(head,optimizer(head),torch.load(path,weights_only=True,map_location='cpu'),row['identity'],norms,scale)!=6000:raise ValueError('fixedutility finalcursor changed')
        head.eval();tau=thresholds[(seed,arm)]
        for table in TABLES:
            for half in HALVES:
                role=role_for(table,half);qs,a,j,b,ids=banks[role];cache=current_cache(row,riskrows[seed],tau,role,banks[role])
                for i in range(cache.cursor,len(qs)):cache.run_one(i,lambda i=i:infer(head,riskheads[seed],norms,scale,a[i],j[i],b[i],tau))
                values=[cache.load(i) for i in range(len(qs))]
                if table=='main':main_outputs[(seed,arm,half)]=values
                info={'path':str(cache.path.relative_to(ROOT)),'identity':cache.identity,'journal_sha256':sha(cache.journal),'sequences':len(qs),'source_cache_identities':ids,'utility_weight_sha256':row['sha256'],'risk_weight_sha256':riskrows[seed]['sha256'],'tau':tau,'physical_utility_and_each_risk_module_forward_calls':sum(bool(v['actions']) for v in values),'physical_candidate_score_vectors':sum(len(v['score_matrix']) for v in values),'ENC_share_single_source_score_forward':True};journals.append(info)
                for policy in ['E','N','C']:
                    name=system(seed,arm+'-'+policy);pred={q['id']:v[policy]['prediction'] for q,v in zip(qs,values)};source_stats=action_summary(qs,b,values,policy,arm,inputs['F_fit']);fullstats=source_stats
                    if table!='main':
                        group=full_values(table,half,values,main_outputs[(seed,arm,half)],banks,ctx[table][half]);fullstats=action_summary(*group,policy,arm,inputs['F_fit']);pred=compose_stress_predictions(predictions['main'][half][name],pred,banks[half][0],qs)
                    if any(majority(v[policy]['prediction'])!=majority(base) for v,base in zip(values,b)):raise ValueError('allsequenceG invariant failed')
                    publish(table,half,name,pred,{'role':'policy','seed':seed,'arm':arm,'policy':policy,'step':6000,'weight_sha256':row['sha256'],'risk_weight_sha256':riskrows[seed]['sha256'],'tau':tau,'policy_cache':info,'action_statistics_source_role':source_stats,'action_statistics_full_task':fullstats,'candidate_count':fullstats['candidate_count'],'accepted_actions':fullstats['accepted_actions'],'source_role':role,'all_sequences_G_majority_unchanged':True,'native_pair_predictions_reused_identically':table!='main','ENC_same_top_no_fallback_shared_weights_scores':True,'shared_risk_weights_across_four_utility_arms_but_actual_inference_repeated_and_charged':True,'full_system_inference_seconds':None,'cached_heads_do_not_include_A08_JEPA_producer_latency':True})
                write(RUN/'runtimecoordination.json',{'phase':'frozen314_ENC_readout','model_pid':os.getpid(),'seed':seed,'arm':arm,'table':table,'half':half,'metric_rows':len(rows),'epoch':time.time(),'next_action':'finish314 thenfreeze/privateoriginalaxisrecount240CIs60oracles56native'})
    expected={(r['table'],r['half'],r['system'],r['role']) for r in plan['rows']};actual=[(r['table'],r['half'],r['system'],r['role']) for r in rows]
    if len(rows)!=314 or set(actual)!=expected or len(journals)!=80 or len(history)!=20:raise ValueError('exact314rows/80newjointcells/20historicalbindings required')
    path=RUN/'readout_predictions.private.npy';save_array(path,predictions)
    if torch.cuda.is_initialized():raise ValueError('CPUreadout initialized unexpected CUDA')
    write(PUB/'outer_metrics.json',{'rows':rows,'metric_rows':314,'policy_metric_rows':240,'reference_metric_rows':74,'readout_freeze_git':freeze['git_commit'],'models_manifest_sha256':sha(PUB/'models_manifest.json'),'calibration_sha256':sha(PUB/'calibration.json'),'readout_plan_sha256':sha(PUB/'readout_plan.json'),'source_outer_cache_manifest_sha256':sha(PUB/'cache_outer_manifest.json'),'private_predictions':array_pin(path),'joint_policy_cache_cells':journals,'historical_A60_cache_cells':history,'all10finals24inner8taus_actualGit_frozen_before_outer':True,'historical_A08_all3_reproduced':True,'no_outer_checkpoint_or_threshold_selection':True,'actual_utility_MLP_forwards':sum(r['physical_utility_and_each_risk_module_forward_calls'] for r in journals),'actual_localrisk_MLP_forwards':2*sum(r['physical_utility_and_each_risk_module_forward_calls'] for r in journals),'actual_historical_newrole_MLP_forwards':sum(r['new_physical_MLP_forwards'] for r in history),'cached_runtime_excludes_frozenproducer_work':True,'epoch':time.time()})
    write(RUN/'runtimecoordination.json',{'phase':'all314_readouts_complete_pending240CIs60oracle56native','model_pid':None,'metric_rows':314,'epoch':time.time(),'next_action':'actualGitfreeze314;independentoriginalaxis240paired5000/60oracle16reproductions/56native'})
if __name__=='__main__':main()
