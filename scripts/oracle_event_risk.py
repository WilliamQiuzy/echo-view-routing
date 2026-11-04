#!/usr/bin/env python3
"""Post-readout6 new full-domain ceilings +30 readonly references +48 N-top supports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from scripts.event_risk_execution import bank,role_for,score_values,decoded,old_records,contexts
from echo_routing.temporal.event_risk_metrics import plan,SEEDS
from echo_routing.temporal.event_risk_kind import support
from echo_routing.temporal.safety_tasks import compose
from echo_routing.temporal.safety_oracle import options,coverage,solve


def main():
    p=execution_parser(__doc__);a=p.parse_args()
    if not authorize(a):return 0
    freeze=verify_freeze('oracle_freeze.json');summary=read(PUB/'summary.json');cal=read(PUB/'calibration.json');ctx=contexts();fit=read(PUB/'training_inputs.json')['F_fit'];pplan=plan();old=old_records()['action_space_diagnostic.json'];reused=[r for r in old['oracle_domains'] if r['domain']=='expanded']
    if summary['metric_rows']!=156 or summary['paired_comparison_count']!=240 or len(reused)!=30 or {(r['table'],r['half'],r['safety']) for r in reused}!={(r['table'],r['half'],r['safety']) for r in pplan['reused_oracle_cells']}:raise ValueError('sealed156/240 and exact30 previously frozen expandedG domains required')
    oracle=[];supports=[];private=[]
    for item in pplan['new_oracle_cells']:
        qs,pa,pj,b=bank(role_for(item['table'],item['half']));pools=[];cover={}
        for q,x,j,z in zip(qs,pa,pj,b):
            if q['kind']!='multi':raise ValueError('new oracle only120 new multi axes')
            pools.append(options(z,x.argmax(1),j.argmax(1),q['labels'],'expanded',item['safety']));c=coverage(z,x.argmax(1),j.argmax(1),q['labels'],'expanded',item['safety'])
            for k,v in c.items():cover[k]=cover.get(k,0)+v
        result=solve(pools,'expanded',item['safety']);choices=result.pop('choices');private.append({'item':item,'choices':choices});oracle.append({**item,**result,**cover,'multi_streams':len(qs),'union_is_jointly_reachable_claim':False,'unsafe_learned_policy_bounded_by_safe_oracle':False})
    for half in ['selection','calibration']:
        f=[next(r['F1'] for r in oracle if r['half']==half and r['safety']==s) for s in ['unrestricted','net-safe','gross-safe']]
        if not f[0]>=f[1]>=f[2]>=summary['systems']['a62_confirmation'][half]['A08_s0_original_B3']['segment_f1_50_multi']-1e-12:raise ValueError('nested safety/noop bounds failed')
    cache={}
    for item in pplan['support_cells']:
        seed,t,h=item['seed'],item['table'],item['half'];key=(seed,t,h)
        if key not in cache:
            role=role_for(t,h);qs,pa,pj,b=bank(role);raw,_=score_values(seed,role,readonly=True);taus={r['policy']:r['tau'] for r in cal['thresholds']+cal['legacy_thresholds'] if r['seed']==seed};v=decoded(raw,b,taus)
            if t!='main':
                mq,ma,mj,mb=bank(h);mr,_=score_values(seed,h,readonly=True);mv=decoded(mr,mb,taus);qs,b,v=compose(mq,[mb,mv],qs,[b,v],ctx[t][h]['sequences'])
            cache[key]=(qs,b,v)
        result=support(*cache[key],item['safety'],fit,private_details=True);detail=result.pop('private_details');private.append({'item':item,'choices':detail});supports.append({**item,**result})
    if len(oracle)!=6 or len(supports)!=48:raise ValueError('full6new/30readonly/48support required')
    path=RUN/'oracle_support_choices.private.npy';save_array(path,np.asarray(private,dtype=object));write(PUB/'action_space_diagnostic.json',{'new_oracle_domains':oracle,'readonly_A61_reference_domains':reused,'new_full_oracle_count':6,'readonly_old_oracle_count':30,'old_oracle_reference_sha256':read(CFG)['prior_A61']['action_space_diagnostic.json']['sha256'],'old30_were_not_resolved_again':True,'fixed_N_top_support':supports,'support_count':48,'private_choice_proof':pin(path),'full_oracle_scope':'only multi pooled IoU50 maxONE G/noop; true safety is label assisted; unsafe learned policies are not bounded by safe domains','support_scope':'original N-top/noop only, no q cutoff representability guaranteed; candidate union is not jointly achievable','summary_sha256':sha(PUB/'summary.json'),'oracle_freeze_git':freeze['git_commit'],'epoch':time.time()});print({'new_full':6,'old_refs':30,'Ntop_support':48},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
