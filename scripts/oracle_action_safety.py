#!/usr/bin/env python3
"""A61 post-frozen314/240 exact60 G-only unconstrained/net/gross-safe ceilings."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.safety_common import *
from scripts.readout_action_safety import contexts,role_for,bank
from echo_routing.temporal.safety_oracle import options,solve,coverage,SAFETIES
from echo_routing.temporal.safety_readout_plan import TABLES,HALVES


def main():
    freeze=verify_freeze('oracle_freeze.json');summary=read(PUB/'summary.json')
    if summary['metric_rows']!=314 or summary['paired_comparison_count']!=240:raise ValueError('all314/240 must freeze before label-assisted oracle')
    old=read(verify_pin(read(CFG)['prior_A60']['action_space_diagnostic.json']));expected={(r['table'],r['half'],r['domain']):r for r in old['oracle_domains'] if r['guard']=='G'}
    if len(expected)!=16:raise ValueError('exact16 oldA60 G domains required')
    ctx=contexts(read(CFG));rows=[];reproduced=0
    for table in TABLES:
        for half in HALVES:
            lookup={}
            for role in [half] if table=='main' else [half,role_for(table,half)]:
                q,a,j,b,_=bank(role);lookup.update({x['id']:(x,pa,pj,base) for x,pa,pj,base in zip(q,a,j,b)})
            multi=[q for q in ctx[table][half]['sequences'] if q['kind']=='multi']
            for domain in ['old','expanded']:
                for safety in SAFETIES:
                    pools=[];cover={k:0 for k in ['baseline_missed_true_units','candidate_union_recovered_true_units','candidate_union_recovered_short10_true_units','candidate_union_recovered_short16_true_units','baseline_missed_short10_true_units','baseline_missed_short16_true_units','eligible_actions']}
                    for q in multi:
                        _,a,j,b=lookup[q['id']];pools.append(options(b,a.argmax(1),j.argmax(1),q['labels'],domain,safety));c=coverage(b,a.argmax(1),j.argmax(1),q['labels'],domain,safety)
                        for key in cover:cover[key]+=c[key]
                    solved=solve(pools,domain,safety);row={'table':table,'half':half,**{k:v for k,v in solved.items() if k!='choices'},'multi_streams':len(multi),**cover,'union_is_jointly_reachable_claim':False,'per_sequence_choices_exported':False}
                    if safety=='unrestricted' and table in TABLES[:-1]:
                        prior=expected[(table,half,domain)]
                        for key in ['M','P','T','numerator','denominator','F1','baseline_missed_true_units','candidate_union_recovered_true_units','candidate_union_recovered_short10_true_units','candidate_union_recovered_short16_true_units','baseline_missed_short10_true_units','baseline_missed_short16_true_units']:
                            if row[key]!=prior[key]:raise ValueError('A60 unrestricted G domain fails exact reproduction')
                        row['overlapping_A60_G_domain_exactly_reproduced']=True;reproduced+=1
                    rows.append(row)
    index={(r['table'],r['half'],r['domain'],r['safety']):r for r in rows}
    for table in TABLES:
        for half in HALVES:
            for domain in ['old','expanded']:
                values=[index[(table,half,domain,s)]['F1'] for s in SAFETIES]
                if not values[0]>=values[1]>=values[2]:raise ValueError('oracle true-safety subset monotonicity failed')
            for safety in SAFETIES:
                if index[(table,half,'old',safety)]['F1']>index[(table,half,'expanded',safety)]['F1']:raise ValueError('expanded domain superset monotonicity failed')
            base=summary['systems'][table][half]['A08_s0_original_B3']['segment_f1_50_multi']
            if any(r['F1']<base-1e-12 for r in rows if r['table']==table and r['half']==half):raise ValueError('always-feasible no-op baseline bound failed')
    if len(rows)!=60 or reproduced!=16:raise ValueError('exact60oracle/16oldG reproduction required')
    write(PUB/'action_space_diagnostic.json',{'oracle_domains':rows,'domain_count':60,'overlapping_A60_G_oracles_exactly_reproduced':reproduced,'all_maxone_G_and_noop':True,'net_safe_actual_constraints':'dE<=0 and dFP16<=0','gross_safe_actual_constraints':'H==0 and dFP16<=0','unrestricted_is_the_only_domain_bounding_policies_that_violate_true_safety':True,'grosssafe_oracle_minus_C_gap_not_pure_ranking_or_always_attainable_safe_gain':True,'native_pair_not_in_oracle_objective':True,'no_oracle_used_for_model_or_threshold_selection':True,'reference_labels_not_expert_frame_visibility':True,'summary_sha256':sha(PUB/'summary.json'),'oracle_freeze_git':freeze['git_commit'],'epoch':time.time()})
    print({'oracles':60,'old_G_exact_reproductions':16},flush=True)
if __name__=='__main__':main()
