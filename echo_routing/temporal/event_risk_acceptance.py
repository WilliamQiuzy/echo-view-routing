"""A62 declared R2 goal and matched attribution assessment, without selection.

Input records are already frozen aggregates. Runtime validity, private audit,
resource closure and main publication are separate checks. Scientific failure
is returned with every failed cell, never an instruction to retrain or censor.
"""
import math
import numpy as np
from echo_routing.temporal.event_risk_metrics import plan,BASE,SEEDS,TABLES,HALVES,system


def _count(v,nonnegative=True):
    if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or nonnegative and v<0:raise ValueError('exact aggregate count required')
    return int(v)


def _f1(v):
    if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number)) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('complete finite pooled F1 required')
    return float(v)


def _interval(v):
    if v is None:return None
    if not isinstance(v,(list,tuple)) or len(v)!=2 or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,float,np.number)) or not math.isfinite(x) for x in v) or not -1<=v[0]<=v[1]<=1:raise ValueError('finite ordered pooled-F1 delta interval or undefined null required')
    return list(map(float,v))


def _safety(row):
    gross=row['gross_reference_position_effects']
    if set(gross)!={'native','multi','pair'}:raise ValueError('complete native/multi/pair gross aggregate required')
    h=repairs=0
    for value in gross.values():
        H=_count(value['H']);R=_count(value['R'])
        if _count(value['dE'],False)!=H-R:raise ValueError('gross H/R cannot be replaced by net error only')
        h+=H;repairs+=R
    stat=row['action_statistics_full_task'];fp=_count(stat['accepted_effect_totals']['dFP'],False);lost=_count(stat['accepted_short16_true_units_lost'])
    return {'H':h,'R':repairs,'net_error_delta':h-repairs,'dFP16_full_task':fp,'short16_gross_lost':lost,'safe':h==0 and fp<=0 and lost==0}


def assess(rows,comparisons,calibration_information):
    p=plan();expected={(r['table'],r['half'],r['system']):r for r in p['rows']};actual={(r['table'],r['half'],r['system']):r for r in rows}
    if len(rows)!=156 or len(actual)!=156 or set(actual)!=set(expected):raise ValueError('exact complete156 outer rows required; no dropped arms/cells or duplicates')
    for key,row in actual.items():
        declaration=expected[key]
        if declaration['role']=='policy' and (row['seed'],row['policy'])!=(declaration['seed'],declaration['policy']):raise ValueError('row policy/seed differs from frozen system identity')
        _f1(row['segment_f1_50_multi'])
    expected_ci={(r['table'],r['half'],r['seed'],r['contrast']):r for r in p['comparisons']};ci={(r['table'],r['half'],r['seed'],r['contrast']):r for r in comparisons}
    if len(comparisons)!=240 or len(ci)!=240 or set(ci)!=set(expected_ci):raise ValueError('exact complete240 fixed paired comparisons required')
    for key,row in ci.items():
        declaration=expected_ci[key]
        if row['left']!=declaration['left'] or row['right']!=declaration['right'] or row['bootstrap_replicates']!=5000:raise ValueError('paired original endpoint or replicate count changed')
        _interval(row['delta95']['segment_f1_50_multi'])
        if declaration['primary_family']:
            if 'delta99_375' not in row:raise ValueError('eight prespecified primary simultaneous intervals missing')
            _interval(row['delta99_375']['segment_f1_50_multi'])
    if len(calibration_information)!=2 or {int(s) for s in calibration_information}!=set(SEEDS):raise ValueError('complete two-seed CAL information required')
    info=[]
    for seed in SEEDS:
        record=calibration_information[str(seed)] if str(seed) in calibration_information else calibration_information[seed]
        n=_count(record['N_admitted']);bad=_count(record['unsafe_N_top'])
        if bad>n:raise ValueError('unsafe CAL tops exceed original N-admitted tops')
        info.append({'seed':seed,'N_admitted':n,'unsafe_N_top':bad,'calibration_limited':n<20 or bad<5})
    cells=[]
    for table in TABLES:
        for half in HALVES:
            base=actual[(table,half,BASE)]
            for seed in SEEDS:
                r=actual[(table,half,system(seed,'R2'))];safe=_safety(r);stat=r['action_statistics_source_role'];recovered=_count(stat['accepted_new_JEPA_short16_true_units_recovered'])
                checks={'H_zero':safe['H']==0,'short16_FP_not_above_Base':safe['dFP16_full_task']<=0,'short16_gross_lost_zero':safe['short16_gross_lost']==0}
                if table=='main':
                    checks['main_native_H_zero']=_count(r['gross_reference_position_effects']['native']['H'])==0
                    checks['main_segment_F1_point_not_below_Base']=_f1(r['segment_f1_50_multi'])>=_f1(base['segment_f1_50_multi'])
                required=None
                if table=='a61_confirmation':
                    required=([6,5] if half=='selection' else [4,3])[SEEDS.index(seed)]
                    checks['retained_A61_JEPA_short16_recovery']=recovered>=required
                if table=='a62_confirmation':
                    required=1;checks['new_A62_JEPA_short16_recovery']=recovered>=1
                cells.append({'table':table,'half':half,'seed':seed,**safe,'JEPA_exclusive_short16_recovered':recovered,'required_recovery':required,'F1_delta_to_Base':_f1(r['segment_f1_50_multi'])-_f1(base['segment_f1_50_multi']),'delta95_to_Base':ci[(table,half,seed,'R2-Base')]['delta95']['segment_f1_50_multi'],'checks':checks,'failed':[k for k,v in checks.items() if not v],'passed':all(checks.values())})
    joint=all(c['passed'] for c in cells);limited=any(r['calibration_limited'] for r in info);attribution={}
    for hypothesis,left,right in [('H1','R1','M0'),('H2','R2','R1')]:
        evidence=[]
        for half in HALVES:
            for seed in SEEDS:
                a=actual[('a62_confirmation',half,system(seed,left))];b=actual[('a62_confirmation',half,system(seed,right))];comparison=ci[('a62_confirmation',half,seed,left+'-'+right)];delta=_f1(a['segment_f1_50_multi'])-_f1(b['segment_f1_50_multi']);interval=_interval(comparison['delta99_375']['segment_f1_50_multi']);safe=_safety(a)
                evidence.append({'half':half,'seed':seed,'F1_delta':delta,'delta99_375':interval,'positive_direction':delta>0,'interval_lower_positive':interval is not None and interval[0]>0,'left_system_safe':safe['safe']})
        gain=all(r['positive_direction'] and r['interval_lower_positive'] for r in evidence);safe=all(r['left_system_safe'] for r in evidence)
        # The information line explicitly bars identifying H1 when CAL is sparse.
        calibration_eligible=not limited if hypothesis=='H1' else True
        attribution[hypothesis]={'contrast':left+'-'+right,'cells':evidence,'F1_increment_evidence_passed':gain,'new_confirmation_left_system_safety_passed':safe,'R2_joint_empirical_target_passed':joint,'CAL_information_eligible_for_identification':calibration_eligible,'claim_supported':gain and safe and joint and calibration_eligible,'scope':'conditional constructed banks and two fixed seeds; H1 packages target/aggregation/capacity changes, H2 adds context under matched architecture'}
    return {'endpoint_coverage_passed':True,'endpoint_coverage_scope':'156 outer metric records and240 paired records only; does not substitute model/182total/state/OS/cost/publication audits','primary_system':'R2','choose_R1_after_R2_failure':False,'automatic_next_round':False,'R2_joint_empirical_target':{'passed':joint,'evaluated_cells':24,'failed_cells':sum(not c['passed'] for c in cells),'cells':cells},'attribution':attribution,'CAL_information':info,'calibration_limited':limited,'CI_contains_zero_is_equivalence':False,'clinical_or_finite_sample_safety_guarantee':False,'negative_results_retained':True}
