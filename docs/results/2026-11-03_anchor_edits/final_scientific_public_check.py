"""Public aggregate-only scientific review; no private axes, inference or experiments."""
import csv,hashlib,json,math
from fractions import Fraction
from pathlib import Path
P=Path(__file__).resolve().parent
read=lambda name: json.loads((P/name).read_text())
sha=lambda name: hashlib.sha256((P/name).read_bytes()).hexdigest()
s=read('summary.json'); o=read('action_space_diagnostic.json'); f=read('figure_receipt.json')
assert s['metric_rows']==70 and s['paired_comparison_count']==96 and s['formal_models']==4 and s['formal_optimizer_updates']==24000
assert f['summary_sha256']==sha('summary.json')==o['summary_sha256']
assert f['action_space_diagnostic_sha256']==sha('action_space_diagnostic.json')
systems={(t,h,n):v for t,hs in s['systems'].items() for h,ns in hs.items() for n,v in ns.items()}
assert len(systems)==70
trades=list(csv.DictReader((P/'task_tradeoffs.csv').open()))
assert len(trades)==70 and {(r['table'],r['half'],r['system']) for r in trades}==set(systems)
for row in trades:
    metric=systems[(row['table'],row['half'],row['system'])]
    for key,value in row.items():
        if key not in ['table','half','system']:
            assert (not value and metric[key] is None) or math.isclose(float(value),metric[key],abs_tol=1e-12,rel_tol=0)
for (table,half,name),v in systems.items():
    c=v['native_confusion']; n=sum(map(sum,c)); assert n=={'selection':284,'calibration':237}[half]
    f1=[]
    for k in range(5):
        den=sum(c[k])+sum(row[k] for row in c)
        f1.append(2*c[k][k]/den if den else 0.)
    assert math.isclose(sum(f1)/5,v['native_macro_f1'],abs_tol=1e-12)
    assert math.isclose(sum(c[k][k] for k in range(5))/n,v['native_accuracy'],abs_tol=1e-12)
    counts=v['counts']
    for kind,suffix in [('multi','multi'),('pair','pairs')]:
        assert math.isclose(2*counts[kind+'_matched_segments']/(counts[kind+'_predicted_segments']+counts[kind+'_true_segments']),v['segment_f1_50_'+suffix],abs_tol=1e-12)
    if name.startswith('seed') and '-G_' in name: assert c==systems[(table,half,'A08_s0_original_B3')]['native_confusion']
    if table!='main':
        main=systems[('main',half,name)]
        for key in ['native_confusion','native_macro_f1','native_frame_accuracy','native_internal_cuts_per_clip','segment_f1_50_pairs','boundary_precision_pairs','boundary_recall_pairs']:
            assert v[key]==main[key]
cs=s['paired_comparisons']; assert len(cs)==96
assert len({(c['table'],c['half'],c['seed'],c['contrast']) for c in cs})==96
for c in cs:
    assert c['bootstrap']['replicates']==5000 and c['bootstrap']['seed']==2026110359
    assert c['bootstrap']['native_sessions']==36 and c['bootstrap']['multi_streams']==120
    assert c['bootstrap']['pair_streams']=={'selection':137,'calibration':115}[c['half']]
    assert all(0<=n<=5000 for n in c['undefined_bootstrap_replicates'].values())
print('Verified public identities/counts:70rows,70CSVrows,96comparisons x5000,all G confusion equal Base; no private-axis re-read')
print('PRIMARY SEGMENT EFFECTS (percentage points):')
for c in cs:
    if c['contrast'] not in ['J-G-A-G','J-G-A08_s0_original_B3']:continue
    d=c['delta']['segment_f1_50_multi'];ci=c['delta95']['segment_f1_50_multi']
    verdict='positive' if ci[0]>0 else 'negative' if ci[1]<0 else 'crosses_or_touches_zero'
    print(c['table'],c['half'],c['seed'],c['contrast'],f'{100*d:+.6f}',f'[{100*ci[0]:+.6f},{100*ci[1]:+.6f}]',verdict)
print('JG short16 FN/true, FP; native extra frame errors/cuts relative Base:')
for table in ['main','old_stress','fresh_confirmation']:
    for half in ['selection','calibration']:
        b=systems[(table,half,'A08_s0_original_B3')]
        print(table,half,'Base',b['short_true_unmatched_16samples'],b['short_true_segments_16samples'],b['short_spurious_segments_16samples'])
        for seed in [2026110359,2026110360]:
            v=systems[(table,half,f'seed{seed}_J-G_fixed6000')]
            extra_errors=(b['native_frame_accuracy']-v['native_frame_accuracy'])*b['counts']['native_positions']
            extra_cuts=(v['native_internal_cuts_per_clip']-b['native_internal_cuts_per_clip'])*sum(map(sum,b['native_confusion']))
            print(seed,v['short_true_unmatched_16samples'],v['short_true_segments_16samples'],v['short_spurious_segments_16samples'],'native_extra_errors/cuts',round(extra_errors),round(extra_cuts))
assert len(o['oracle_domains'])==o['domain_count']==12
assert o['label_assisted_not_a_system_or_gate_or_rescue'] and o['no_oracle_choices_used_for_learning_or_selection'] and o['coverage_union_not_jointly_reachable']
assert len({(r['table'],r['half'],r['domain']) for r in o['oracle_domains']})==12
csv_oracle=list(csv.DictReader((P/'action_space_oracle.csv').open())); assert len(csv_oracle)==12
print('ORACLE exact declared domain ceilings, excess over Base(pp), union recoverable/baseline missed:')
for r in o['oracle_domains']:
    b=systems[(r['table'],r['half'],'A08_s0_original_B3')];bc=b['counts'];bf=Fraction(2*bc['multi_matched_segments'],bc['multi_predicted_segments']+bc['multi_true_segments'])
    exact=Fraction(r['numerator'],r['denominator']); assert exact>=bf and math.isclose(float(exact),r['F1'],abs_tol=1e-15)
    same=next(x for x in csv_oracle if (x['table'],x['half'],x['domain'])==(r['table'],r['half'],r['domain']))
    assert int(same['numerator'])==r['numerator'] and int(same['denominator'])==r['denominator']
    print(r['table'],r['half'],r['domain'],f'{r["F1"]:.12f}',f'{100*float(exact-bf):+.6f}',r['candidate_union_recovered_true_units'],r['baseline_missed_true_units'],'equalBase',exact==bf)
