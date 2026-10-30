#!/usr/bin/env python3
"""Package ONLY independently verified aggregates for the current results page."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'docs/results/2026-10-30_external_original_tests'
SOURCE=ROOT/'demo/external_results'
EXPECTED={'ev9v':800,'dynamic':1277,'lvh':343,'camus':50}


def validate_aggregate_inputs(native,verification,stfm):
    if set(native)!=set(EXPECTED):raise ValueError('incomplete native datasets')
    for d,n in EXPECTED.items():
        for dec in ['argmax','frozen_B3']:
            for s in range(3):
                if native[d].get(f'{dec}_s{s}',{}).get('n_clips')!=n:raise ValueError('incomplete model denominator')
    if verification.get('passed') is not True:raise ValueError('independent verification must pass before packaging')
    metadata=verification.get('hashes',{}).get('source_metadata',{})
    if set(metadata)!={'file_count','aggregate_sha256'}:
        raise ValueError('public source metadata must contain aggregate hashes only')
    for key in ['legacy_per_key_prior','corrected_prior_once_same_old_weights']:
        r=stfm.get(key,{})
        if r.get('original_nine_code',{}).get('n_clips')!=888 or r.get('shared_five_family',{}).get('n_clips')!=800:
            raise ValueError('incomplete original-native baseline')


def read(name):return json.loads((RESULT/name).read_text())
def mean(values):return sum(values)/len(values)


def per_class_metrics(result):
    """Derive readable class diagnostics from the already-audited confusion."""
    matrix=result['confusion'];rows=[]
    for i,name in enumerate(['PLAX','PSAX','A4C','A5C','SC4C']):
        support=sum(matrix[i]);predicted=sum(row[i] for row in matrix);correct=matrix[i][i]
        rows.append({'name':name,'n':support,'correct':correct,'predicted':predicted,
            'precision':correct/predicted if predicted else 0.,'recall':correct/support if support else 0.,
            'f1':2*correct/(support+predicted) if support+predicted else 0.})
    if abs(mean([r['f1'] for r in rows])-result['five_class_macro_f1'])>1e-12:
        raise ValueError('per-class diagnostics disagree with audited macro-F1')
    return rows


def validate_audit_binding(verification,files):
    for key,path in files.items():
        actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if verification.get('hashes',{}).get(key)!=actual:
            raise ValueError('stale independent audit binding: '+key)


def payload():
    native,verify,stfm,unc=read('native_results.json'),read('verification.json'),read('stfm_native_results.json'),read('uncertainty.json')
    validate_aggregate_inputs(native,verify,stfm)
    validate_audit_binding(verify,{k:RESULT/f for k,f in [('native_results','native_results.json'),
        ('stfm_native_results','stfm_native_results.json'),('uncertainty','uncertainty.json')]})
    ext=[];seedrows=[]
    names={'ev9v':'EV9V','dynamic':'EchoNet-Dynamic','lvh':'EchoNet-LVH','camus':'CAMUS'}
    labels={'dynamic':'A4C','lvh':'PLAX','camus':'4CH code → A4C'}
    notes={'dynamic':'Official TEST; patient relationships unknown','lvh':'Official test; distinct recordings',
        'camus':'Official testing; 0.18–0.51 s half-cycles; source code not view-adjudicated'}
    for d in ['dynamic','lvh','camus']:
        rows=[native[d][f'frozen_B3_s{s}'] for s in range(3)];u=unc['datasets'][d]
        ext.append({'name':names[d],'label':labels[d],'n':EXPECTED[d],'mean_recall':mean([r['accuracy'] for r in rows]),
            'ci95':u['mean_B3']['ci95'],'correct':[u['models'][f'frozen_B3_s{s}']['correct'] for s in range(3)],
            'cuts_per_clip':mean([r['internal_cuts_per_clip'] for r in rows]),'note':notes[d]})
        for dec in ['frozen_B3','argmax']:
            for s in range(3):
                r=native[d][f'{dec}_s{s}'];seedrows.append({'dataset':names[d],'decoder':'Frozen B3' if dec=='frozen_B3' else 'Direct argmax',
                    'seed':s,'recall':r['accuracy'],'cuts_per_clip':r['internal_cuts_per_clip']})
    ours=[native['ev9v'][f'frozen_B3_s{s}'] for s in range(3)];u=unc['datasets']['ev9v']
    original=[{'name':'A08 video Transformer + frozen B3','ours':True,'seed':'Mean of 0 / 1 / 2','n':800,
        'accuracy':mean([r['accuracy'] for r in ours]),'macro_f1':mean([r['five_class_macro_f1'] for r in ours]),
        'macro_ci95':u['mean_B3']['macro_ci95'],'a4c':mean([r['class_recall']['A4C'] for r in ours]),'a5c':mean([r['class_recall']['A5C'] for r in ours])}]
    for s,r in enumerate(ours):
        original.append({'name':'A08 + frozen B3','ours':True,'seed':str(s),'n':800,'accuracy':r['accuracy'],
            'macro_f1':r['five_class_macro_f1'],'macro_ci95':u['models'][f'frozen_B3_s{s}']['macro_f1_ci95'],
            'a4c':r['class_recall']['A4C'],'a5c':r['class_recall']['A5C']})
    details=[]
    for key,title in [('legacy_per_key_prior','STFM native rule (frozen legacy weights)'),('corrected_prior_once_same_old_weights','STFM prior once (same legacy weights)')]:
        r=stfm[key]['shared_five_family'];ci=unc['baseline'][key]
        original.append({'name':title,'ours':False,'seed':'100','n':800,'accuracy':r['accuracy'],'macro_f1':r['five_class_macro_f1'],
            'macro_ci95':ci['macro_f1_ci95'],'a4c':r['class_recall']['A4C'],'a5c':r['class_recall']['A5C']})
        r=stfm[key]['original_nine_code'];details.append({'name':title,'ours':False,'population':'Nine codes / 888','accuracy':r['accuracy'],'macro_f1':r['macro_f1']})
    for s in range(3):
        r=native['ev9v'][f'argmax_s{s}'];details.append({'name':f'A08 direct argmax, s{s}','ours':True,'population':'Five classes / 800','accuracy':r['accuracy'],'macro_f1':r['five_class_macro_f1']})
    old=json.loads((ROOT/'docs/results/2026-10-23_frozen_external_benchmark/evaluation_v1.json').read_text())['values']
    historical=[]
    groups=[(['A08_s0','A08_s2'],'A08 + B3 (s0/s2 mean)',True),(['EchoViewCLIP_stage1_windowed'],'EchoViewCLIP stage1',False),
        (['STFM_numeric_s100'],'STFM numeric frame order, s100',False),(['ASFormer_v2_frameorder'],'ASFormer (EV9V adaptation)',False),
        (['EchoPrime_view_B3'],'EchoPrime + B3',False),(['EchoPrime_view_per_frame'],'EchoPrime per frame',False)]
    keys={'native':'clip macro-F1','pairs':'segment F1@50, pairs','multi':'segment F1@50, multi',
        'fp':'multi FP segments','fp_short':'multi FP segments <= 16','fn':'multi FN segments','fn_short':'multi missed true segments <= 16'}
    for ids,title,isours in groups:
        historical.append({'name':title,'ours':isours,**{k:mean([old[i][v]['value'] for i in ids]) for k,v in keys.items()}})
    paired=[{'name':title,**unc['paired_original'][key]['mean_B3']} for key,title in
        [('legacy_per_key_prior','STFM native rule'),('corrected_prior_once_same_old_weights','STFM prior once (same weights)')]]
    classes=[per_class_metrics(r) for r in ours]
    baseline_classes=per_class_metrics(stfm['legacy_per_key_prior']['shared_five_family'])
    class_rows=[{'name':base['name'],'n':base['n'],'share':base['n']/800,
        **{f'a08_{key}':mean([seed[i][key] for seed in classes]) for key in ['precision','recall','f1']},
        'stfm_f1':base['f1']} for i,base in enumerate(baseline_classes)]
    prep=read('preparation.json');short=', '.join(v['sha256'][:12] for v in prep['weights'].values())
    return {'summary':'800 recordings · five shared view classes · three frozen A08 seeds. STFM’s original test is EV9V, already used in this project; this is not new external confirmation.',
        'external':ext,'seed_rows':seedrows,'original':original,'original_details':details,'history':historical,
        'availability':read('availability.json'),'paired_original':paired,'class_breakdown':class_rows,
        'paper_reported':read('paper_reported.json'),
        'metric_example_s0_A5C':classes[0][3],'resources':read('resources.json'),
        'model_identity':f'A08 all-neighbour-smoothed video Transformer + frozen B3; s0/s1/s2; weight SHA prefixes: {short}',
        'verification_note':'Independent recount reproduced full label sequences, counts and metrics from saved logits. Development-logit, frame-rate, window and target-marker checks passed. Intervals are descriptive grouped bootstrap intervals for fixed models. Patient independence and upstream pretraining overlap remain limited or unknown.',
        'verification':verify,'uncertainty':unc,'native_results':native,'stfm_original_results':stfm,
        'source_hashes':{name:hashlib.sha256((RESULT/name).read_bytes()).hexdigest() for name in
            ['native_results.json','stfm_native_results.json','verification.json','uncertainty.json','availability.json','input_audit.json','resources.json','paper_reported.json']}}


def build(out):
    data=payload();out=Path(out)
    allowed={'index.html','style.css','app.js','data.js','results.json','results.csv'}
    if out.exists() and any(p.name not in allowed for p in out.iterdir()):
        raise ValueError('output directory contains non-report files; choose a fresh aggregate-only directory')
    out.mkdir(parents=True,exist_ok=True)
    for name in ['index.html','style.css','app.js']:
        src=SOURCE/name;dst=out/name
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    text=json.dumps(data,ensure_ascii=False,indent=2)
    (out/'results.json').write_text(text)
    (out/'data.js').write_text('window.EXTERNAL_RESULTS = '+text.replace('</',r'<\/')+';\n')
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['table','dataset','model','condition','n','metric','value','evidence'])
        for r in [data['original'][0],data['original'][4]]:
            for key in ['accuracy','macro_f1','a4c','a5c']:
                w.writerow(['matched_native','EV9V shared five',r['name'],r['seed'],r['n'],key,r[key],'A50 independent recount'])
        for r in data['class_breakdown']:
            for key,name in [('a08_f1','A08 three-seed mean'),('stfm_f1','STFM native s100')]:
                w.writerow(['matched_class','EV9V',name,r['name'],r['n'],'F1',r[key],'A50 audited confusion'])
        for r in data['history']:
            for key in ['native','pairs','multi']:
                w.writerow(['historical_matched','EV9V',r['name'],'A26',{'native':800,'pairs':240,'multi':120}[key],key,r[key],'A26 frozen report'])
        for r in data['paper_reported']['results']:
            for key in ['accuracy','f1','ovr_auc']:
                if r[key] is not None:
                    w.writerow(['paper_reported',r['dataset'],r['model'],str(r['classes'])+' classes; A08 not evaluated',r['n'],key,r[key],r['source']])
    print(json.dumps({'out':str(out),'published_files':['index.html','style.css','app.js','data.js','results.json','results.csv'],
        'result_sha256':hashlib.sha256(text.encode()).hexdigest(),'source_sequences':sum(EXPECTED.values())}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=SOURCE);a=ap.parse_args();build(a.out)
