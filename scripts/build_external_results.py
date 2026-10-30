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
    labels={'dynamic':'A4C','lvh':'PLAX','camus':'4CH码 → A4C'}
    notes={'dynamic':'官方TEST；患者关系未知','lvh':'官方test；按录像去重','camus':'官方testing；半周期0.18–0.51秒；4CH码未经切面复核'}
    for d in ['dynamic','lvh','camus']:
        rows=[native[d][f'frozen_B3_s{s}'] for s in range(3)];u=unc['datasets'][d]
        ext.append({'name':names[d],'label':labels[d],'n':EXPECTED[d],'mean_recall':mean([r['accuracy'] for r in rows]),
            'ci95':u['mean_B3']['ci95'],'correct':[u['models'][f'frozen_B3_s{s}']['correct'] for s in range(3)],
            'cuts_per_clip':mean([r['internal_cuts_per_clip'] for r in rows]),'note':notes[d]})
        for dec in ['frozen_B3','argmax']:
            for s in range(3):
                r=native[d][f'{dec}_s{s}'];seedrows.append({'dataset':names[d],'decoder':'固定B3' if dec=='frozen_B3' else '直接argmax',
                    'seed':s,'recall':r['accuracy'],'cuts_per_clip':r['internal_cuts_per_clip']})
    ours=[native['ev9v'][f'frozen_B3_s{s}'] for s in range(3)];u=unc['datasets']['ev9v']
    original=[{'name':'视频Transformer＋固定B3（A08）','ours':True,'seed':'0 / 1 / 2 均值','n':800,
        'accuracy':mean([r['accuracy'] for r in ours]),'macro_f1':mean([r['five_class_macro_f1'] for r in ours]),
        'macro_ci95':u['mean_B3']['macro_ci95'],'a4c':mean([r['class_recall']['A4C'] for r in ours]),'a5c':mean([r['class_recall']['A5C'] for r in ours])}]
    for s,r in enumerate(ours):
        original.append({'name':'A08＋固定B3','ours':True,'seed':str(s),'n':800,'accuracy':r['accuracy'],
            'macro_f1':r['five_class_macro_f1'],'macro_ci95':u['models'][f'frozen_B3_s{s}']['macro_f1_ci95'],
            'a4c':r['class_recall']['A4C'],'a5c':r['class_recall']['A5C']})
    details=[]
    for key,title in [('legacy_per_key_prior','STFM 原生规则（旧冻结权重）'),('corrected_prior_once_same_old_weights','STFM 先验加一次（同一旧权重）')]:
        r=stfm[key]['shared_five_family'];ci=unc['baseline'][key]
        original.append({'name':title,'ours':False,'seed':'100','n':800,'accuracy':r['accuracy'],'macro_f1':r['five_class_macro_f1'],
            'macro_ci95':ci['macro_f1_ci95'],'a4c':r['class_recall']['A4C'],'a5c':r['class_recall']['A5C']})
        r=stfm[key]['original_nine_code'];details.append({'name':title,'ours':False,'population':'九码 / 888','accuracy':r['accuracy'],'macro_f1':r['macro_f1']})
    for s in range(3):
        r=native['ev9v'][f'argmax_s{s}'];details.append({'name':f'A08 直接argmax，s{s}','ours':True,'population':'五类 / 800','accuracy':r['accuracy'],'macro_f1':r['five_class_macro_f1']})
    old=json.loads((ROOT/'docs/results/2026-10-23_frozen_external_benchmark/evaluation_v1.json').read_text())['values']
    historical=[]
    groups=[(['A08_s0','A08_s2'],'A08＋B3（s0/s2均值）',True),(['EchoViewCLIP_stage1_windowed'],'EchoViewCLIP stage1',False),
        (['STFM_numeric_s100'],'STFM 数值帧序 s100',False),(['ASFormer_v2_frameorder'],'ASFormer（EV9V适应）',False),
        (['EchoPrime_view_B3'],'EchoPrime＋B3',False),(['EchoPrime_view_per_frame'],'EchoPrime 逐帧',False)]
    keys={'native':'clip macro-F1','pairs':'segment F1@50, pairs','multi':'segment F1@50, multi',
        'fp':'multi FP segments','fp_short':'multi FP segments <= 16','fn':'multi FN segments','fn_short':'multi missed true segments <= 16'}
    for ids,title,isours in groups:
        historical.append({'name':title,'ours':isours,**{k:mean([old[i][v]['value'] for i in ids]) for k,v in keys.items()}})
    paired=[{'name':title,**unc['paired_original'][key]['mean_B3']} for key,title in
        [('legacy_per_key_prior','STFM 原生规则'),('corrected_prior_once_same_old_weights','STFM 先验加一次（同旧权重）')]]
    prep=read('preparation.json');short=', '.join(v['sha256'][:12] for v in prep['weights'].values())
    return {'summary':'A08的3个冻结种子已完成2,470条来源序列。可取得且兼容的原始切面test来自STFM的EV9V；另完成3个外部来源检查。尚无新的独立A4C/A5C cine测试证据。',
        'external':ext,'seed_rows':seedrows,'original':original,'original_details':details,'history':historical,
        'availability':read('availability.json'),'paired_original':paired,'resources':read('resources.json'),'model_identity':f'A08全相邻平滑视频Transformer＋冻结B3；s0/s1/s2；权重SHA前缀：{short}',
        'verification_note':'逐例缓存已独立重建概率、完整标签序列、计数和指标；开发集输出复现、原始帧率/窗口/标记核查通过。95%区间为固定模型下分组重采样的描述性区间；患者独立性和上游预训练重叠仍有限制。',
        'verification':verify,'uncertainty':unc,'native_results':native,'stfm_original_results':stfm,
        'source_hashes':{name:hashlib.sha256((RESULT/name).read_bytes()).hexdigest() for name in
            ['native_results.json','stfm_native_results.json','verification.json','uncertainty.json','availability.json','input_audit.json','resources.json']}}


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
        w=csv.writer(f);w.writerow(['table','dataset_or_model','condition','n','accuracy_or_recall','macro_f1'])
        for r in data['external']:w.writerow(['external',r['name'],'A08 fixed B3 three-seed mean',r['n'],r['mean_recall'],''])
        for r in data['original']:w.writerow(['original_shared',r['name'],r['seed'],r['n'],r['accuracy'],r['macro_f1']])
        for r in data['original_details']:w.writerow(['original_detail',r['name'],r['population'],'',r['accuracy'],r['macro_f1']])
    print(json.dumps({'out':str(out),'published_files':['index.html','style.css','app.js','data.js','results.json','results.csv'],
        'result_sha256':hashlib.sha256(text.encode()).hexdigest(),'source_sequences':sum(EXPECTED.values())}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=SOURCE);a=ap.parse_args();build(a.out)
