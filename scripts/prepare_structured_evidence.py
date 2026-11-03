#!/usr/bin/env python3
"""A58 metadata/source qualification. No media decode or model forward."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.structured_common import ROOT,RUN,PUB,CFG,read,write,sha,sequences,verify_freeze
from scripts.cache_structured_evidence import build_a08,build_jepa
from echo_routing.temporal.structured_inputs import verify_roles,windows
from echo_routing.compose.recipe_builder import Pool,make_multi_recipes,recipe_to_dict
from echo_routing.evaluate.failure_units import session_of

def source_qualification():
    verify_freeze('preparation_freeze.json');records={}
    for name,build in [('a08',build_a08),('jepa',build_jepa)]:
        model=build('cpu');state=model.state_dict();spec=[(k,list(v.shape),str(v.dtype)) for k,v in state.items()];records[name]={'strict_state_keys':len(state),'parameters':sum(p.numel() for p in model.parameters()),'all_parameters_frozen':all(not p.requires_grad for p in model.parameters()),'state_key_shape_dtype_digest':hashlib.sha256(json.dumps(spec).encode()).hexdigest(),'source_file_sha256':sha(ROOT/read(CFG)['source'][name]['path']),'model_forwards':0};del model
    assert records['a08']['strict_state_keys']==400 and records['jepa']['strict_state_keys']==214
    write(PUB/'source_qualification.json',{'passed':True,'sources':records,'GPU_forwards':0,'new_optimizer_updates':0,'pretrained_publication_parity_claim':False,'probability_trajectory_not_anatomical_hidden_feature':True,'epoch':time.time()});print(json.dumps({'strict_source_qualification_passed':True,'records':records}),flush=True)

def metadata():
    verify_freeze('preparation_freeze.json');cfg=read(CFG);fit,inner=sequences('fit'),sequences('inner_cal');rolecheck=verify_roles({'fit':fit,'inner_cal':inner});old=ROOT/cfg['data']['old_outer_metadata'];assert sha(old)==cfg['data']['old_outer_metadata_sha256'];outer=np.load(old,allow_pickle=True).item();roles={};outer_sources=set();training_sources={r[0] for q in fit+inner for r in q['refs']}
    for role,qs in [('fit',fit),('inner_cal',inner)]:roles[role]={**cfg['data']['roles'][role],'sequences':len(qs),'native':sum(q['kind']=='native' for q in qs),'positions':sum(len(q['refs']) for q in qs),'a08_stride2_windows':sum(len(windows(len(q['refs']),2)['centre']) for q in qs),'jepa_chunks':sum((len(q['refs'])+31)//32 for q in qs),'max_length':max(len(q['refs']) for q in qs)}
    for i,half in enumerate(['selection','calibration']):
        qs=outer[half]['sequences'];native=[q for q in qs if q['kind']=='native'];sources={r[0] for q in qs for r in q['refs']};assert not sources&training_sources;outer_sources|=sources;path=RUN/'metadata'/f'{half}.private.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,np.array(qs,dtype=object),allow_pickle=True);roles[half]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'sequences':len(qs),'positions':sum(len(q['refs']) for q in qs),'a08_stride2_windows':sum(len(windows(len(q['refs']),2)['centre']) for q in qs),'jepa_chunks':sum((len(q['refs'])+31)//32 for q in qs)}
        n_of={q['id']:len(q['refs']) for q in native};by_id={q['id']:q for q in native};pool=Pool.from_index([q['id'] for q in native],[int(q['labels'][0]) for q in native],[len(q['labels']) for q in native]);s=cfg['stress'];recipes=make_multi_recipes(pool,s['streams_per_half'],np.random.default_rng([s['seed'],58,i]),n_fragments=tuple(s['n_fragments']),edit_variant=s['edit_variant'],edit_prob=s['edit_prob'],min_len=s['min_len'],max_len=s['max_len'],reuse=True,p_same_next=s['p_same_next']);assert len(recipes)==120;stress=[];lengths=[]
        for j,r in enumerate(recipes):
            refs=[];labels=[];joins=[]
            for fr in r.fragments:
                if refs:joins.append(len(refs))
                source=by_id[fr.video_id];refs.extend([(v,int(f),.9 if fr.variant=='gamma090' else 1.) for v,f,_ in source['refs'][fr.start:fr.end]]);labels.extend([fr.label]*(fr.end-fr.start));lengths.append(fr.end-fr.start)
            stress.append({'id':f'a58_stress_{half}_{j:04d}','kind':'multi','refs':refs,'labels':np.array(labels,np.int64),'joins':np.array(joins,np.int64)})
        path=RUN/'metadata'/f'stress_{half}.private.npy';np.save(path,np.array(stress,dtype=object),allow_pickle=True);write(RUN/'metadata'/f'stress_{half}_recipes.private.json',[recipe_to_dict(r) for r in recipes]);roles['stress_'+half]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'sequences':len(stress),'positions':sum(len(q['refs']) for q in stress),'a08_stride2_windows':sum(len(windows(len(q['refs']),2)['centre']) for q in stress),'jepa_chunks':sum((len(q['refs'])+31)//32 for q in stress),'realized_fragment_count':len(lengths),'realized_fragment_minmax':[min(lengths),max(lengths)],'realized_fragment_quantiles':np.quantile(lengths,[.1,.5,.9]).tolist(),'requested_conditional_uniform':'4..min(80,source_n)','recipe_sha256':sha(RUN/'metadata'/f'stress_{half}_recipes.private.json'),'source_footprint_outside_half':sum(r[0] not in sources for q in stress for r in q['refs'])}
    plans={};native=np.array([i for i,q in enumerate(fit) if q['kind']=='native']);mixed=np.array([i for i,q in enumerate(fit) if q['kind']!='native'])
    for seed in cfg['training']['seeds']:
        rng=np.random.default_rng([seed,58]);columns=[]
        for pool in [native,mixed]:
            out=[]
            while len(out)<6000:out.extend(rng.permutation(pool).tolist())
            columns.append(out[:6000])
        plan=np.stack(columns,1).astype(np.int64);path=RUN/f'plan_seed{seed}.private.npy';np.save(path,plan);plans[str(seed)]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'native_unique':len(set(plan[:,0])),'constructed_unique':len(set(plan[:,1])),'updates':6000,'physical_sequences':12000}
    # Old FIT-only A31 raw logits are a parity target, never source-patient metadata.
    lookup={q['id']:i for i,q in enumerate(fit) if q['kind']=='native'};candidate=[];rawpins=[]
    for f,want in enumerate(['96d1fb683e1a261d5be9a3cc7ec393637577bbf83b56140cec41f544c4f3b486','08255192dae3a90644ea062880bf6b2b298181ed12e9890d56ca04630f5adef4']):
        poolpath=ROOT/f'runs/video_session_correction_v1/pool_{f}.npy';assert sha(poolpath)==want;pool=np.load(poolpath,allow_pickle=True).tolist();logpath=ROOT/f'runs/video_session_correction_v1/outputs_s0_f{f}.npz';wantlog=['5884435770e8b65959b22ff77f74b0c2053cadf7fd2a5a28c842104df2f1a512','0a99be35b0cd78d650f5014c8ddc783173e6fb849ee556d6fa2a52c73f33d360'][f];assert sha(logpath)==wantlog;rawpins.append({'pool':str(poolpath.relative_to(ROOT)),'pool_sha256':want,'logits':str(logpath.relative_to(ROOT)),'logits_sha256':wantlog})
        with np.load(logpath) as z:
            for j,q in enumerate(pool):
                if q['kind']!='native' or q['id'] not in lookup:continue
                target=fit[lookup[q['id']]];assert q['refs']==target['refs'];ix=np.flatnonzero(z['seq']==j);w=windows(len(q['refs']),5);dense=set(windows(len(q['refs']),2)['centre'].tolist());assert np.array_equal(z['centre'][ix],w['centre'])
                for k,row in enumerate(ix):
                    if int(w['centre'][k]) in dense:candidate.append({'sequence':lookup[q['id']],'centre':int(w['centre'][k]),'marker':int(w['marker'][k]),'expected_logits':z['logits'][row].tolist()})
    panel=[candidate[i] for i in np.linspace(0,len(candidate)-1,16).round().astype(int)];write(RUN/'a08_FIT_parity.private.json',panel)
    oldpins=read(ROOT/cfg['data']['outer_inputs']);diagnostic={'a08':{half:oldpins['A08_cache_pins'][half]['0'] for half in ['selection','calibration']},'jepa':{}}
    for half in ['selection','calibration']:
        folder=ROOT/'runs/a57_dense_sequence_v1/outer_readout'/cfg['source']['jepa']['sha256']/half;diagnostic['jepa'][half]={'path':str(folder.relative_to(ROOT)),'journal_sha256':sha(folder/'committed.jsonl'),'identity_sha256':sha(folder/'identity.json'),'sequences':len(outer[half]['sequences'])}
    record={'passed':True,'roles':roles,'plans':plans,'role_isolation':rolecheck,'FIT_inner_outer_source_overlap':0,'stress_generated_before_formal_model_results':True,'stress_seed_contract':'default_rng([2026110358,58,half_index])','parity_private_sha256':sha(RUN/'a08_FIT_parity.private.json'),'parity_windows':16,'parity_cache_inputs':rawpins,'diagnostic_inputs':diagnostic,'two_column_metadata_only':True,'source_patient_or_quarantine_files_opened':0,'protected_media_forwards':0,'protected_labels_used_for_training_or_selection':False,'A08_historical_inner_exposure_not_OOF':True,'new_GPU_forwards':0,'epoch':time.time()};write(PUB/'input_manifest.json',record);print(json.dumps({'A58_metadata_ready':True,'roles':roles,'plans':plans}),flush=True)

if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True);globals()[sys.argv[1]]()
