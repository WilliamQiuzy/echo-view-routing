#!/usr/bin/env python3
"""Full authorized protocoltrain union and fresh1500bank; no model forward."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import train_dense_sequence as base,prepare_video_feature_temporal as legacy
from scripts._common import load_cfg
from echo_routing.temporal import dense_sequence as dense,video_sequence_features as refs
from echo_routing.temporal.dense_readout import ReadoutPixels
from echo_routing.compose.recipe_builder import Pool,make_multi_recipes
from echo_routing.evaluate.failure_units import session_of
from scripts.audit_dense_input_exposure import digest
RUN=base.RUN/'full_refit_v1';PUB=base.PUB/'full_refit';CFG=ROOT/'configs/experiments/dense_target_refit_v1.yaml';read,write,sha=base.read,base.write,base.sha

def metadata():
    assert not (PUB/'input_metadata.json').exists(),'existing refit metadata immutable';cfg=read(CFG);fit,cal=base.inputs('fit'),base.inputs('inner_cal');natives=sorted([q for q in fit+cal if q['kind']=='native'],key=lambda q:q['id']);allow={q['id'] for q in natives};protocol=ROOT/'runs/20261002-051140-model_development_protocol-fa45e7f4-s0/protocol.json';assert allow==set(read(protocol)['train']['clips']) and len(allow)==3364 and len({session_of(v) for v in allow})==502
    # Outer membership is checked without any outer label or media use.
    outer=np.load(base.RUN/'outer_metadata.private.npy',allow_pickle=True).item();outer_ids={r[0] for h in outer.values() for q in h['sequences'] for r in q['refs']};assert not allow&outer_ids
    name='configs/experiments/video_feature_temporal_v1.yaml';ctx=legacy.Ctx(load_cfg(SimpleNamespace(config=name,set=[])),name);rc,mc=ctx.cfg['recipes'],ctx.cfg.get('multi',{});pool=Pool.from_index([q['id'] for q in natives],[int(q['labels'][0]) for q in natives],[len(q['labels']) for q in natives]);recipes=make_multi_recipes(pool,1500,np.random.default_rng(cfg['recipe_seed']),n_fragments=tuple(mc.get('n_fragments',(4,8))),edit_variant=rc['edit_variant'],edit_prob=float(mc.get('edit_prob',.5)),min_len=rc['min_len'],max_len=rc['max_len'],reuse=True,p_same_next=float(mc.get('p_same_next',.35)));frames={q['id']:np.asarray([r[1] for r in q['refs']],np.int64) for q in natives};seqs=list(natives)
    for i,r in enumerate(recipes):
        labels,joins=refs.stream_labels_joins(r.fragments);seqs.append({'id':f'a57_fullrefit_{i:04d}','kind':'mixed','refs':refs.stream_refs(r.fragments,lambda v:frames[v]),'labels':labels,'joins':joins})
    assert all(len(q['refs'])==len(q['labels']) and {r[0] for r in q['refs']}<=allow for q in seqs);plan=base.make_plan(seqs,cfg['seed']);valid=0;exposed=set();class_valid=np.zeros(5,np.int64)
    for u in range(6000):
        exposed.add(seqs[int(plan['sequence'][u,0])]['id'])
        for i,start in zip(plan['sequence'][u],plan['start'][u]):
            q=seqs[int(i)];ix,mask=dense.chunk_positions(len(q['labels']),int(start));assert {q['refs'][k][0] for k in ix}<=allow;valid+=int(mask.sum());class_valid+=np.bincount(q['labels'][ix][mask],minlength=5)
    assert exposed==allow;np.save(RUN/'train_sequences.private.npy',np.array(seqs,dtype=object),allow_pickle=True);np.savez(RUN/'plan.private.npz',**plan)
    parent=read(base.PUB/'models_manifest.json');row=next(x for x in parent['outer_rows'] if x['row']=='seed2026110257_J-high_fixed6000');oldchoices=read(base.PUB/'decoder_choices.json');transfer={'parent_model_sha256':row['sha256'],'parent_decoder_choices_sha256':sha(base.PUB/'decoder_choices.json'),'params':oldchoices['choices_by_checkpoint_sha256'][row['sha256']]['params'],'scope':'unchanged parent innercalibrated numerical parameters transferred;notrefitcalibrated;former628innernowTRAINING'};write(PUB/'B3_transfer.json',transfer)
    record={'passed':True,'native_clips':3364,'sessions':502,'native_class_counts':np.bincount([int(q['labels'][0]) for q in natives],minlength=5).tolist(),'constructed_streams':1500,'recipe_seed':cfg['recipe_seed'],'native_unique_sources_exposed_in6000':len(exposed),'whole64footprints_checked':12000,'authorized_source_violations':0,'outer_source_overlap':0,'former628_inner_role':'TRAINING;noinnerselection','sequence_sha256':sha(RUN/'train_sequences.private.npy'),'plan_file_sha256':sha(RUN/'plan.private.npz'),'plan_canonical_sha256':digest(plan),'valid_original_positions':valid,'padded_original_positions':6000*128-valid,'training_positions_by_class':class_valid.tolist(),'protocol_sha256':sha(protocol),'B3_transfer_sha256':sha(PUB/'B3_transfer.json'),'new_model_forwards':0};write(PUB/'input_metadata.json',record);print(json.dumps(record),flush=True)

def smoke_pixels():
    meta=read(PUB/'input_metadata.json');seqs=np.load(RUN/'train_sequences.private.npy',allow_pickle=True).tolist();assert sha(RUN/'train_sequences.private.npy')==meta['sequence_sha256'];plan=dict(np.load(RUN/'plan.private.npz'));assert digest(plan)==meta['plan_canonical_sha256'];p=ReadoutPixels(seqs);batch=p.batch([seqs[int(i)] for i in plan['sequence'][0]],plan['start'][0],'cpu');np.savez(RUN/'FIT_smoke.private.npz',**{k:t.numpy() for k,t in zip(['frames','labels','mask'],batch)});write(PUB/'smoke_input.json',{'sha256':sha(RUN/'FIT_smoke.private.npz'),'shape':[2,64,224,224,3],'source':'firstfixedfulltrainplan;formerinneristraining','new_model_forwards':0})

if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True);start=time.time();ok=False;mode=sys.argv[1]
    try:globals()[mode]();ok=True
    finally:
        r=resource.getrusage(resource.RUSAGE_SELF);receipt={'stage':'refit_prepare_'+mode,'model_pid':os.getpid(),'ok':ok,'start_epoch':start,'end_epoch':time.time(),'process_CPU_seconds':r.ru_utime+r.ru_stime,'full_process_wall_seconds':time.time()-start,'GPU_job':False};write(RUN/(mode+'_process.json'),receipt)
        with (base.RUN/'auxiliary_CPU_receipts.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
