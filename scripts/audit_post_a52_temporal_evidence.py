#!/usr/bin/env python3
"""POST-A52-D: immutable metadata/score audit, zero image or model operations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from collections import Counter,defaultdict
import hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from echo_routing.temporal.temporal_evidence_audit import singleton_view,informative,auc_partition,dense_gate
O=ROOT/'runs/post_a52_temporal_evidence_v1';P=ROOT/'docs/results/2026-11-01_post_a52_temporal_evidence'
V=ROOT/'runs/a51_verified_direct_view_pilot_v1';I=ROOT/'runs/a52_point_detail_v1';S=ROOT/'runs/a51_discriminative_evidence_v1/source'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,z):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')

def main():
 start=time.time();dest=O/'tokens.private.json';assert not dest.exists(),'audit already sealed; no repeat/reselection'
 b=read(V/'bound.private.json');sel=read(V/'selection.private.json');meta=read(S/'private_metadata_marker.json');chrono=read(S/'private_chronology.json');p51=ROOT/'docs/results/2026-11-01_direct_view_evidence/pilot';p52=ROOT/'docs/results/2026-11-01_point_detail_a5c'
 assert sha(V/'bound.private.json')==read(p51/'input_manifest.json')['bound_sha256'];assert sha(V/'selection.private.json')==b['selection_sha256'];assert sha(S/'private_metadata_marker.json')==sel['original_metadata_sha256'];assert sha(S/'private_chronology.json')==sel['parent_chronology_sha256'];assert len(b['requests'])==702
 assert [x['id'] for x in b['requests']]==[x['id'] for x in sel['requests']] and len({x['id'] for x in b['requests']})==702
 labels=defaultdict(list)
 for r in meta['records']:labels[r['directory_label'],r['video']].append(r)
 axes={(v['folder'],v['video']):v['native_pts_indices'] for v in chrono['videos']};tokens=[];role_counts={};unique_refs={role:set() for role in ['fit','diagnostic']}
 for r in b['requests']:
  key=r['directory_label'],r['video'];assert r['centre']==r['frames'][8];a=axes[key];pairs=[(int(r['frames'][j]),int(a[r['frames'][j]])) for j in range(0,16,2)];ys=[singleton_view(labels[key],ordinal,pts) for ordinal,pts in pairs];refs=[(key[0],key[1],k[0]) for k in pairs]
  marker=singleton_view(labels[key],r['centre'],a[r['centre']]);assert marker==int(r['true_label']=='A5C')
  flag=informative(ys,refs,marker);tokens.append({'id':r['id'],'patient':r['patient'],'role':r['role'],'video':list(key),'marker_label':marker,'labels':ys,'references':refs,'native_pts':pairs,'informative':flag})
  unique_refs[r['role']].update(refs)
 for role in ['fit','diagnostic']:
  rr=[r for r in tokens if r['role']==role];qualified=[r for r in rr if r['informative']];hist=Counter('UNKNOWN' if y is None else ['A4C','A5C'][y] for r in rr for y in r['labels']);opposite=sum(y is not None and y!=r['marker_label'] for r in rr for y in r['labels']);changes=sum(a is not None and b is not None and a!=b for r in rr for a,b in zip(r['labels'],r['labels'][1:]));unique_changes={(tuple(r['video']),tuple(r['references'][j]),tuple(r['references'][j+1]),r['labels'][j],r['labels'][j+1]) for r in rr for j in range(7) if r['labels'][j] is not None and r['labels'][j+1] is not None and r['labels'][j]!=r['labels'][j+1]};two=[r for r in rr if {y for y in r['labels'] if y is not None}=={0,1}]
  role_counts[role]={'requests_total':len(rr),'tokens_total':8*len(rr),'token_label_counts':dict(hist),'known_nonmarker_label_slots':opposite,'requests_with_both_known_views':len(two),'patients_with_both_known_views':len({r['patient'] for r in two}),'adjacent_known_token_label_change_pairs':changes,'unique_label_change_endpoint_pairs':len(unique_changes),'unique_physical_token_references':len(unique_refs[role]),'informative':{'requests':len(qualified),'patients':len({r['patient'] for r in qualified}),'videos':len({tuple(r['video']) for r in qualified}),'marker_classes':len({r['marker_label'] for r in qualified})}}
 gate=dense_gate({k:v['informative'] for k,v in role_counts.items()});dump(dest,{'source_selection_sha256':sha(V/'selection.private.json'),'bound_sha256':sha(V/'bound.private.json'),'source_metadata_sha256':sha(S/'private_metadata_marker.json'),'chronology_sha256':sha(S/'private_chronology.json'),'tokens':tokens})
 # Exact score order and frozen raw/fit cache identities; all models use manual truth.
 pbound=read(I/'bound.private.json');assert [r['id'] for r in pbound['rows']]==[r['id'] for r in b['requests']];assert sha(I/'bound.private.json')==read(p52/'input_manifest.json')['bound_sha256']
 raw51=np.load(V/'source_scores.private.npz');raw52=np.load(I/'scores.private.npz');assert sha(V/'source_scores.private.npz')==read(p51/'source_readout.json')['raw_score_cache_sha256'];assert sha(I/'scores.private.npz')==read(p52/'readout.json')['score_cache_sha256']
 models={'diagnostic':{**{a:raw51[a] for a in ['G','L','W']},**{a:raw52[a] for a in ['H','R']},'video_geometry':raw51['geometry'],'video_constant':raw51['constant'],'point_geometry':raw52['geometry'],'point_constant':raw52['constant']},'fit':{}}
 for root,arms,manifest in [(V,['L','W'],read(p51/'source_models_manifest.json')),(I,['H','R'],read(p52/'models_manifest.json'))]:
  for arm in arms:
   path=root/'source_train'/arm/'fit_final.npy';assert sha(path)==manifest['models'][arm]['fit_final_sha256'];models['fit'][arm]=np.load(path)
 score_results={}
 for role,cache in models.items():
  rr=[r for r in b['requests'] if r['role']==role];y=np.array([r['true_label']=='A5C' for r in rr]);group_video=[r['directory_label']+'/'+r['video'] for r in rr];group_patient=[r['patient'] for r in rr];score_results[role]={}
  for arm,s in cache.items():
   assert s.shape==(len(rr),) and np.isfinite(s).all();v=auc_partition(y,s,group_video);p=auc_partition(y,s,group_patient);pred=s>0
   score_results[role][arm]={'A4C_false_positive':int((pred&~y).sum()),'A5C_correct':int((pred&y).sum()),'A4C_total':int((~y).sum()),'A5C_total':int(y.sum()),'video':v,'patient':p}
 assert {r['patient'] for r in tokens if r['role']=='fit'}.isdisjoint({r['patient'] for r in tokens if r['role']=='diagnostic'})
 rec={'run_id':'POST-A52-D','cache_only_no_model_image_or_GPU_operations':True,'original_requests':702,'by_role':role_counts,'dense_supervision_qualification':gate,'score_discrimination':score_results,'manual_labels_used_for_every_model_including_W':True,'G_fit_unavailable_not_inferred':True,'private_token_table_sha256':sha(dest),'existing_inputs_sha256':{'bound':sha(V/'bound.private.json'),'selection':sha(V/'selection.private.json'),'source_metadata':sha(S/'private_metadata_marker.json'),'chronology':sha(S/'private_chronology.json'),'A51_scores':sha(V/'source_scores.private.npz'),'A52_scores':sha(I/'scores.private.npz')},'new_training_or_target_readout':0,'descriptive_exploratory_reused_source_diagnostic':True,'no_group_conditioned_threshold_or_normalization':True,'scope':'Eight nominal even temporal positions; endpoint labels only, UNKNOWN is not a view or transition-boundary truth; overlapping requests not independent samples.'}
 dump(P/'audit.json',rec);print(json.dumps({'by_role':role_counts,'dense_gate':gate,'diag_model_AUC':{a:{'global':v['video']['global_AUC'],'within_video':v['video']['within_AUC'],'within_patient':v['patient']['within_AUC'],'within_video_pairs':v['video']['within_pairs']} for a,v in score_results['diagnostic'].items() if a in ['G','L','W','H','R']}}))
 u=resource.getrusage(resource.RUSAGE_SELF);dump(O/'audit_process.json',{'pid':os.getpid(),'wall_seconds':time.time()-start,'CPU_seconds':u.ru_utime+u.ru_stime,'GPU_job_wall_seconds':0})
if __name__=='__main__':main()
