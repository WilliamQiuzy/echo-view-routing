import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib,time,resource,collections
from pathlib import Path
import numpy as np
ROOT=Path('/home/william/echo-view-routing');sys.path.insert(0,str(ROOT))
from echo_routing.temporal.proposal_actions import propose,partition,apply
from echo_routing.temporal.safety_targets import effects,position_labels
from echo_routing.temporal.safety_runtime import policies
from echo_routing.temporal.safety_features import utility_features,position_features
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments
from echo_routing.temporal.baselines.registry import decode_with_prob
P=ROOT/'docs/results/2026-11-04_action_safety_ranking'
START=time.time();pins={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):pins[str(Path(p).relative_to(ROOT))]=sha(p);return json.loads(Path(p).read_text())
outer=read(P/'outer_metrics.json');inp=read(P/'input_manifest.json');tr=read(P/'training_inputs.json');cfg=read(ROOT/'configs/experiments/action_safety_ranking_v1.json');fresh=read(P/'cache_outer_manifest.json');norm=json.loads((ROOT/tr['normalizers']['path']).read_text());fit=tr['F_fit'];classes=cfg['channel_mapping']['classes']
def loadq(role):
 row=inp['roles'][role];f=ROOT/row['path'];assert sha(f)==row['sha256'];pins[row['path']]=row['sha256'];return np.load(f,allow_pickle=True).tolist()
def payload(folder,index,journal=None):
 folder=ROOT/folder
 if journal is None:journal=[json.loads(x) for x in (folder/'committed.jsonl').read_text().splitlines()]
 f=folder/f'sequence{index:06d}.private.npz';assert sha(f)==journal[index]['sha256'];pins[str(f.relative_to(ROOT))]=journal[index]['sha256'];return np.load(f,allow_pickle=False)
def sources(role,qs):
 result={}
 for s in ['a08','jepa']:
  row=inp['source_cache_roles'].get(role,{}).get(s)
  if row is None:row=next(x for x in fresh['records'] if x['role']==role and x['source']==s)
  folder=ROOT/row['path'];assert sha(folder/'committed.jsonl')==row['journal_sha256'];pins[str((folder/'committed.jsonl').relative_to(ROOT))]=row['journal_sha256'];journal=[json.loads(x) for x in (folder/'committed.jsonl').read_text().splitlines()]
  result[s]=(row['path'],journal)
 return result

def scorecache(row,index):
 info=row['policy_cache'];folder=ROOT/info['path'];assert sha(folder/'committed.jsonl')==info['journal_sha256'];pins[str((folder/'committed.jsonl').relative_to(ROOT))]=info['journal_sha256'];z=payload(info['path'],index);return {k:z[k].copy() for k in z.files}
def qstats(v):
 a=np.asarray(v,float);return None if not a.size else {'min':float(a.min()),'q25':float(np.quantile(a,.25)),'median':float(np.median(a)),'q75':float(np.quantile(a,.75)),'max':float(a.max()),'mean':float(a.mean())}
def matchinfo(b,a,y):
 e=apply(b,a);t=labels_to_segments(y);before=set(match_segments(labels_to_segments(b),t));after=set(match_segments(labels_to_segments(e),t));rec=after-before;lost=before-after
 return rec,lost,{ 'recovered_true_units':len(rec),'lost_true_units':len(lost),'recovered_short16':sum(t[k].length<=16 for k in rec),'lost_short16':sum(t[k].length<=16 for k in lost),'dFN16':sum(t[k].length<=16 for k in lost)-sum(t[k].length<=16 for k in rec),'short16_recovered_lengths':[t[k].length for k in sorted(rec) if t[k].length<=16],'short16_lost_lengths':[t[k].length for k in sorted(lost) if t[k].length<=16]}
def summary(b,a,y,score,tau):
 e=apply(b,a);ch=e!=b;h=ch&(b==y)&(e!=y);r=ch&(b!=y)&(e==y);neutral=ch&~(h|r);co=effects(b,a,y,fit);rec,lost,m=matchinfo(b,a,y)
 def flows(mask):return [{'truth':classes[t],'Base':classes[v],'edited':classes[w],'positions':n} for (t,v,w),n in sorted(collections.Counter(zip(y[mask].tolist(),b[mask].tolist(),e[mask].tolist())).items())]
 tb=np.flatnonzero(y[1:]!=y[:-1])+1;pb=np.flatnonzero(b[1:]!=b[:-1])+1;eb=np.flatnonzero(e[1:]!=e[:-1])+1
 return {'kind':'both' if a.old and a.jepa else 'old_raw_disagreement' if a.old else 'JEPA_run','interval_length':a.end-a.start,'changed_K':int(ch.sum()),'neutral_changed':int(neutral.sum()),'T':len(b),'target_class_distribution':{classes[int(k)]:int(n) for k,n in zip(*np.unique(e[ch],return_counts=True))},'H_flows':flows(h),'R_flows':flows(r),'neutral_flows':flows(neutral),'true_transitions_inside_interval':int(((tb>a.start)&(tb<a.end)).sum()),'true_transition_on_interval_endpoint':bool(a.start in tb or a.end in tb),'Base_cuts_inside':int(((pb>a.start)&(pb<a.end)).sum()),'edited_cuts_inside':int(((eb>a.start)&(eb<a.end)).sum()),**co,**m,'predicted_utility':float(score[0]),'predicted_H':float(score[1]),'predicted_R':float(score[2]),'predicted_neutral':float(score[3]),'pFP':float(score[4]),'r':float(score[5]),'tau':float(tau),'gate_margins':{'positive_utility':float(score[0]),'N_predicted_R_minus_H':float(score[2]-score[1]),'N_half_minus_pFP_strict':float(.5-score[4]),'C_tau_minus_r_strict':float(tau-score[5])},'pH_distribution_not_cached_no_new_forward':True,'mean_pH_from_cached_sum':float(score[1]/ch.sum()),'max_pH_lower_bound':float(score[1]/ch.sum()),'max_pH_upper_bound_from_r':float(score[5])}

private=[];public=[];roles=['a60_confirmation_selection','a61_confirmation_calibration']
for role in roles:
 qs=loadq(role);src=sources(role,qs);table='a60_confirmation' if role.startswith('a60') else 'a61_confirmation';half='selection' if role.endswith('selection') else 'calibration';primary=[r for r in outer['rows'] if r['table']==table and r['half']==half and r.get('arm')=='S-L' and r.get('policy')=='C' and (r['seed']==2026110461 or table=='a61_confirmation')]
 bank={}
 for ix,q in enumerate(qs):
  pa=payload(src['a08'][0],ix,src['a08'][1])['prob'];pj=payload(src['jepa'][0],ix,src['jepa'][1])['prob'];b=decode_with_prob('b3',pa,{'decode':{'b3':cfg['source']['a08']['original_B3']}})[0];y=np.asarray(q['labels']);bank[ix]=(pa,pj,b,y)
 for row in primary:
  for ix,q in enumerate(qs):
   z=scorecache(row,ix);pa,pj,b,y=bank[ix];pred=z['labels_C'];H=int(((b==y)&(pred!=y)).sum())
   if not H:continue
   acts,rej=partition(b,propose(b,pa.argmax(1),pj.argmax(1)));sm=z['score_matrix'];v=policies(b,acts,sm,float(z['tau']));chosen=v['C']['accepted_index'];assert chosen is not None and np.array_equal(v['C']['prediction'],pred)
   case='Case A' if table=='a60_confirmation' else 'Case B';a=acts[chosen];cost=[effects(b,ac,y,fit) for ac in acts];rec,lost,_=matchinfo(b,a,y);true=labels_to_segments(y);shortrec={k for k in rec if true[k].length<=16};safe=[i for i,c in enumerate(cost) if c['safe']];benef=[i for i in safe if cost[i]['u']>0]
   shared=[i for i in safe if shortrec & matchinfo(b,acts[i],y)[0]]
   selected=summary(b,a,y,sm[chosen],float(z['tau']));fx=utility_features(pa,pj,b,a);px=position_features(pa,pj,b,a);labs=position_labels(b,a,y);e=apply(b,a);ch=np.flatnonzero(e!=b)
   def pp(prob,ind):return {'A08_proposed_minus_Base':qstats(pa[ch[ind],e[ch[ind]]]-pa[ch[ind],b[ch[ind]]]),'JEPA_proposed_minus_Base':qstats(pj[ch[ind],e[ch[ind]]]-pj[ch[ind],b[ch[ind]]]),'A08_max_prob':qstats(pa[ch[ind]].max(1)),'JEPA_max_prob':qstats(pj[ch[ind]].max(1))}
   features={'position47_context':'current A/J posterior, Base/edited class, mean16-neighbor posterior per side, has-flank, logT/L/K and coordinates; no explicit true-transition or frame-correctness observable','utility109_context':'pooled action/flank posterior statistics and occupancy/geometry, not positionwise labels','FIT_zscore_abs_utility':qstats(np.abs((fx-np.asarray(norm['candidate']['mean']))/np.asarray(norm['candidate']['std']))),'FIT_zscore_abs_position':qstats(np.abs((px-np.asarray(norm['position']['mean']))/np.asarray(norm['position']['std']))),'H_positions_posterior_evidence':pp(pa,labs==0),'R_positions_posterior_evidence':pp(pa,labs==1),'risk_position_probability_distribution_available':False,'reason':'score cache retains aggregate sums H/R/neutral, pFP and max evidence only; no position logits/probability array; no forbidden recompute'}
   armrows=[]
   for ar in [r for r in outer['rows'] if r['table']==table and r['half']==half and r.get('role')=='policy' and r.get('policy')=='C']:
    az=scorecache(ar,ix);sc=az['score_matrix'];av=policies(b,acts,sc,float(az['tau']));target='v' if ar['arm'].startswith('S') else 'u';trueorder=sorted(range(len(acts)),key=lambda i:(-cost[i][target],acts[i].key));toptrue=next((i for i in trueorder if cost[i][target]>0),None)
    def outcome(i):return {'candidate':None if i is None else i,'noop':i is None,'cost':None if i is None else cost[i]}
    def existing_gate(i):return i is not None and sc[i,1]-sc[i,2]<=0 and sc[i,4]<.5 and sc[i,5]<float(az['tau'])
    top=av['C']['top_index'];orig={p:outcome(av[p]['accepted_index']) for p in ['E','N','C']}
    safe_ranking=sorted(safe,key=lambda i:(-sc[i,0],acts[i].key));safe_rows=[{'candidate':i,'u':cost[i]['u'],'v':cost[i]['v'],'predicted_utility':float(sc[i,0]),'predicted_rank_including_noop':1+sum(sc[:,0]>sc[i,0])+int(0>sc[i,0]),'positive':bool(sc[i,0]>0),'N_pass':bool(sc[i,1]<=sc[i,2] and sc[i,4]<.5),'C_pass':bool(existing_gate(i)),'is_original_top':top==i,'recovers_same_selected_short_unit':bool(shortrec & matchinfo(b,acts[i],y)[0])} for i in safe_ranking]
    armrows.append({'seed':ar['seed'],'arm':ar['arm'],'tau':float(az['tau']),'top':outcome(top),'E_N_C':orig,'selected_unsafe_candidate_score':sc[chosen].tolist(),'safe_alternatives':safe_rows,'counterfactual_labels_posthoc_not_deployable':{'true_risk_veto_original_top_no_fallback':outcome(top if top is not None and cost[top]['safe'] else None),'true_arm_target_ranking_with_existing_N_C_no_fallback':outcome(toptrue if existing_gate(toptrue) else None),'true_target_ranking_and_true_risk_no_fallback':outcome(toptrue if toptrue is not None and cost[toptrue]['safe'] else None)},'true_arm_target_ranking_top':outcome(toptrue)})
   entry={'case':case,'seed':row['seed'],'source_role':role,'kind':q['kind'],'selected':selected,'eligible_candidates':len(acts),'G_rejected_candidates':len(rej),'true_safe_candidates':len(safe),'true_safe_beneficial_u_positive_candidates':len(benef),'safe_alternative_recovers_any_same_recovered_short_unit':len(shared)>0,'selected_short16_recovered_unit_count':len(shortrec),'safe_alternative_count_same_short':len(shared),'features_and_posterior_evidence':features,'all8_matched_arms':armrows,'evidence':{'primary_policy_payload_sha256':sha(ROOT/row['policy_cache']['path']/f'sequence{ix:06d}.private.npz'),'role_metadata_sha256':inp['roles'][role]['sha256']}}
   public.append(entry);private.append({**entry,'sequence_id':q['id'],'sequence_index':ix,'action_key':a.key,'actions':[{'start':ac.start,'end':ac.end,'replacement':ac.replacement,'old':ac.old,'jepa':ac.jepa} for ac in acts],'truth':y.tolist(),'Base':b.tolist(),'A08':pa.tolist(),'JEPA':pj.tolist(),'all_candidate_effects':cost,'score_matrix':sm.tolist(),'utility109':fx.tolist(),'position47':px.tolist()})
assert len(public)==3,('expected 3 unsafe accepted actions',len(public))
B=[r for r in private if r['case']=='Case B'];sameB=B[0]['sequence_id']==B[1]['sequence_id'];sameaction=B[0]['action_key']==B[1]['action_key'];assert sorted([r['selected']['H'] for r in public])==[9,9,40]
common={'postclosure_forensic_only':True,'A62_started':False,'new_NN_forwards':0,'optimizer_updates':0,'thresholds_changed':False,'original_A61_files_modified':False,'class_order':classes,'Case_B_same_axis_across_seeds':sameB,'Case_B_same_action_across_seeds':sameaction,'Case_A_H40_is_single_action':sum(r['case']=='Case A' for r in private)==1,'input_sha256':pins,'script_sha256':sha('/tmp/a61_forensic_postclosure_v1.py'),'FIT_summary':{k:tr[k] for k in ['F_fit','target_ranges','kind_candidate_counts','kind_nonempty_query_counts','max_C','max_K','utility_features_nonconstant','position_features_nonconstant']},'scope':'selected existing unsafe accepted actions only; no full metric rerun; labels used only for retrospective oracle diagnosis; no expert frame visibility'}
Path('/tmp/a61_forensic_postclosure_v1.private.json').write_text(json.dumps({**common,'cases':private},allow_nan=False))
s=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
report={**common,'cases':public,'private_detailed_evidence_sha256':sha('/tmp/a61_forensic_postclosure_v1.private.json'),'CPU_SELF_plus_waitedchildren_seconds_through_sample':s.ru_utime+s.ru_stime+c.ru_utime+c.ru_stime,'wall_seconds_through_sample':time.time()-START,'SELF_observed_OS_exit':None,'final_write_exit_tail_unmeasured':True}
# Paths/pins are server provenance only and contain no case IDs; journal/payload paths contain no source patient IDs.
Path('/tmp/a61_forensic_postclosure_v1.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'cases':len(public),'H':[r['selected']['H'] for r in public],'CPU_seconds':report['CPU_SELF_plus_waitedchildren_seconds_through_sample'],'wall':report['wall_seconds_through_sample'],'public_sha256':sha('/tmp/a61_forensic_postclosure_v1.public.json'),'same_B_axis':sameB,'same_B_action':sameaction}),flush=True)
