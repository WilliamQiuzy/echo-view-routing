#!/usr/bin/env python3
"""Independent synthetic full-cache IO measurement; never scientific output."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCE='scripts/a63_probe_completed_supply_selected_risk_cost.py'
WAITER='scripts/a63_wait_completed_supply_selected_risk_cost.py'
ROLES=('F0_native','F0_mixed','F1_native','F1_mixed')
SEEDS=(2026110461,2026110462)
PURPOSE='cost_only_synthetic_nonadoptable'
LIMITS={'CPU_seconds':600,'wall_seconds':600,'temp_bytes':80000000}


CUDA_MARKER='A63_COST_CUDA_OBSERVATION '
def cuda_observation():
 module=sys.modules.get('torch')
 try:initialized=False if module is None else bool(module.cuda.is_initialized());known=True
 except BaseException:initialized=None;known=False
 value={'known':known,'torch_imported':module is not None,'initialized':initialized,'scope':'CUDA hidden before project imports; Torch module context inspected if present; no CUDA library or model is instantiated by this source'}
 if not known or initialized:
  print(CUDA_MARKER+json.dumps(value),file=sys.stderr,flush=True)
  raise ValueError('unexpected or unknown CUDA context forbids CPU-only cost work')
 return value


def pin(root,path):
 p=Path(path);return {'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def files(root):
 root=Path(root);out=[]
 if not root.exists():return out
 for p in sorted(root.rglob('*')):
  if p.is_symlink():raise ValueError('cost-only payload symlink refused')
  if p.is_file():out.append(p)
  elif not p.is_dir():raise ValueError('ordinary cost-only payload files/directories required')
 return out


def file_manifest(folder):
 return [{'path':str(p.relative_to(folder)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files(folder)]


def cleanup_owned(folder,manifest):
 folder=Path(folder)
 if folder.is_symlink() or not (folder.name.startswith('cost_only_synthetic') or folder.name=='payloads' and folder.parent.parent.name=='cost_only_synthetic_cache_probe'):raise ValueError('exclusive synthetic payload root required')
 observed=file_manifest(folder)
 if observed!=manifest:raise ValueError('owned-only immutable cleanup manifest differs; preserve all files')
 for row in manifest:(folder/row['path']).unlink()
 for p in sorted(folder.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  if p.is_dir():p.rmdir()
 folder.rmdir()
 return {'namespace':PURPOSE,'owned_payload_root':str(folder),'before_regular_bytes':sum(r['bytes'] for r in manifest),'after_regular_bytes':0,'fully_removed':not folder.exists(),'owned_only':True}


def _value(actions,b,positive):
 import numpy as np
 from echo_routing.temporal.proposal_actions import apply
 rows=[]
 for a in actions:
  k=int(np.count_nonzero(apply(b,a)!=b))
  if not k:raise ValueError('admitted action must change positions')
  rows.append([1. if positive else 0.,.2*k,.7*k,.1*k,.1,.2])
 c6=np.asarray(rows,dtype=np.float64).reshape(len(actions),6)
 return {'D_C6':c6.copy(),'X_C6':c6.copy(),'x109':np.zeros((len(actions),109),np.float64),'context96':np.zeros((len(actions),96),np.float64)}


def build_pool(folder,banks,identity):
 from echo_routing.temporal.selected_risk_compact_cache import CompactSupplyCache,payload_upper
 from echo_routing.temporal.proposal_actions import propose,partition
 from echo_routing.temporal.selected_risk_resume_contract import digest
 from echo_routing.temporal.anchor_cache import sha
 folder=Path(folder)
 if folder.exists() or folder.is_symlink():raise FileExistsError('fresh nonadoptable synthetic pool only')
 folder.mkdir(parents=True);first={};records=[];axes=admitted=allg=bound=0;role_counts={}
 for role,(qs,aa,jj,bb) in banks.items():
  axes+=len(qs);role_counts[role]=0
  for a,j,b in zip(aa,jj,bb):
   acts=propose(b,a.argmax(1),j.argmax(1));g=partition(b,acts)[0];admitted+=len(g);allg+=len(acts);role_counts[role]+=len(g)
   bound+=payload_upper(len(g),reference=False)+payload_upper(len(g),reference=True)
 for seed in SEEDS:
  for role,(qs,aa,jj,bb) in banks.items():
   raw_a=[a.argmax(1) for a in aa];raw_j=[j.argmax(1) for j in jj]
   ident=digest({'purpose':PURPOSE,'bindings':identity,'seed':seed,'role':role})
   rel=f'seed{seed}/{role}';c=CompactSupplyCache(folder/rel,ident,bb,raw_a,raw_j,feature_reference=first.get(role))
   for i in range(len(qs)):
    acts=partition(bb[i],propose(bb[i],raw_a[i],raw_j[i]))[0]
    c.run_one(i,lambda acts=acts,i=i:_value(acts,bb[i],seed==SEEDS[1]))
   records.append({'seed':seed,'role':role,'relative_path':rel,'identity':ident,'journal_sha256':sha(c.journal),'axes':c.cursor,'admitted_G':role_counts[role],'positive_u':seed==SEEDS[1],'feature_reference_seed':None if seed==SEEDS[0] else SEEDS[0]})
   if seed==SEEDS[0]:first[role]=CompactSupplyCache(folder/rel,ident,bb,raw_a,raw_j,readonly=True)
 return {'schema':'A63_cost_only_cache_manifest_v1','purpose':PURPOSE,'axes':axes,'admitted_G':admitted,'ALL_G':allg,'seeds':list(SEEDS),'cache_records':len(records),'records':records,'structural_cache_payload_upper_bytes':bound}


def verify_pool(folder,banks,manifest,*,check_synthetic=False):
 import numpy as np
 from echo_routing.temporal.selected_risk_compact_cache import CompactSupplyCache,payload_upper
 from echo_routing.temporal.anchor_cache import sha
 if manifest['purpose']!=PURPOSE or manifest['seeds']!=list(SEEDS):raise ValueError('nonadoptable synthetic identity required')
 first={};count=0
 for seed in SEEDS:
  for role,(qs,aa,jj,bb) in banks.items():
   row=next(r for r in manifest['records'] if (r['seed'],r['role'])==(seed,role))
   if row['relative_path']!=f'seed{seed}/{role}':raise ValueError('canonical isolated synthetic cache path required')
   c=CompactSupplyCache(Path(folder)/row['relative_path'],row['identity'],bb,[a.argmax(1) for a in aa],[j.argmax(1) for j in jj],readonly=True,feature_reference=first.get(role))
   if c.cursor!=len(qs) or c.pending.exists() or sha(c.journal)!=row['journal_sha256']:raise ValueError('complete readonly cache/journal/noPending required')
   if check_synthetic:
    for i in range(c.cursor):
     v=c.load(i);expected=_value(v['actions'],bb[i],seed==SEEDS[1])
     if any(v[k].dtype!=a.dtype or v[k].shape!=a.shape or v[k].tobytes()!=a.tobytes() for k,a in expected.items()):raise ValueError('exact synthetic mass/zero-feature/reference bit mismatch')
     if c.file(i).stat().st_size>payload_upper(len(v['actions']),reference=seed==SEEDS[1]):raise ValueError('same4096 header bound exceeded')
     with zipfile.ZipFile(c.file(i)) as z:
      if any(info.compress_type!=zipfile.ZIP_STORED for info in z.infolist()):raise ValueError('original uncompressed NPZ_STORED layout required')
     count+=1
   else:count+=c.cursor
   if seed==SEEDS[0]:first[role]=c
 return {'complete':True,'cache_records':len(manifest['records']),'verified_payloads':count,'read_only_cache_payloads_verified':True,'canonical_actions_verified':True,'reference_raw_features_bitwise_verified':True,'no_pending':True,**({'NPZ_STORED':True} if check_synthetic else {})}


def setup(root,spec_pin,owner_sha,*,verify_contract=True):
 from echo_routing.temporal.selected_risk_resume_context import Context
 from echo_routing.temporal.selected_risk_resume_runtime import bootstrap
 from echo_routing.temporal.selected_risk_resume_contract import RUN,digest
 from echo_routing.temporal.selected_risk_contracts import source_closure
 from echo_routing.temporal.safety_git import verify
 from scripts.a63_resume_completed_supply_selected_risk import EXTRA_SOURCES
 bare=object.__new__(Context);bare.root=Path(root).resolve();spec=bare.read(spec_pin)
 if spec.get('schema')!='A63_full_cache_verifier_cost_probe_spec_v1' or spec.get('limits')!=LIMITS or spec.get('purpose')!=PURPOSE or spec.get('runtime_authorization_sha256')!=owner_sha:raise ValueError('exact separately raw-pinned cost-only limits/authority required')
 nonce=spec['nonce']
 if not isinstance(nonce,str) or len(nonce)!=32 or any(c not in '0123456789abcdef' for c in nonce) or spec['output_prefix']!=RUN+'/cost_only_synthetic_cache_probe/'+nonce:raise ValueError('new exclusive cost-only synthetic namespace required')
 ctx,source=bootstrap(root,spec['source_runtime_spec_pin'],owner_sha)
 old=ctx.verify_stage(source['stage'],source['freeze'],required=set(EXTRA_SOURCES))
 if spec['geometry_pin']!=source['predecessors']['observe_geometry']['payload_pin'] or spec['input_manifest_pin']!=source['metadata'] or spec['core_pins']!={k:source['arguments']['core'][k] for k in ['models_manifest','audit','training_inputs']}:raise ValueError('same original source/input/geometry/core required')
 if verify_contract:
  c=ctx.read(spec['source_Git_contract_pin'])
  if c.get('schema')!='A63_full_cache_verifier_cost_probe_sources_v1' or c['runtime_authorization_sha256']!=owner_sha or c['readonly_source_runtime_spec_pin']!=spec['source_runtime_spec_pin'] or c['readonly_source_freeze_pin']!=source['freeze'] or c['probe_spec_semantics_sha256']!=digest({k:v for k,v in spec.items() if k!='source_Git_contract_pin'}):raise ValueError('two-scope actual-Git source contract differs')
  new=set(source_closure(root,[SOURCE,WAITER]))-set(old['objects'])
  if set(c['objects'])!=new or c['historical_frozen_imports']!={}:raise ValueError('exact additional current source closure required')
  verify(ctx.contract_proof(c,c['actual_Git_object_proof']),c['git_commit'],c['objects'])
  for p,h in c['objects'].items():ctx.raw({'path':p,'sha256':h})
  combined={**old['objects'],**c['objects']}
  if c['combined_source_objects_sha256']!=digest(combined):raise ValueError('explicit disjoint two-scope combined object identity differs')
 else:combined=old['objects']
 # Core arrays/weights are HASHED only; no torch load, normalizer or head exists.
 from echo_routing.temporal.selected_risk_resume_supply import Supply
 if verify_contract:Supply(ctx,None,None,None)._aux(source['arguments']['core'])
 else:
  core=source['arguments']['core'];verify(ctx.read(core['actual_Git_object_proof']),core['git_commit'],{core[k]['path']:core[k]['sha256'] for k in ['models_manifest','audit','training_inputs']})
  for k in ['models_manifest','audit','training_inputs']:ctx.read(core[k])
 binding={k:spec[k] for k in ['source_runtime_spec_pin','source_Git_contract_pin','input_manifest_pin','geometry_pin','core_pins']}
 binding.update(probe_spec_pin=spec_pin,source_pins=[{'path':p,'sha256':h} for p,h in sorted(combined.items()) if p.endswith('.py')])
 return ctx,source,spec,binding


def make_contract(root,spec_pin,owner_sha,commit):
 from echo_routing.temporal.selected_risk_contracts import source_closure
 from echo_routing.temporal.safety_git import build,verify
 from echo_routing.temporal.selected_risk_resume_contract import digest
 ctx,source,spec,_=setup(root,spec_pin,owner_sha,verify_contract=False);old=ctx.read(source['freeze'])
 paths=set(source_closure(root,[SOURCE,WAITER]))-set(old['objects']);objects={p:pin(ctx.root,ctx.root/p)['sha256'] for p in sorted(paths)}
 proof=build(root,commit,sorted(objects));verify(proof,commit,objects)
 pp=ctx.write_json(spec['output_prefix'].split('/',2)[-1]+'/source_current.private.json',proof)
 c={'schema':'A63_full_cache_verifier_cost_probe_sources_v1','runtime_authorization_sha256':owner_sha,'readonly_source_runtime_spec_pin':spec['source_runtime_spec_pin'],'readonly_source_freeze_pin':source['freeze'],'probe_spec_semantics_sha256':digest({k:v for k,v in spec.items() if k!='source_Git_contract_pin'}),'objects':objects,'git_commit':commit,'actual_Git_object_proof':pp,'historical_frozen_imports':{},'combined_source_objects_sha256':digest({**old['objects'],**objects})}
 return ctx.write_json(spec['output_prefix'].split('/',2)[-1]+'/source_contract.json',c)


def _banks(ctx,source):
 from echo_routing.temporal.selected_risk_resume_inputs import ReadOnlyInputs
 reader=ReadOnlyInputs(ctx,source['metadata'],FIT_producer_pin=source['FIT_producer'])
 return {r:reader.bank(r,purpose='selector') for r in ROLES}


def _write(ctx,spec,name,value):
 from echo_routing.temporal.selected_risk_resume_contract import RUN
 return ctx.write_json(spec['output_prefix'][len(RUN)+1:]+'/'+name,value)


def _pool_shape(manifest):return {k:manifest[k] for k in ['purpose','axes','admitted_G','ALL_G','seeds','cache_records']}

def _zeros():return {'NN_calls':0,'SGD_updates':0,'model_instances':0,'GT_or_head_effect_used_for_cost':False,'cuda_context_observation':cuda_observation(),'metadata_label_parse_exception':'Original pinned metadata loader parses labels/length for alignment; no reference values or head effects construct or select this synthetic cost.'}


def prepare(ctx,source,spec,binding):
 banks=_banks(ctx,source);folder=ctx.root/spec['output_prefix']/'payloads'
 # Declare exact owned paths before any payload creation, including possible
 # interrupted writer temporaries. Cleanup never infers ownership from totals.
 planned=[]
 for seed in SEEDS:
  for role,(qs,*_) in banks.items():
   parent=f'seed{seed}/{role}/';planned += [parent+n for n in ['identity.json','identity.json.tmp','pending.json','pending.json.tmp','committed.jsonl']]
   for i in range(len(qs)):planned += [parent+f'sequence{i:06d}.private.npz',parent+f'sequence{i:06d}.private.npz.tmp']
 plan=_write(ctx,spec,'owned_plan.json',{'purpose':PURPOSE,'payload_root':str(folder.relative_to(ctx.root)),'allowed_files':planned})
 m=build_pool(folder,banks,binding);prepared_checks=verify_pool(folder,banks,m,check_synthetic=True);shape=_pool_shape(m)
 if shape!={'purpose':PURPOSE,'axes':5736,'admitted_G':8111,'ALL_G':8522,'seeds':list(SEEDS),'cache_records':8}:raise ValueError('full5736/8111/8522/8 exact observable pool required')
 manifest=_write(ctx,spec,'cache_manifest.private.json',m);deleted=file_manifest(folder)
 if not {r['path'] for r in deleted}<=set(planned):raise ValueError('payload outside predeclared ownership plan')
 dp=_write(ctx,spec,'deleted_file_manifest.private.json',deleted);actual=sum(r['bytes'] for r in deleted);bound=m['structural_cache_payload_upper_bytes']+8000000
 if actual>bound or bound>LIMITS['temp_bytes']:raise ValueError('complete cost-only payload/journal/metadata bound exceeds80MB')
 return _write(ctx,spec,'prepare_summary.json',{'schema':'A63_cost_only_cache_prepare_summary_v1','passed':True,'complete':True,'bindings':binding,'synthetic_pool':shape,'cache_manifest_pin':manifest,'deleted_file_manifest_pin':dp,'owned_plan_pin':plan,'payload_regular_bytes':actual,'structural_peak_upper_bytes':bound,'NPZ_STORED':prepared_checks['NPZ_STORED'],'payload_4096_header_verified':True,'synthetic_values_and_dtype_verified':True,**_zeros()})


def verify_cache(ctx,source,spec,binding,index):
 if index not in [0,1,2]:raise ValueError('three separate fresh verifier processes only')
 prep=ctx.read(pin(ctx.root,ctx.root/spec['output_prefix']/'prepare_summary.json'));m=ctx.read(prep['cache_manifest_pin'])
 if prep['bindings']!=binding:raise ValueError('fresh process input/core/Git source bindings changed')
 out=verify_pool(ctx.root/spec['output_prefix']/'payloads',_banks(ctx,source),m)
 return _write(ctx,spec,f'verify{index}_summary.json',{'schema':'A63_cost_only_cache_verifier_summary_v1','passed':True,'bindings':binding,'synthetic_pool':_pool_shape(m),'cache_manifest_pin':prep['cache_manifest_pin'],**out,**_zeros()})


def cost_parent(root,spec_pin,owner_sha,lock_fd):
 from scripts.a63_wait_completed_supply_selected_risk_cost import run_process,validate_lock
 ctx,source,spec,binding=setup(root,spec_pin,owner_sha);validate_lock(ctx,lock_fd,spec_pin)
 base=[sys.executable,str(ctx.root/SOURCE),'--execute','--probe-spec',spec_pin['path'],'--probe-spec-sha',spec_pin['sha256'],'--owner-sha',owner_sha,'--lock-fd',str(lock_fd)]
 rows=[]
 for mode in ['prepare','verify0','verify1','verify2']:
  row=run_process(ctx,spec,[*base,'--mode',mode],mode,lock_fd)
  rows.append(row)
  if row['actual_OS_exit_code']!=0 or not row['full_accounting_complete']:raise ValueError('cost-only child failed/unknown; retain all raw evidence')
 # stdout is deliberately exact one-line JSON, not another administrative fee.
 def output(row):return json.loads(ctx.raw(row['raw_stdout_pin']))['output_pin']
 prep=ctx.read(output(rows[0]));folder=ctx.root/spec['output_prefix']/'payloads';deleted=ctx.read(prep['deleted_file_manifest_pin'])
 clean=cleanup_owned(folder,deleted);clean['owned_payload_root']=str(folder.relative_to(ctx.root));clean['deleted_file_manifest_pin']=prep['deleted_file_manifest_pin']
 return _write(ctx,spec,'payload.json',{'schema':'A63_full_cache_verifier_cost_probe_v1','passed':True,'runtime_authorization_sha256':owner_sha,**binding,'synthetic_pool':prep['synthetic_pool'],'cache_manifest_pin':prep['cache_manifest_pin'],'prepare_process':rows[0],'three_fresh_verifier_processes':rows[1:],'cleanup':clean,'prepare_summary_pin':output(rows[0]),'verification_summary_pins':[output(r) for r in rows[1:]],'all_actual_attempts_charged_by_outer_whole_wait4':True,**_zeros()})


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--mode',choices=['prepare','verify0','verify1','verify2','parent','make-contract']);p.add_argument('--probe-spec');p.add_argument('--probe-spec-sha');p.add_argument('--owner-sha');p.add_argument('--lock-fd',type=int);p.add_argument('--git-commit');a=p.parse_args(argv)
 if not a.execute:print(json.dumps({'preparation_only':True,'NN':0,'SGD':0,'new_process':False}));return 0
 if not all([a.mode,a.probe_spec,a.probe_spec_sha,a.owner_sha]):p.error('independent raw source/probe/new-owner pins required')
 spec_pin={'path':a.probe_spec,'sha256':a.probe_spec_sha};cuda_observation()
 if a.mode=='make-contract':
  if not a.git_commit:p.error('actual source Git commit required')
  print(json.dumps({'contract_pin':make_contract(ROOT,spec_pin,a.owner_sha,a.git_commit)}));return 0
 from scripts.a63_wait_completed_supply_selected_risk_cost import validate_lock
 ctx,source,spec,binding=setup(ROOT,spec_pin,a.owner_sha);validate_lock(ctx,a.lock_fd,spec_pin)
 if a.mode=='parent':out=cost_parent(ROOT,spec_pin,a.owner_sha,a.lock_fd);print(json.dumps({'passed':True,'payload_pin':out}));return 0
 out=prepare(ctx,source,spec,binding) if a.mode=='prepare' else verify_cache(ctx,source,spec,binding,int(a.mode[-1]))
 print(json.dumps({'passed':True,'output_pin':out}));return 0

if __name__=='__main__':raise SystemExit(main())
