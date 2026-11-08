import importlib
from pathlib import Path
import numpy as np
import pytest

def modules():
 return importlib.import_module('scripts.a63_probe_completed_supply_selected_risk_cost'),importlib.import_module('scripts.a63_wait_completed_supply_selected_risk_cost')

def banks():
 class TruthBomb:
  def __array__(self,*a,**k):raise AssertionError('GT value used')
 b=np.zeros(8,np.int64);a=np.eye(5,dtype=np.float64)[b];j=b.copy();j[3]=1;j=np.eye(5,dtype=np.float64)[j]
 return {'F0_mixed':([{'labels':TruthBomb()}],[a],[j],[b])}

def test_full_codec_synthetic_branches_reference_and_tamper(tmp_path):
 m,_=modules();folder=tmp_path/'cost_only_synthetic_payloads';v=m.build_pool(folder,banks(),{'purpose':'cost_only_synthetic_nonadoptable'})
 assert len(v['records'])==2 and v['axes']==1 and v['admitted_G']>0
 assert {r['seed'] for r in v['records']}=={2026110461,2026110462}
 out=m.verify_pool(folder,banks(),v);assert out['cache_records']==2 and out['complete']
 first=v['records'][0];last=v['records'][1]
 assert first['positive_u'] is False and last['positive_u'] is True and last['feature_reference_seed']==2026110461
 f=next((folder/last['relative_path']).glob('sequence*.npz'));raw=f.read_bytes();f.write_bytes(raw+b' ')
 with pytest.raises(ValueError):m.verify_pool(folder,banks(),v)

def test_owned_cleanup_refuses_foreign_and_keeps_evidence(tmp_path):
 m,_=modules();folder=tmp_path/'cost_only_synthetic_payloads';v=m.build_pool(folder,banks(),{'purpose':'cost_only_synthetic_nonadoptable'});manifest=m.file_manifest(folder)
 (folder/'foreign.txt').write_text('preserve')
 with pytest.raises(ValueError):m.cleanup_owned(folder,manifest)
 assert (folder/'foreign.txt').read_text()=='preserve' and len(list(folder.rglob('*.npz')))>0
 (folder/'foreign.txt').unlink();out=m.cleanup_owned(folder,manifest)
 assert out['fully_removed'] and out['owned_only'] and out['before_regular_bytes']>0 and out['after_regular_bytes']==0 and not folder.exists()

def test_cost_only_caps_preserve_reserves_and_default_no_spawn(monkeypatch,capsys):
 _,m=modules();seen=[]
 monkeypatch.setattr(m,'forecast',lambda *a,**kw:seen.append((a,kw)) or {'all_caps_fit':True})
 assert m.probe_admission({'spent':1},800000000,12,{'owner':1})['all_caps_fit']
 a,kw=seen[0];assert a[1:4]==(0,4200,1008000000) and a[5]==7800
 monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('dryrun spawned')))
 assert m.main([])==0 and 'preparation_only' in capsys.readouterr().out


def test_probe_actual_storage_scalar_and_no_scientific_admission(monkeypatch):
 import contextlib,types
 _,m=modules();import echo_routing.temporal.selected_risk_resume_runtime as runtime
 class State:
  def __init__(self,ctx,source,pin):self.spec=source
  @contextlib.contextmanager
  def queue(self):yield 9
  def spent(self):return {'canonical':1}
  def storage(self):return 800000000
  def admission(self):raise AssertionError('scientific full-matrix admission must not launch cost measurement')
 monkeypatch.setattr(runtime,'RuntimeState',State);monkeypatch.setattr(m,'inventory',lambda *a:None)
 monkeypatch.setattr(m,'_write',lambda *a:{'path':'retained-refusal','sha256':'a'*64})
 seen=[]
 monkeypatch.setattr(m,'probe_admission',lambda spent,current,now,owner:seen.append(current) or {'all_caps_fit':False})
 with pytest.raises(ValueError,match='cost-only600'):m.execute(types.SimpleNamespace(owner={}),{'arguments':{}},{'source_runtime_spec_pin':{},'output_prefix':'unused'},None,{})
 assert seen==[800000000]


def test_cost_ticket_rejects_different_raw_probe_pin(tmp_path,monkeypatch):
 import os,types
 _,m=modules();from echo_routing.temporal.selected_risk_resume_runtime import SHARED_QUEUE
 lock=tmp_path/SHARED_QUEUE;lock.parent.mkdir(parents=True);lock.write_text('')
 fd=os.open(lock,os.O_RDWR)
 old={'path':'old.json','sha256':'a'*64};new={'path':'new.json','sha256':'b'*64}
 spec={'source_runtime_spec_pin':{'path':'source','sha256':'c'*64},'source_Git_contract_pin':{'path':'contract','sha256':'d'*64},'purpose':'cost_only_synthetic_nonadoptable','limits':m.LIMITS}
 intent={'runtime_authorization_sha256':'e'*64,'boot_id':'boot','parent_pid':os.getpid(),'parent_start_ticks':7,'probe_spec_pin':old,**spec}
 ctx=types.SimpleNamespace(root=tmp_path,owner_sha='e'*64,read=lambda p:intent if p.get('path')=='intent' else spec)
 monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setattr(m,'boot',lambda:'boot');monkeypatch.setattr(m,'ticks',lambda pid:7)
 monkeypatch.setenv('A63_COST_PROBE_INTENT_PATH','intent');monkeypatch.setenv('A63_COST_PROBE_INTENT_SHA','f'*64)
 try:
  with pytest.raises(ValueError,match='probe'):m.validate_lock(ctx,fd,new)
 finally:os.close(fd)


def test_monitor_exception_stops_waits_and_records_uncertainty(tmp_path,monkeypatch):
 import types,resource
 _,m=modules();ctx=types.SimpleNamespace(root=tmp_path,output=lambda name:tmp_path/name)
 spec={'output_prefix':'runs/a63_selected_action_risk_resume_v1/cost_only_synthetic_cache_probe/'+'1'*32}
 child=types.SimpleNamespace(pid=42,returncode=None)
 monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:child);monkeypatch.setattr(m,'ticks',lambda pid:8);monkeypatch.setattr(m,'boot',lambda:'boot')
 monkeypatch.setattr(m.os,'wait4',lambda *a:(0,0,None));monkeypatch.setattr(m,'sample_owned_bytes',lambda *a:(_ for _ in ()).throw(ValueError('injected owned stat failure')))
 stopped=[];usage=types.SimpleNamespace(ru_utime=2.,ru_stime=.5)
 monkeypatch.setattr(m,'stop_owned',lambda c,t,outer:stopped.append((c.pid,t,outer)) or (1,usage,True))
 monkeypatch.setattr(m,'_write',lambda *a:{'path':'record','sha256':'a'*64})
 row=m.run_process(ctx,spec,['owned-cost-only-child'],'test',9,outer=True)
 assert stopped==[(42,8,True)] and not row['full_accounting_complete'] and row['CPU_core_seconds'] is None and row['known_partial_CPU_core_seconds']==2.5


def test_stop_never_follows_reused_parent_into_foreign_children(monkeypatch):
 import types
 _,m=modules();original=Path.read_text
 def read(path,*a,**k):
  if str(path)=='/proc/42/task/42/children':return '43'
  if str(path)=='/proc/43/task/43/children':return ''
  return original(path,*a,**k)
 monkeypatch.setattr(Path,'read_text',read);monkeypatch.setattr(m,'ticks',lambda p:999 if p==42 else 1)
 monkeypatch.setattr(m.os,'wait4',lambda *a:(42,256,types.SimpleNamespace(ru_utime=1.,ru_stime=0.)))
 monkeypatch.setattr(m.os,'kill',lambda *a:(_ for _ in ()).throw(AssertionError('foreign/reused process signalled')))
 monkeypatch.setattr(m.os,'killpg',lambda *a:(_ for _ in ()).throw(AssertionError('foreign/reused group signalled')))
 code,usage,unknown=m.stop_owned(types.SimpleNamespace(pid=42,returncode=None),8,True)
 assert code==1 and unknown


def test_monitor_cleanup_disappearance_only_not_link_or_permission(tmp_path,monkeypatch):
 import os
 _,m=modules();folder=tmp_path/'owned';folder.mkdir();file=folder/'payload';file.write_text('123')
 old=Path.lstat
 def gone(p,*a,**k):
  if p==file:raise FileNotFoundError('same-owner cleanup')
  return old(p,*a,**k)
 monkeypatch.setattr(Path,'lstat',gone)
 assert m.sample_owned_bytes(folder)==0
 monkeypatch.setattr(Path,'lstat',old);os.symlink('payload',folder/'link')
 with pytest.raises(ValueError):m.sample_owned_bytes(folder)
 (folder/'link').unlink()
 def denied(p,*a,**k):
  if p==file:raise PermissionError('must not swallow')
  return old(p,*a,**k)
 monkeypatch.setattr(Path,'lstat',denied)
 with pytest.raises(PermissionError):m.sample_owned_bytes(folder)
