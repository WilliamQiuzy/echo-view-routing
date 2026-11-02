#!/usr/bin/env python3
"""A56 own-PID receipt accounting, active owner segments and immutable history."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/a56_source_retention_v1'
p=argparse.ArgumentParser();p.add_argument('mode',choices=['snapshot','begin','end','close'],default='snapshot',nargs='?');p.add_argument('--root-reviewer-upper-hours',type=float);a=p.parse_args();now=time.time();c=json.loads((OUT/'coordination.json').read_text());segments=c['owner_segments']
if a.mode=='begin' and (not segments or segments[-1]['end'] is not None):segments.append({'start':now,'end':None})
if a.mode in ('end','close') and segments and segments[-1]['end'] is None:segments[-1]['end']=now
if a.root_reviewer_upper_hours is not None:c['root_reviewer_CPU_upper_hours']=a.root_reviewer_upper_hours
code='''import json,time,os
from pathlib import Path
out=Path('/home/william/echo-view-routing/runs/a56_source_retention_v1');records=[]
for p in list(out.glob('*worker_process.json'))+list(out.glob('process_*.json')):records.append(json.loads(p.read_text()))
p=out/'active_process.json'
if p.exists():
 z=json.loads(p.read_text());stat=Path('/proc/'+str(z['model_pid'])+'/stat')
 if stat.exists():
  fields=stat.read_text().split();records.append({'model_pid':z['model_pid'],'process_CPU_seconds':(int(fields[13])+int(fields[14]))/os.sysconf('SC_CLK_TCK'),'GPU_job_wall_seconds':time.time()-z['start_epoch'] if z['GPU_job'] else 0})
sizes=[]
for p in out.rglob('*'):
 try:
  if p.is_file():sizes.append((p.name,p.stat().st_size))
 except FileNotFoundError:pass
print(json.dumps({'records':records,'namespace_bytes':sum(v for n,v in sizes),'largest_checkpoint_bytes':max([v for n,v in sizes if n.endswith('.pt')],default=0)}))'''
proc=subprocess.run(['ssh','-o','ConnectTimeout=10','echo-l40s','/home/william/echo-view-routing/.venv/bin/python -'],input=code,text=True,capture_output=True,check=True);remote=json.loads(proc.stdout);by={}
for z in remote['records']:
 key=str(z.get('model_pid',z.get('pid')));old=by.get(key,[0.,0.]);cpu=z.get('process_CPU_seconds',z.get('CPU_seconds',0.));gpu=z.get('full_process_wall_seconds',0.) if z.get('GPU_job') else z.get('GPU_job_wall_seconds',0.);by[key]=[max(old[0],cpu),max(old[1],gpu)]
owner=sum((s['end'] or now)-s['start'] for s in segments)/3600;child=sum(x[0] for x in by.values())/3600;gpu=sum(x[1] for x in by.values())/3600;extra=(30+c.get('completed_unmetered_coordination_CPU_upper_seconds',0))/3600;cpu=owner+child+c['root_reviewer_CPU_upper_hours']+extra;previous=json.loads((OUT/'resource_ledger.json').read_text()) if (OUT/'resource_ledger.json').exists() else {};observed=max(previous.get('observed_namespace_peak_bytes',0),remote['namespace_bytes']);peak=max(previous.get('new_phase_peak_upper_bytes',0),observed+remote['largest_checkpoint_bytes']+10000000)
v={'asof_epoch':now,'program_wall_elapsed_upper_hours':(now-c['program_start_epoch'])/3600,'program_start_origin':c['program_start_origin'],'new_phase_CPU_upper_hours':cpu,'new_phase_GPU_job_wall_hours':gpu,'new_phase_peak_upper_bytes':peak,'actual_namespace_bytes':remote['namespace_bytes'],'observed_namespace_peak_bytes':observed,'largest_completed_checkpoint_bytes':remote['largest_checkpoint_bytes'],'owner_active_wall_upper_hours':owner,'root_reviewer_CPU_upper_hours':c['root_reviewer_CPU_upper_hours'],'worker_CPU_hours':child,'completed_unwrapped_metadata_CLI_CPU_upper_seconds':30,'completed_unmetered_coordination_CPU_upper_seconds':c.get('completed_unmetered_coordination_CPU_upper_seconds',0),'PID_deduplicated_CPU_GPU_seconds':by,'historical_receipt':c['historical_receipt'],'historical_lifetime_CPU_upper_hours':c['historical_lifetime_CPU_upper_hours'],'historical_lifetime_GPU_job_wall_hours':c['historical_lifetime_GPU_job_wall_hours'],'charged_lifetime_CPU_upper_hours':cpu+c['historical_lifetime_CPU_upper_hours'],'charged_lifetime_GPU_job_wall_hours':gpu+c['historical_lifetime_GPU_job_wall_hours'],'caps':c['new_phase_caps'],'scope':'Own worker PID receipts, active execution-owner wall upper and separately declared root/reviewer active upper; overlapping upper charges are explicit, not total host CPU. Owner idle waits excluded by closed segments. Observed monotone namespace peak plus one largest existing checkpoint bounds completed atomic coexistence, plus10MBcode allowance. No future cost charged.','old_phase_unchanged':True,'future_charges':0,'idle_wait_CPU_charges':0}
(OUT/'coordination.json').write_text(json.dumps(c,indent=2)+'\n');(OUT/'resource_ledger.json').write_text(json.dumps(v,indent=2)+'\n');assert cpu<6 and gpu<3 and peak<5000000000 and now<c['wall_hard_deadline_epoch'],'A56 resource cap reached; stop discretionary work'
print(json.dumps({k:v[k] for k in ['new_phase_CPU_upper_hours','new_phase_GPU_job_wall_hours','new_phase_peak_upper_bytes','program_wall_elapsed_upper_hours']}))
