"""A49 fixed, source-only acquisition. No model or target outcome readout."""
import sys,json,pathlib,urllib.request,hashlib,time,concurrent.futures,threading,traceback
sys.path.insert(0,'.')
from scripts.lvot_caliper_contract import stream_verified_tar
P=pathlib.Path('runs/a49_lvot_evidence_qualification_v1'); REV='7a36922880061791519a8f01553c15cad5df564f'; REPO='Ahus-AIM/EchoXFlow'
def freeze():
 a=json.loads((P/'main_caliper_candidates.private.json').read_text()); package=json.load(open('runs/a47_public_evidence_intake_v1/package_manifest.json')); exams=[]
 for f in range(5):
  exams += sorted({v['exam'] for v in a if v['fold']==f})[:24]
 paths=['exams/'+e+'.tar' for e in exams]
 req=urllib.request.Request(f'https://huggingface.co/api/datasets/{REPO}/paths-info/{REV}',data=json.dumps({'paths':paths,'expand':False}).encode(),headers={'Content-Type':'application/json'})
 objects={x['path']:x for x in json.load(urllib.request.urlopen(req,timeout=40))}
 rows=[]
 for e in exams:
  candidates=[v for v in a if v['exam']==e]; meta=objects['exams/'+e+'.tar']; assert meta['size']==package['tar_bytes'][e] and meta['lfs']
  prefixes=set()
  for v in candidates:
   r=v['recording_zarr_path']; prefixes.update([r+'/.zattrs',r+'/.zmetadata',r+'/data/2d_brightness_mode',r+'/'+v['point_array']])
  rows.append({'exam':e,'fold':candidates[0]['fold'],'publisher_object':meta,'candidates':candidates,'prefixes':sorted(prefixes)})
 plan={'revision':REV,'repo':REPO,'rows':rows,'concurrency':4,'deadline_seconds':2700,'traffic_bytes_cap':100_000_000_000,'per_archive_retained_cap':3_000_000,'persistent_new_cap':700_000_000,'prior_contract':'demo_contract_receipt.private.json','initial_qualification_closed_before_preparation':True}
 body=json.dumps(plan,sort_keys=True,indent=2).encode(); path=P/'preparation_selection.private.json'; assert not path.exists(); path.write_bytes(body)
 aggregate={'phase':'preparation_frozen_before_media','revision':REV,'repo':REPO,'archive_count':len(rows),'archive_bytes':sum(v['publisher_object']['size'] for v in rows),'potential_records':len({v['record'] for r in rows for v in r['candidates']}),'selection_sha256':hashlib.sha256(body).hexdigest(),'per_fold_exams':{str(f):sum(r['fold']==f for r in rows) for f in range(5)},'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(pathlib.Path('scripts/lvot_caliper_contract.py').read_bytes()).hexdigest(),'gpu_hours':0}
 (P/'preparation_freeze.json').write_text(json.dumps(aggregate,indent=2)); print(json.dumps(aggregate,indent=2))
def run():
 plan=json.loads((P/'preparation_selection.private.json').read_text()); t=time.time(); deadline=min(t+plan['deadline_seconds'], (P/'preparation_selection.private.json').stat().st_mtime+plan['deadline_seconds']); lock=threading.Lock(); state={'traffic_bytes':0}; (P/'archive_receipts').mkdir(exist_ok=True); (P/'archive_quarantine').mkdir(exist_ok=True)
 def work(row):
  e=row['exam']; receipt=P/'archive_receipts'/ (e+'.private.json'); dst=P/'archive_quarantine'/e
  if receipt.exists(): raise RuntimeError('no automatic repeat of existing exam')
  started=time.time(); attempts=[]; good=False
  for attempt in range(2):
   # Only retry transport failures before retaining data; never overwrite quarantine.
   if dst.exists(): break
   try:
    meta=row['publisher_object']; url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/'+meta['path']; response=urllib.request.urlopen(url,timeout=30)
    class Bounded:
     def read(self,n=-1):
      with lock:
       if time.time()>=deadline or state['traffic_bytes']+max(n,0)>plan['traffic_bytes_cap']: raise RuntimeError('preparation time/traffic cap reached')
      b=response.read(n)
      with lock:
       state['traffic_bytes']+=len(b)
       if state['traffic_bytes']>plan['traffic_bytes_cap']: raise RuntimeError('preparation traffic cap reached')
      return b
    proof=stream_verified_tar(Bounded(),meta,set(row['prefixes']),dst,plan['per_archive_retained_cap']); response.close(); good=True; attempts.append({'attempt':attempt+1,'ok':True,'proof':proof}); break
   except Exception as error:
    attempts.append({'attempt':attempt+1,'ok':False,'error_type':type(error).__name__,'error':str(error)[:400]})
    if time.time()>=deadline: break
  out={'exam':e,'fold':row['fold'],'publisher_verified':good,'started_unix':started,'finished_unix':time.time(),'attempts':attempts}; receipt.write_text(json.dumps(out,indent=2))
  with lock:
   progress={'finished_archives':len(list((P/'archive_receipts').glob('*.json'))),'verified_archives':sum(json.loads(x.read_text())['publisher_verified'] for x in (P/'archive_receipts').glob('*.json')),'traffic_bytes':state['traffic_bytes'],'elapsed_seconds':time.time()-t,'gpu_hours':0}; (P/'preparation_progress.json').write_text(json.dumps(progress,indent=2))
  print(json.dumps(progress),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[]
  # Bounded 4-job batches; never enqueue all 120 beyond the deadline.
  for i in range(0,len(plan['rows']),4):
   if time.time()>=deadline: break
   batch=[pool.submit(work,row) for row in plan['rows'][i:i+4]]
   for future in batch: future.result()
 (P/'preparation_run_receipt.json').write_text(json.dumps({'started_unix':t,'finished_unix':time.time(),'elapsed_seconds':time.time()-t,**state,'gpu_hours':0,'fixed_selection_sha256':hashlib.sha256((P/'preparation_selection.private.json').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':
 (freeze if sys.argv[1]=='freeze' else run)()
