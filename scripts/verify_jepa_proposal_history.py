#!/usr/bin/env python3
"""Local actual-Git proof for completed preparation versions, no neural work."""
import hashlib,json,os,resource,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'docs/results/2026-11-03_jepa_proposals';RUN=ROOT/'runs/a60_jepa_proposals_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    start=time.time();identity=json.loads((PUB/'preparation_identity_v3.json').read_text());names=['preparation_freeze.json','training_prepare_freeze.json'];pins={}
    for name in names:
        value=json.loads((RUN/name).read_text())
        if value!=identity:raise ValueError('historical preparation freeze differs from public v3 identity')
        pins[name]=sha(RUN/name)
    for p,h in identity['objects'].items():
        raw=subprocess.check_output(['git','show',identity['git_commit']+':'+p],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('historical actualGit object mismatch')
    divergent=sorted(p for p,h in identity['objects'].items() if sha(ROOT/p)!=h)
    if set(divergent)-{'scripts/supervise_jepa_proposals.py'}:raise ValueError('unexpected changed historical source dependency')
    end=time.time();own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    digest=hashlib.sha256(json.dumps(identity['objects'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    value={'passed':True,'all_actual_Git_objects_verified':True,'validated_freezes':names,'historical_freeze_file_sha256':pins,'historical_source_git_commit':identity['git_commit'],'object_count':len(identity['objects']),'objects_SHA256':digest,'preparation_identity_v3_sha256':sha(PUB/'preparation_identity_v3.json'),'current_divergent_paths':divergent,'difference_scope':'completed metadata stages used earlier supervisor entry-accounting revision;subsequent repair changes no proposal/data/model/prediction recipe;originalversion retained inactualGit','stage':'local_historical_source_Git_verification','measurement_host':socket.gethostname(),'model_pid':os.getpid(),'started_epoch':start,'finished_epoch':end,'full_process_wall_seconds':end-start,'measured_CPU_core_seconds':own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime,'observed_worker_exit_code':None,'GPU_job':False,'CUDA_context_occupied':False,'scope':'runningSELF fullCPU+waitedGit children;actualGit hash inspection only;finalOSnotyetobservable;finalJSONbookkeeping unmetered;no source/model rerun'}
    (PUB/'historical_source_version_verification.json').write_text(json.dumps(value,sort_keys=True)+'\n');print({'passed':True,'objects':len(identity['objects']),'divergent':divergent,'CPU_seconds':value['measured_CPU_core_seconds']})
if __name__=='__main__':main()
