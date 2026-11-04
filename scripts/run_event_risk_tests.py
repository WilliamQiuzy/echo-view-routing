#!/usr/bin/env python3
"""A62 bounded local checks; durable pre-spawn receipt reservation and actual CPU."""
import argparse,hashlib,json,math,os,resource,signal,socket,subprocess,sys,time
from pathlib import Path


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def durable_folder(path):
    missing=[];current=path
    while not current.exists():missing.append(current);current=current.parent
    path.mkdir(parents=True,exist_ok=True)
    for new in reversed(missing):sync_directory(new.parent);sync_directory(new)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--timeout-seconds',type=float,default=300.)
    parser.add_argument('stage');parser.add_argument('tests',nargs='+');args=parser.parse_args()
    if not args.stage.replace('_','').isalnum() or not math.isfinite(args.timeout_seconds) or args.timeout_seconds<=0:
        raise ValueError('plain stage and finite positive timeout required')
    root=Path(__file__).resolve().parents[1];folder=root/'docs/results/2026-11-04_anchor_context_risk'
    durable_folder(folder);path=folder/(args.stage+'.json');claim=folder/(args.stage+'.claim.json')
    if path.exists():raise ValueError('earlier receipt already exists; child not launched')
    child_before=resource.getrusage(resource.RUSAGE_CHILDREN);self_before=resource.getrusage(resource.RUSAGE_SELF);start=time.time()
    with claim.open('x') as h:
        json.dump({'stage':args.stage,'supervisor_pid':os.getpid(),'started_epoch':start},h);h.flush();os.fsync(h.fileno())
    sync_directory(folder)
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    command=[sys.executable,'-m','pytest','-q','--tb=short',*args.tests]
    process=None;output='';reason=None;exception=None
    try:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        try:output=process.communicate(timeout=args.timeout_seconds)[0]
        except subprocess.TimeoutExpired:
            reason='timeout';os.killpg(process.pid,signal.SIGTERM)
            try:output=process.communicate(timeout=5)[0]
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);output=process.communicate()[0]
    except BaseException as error:
        exception=type(error).__name__+': '+str(error)
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);output=process.communicate()[0]
    finish=time.time();ca=resource.getrusage(resource.RUSAGE_CHILDREN);sa=resource.getrusage(resource.RUSAGE_SELF)
    cc=ca.ru_utime+ca.ru_stime-child_before.ru_utime-child_before.ru_stime
    sc=sa.ru_utime+sa.ru_stime-self_before.ru_utime-self_before.ru_stime
    code=None if process is None else process.returncode
    status=0 if code==0 and reason is None and exception is None else 1
    value={'stage':args.stage,'command':command,'measurement_host':socket.gethostname(),'model_pid':None if process is None else process.pid,
      'supervisor_pid':os.getpid(),'started_epoch':start,'finished_epoch':finish,'full_process_wall_seconds':finish-start,
      'observed_worker_exit_code':code,'stage_exit_code':status,'termination_reason':reason,'exception':exception,
      'GPU_job':False,'CUDA_context_occupied':False,'CUDA_visibility_hidden_before_child_import':True,
      'measured_child_CPU_seconds':cc,'measured_supervisor_CPU_seconds':sc,'measured_CPU_core_seconds':cc+sc,
      'durable_claim_before_Popen':True,'scope':'local synthetic checks;SELF region+waited CHILDREN;final JSON bookkeeping and wrapper imports separate metadata',
      'output':output,'output_sha256':hashlib.sha256(output.encode()).hexdigest()}
    temporary=path.with_suffix('.json.tmp')
    with temporary.open('x') as h:json.dump(value,h,indent=2,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
    os.link(temporary,path);sync_directory(folder);temporary.unlink();claim.unlink();sync_directory(folder)
    print(output);print(json.dumps({k:v for k,v in value.items() if k!='output'}));return status

if __name__=='__main__':raise SystemExit(main())
