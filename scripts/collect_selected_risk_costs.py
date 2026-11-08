#!/usr/bin/env python3
"""Explicit registered A63 local fees; raw aggregates/children never auto-scanned."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,resource,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.selected_risk_common import *
from echo_routing.temporal.selected_risk_accounting import collect


def main():
    p=execution_parser(__doc__);p.add_argument('--registrations',required=True);a=p.parse_args()
    if not authorize(a):return 0
    specpath=checked_output(ROOT/a.registrations);specs=read(specpath)
    # Only this collector's explicitly named own raw receipts are auto-added;
    # server/admission/resource/aggregate summary files are never discovered.
    for f in PUB.glob('admin_fee_collection_*.json'):
        rp=relative(f)
        if not any(s['path']==rp for s in specs):specs.append({'path':rp,'sha256':sha(f),'charge_group':'fee_collection:'+f.name,'CPU_field':'CPU_SELF_plus_waitedchildren_seconds','kind':'admin'})
    canonical=PUB/'local_cpu_receipts.json'
    versions=[(int(m.group(1)),f) for f in PUB.glob('local_cpu_receipts_v*.json') if (m:=re.fullmatch(r'local_cpu_receipts_v([1-9][0-9]*)\.json',f.name))]
    if versions and (not canonical.exists() or canonical.read_bytes()!=max(versions)[1].read_bytes()):raise ValueError('canonical/latest version mismatch needs explicit reconciliation')
    prior=read(canonical) if canonical.exists() else None
    out=collect(ROOT,specs,prior=prior);sample=time.time();r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
    own=PUB/f'admin_fee_collection_{os.getpid()}_{int(sample*1e6)}.json';write(own,{'CPU_SELF_plus_waitedchildren_seconds':r.ru_utime+r.ru_stime+c.ru_utime+c.ru_stime,'SELF_OS_exit_code':None,'scope':'whole fresh explicit local-fee parser SELF plus waitedchildren through sample; no server stages charged here; finalwrite/merge/exit/tool tails unknown','sampled_epoch':sample})
    out=collect(ROOT,[{'path':relative(own),'sha256':sha(own),'charge_group':'fee_collection:'+own.name,'CPU_field':'CPU_SELF_plus_waitedchildren_seconds','kind':'admin'}],prior=out);out['own_raw_receipt']=pin(own);out['registration_pin']=pin(specpath)
    versions=[int(m.group(1)) for f in PUB.glob('local_cpu_receipts_v*.json') if (m:=re.fullmatch(r'local_cpu_receipts_v([1-9][0-9]*)\.json',f.name))];v=max(versions,default=0)+1
    version=PUB/f'local_cpu_receipts_v{v}.json';write(version,out);write(PUB/'local_cpu_receipts.json',out)
    print({'snapshot':pin(version),'unique_receipts':len(out['receipts']),'known_measured_CPU_seconds':out['known_measured_CPU_seconds'],'separate_admin_allowance_seconds':out['administrative_CPU_allowance_seconds']},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
