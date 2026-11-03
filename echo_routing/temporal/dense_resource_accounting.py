"""Closed-PID actual costs, with no conversion of idle/coordinator wall to CPU."""
import math,fcntl
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def terminal_queue_lock(path):
    """Use the supervisor's actual lock throughout terminal checks and writes."""
    with Path(path).open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('queue is owned; refuse concurrent terminal collection') from e
        try:yield
        finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def closed_totals(records):
    seen=set();receipts=set();PID_hosts=[];gpu=[];cpu=wall=failed_wall=0.;failures=receipt_only=0;uncertain=[]
    for row in records:
        pid=row.get('model_pid');host=row.get('measurement_host');start=row.get('start_epoch');receipt=row.get('receipt_identity')
        real_pid=isinstance(pid,int) and not isinstance(pid,bool) and pid>0
        if real_pid:
            if any(pid==old_pid and (not host or not old_host or host==old_host) for old_pid,old_host in PID_hosts):raise ValueError('duplicate PID or unresolved host collision; reject double charging')
            PID_hosts.append((pid,host));key=('host_PID',host,pid)
        else:
            identity=receipt or (pid if isinstance(pid,str) and pid else None)
            if identity is None:raise ValueError('actual PID or explicitly receipt-only identity required')
            key=('receipt_only',identity);receipt_only+=1
        if receipt is not None:
            if receipt in receipts:raise ValueError('duplicate measurement receipt; reject double charging')
            receipts.add(receipt)
        if key in seen:raise ValueError('duplicate measured worker identity; reject double charging')
        seen.add(key)
        if not (real_pid and host and start is not None and math.isfinite(float(start))):uncertain.append({'receipt_identity':receipt,'recorded_PID':pid if real_pid else None,'host_recorded':bool(host),'start_recorded':start is not None,'identity_basis':'recordedhost/PIDwhereavailable;otherwiseexplicitreceiptidentity;crossreceiptCPUoverlapnotproven'})
        c=float(row['process_CPU_seconds']);w=float(row.get('full_process_wall_seconds',0.))
        if not math.isfinite(c) or not math.isfinite(w) or min(c,w)<0:raise ValueError('finite nonnegative actual costs required')
        cpu+=c;failed=row.get('exit_code',0)!=0 or not row.get('ok',True);failures+=bool(failed)
        if row['GPU_job']:
            start,end=float(row['start_epoch']),float(row['end_epoch'])
            if not math.isfinite(start) or not math.isfinite(end) or end<start:raise ValueError('finite closed GPU worker interval required')
            if abs(w-(end-start))>.01:raise ValueError('GPU actual wall differs from closed worker interval')
            gpu.append((start,end));wall+=w;failed_wall+=w if failed else 0.
    gpu.sort()
    if any(b[0]<a[1]-1e-3 for a,b in zip(gpu,gpu[1:])):raise ValueError('overlapping GPU workers violate the single queue')
    return {'GPU_job_wall_seconds':wall,'measured_CPU_core_seconds':cpu,'failed_GPU_job_wall_seconds':failed_wall,'closed_measured_receipts':len(seen),'failed_measured_receipts':failures,'GPU_jobs':len(gpu),'single_GPU_queue_intervals_disjoint':True,'receipt_only_identity_records':receipt_only,'complete_host_PID_start_provenance':not bool(uncertain),'identity_provenance_uncertainties':uncertain,'CPU_cross_receipt_overlap_excluded_by_complete_provenance':not bool(uncertain)}
