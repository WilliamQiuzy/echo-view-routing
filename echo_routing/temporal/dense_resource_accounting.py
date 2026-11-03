"""Closed-PID actual costs, with no conversion of idle/coordinator wall to CPU."""
import math

def closed_totals(records):
    seen=set();gpu=[];cpu=wall=failed_wall=0.;failures=0
    for row in records:
        key=row['model_pid']
        if key in seen:raise ValueError('duplicate measured worker identity; reject double charging')
        seen.add(key);c=float(row['process_CPU_seconds']);w=float(row.get('full_process_wall_seconds',0.))
        if not math.isfinite(c) or not math.isfinite(w) or min(c,w)<0:raise ValueError('finite nonnegative actual costs required')
        cpu+=c;failed=row.get('exit_code',0)!=0 or not row.get('ok',True);failures+=bool(failed)
        if row['GPU_job']:
            start,end=float(row['start_epoch']),float(row['end_epoch'])
            if not math.isfinite(start) or not math.isfinite(end) or end<start:raise ValueError('finite closed GPU worker interval required')
            if abs(w-(end-start))>.01:raise ValueError('GPU actual wall differs from closed worker interval')
            gpu.append((start,end));wall+=w;failed_wall+=w if failed else 0.
    gpu.sort()
    if any(b[0]<a[1]-1e-3 for a,b in zip(gpu,gpu[1:])):raise ValueError('overlapping GPU workers violate the single queue')
    return {'GPU_job_wall_seconds':wall,'measured_CPU_core_seconds':cpu,'failed_GPU_job_wall_seconds':failed_wall,'closed_measured_workers':len(seen),'failed_measured_workers':failures,'GPU_jobs':len(gpu),'single_GPU_queue_intervals_disjoint':True}
