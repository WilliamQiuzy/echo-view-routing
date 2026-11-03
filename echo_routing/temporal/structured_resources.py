"""Terminal costs: observed status, sampled CPU and accounting allowances stay distinct."""
import math

def closed_costs(records):
    seen=set();gpu=[];cpu=wall=allowance=0.;exits=[];unknown=failures=0
    for row in records:
        pid=row.get('model_pid');host=row.get('measurement_host');start=float(row['start_epoch']);end=float(row['end_epoch']);w=float(row['full_process_wall_seconds']);c=float(row['process_CPU_seconds']);extra=float(row.get('unmeasured_completed_CPU_accounting_allowance_seconds',0));key=(host,pid)
        if not host or not isinstance(pid,int) or isinstance(pid,bool) or pid<=0 or key in seen:raise ValueError('unique actual host/PID provenance required')
        seen.add(key)
        if any(not math.isfinite(v) for v in [start,end,w,c,extra]) or end<start or min(w,c,extra)<0 or abs(w-(end-start))>.01:raise ValueError('finite closed interval and reconciled nonnegative costs required')
        observed=row.get('exit_code') if row.get('child_exit_code_directly_waited',True) else None;manifest=row.get('completion_verified_by_frozen_complete_cache_manifest');unknown+=observed is None;failed=(observed is not None and observed!=0) or bool(row.get('termination_reason')) or bool(row.get('guardian_exception')) or manifest is False;failures+=bool(failed);exits.append({'stage':row['stage'],'model_pid':pid,'measurement_host':host,'observed_worker_exit_code':observed,'manifest_complete':manifest,'raw_receipt_inferred_exit_code_ignored':row.get('exit_code') if observed is None else None,'OS_status_inferred_from_manifest':False});cpu+=c;allowance+=extra
        if row['GPU_job']:gpu.append((start,end));wall+=w
    gpu.sort()
    if any(b[0]<a[1]-1e-3 for a,b in zip(gpu,gpu[1:])):raise ValueError('overlapping GPU intervals violate exclusive queue')
    return {'GPU_job_wall_seconds':wall,'summed_measured_CPU_receipt_seconds':cpu,'completed_unmeasured_CPU_accounting_allowance_seconds':allowance,'closed_receipts':len(records),'GPU_jobs':len(gpu),'single_GPU_queue_intervals_disjoint':True,'failed_or_guard_terminated_stages':failures,'unknown_worker_OS_exit_statuses':unknown,'worker_exit_statuses':exits,'GPU_scope':'fullclosedjobintervals;orphanintervalendsatguardianobservationandmayinclude<=pollintervaltail','CPU_scope':'sumofmeasuredreceipts;adoptedworkeruseslastprocCPUplusmeasuredguardian;originaldeadsupervisorandlastpollCPUtailnotclaimedmeasured','CPU_cross_receipt_and_orphan_provenance_caveat':'singlequeueworkerintervalsrecorded;lostsupervisorCPUunrecoverable;separatecompletedallowanceisbudgetaccountingnotactualCPU'}
