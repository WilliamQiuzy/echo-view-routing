"""A61 terminal success gates and honest OS/process-identity statements."""


def validate_science(summary,oracle,native,models):
    if not (summary['formal_models']==10 and summary['formal_optimizer_updates']==60000 and summary['metric_rows']==314 and summary['paired_comparison_count']==240 and summary['all314_metrics_independently_recounted'] and oracle['all_maxone_G_and_noop'] and oracle['domain_count']==60 and oracle['overlapping_A60_G_oracles_exactly_reproduced']==16 and native['passed'] and native['cell_count']==56 and models['passed'] and models['all10finals24inner8taus56sets_frozen']):raise ValueError('all60000/314/240/60+16/56 actual scientific stages required')
    return True


def worker_status(records):
    return {'all_new_worker_statuses_directly_waited':all(r.get('child_exit_code_directly_waited') is True and r.get('observed_worker_exit_code') is not None and not r.get('OS_status_inferred_from_manifest',False) for r in records),'unknown_server_worker_OS_status_count':sum(r.get('observed_worker_exit_code') is None for r in records)}


def process_relation(record,pid,sample):
    if sample is None:return 'absent'
    # A later boot/PID reuse is not evidence that the old A61 job remains alive.
    if sample['start_epoch']>record['finished_epoch']+1.:return 'reused_after_record'
    if pid==record.get('model_pid') and record.get('process_start_ticks') is not None and sample['start_ticks']!=record['process_start_ticks']:return 'reused_different_start_ticks'
    return 'possibly_same_owned_live_process'
