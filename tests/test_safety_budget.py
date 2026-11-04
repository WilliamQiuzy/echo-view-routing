import importlib
import numpy as np
from echo_routing.temporal.proposal_actions import propose,partition,apply


def test_structural_cost_cases_cover_dense_C_and_long_K_without_truth_or_training():
    m=importlib.import_module('echo_routing.temporal.safety_budget')
    cases=m.structural_cases(701);assert set(cases)=={'dense_C','long_K'}
    counts={}
    for name,(a,j,b) in cases.items():
        actions,rejected=partition(b,propose(b,a.argmax(1),j.argmax(1)))
        counts[name]=(len(actions),sum(np.count_nonzero(apply(b,x)!=b) for x in actions))
        assert len(a)==len(j)==len(b)==701
    assert counts['dense_C'][0]>701 and counts['long_K'][1]>=700


def test_complete_budget_never_omits_fixed_matrix_and_fullstate_storage():
    m=importlib.import_module('echo_routing.temporal.safety_budget')
    roles={'inner_cal':{'sequences':928,'positions':79000,'max_length':561},'selection':{'sequences':541,'positions':37000,'max_length':701},'a61_confirmation_selection':{'sequences':120,'positions':16000,'max_length':339,'a08_stride5_windows':3200,'jepa_chunks':550},'a61_confirmation_calibration':{'sequences':120,'positions':16500,'max_length':293,'a08_stride5_windows':3350,'jepa_chunks':570}}
    checks={'timings':{k:{'seconds_per_update_including_plan_batch_forward_backward_trace_and_full_checkpoint_IO':.03} for k in ['utility','risk']}}
    geometry={k:{'sequences':v['sequences'],'T_sum':v['positions'],'C_sum':1000,'K_sum':2000,'TC_sum':400000} for k,v in roles.items() if not k.startswith('a61')}
    costs={'role_geometry':geometry,'structural_cases':[{'name':'dense_C','T':701,'C':1051,'K':1051,'max_seconds':1.},{'name':'long_K','T':701,'C':1,'K':700,'max_seconds':.1}],'largest_full_checkpoint_bytes':90000,'maximum_engineering_trace_row_bytes':4000}
    out=m.complete(roles,checks,costs,100000000)
    assert out['formal_updates_remaining']==60000 and out['fullstate_checkpoint_files']==40
    assert out['mandatory_remaining_matrix']=={'metric_rows':314,'comparisons':240,'bootstrap_replicates_each':5000,'oracles':60,'old_G_reproductions':16,'native_cells':56}
    assert out['remaining_CPU_seconds']>1800 and out['remaining_GPU_seconds']>0 and out['incremental_tracked_path_peak_upper_bytes']>4000*60000
