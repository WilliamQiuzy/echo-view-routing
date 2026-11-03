"""Privacy-whitelisted fixed training bins; these are FIT objectives, not validation scores."""
import math

def binned_curve(rows,block=100,expected_updates=6000):
    if len(rows)!=expected_updates or [r['u'] for r in rows]!=list(range(expected_updates)) or expected_updates%block:raise ValueError('complete fixed contiguous training trace and exact bins required')
    out=[]
    for start in range(0,len(rows),block):
        sample=rows[start:start+block];keys=['loss','native_NLL','constructed_NLL','gradient_norm','lr','step_wall_seconds_before_journal']
        if any(not math.isfinite(r[k]) for r in sample for k in keys) or any(r['valid_original_positions']<1 for r in sample):raise ValueError('finite objective/timing and positive exposure records required')
        mean=lambda key:math.fsum(r[key] for r in sample)/len(sample)
        out.append({'first_update':start,'last_update_exclusive':start+block,'optimizer_updates':block,'training_NLL_equal_update_mean':mean('loss'),'native_NLL_equal_update_mean':mean('native_NLL'),'constructed_NLL_equal_update_mean':mean('constructed_NLL'),'unclipped_gradient_norm_mean':mean('gradient_norm'),'unclipped_gradient_norm_max':max(r['gradient_norm'] for r in sample),'unclipped_gradient_norm_gt1_fraction':sum(r['gradient_norm']>1 for r in sample)/len(sample),'first_lr':sample[0]['lr'],'last_lr':sample[-1]['lr'],'valid_original_positions':sum(r['valid_original_positions'] for r in sample),'optimizer_step_wall_before_journal_mean_seconds':mean('step_wall_seconds_before_journal')})
    return out
