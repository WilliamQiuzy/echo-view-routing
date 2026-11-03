"""Independent exact exposure/schedule/liveness checks of every physical update."""
import math
from echo_routing.temporal.dev_train import lr_factor

def audit_trace(rows,plan,identity,lengths,active_names,expected_updates=6000):
    if len(rows)!=expected_updates or len(plan)<expected_updates:raise ValueError('complete fixed physical optimizer prefix required')
    for u,row in enumerate(rows):
        expected=[int(i) for i in plan[u]]
        if row['u']!=u or row['identity']!=identity or row['plan_row']!=expected or row['physical_sequences']!=2 or row['valid_original_positions']!=sum(lengths[i] for i in expected):raise ValueError('physical update/exposure/plan binding changed')
        if not row['finite_gradients'] or set(row['nonzero_gradient_tensors'])!=set(active_names) or any(not math.isfinite(row[k]) for k in ['loss','native_NLL','constructed_NLL','gradient_norm','lr']):raise ValueError('finite complete active gradient/loss record required')
        if not math.isclose(row['lr'],.001*lr_factor(u,100,6000),rel_tol=1e-12,abs_tol=1e-15):raise ValueError('fixed full6000 schedule differs')
    if not all(any(row['nonzero_gradient_tensors'][name] for row in rows[1:]) for name in active_names):raise ValueError('every selected head tensor must show nonzero later gradient liveness across history')
    return {'physical_optimizer_updates':len(rows),'valid_original_positions':sum(r['valid_original_positions'] for r in rows),'full_plan_exact':True,'schedule_total':6000,'all_selected_tensors_later_gradient_liveness_observed':True,'final_model_digest':rows[-1]['model_digest_after_update']}
