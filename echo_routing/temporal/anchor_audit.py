"""Independent exact matrix/draw/schedule/liveness recount, never last-batch gates."""
import math
import numpy as np
from echo_routing.temporal.anchor_training import lr_factor


def exact_models(rows):
    expected={(s,a) for s in [2026110359,2026110360] for a in ['A','J']}
    actual=[(r['seed'],r['arm']) for r in rows]
    if len(actual)!=4 or set(actual)!=expected:
        raise ValueError('exact two-seed two-arm physical matrix required')
    return True


def validate_trace(rows,data,plan,identity,updates=6000):
    if len(rows)!=updates:
        raise ValueError('complete physical update trace required')
    names=None; live={}
    for u,row in enumerate(rows):
        indices=np.concatenate([data['kind_indices'][k][plan[k][u]] for k in ['native','constructed']]).tolist()
        positions=sum(data['candidate_rows'][i]['feature_positions'] for i in indices)
        if (row['u']!=u or row['identity']!=identity or row['candidate_indices']!=indices
            or row['physical_candidates']!=len(indices) or row['physical_feature_positions']!=positions
            or not math.isclose(row['lr'],.001*lr_factor(u),rel_tol=1e-12,abs_tol=1e-15)
            or row['finite_gradients'] is not True or not all(math.isfinite(row[k]) for k in ['loss','gradient_norm'])):
            raise ValueError('physical plan/exposure/schedule/finite trace mismatch')
        gradients=row['nonzero_gradient_tensors']
        if names is None:names=set(gradients);live={n:False for n in names}
        if set(gradients)!=names or any(type(v) is not bool for v in gradients.values()):
            raise ValueError('complete per-tensor gradient trace required')
        for n,v in gradients.items():live[n]|=v
    return live
