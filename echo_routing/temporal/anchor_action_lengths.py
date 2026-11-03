"""Card44 descriptors on complete original axes; no scores/labels/GT required."""
import numpy as np


def distribution(values):
    values=np.asarray(values,dtype=np.float64)
    return {'count':len(values),'sum':float(values.sum()),'mean':float(values.mean()) if len(values) else None,
            **{key:float(v) for key,v in zip(['minimum','q25','median','q75','maximum'],np.quantile(values,[0,.25,.5,.75,1]))}} if len(values) else {
            'count':0,'sum':0.,'mean':None,'minimum':None,'q25':None,'median':None,'q75':None,'maximum':None}


def summarize(units):
    if not units:raise ValueError('complete positive original axes required')
    candidate,ratios,accepted,accepted_ratios,all_ratios,axes=[],[],[],[],[],[]
    with_candidates=0
    for unit in units:
        n=unit['T'];spans=[tuple(s) for s in unit['spans']];action=None if unit['accepted'] is None else tuple(unit['accepted'])
        if not isinstance(n,(int,np.integer)) or n<1:raise ValueError('positive integer original T required')
        if (any(len(s)!=2 or any(not isinstance(v,(int,np.integer)) for v in s) or not 0<=s[0]<s[1]<=n for s in spans)
            or any(a[1]>=b[0] for a,b in zip(spans,spans[1:])) or action is not None and action not in spans):
            raise ValueError('ordered disjoint maximal raw-support spans and one admitted action required')
        axes.append(n);with_candidates+=bool(spans)
        candidate.extend(e-s for s,e in spans);ratios.extend((e-s)/n for s,e in spans)
        length=0 if action is None else action[1]-action[0]
        all_ratios.append(length/n)
        if action is not None:accepted.append(length);accepted_ratios.append(length/n)
    positions=sum(axes);selected=sum(accepted)
    return {'sequences':len(units),'positions':positions,'sequence_lengths':distribution(axes),
            'sequences_with_candidates':int(with_candidates),'candidate_count':len(candidate),
            'candidate_lengths':distribution(candidate),'candidate_length_over_T':distribution(ratios),
            'candidate_union_positions':sum(candidate),'candidate_union_positions_over_all_positions':sum(candidate)/positions,
            'accepted_count':len(accepted),'accepted_lengths':distribution(accepted),'accepted_length_over_T':distribution(accepted_ratios),
            'all_sequence_accepted_length_over_T':distribution(all_ratios),
            'accepted_positions':selected,'accepted_positions_over_all_positions':selected/positions,
            'no_action_sequences':len(units)-len(accepted),'unchanged_positions':positions-selected,
            'T_definition':'complete original10Hz sequence positions,not windows/tubelets/candidate length',
            'quantiles':'unweighted complete listed units;whole-axis fraction separately position-weighted'}
