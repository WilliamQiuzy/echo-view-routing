"""A62 prospective source-grammar work bounds, with no truth/head interface.

Independent engineering module: it changes neither the executed metadata/cost
contracts nor proposal, producer or feature semantics. Counts are work volumes,
not elapsed-time predictions, physical minima, or actual resource expenditures.

For length T: old/raw disagreement components are disjoint (at most ceil(T/2));
J maximal runs are disjoint (at most T). Their changed-position sums are at most
T each. Deduplication and full-axis G only remove actions. Thus C<=ceil(T/2)+T,
sum(K)<=2T, but full-context TC MUST still be T*C, not sum(K).

B3 NMS5 keeps internal cuts at distance >=6. Edges need not have length6.
Rb<=1+ceil((T-1)/6). Actual A08 sequence uses structured_inputs.windows(T,5),
then nearest_centre allocation; so Ra<=W=len(window_centres(T,5)).

Replacing one interval introduces at most two edge cuts: edited run count
Re<=Rb+replacement_runs+1. For nA disjoint RA intervals, the sum of internal
raw-A cuts is <=Ra-1, hence sum(replacement_runs)<=Ra-1+nA. A J action is
constant, giving Re<=Rb+2. Over all C actions this proves
sum(Rb+Re)<=C*(2Rb+2)+Ra-1. We retain the looser C*(2Rb+3)+W bound below.
It applies before and after G/dedup, and never assumes J has stride5 structure.
"""
from numbers import Integral

import numpy as np

from echo_routing.temporal.anchor_edits import aligned
from echo_routing.temporal.proposal_actions import propose, partition, apply


def _length(T):
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, Integral) or T < 1:
        raise ValueError('positive original-axis integer length required')
    return int(T)


def _centres(T):
    # Exact pure windowed.window_centres(T,5) expression. Avoid importing its
    # features.extract->torch dependency in this no-NN geometry module.
    centres = np.arange(2, T, 5, dtype=np.int64)
    if not len(centres) or centres[-1] < T - 3:
        centres = np.append(centres, T - 1)
    return centres


def _window_count(T):
    count = 0 if T <= 2 else (T - 3) // 5 + 1
    if count == 0 or 2 + 5 * (count - 1) < T - 3:
        count += 1
    return count


def axis_bounds(T):
    """Conservative scalar work volumes for one unobserved original axis.

    K is SUM changed positions across all candidates, KT is T times that sum;
    T2/C2 retain independent quadratic capacities. N1 aliases N, not another
    chargeable axis. Rb and W are grammar bounds, never observed model outputs.
    """
    T = _length(T)
    C = (T + 1) // 2 + T
    Rb = 1 + (T + 4) // 6
    W = _window_count(T)
    return {'N': 1, 'N1': 1, 'T': T, 'T2': T * T,
            'C': C, 'C2': C * C, 'TC': T * C,
            'K': 2 * T, 'KT': 2 * T * T,
            'CR': C * (2 * Rb + 3) + W, 'W': W, 'Rb': Rb}


def aggregate_bounds(lengths):
    """Sum PER-AXIS capacities, including zeros for an empty role.

    C2/TC/T2 are sums of per-axis terms, not squares/products of pooled counts.
    max_T is separate; N1 remains an alias and must not be charged twice.
    """
    total = {key: 0 for key in axis_bounds(1)}
    maximum = 0
    for length in lengths:
        row = axis_bounds(length)
        for key in total:
            total[key] += row[key]
        maximum = max(maximum, row['T'])
    return {**total, 'max_T': maximum}


def _cuts(value):
    return np.flatnonzero(value[1:] != value[:-1]) + 1


def observed_geometry(base, raw_a, raw_j):
    """Observe fixed-teacher labels, proposals and G; never accept GT/scores.

    Canonical C/K/CR count eligible unique actions. all_C/all_K/all_CR also
    retain the full pre-G proposal volume required by propose/partition. Ra/Rj
    are actual observed run counts. Structure failure raises before returning
    an estimate. Raw-A transitions must be actual nearest-centre cell edges;
    checking only the number of runs would not verify this producer grammar.
    """
    base, raw_a = aligned(base, raw_a)
    base, raw_j = aligned(base, raw_j)
    T = len(base)
    bounds = axis_bounds(T)
    bcuts, acuts = _cuts(base), _cuts(raw_a)
    if len(bcuts) > 1 and np.any(np.diff(bcuts) < 6):
        raise ValueError('Base violates fixed B3 NMS5 internal cut spacing')
    centres = _centres(T)
    allowed_a_cuts = (centres[:-1] + centres[1:]) // 2 + 1
    if not set(map(int, acuts)) <= set(map(int, allowed_a_cuts)):
        raise ValueError('raw A08 violates stride5 nearest-centre allocation')
    Rb, Ra, Rj = 1 + len(bcuts), 1 + len(acuts), 1 + len(_cuts(raw_j))
    if Rb > bounds['Rb'] or Ra > bounds['W']:
        raise ValueError('fixed producer run bound exceeded')

    actions = propose(base, raw_a, raw_j)
    eligible, rejected = partition(base, actions)
    eligible_keys = {action.key for action in eligible}
    K = CR = all_K = all_CR = 0
    for action in actions:
        edited = apply(base, action)
        changed = int(np.count_nonzero(edited != base))
        run_work = Rb + 1 + len(_cuts(edited))
        all_K += changed
        all_CR += run_work
        if action.key in eligible_keys:
            K += changed
            CR += run_work
    C, all_C = len(eligible), len(actions)
    result = {'N': 1, 'N1': 1, 'T': T, 'T2': T * T,
              'C': C, 'C2': C * C, 'TC': T * C, 'K': K,
              'KT': T * K, 'CR': CR, 'W': len(centres),
              'Rb': Rb, 'Ra': Ra, 'Rj': Rj, 'G_rejected': len(rejected),
              'all_C': all_C, 'all_C2': all_C * all_C,
              'all_TC': T * all_C, 'all_K': all_K,
              'all_KT': T * all_K, 'all_CR': all_CR}
    for prefix in ('', 'all_'):
        for key in ('C', 'C2', 'TC', 'K', 'KT', 'CR'):
            if result[prefix + key] > bounds[key]:
                raise ValueError('observed proposal volume violates proven bound: ' + prefix + key)
        if result[prefix + 'CR'] > result[prefix + 'C'] * (2 * Rb + 3) + len(centres):
            raise ValueError('observed per-candidate RLE work violates structural proof')
    return result
