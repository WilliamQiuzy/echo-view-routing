"""A63 independent complete-trace/fullstate recounts; no neural forward.

These schema/physical-exposure checks require caller-supplied independently
recomputed identities and frozen draws. Runtime/OS/file receipts, source
isolation and actual Git objects must be audited separately. A fabricated test
state passing here is not evidence that training occurred.
"""
import math
import re
from numbers import Real

import numpy as np

SEEDS = (2026110461, 2026110462)
EVENT_NAMES = {'network.0.weight', 'network.0.bias', 'network.2.weight', 'network.2.bias'}
RISK_NAMES = {f'{part}.{layer}.{field}' for part in ('position', 'candidate')
              for layer in ('0', '2') for field in ('weight', 'bias')}
KINDS = ('native', 'constructed')


def _identity(value):
    keys = {'code', 'config', 'sources', 'input', 'draw_plan', 'normalizer', 'target_scales'}
    if (not isinstance(value, dict) or set(value) != keys or
            any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None for v in value.values())):
        raise ValueError('independently computed complete seven-SHA identity required')
    return value


def _integer(value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError('canonical integer count/cursor required')
    return int(value)


def _indices(value, n):
    arr = np.asarray(value)
    if (arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer) or
            (arr < 0).any() or (arr >= n).any()):
        raise ValueError('exact nonnegative in-domain integer indices required before casting')
    return arr.astype(np.int64, copy=False)


def _pools(kind_indices, n):
    if not isinstance(kind_indices, dict) or set(kind_indices) != set(KINDS):
        raise ValueError('both native and mixed pools required without fallback')
    pools = {k: _indices(kind_indices[k], n) for k in KINDS}
    if any(not len(v) for v in pools.values()) or not np.array_equal(np.sort(np.concatenate(list(pools.values()))), np.arange(n)):
        raise ValueError('complete disjoint nonempty native/mixed candidate partition required')
    return pools


def _finite_nonnegative(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative loss/gradient/physical timing required')
    return float(value)


def _row(row, update, identity, names):
    from echo_routing.temporal.proposal_training import lr_factor
    try:
        if (_integer(row['u']) != update or row['identity'] != identity or row['finite_gradients'] is not True or
                set(row['nonzero_gradient_tensors']) != names or
                any(type(v) is not bool for v in row['nonzero_gradient_tensors'].values())):
            raise ValueError('physical cursor/expected identity/exact gradient tensor coverage changed')
        for key in ('loss', 'gradient_norm', 'optimizer_update_wall_seconds'):
            _finite_nonnegative(row[key])
        if not math.isclose(_finite_nonnegative(row['lr']), .001*lr_factor(update), rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('original fixed6000 LR schedule changed')
    except (KeyError, TypeError) as exc:
        raise ValueError('complete physical row required') from exc


def _exact_indices(value, expected):
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError('complete ordered physical graph list required')
    if any(_integer(v) != int(e) for v, e in zip(value, expected)):
        raise ValueError('ordered physical graph differs from exact frozen draw')


def audit_aux_trace(rows, data, plan, seed, role, identity):
    """Recount S-L full-query and old-risk all-position exposure independently."""
    from echo_routing.temporal.safety_training import draw_plan
    from echo_routing.temporal.safety_data import exact
    identity = _identity(identity)
    if _integer(seed) not in SEEDS or role not in ('utility', 'risk') or len(rows) != 6000:
        raise ValueError('exact declared6000 auxiliary trace required')
    try:
        targets = np.asarray(data['targets'])
        if targets.ndim != 2 or targets.shape[1] != 2 or not len(targets) or not np.isfinite(targets).all():
            raise ValueError('complete finite two-target candidate rows required')
        n = len(targets); pools = _pools(data['kind_indices'], n)
        offsets = np.asarray(data['position_offsets'])
        if (offsets.shape != (n+1,) or not np.issubdtype(offsets.dtype, np.integer) or
                offsets[0] != 0 or (offsets < 0).any() or (offsets > np.iinfo(np.int64).max).any()):
            raise ValueError('canonical complete changed-position offsets required')
        offsets = offsets.astype(np.int64); K = np.diff(offsets)
        if (K < 1).any():
            raise ValueError('every candidate requires all nonempty changed positions')
        groups = [_indices(g, n) for g in data['groups']]
        nonempty = [i for i, g in enumerate(groups) if len(g)]
        if not np.array_equal(np.sort(np.concatenate([groups[i] for i in nonempty])), np.arange(n)):
            raise ValueError('candidate rows must partition into complete source queries')
        group_pools = {k: _indices(data['kind_groups'][k], len(groups)) for k in KINDS}
        if set(data['kind_groups']) != set(KINDS) or any(not len(v) for v in group_pools.values()):
            raise ValueError('both nonempty query kind pools required')
        if not np.array_equal(np.sort(np.concatenate(list(group_pools.values()))), nonempty):
            raise ValueError('exact disjoint nonempty-query partition required')
        for kind in KINDS:
            members = np.concatenate([groups[q] for q in group_pools[kind]])
            if not np.array_equal(np.sort(members), np.sort(pools[kind])):
                raise ValueError('query kind and candidate kind domains differ')
        selected_pools = group_pools if role == 'utility' else pools
        expected = draw_plan(*(len(selected_pools[k]) for k in KINDS), seed, role)
        if expected['fallback'] or not exact(plan, expected):
            raise ValueError('original complete no-fallback role-specific draw plan changed')
        pair_counts = []
        for group in groups:
            t = np.r_[targets[group, 1], 0.]
            pair_counts.append(int(np.count_nonzero(np.triu(t[:, None] != t, 1))))
        names = EVENT_NAMES if role == 'utility' else RISK_NAMES
        live = dict.fromkeys(names, False)
        totals = dict(physical_updates=6000, query_group_exposures=0, candidate_exposures=0,
                      changed_position_exposures=0, unequal_pair_exposures=0,
                      native_candidate_exposures=0, constructed_candidate_exposures=0)
        for u, row in enumerate(rows):
            _row(row, u, identity, names)
            candidates, queries, counts = [], [], []
            for block, kind in enumerate(KINDS):
                selected = selected_pools[kind][expected['indices'][u, 32*block:32*(block+1)]]
                idx = np.concatenate([groups[q] for q in selected]).tolist() if role == 'utility' else selected.tolist()
                candidates.extend(idx); counts.append(len(idx))
                if role == 'utility': queries.extend(selected.tolist())
            positions = sum(int(K[i]) for i in candidates)
            pairs = sum(pair_counts[q] for q in queries)
            _exact_indices(row['candidate_indices'], candidates)
            _exact_indices(row['query_sequence_indices'], queries)
            _exact_indices(row['source_kind_blocks'], [0, 1])
            if (_integer(row['physical_candidates'], 1) != len(candidates) or
                    _integer(row['changed_position_exposures'], 1) != positions or
                    _integer(row.get('physical_groups', 0)) != len(queries) or
                    _integer(row.get('physical_unequal_pairs', 0)) != pairs or
                    role == 'risk' and _integer(row['physical_changed_positions'], 1) != positions):
                raise ValueError('complete query/candidate/allK/unequal-pair physical exposure changed')
            components = ('pointwise_MSE', 'pairwise_logistic') if role == 'utility' else ('position_CE', 'candidate_BCE')
            loss = sum(_finite_nonnegative(row[k]) for k in components)
            if not math.isclose(row['loss'], loss, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError('fixed auxiliary component loss changed')
            for key, value in [('query_group_exposures', len(queries)), ('candidate_exposures', len(candidates)),
                               ('changed_position_exposures', positions), ('unequal_pair_exposures', pairs),
                               ('native_candidate_exposures', counts[0]), ('constructed_candidate_exposures', counts[1])]:
                totals[key] += value
            for name, flag in row['nonzero_gradient_tensors'].items(): live[name] |= flag
        if not all(live.values()):
            raise ValueError('an active auxiliary tensor never received nonzero gradient')
        return {**totals, 'all_tensor_gradient_liveness_over_history': live,
                'original_no_fallback_draw_verified': True, 'no_model_forward_during_audit': True}
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('complete auxiliary graph/trace schema required') from exc


def audit_event_trace(rows, draws, kind_indices, unsafe, identity):
    """Audit C0/C1 using caller-verified global candidate draws, not old4183."""
    identity = _identity(identity)
    if len(rows) != 6000:
        raise ValueError('exact6000 physical event updates required')
    y = np.asarray(unsafe)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('complete binary candidate event targets required')
    pools = _pools(kind_indices, len(y)); draws = np.asarray(draws)
    if (draws.shape != (6000, 64) or not np.issubdtype(draws.dtype, np.integer) or
            (draws < 0).any() or (draws >= len(y)).any() or
            not np.isin(draws[:, :32], pools['native']).all() or
            not np.isin(draws[:, 32:], pools['constructed']).all()):
        raise ValueError('frozen global native32/mixed32 complete candidate draws required')
    live = dict.fromkeys(EVENT_NAMES, False); total = 0
    try:
        for u, row in enumerate(rows):
            _row(row, u, identity, EVENT_NAMES)
            _exact_indices(row['candidate_indices'], draws[u]); _exact_indices(row['source_kind_blocks'], [0, 1])
            count = int(y[draws[u]].sum())
            if (_integer(row['physical_candidates']) != 64 or _integer(row['unsafe_targets']) != count or
                    _finite_nonnegative(row['candidate_BCE']) != row['loss']):
                raise ValueError('exact64 candidate exposure, unsafe count or ordinary BCE changed')
            total += count
            for name, flag in row['nonzero_gradient_tensors'].items(): live[name] |= flag
        if not all(live.values()):
            raise ValueError('an active event tensor never received nonzero gradient')
    except (KeyError, TypeError) as exc:
        raise ValueError('complete physical event trace required') from exc
    return dict(physical_updates=6000, candidate_exposures=384000,
                native_candidate_exposures=192000, constructed_candidate_exposures=192000,
                unsafe_candidate_exposures=total, all_tensor_gradient_liveness_over_history=live,
                caller_frozen_global_draws_verified=True, no_model_forward_during_audit=True)


def _shapes(role):
    def network(prefix, n, out):
        return {prefix+'.0.weight': (32, n), prefix+'.0.bias': (32,),
                prefix+'.2.weight': (out, 32), prefix+'.2.bias': (out,)}
    if role == 'utility': return network('network', 109, 1)
    if role == 'risk': return {**network('position', 47, 3), **network('candidate', 109, 1)}
    return network('network', 205, 1)


def _validate_states(model, opt, states, identity, norms, initial_state, role, scale=None):
    import random
    import torch
    from echo_routing.temporal import safety_training, event_risk_model
    from echo_routing.temporal.event_risk_audit import tensor_equal
    identity = _identity(identity)
    if torch.cuda.is_initialized():
        raise ValueError('CPU audit must not inherit an initialized CUDA context')
    shapes = _shapes(role)
    current = dict(model.named_parameters())
    if (_integer(model.seed) not in SEEDS or set(current) != set(shapes) or
            set(initial_state) != set(shapes) or set(states) != {2000, 4000, 6000, 'latest'} or
            any(type(k) is not int for k in states if k != 'latest')):
        raise ValueError('exact architecture, seed, initializer and four fullstates required')
    if ((role == 'utility' and (model.role, model.arm) != ('utility', 'S-L')) or
            (role == 'risk' and (model.role, model.arm) != ('risk', 'risk')) or
            (role == 'event' and (model.role != 'candidate_unsafe_event' or model.arm not in ('C0', 'C1')))):
        raise ValueError('declared auxiliary S-L/risk or C0/C1 role required')
    for name, value in current.items():
        first = initial_state[name]
        if (tuple(value.shape) != shapes[name] or value.dtype != torch.float32 or value.is_cuda or
                not isinstance(first, torch.Tensor) or first.shape != value.shape or first.dtype != value.dtype or
                first.is_cuda or not bool(torch.isfinite(first).all())):
            raise ValueError('complete CPU FP32 architecture and finite matched initializer required')
    backend = event_risk_model if role == 'event' else safety_training
    # Qualified restore verifies every Adam tensor, identity, exact LR, normalizer,
    # scale and RNG schema. Save global RNG even on failure; audit never draws.
    rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    try:
        for cursor in (2000, 4000, 6000, 'latest'):
            value = states[cursor]; expected = 6000 if cursor == 'latest' else cursor
            keys = {'model', 'optimizer', 'update', 'arm', 'role', 'seed', 'schedule_total', 'identity', 'normalizers', 'rng'}
            keys.add('target_semantics' if role == 'event' else 'utility_scale')
            if (not isinstance(value, dict) or set(value) != keys or _integer(value['update']) != expected or
                    set(value['rng']) != {'torch_CPU', 'numpy', 'python'} or
                    any(not isinstance(v, torch.Tensor) or v.is_cuda or v.requires_grad for v in value['model'].values())):
                raise ValueError('complete canonical CPU detached fullstate/RNG/cursor required')
            args = (model, opt, value, identity, norms) if role == 'event' else (model, opt, value, identity, norms, scale)
            if backend.restore(*args) != expected:
                raise ValueError('qualified restore cursor differs from milestone')
            backend.validate_optimizer(model, value['optimizer'], expected)
            for p in model.parameters():
                if any(not bool(opt.state[p][k].count_nonzero()) for k in ('exp_avg', 'exp_avg_sq')):
                    raise ValueError('an active tensor has no nonzero Adam moments')
        if not tensor_equal(states[6000], states['latest']):
            raise ValueError('latest/final complete model/Adam/RNG/identity state differs')
        if any(torch.equal(initial_state[n], value) for n, value in model.state_dict().items()):
            raise ValueError('an active tensor never drifted from matched initializer')
    except (KeyError, TypeError, RuntimeError, OverflowError) as exc:
        raise ValueError('invalid complete checkpoint/Adam/RNG identity') from exc
    finally:
        torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
    return dict(fullstates=4, milestones=[2000, 4000, 6000], latest_cursor=6000,
                parameters=sum(p.numel() for p in model.parameters()),
                latest_final_complete_state_equal=True, all_active_tensors_drifted=True,
                all_Adam_tensors_and_norm_RNG_identities_validated=True,
                no_model_forward_during_audit=True)


def validate_aux_states(model, opt, states, identity, data, initial_state):
    if model.role not in ('utility', 'risk'):
        raise ValueError('auxiliary role required')
    return _validate_states(model, opt, states, identity, data['normalizers'], initial_state,
                            model.role, data['scale'])


def validate_event_states(model, opt, states, identity, norms, initial_state):
    return _validate_states(model, opt, states, identity, norms, initial_state, 'event')
