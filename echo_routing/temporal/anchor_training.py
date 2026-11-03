"""A59 fixed CPU utility regressor, balanced plan, and fail-closed lifecycle.

The entrypoint must hide CUDA before importing torch. Features and targets are
detached cached tensors; no frozen producer belongs to this optimizer.
"""
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


class UtilityRegressor(torch.nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        if arm not in ['A', 'J']:
            raise ValueError('declared A/J evidence head required')
        self.arm, self.seed = arm, int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network = torch.nn.Sequential(torch.nn.Linear(98, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
            torch.nn.init.zeros_(self.network[2].weight)
            torch.nn.init.zeros_(self.network[2].bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 98:
            raise ValueError('fixed normalized N by 98 features required')
        return self.network(features).squeeze(-1)


def draw_plan(native_candidates, constructed_candidates, seed):
    counts = [native_candidates, constructed_candidates]
    if any(int(n) != n or n < 0 for n in counts):
        raise ValueError('nonnegative complete FIT candidate counts required')
    out = {'no_action_domain': not any(counts)}
    for kind, (name, count) in enumerate(zip(['native', 'constructed'], counts)):
        out[name] = np.random.default_rng([int(seed), 59, kind]).integers(count, size=(6000, 32), dtype=np.int64) if count else np.empty((6000, 0), np.int64)
    return out


def lr_factor(update):
    if not 0 <= update < 6000:
        raise ValueError('fixed six-thousand-update schedule required')
    return (update + 1) / 100 if update < 100 else .5 * (1 + math.cos(math.pi * (update - 100) / 5900))


def optimizer(model):
    named = list(model.named_parameters())
    return torch.optim.AdamW([{'params': [p for _, p in named], 'param_names': [n for n, _ in named],
                               'name': 'anchor_utility', 'base_lr': .001, 'lr': .001}],
                             weight_decay=.01, betas=(.9, .999), eps=1e-8)


def recipe(group):
    return (group.get('name') == 'anchor_utility' and group.get('base_lr') == .001
            and group.get('weight_decay') == .01 and tuple(group.get('betas', ())) == (.9, .999)
            and group.get('eps') == 1e-8 and not group.get('amsgrad', False)
            and not group.get('maximize', False) and not group.get('differentiable', False)
            and math.isfinite(group.get('lr', float('nan'))) and 0 <= group['lr'] <= .001)


def optimizer_binding(model, opt):
    named = list(model.named_parameters())
    return (isinstance(opt, torch.optim.AdamW) and len(opt.param_groups) == 1
            and recipe(opt.param_groups[0]) and opt.param_groups[0]['param_names'] == [n for n, _ in named]
            and [id(p) for p in opt.param_groups[0]['params']] == [id(p) for _, p in named]
            and all(p.requires_grad and not p.is_cuda for _, p in named))


def validate_optimizer_state(model, payload, update):
    if int(update) != update or not 0 <= update <= 6000:
        raise ValueError('invalid checkpoint cursor')
    groups = payload['param_groups']
    named = list(model.named_parameters())
    if (len(groups) != 1 or not recipe(groups[0]) or groups[0].get('param_names') != [n for n, _ in named]
            or len(groups[0]['params']) != len(named) or len(set(groups[0]['params'])) != len(named)):
        raise ValueError('checkpoint optimizer recipe/coverage changed')
    expected_lr = .001 if update == 0 else .001 * lr_factor(update - 1)
    if not math.isclose(groups[0]['lr'], expected_lr, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('optimizer schedule/cursor mismatch')
    state = payload['state']
    if set(state) != (set(groups[0]['params']) if update else set()):
        raise ValueError('optimizer exact active state coverage changed')
    for key, (_, parameter) in zip(groups[0]['params'], named):
        if not update:
            continue
        value = state[key]
        if set(value) != {'step', 'exp_avg', 'exp_avg_sq'} or float(value['step']) != update:
            raise ValueError('cursor differs from actual Adam counters')
        for name in ['exp_avg', 'exp_avg_sq']:
            tensor = value[name]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype or tensor.is_cuda or not bool(torch.isfinite(tensor).all()):
                raise ValueError('optimizer moment shape/dtype/finite contract changed')
        if bool((value['exp_avg_sq'] < 0).any()):
            raise ValueError('negative Adam second moment')


def step(model, opt, kind_batches, update):
    if len(kind_batches) != 2 or not optimizer_binding(model, opt):
        raise ValueError('fixed two-kind CPU optimizer binding required')
    validate_optimizer_state(model, opt.state_dict(), update)
    active = [(kind, value) for kind, value in enumerate(kind_batches) if value is not None]
    if not active:
        raise ValueError('no candidate domain cannot execute pretend updates')
    for _, (features, target) in active:
        if (features.shape != (32, 98) or target.shape != (32,) or features.dtype != torch.float32
                or target.dtype != torch.float32 or features.is_cuda or target.is_cuda
                or features.requires_grad or target.requires_grad
                or not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(target).all())):
            raise ValueError('32 detached finite FP32 CPU candidates per nonempty kind required')
    opt.param_groups[0]['lr'] = .001 * lr_factor(update)
    opt.zero_grad(set_to_none=True)
    losses = [(kind, torch.nn.functional.mse_loss(model(features), target)) for kind, (features, target) in active]
    loss = torch.stack([value for _, value in losses]).mean()
    loss.backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    if not bool(torch.isfinite(loss)) or any(g is None or not bool(torch.isfinite(g).all()) for g in gradients.values()):
        raise ValueError('missing/nonfinite utility loss or gradient')
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
    if not math.isfinite(norm):
        raise ValueError('nonfinite unclipped gradient norm')
    opt.step()
    return {'u': int(update), 'loss': float(loss.detach()),
            'native_MSE': next((float(value.detach()) for kind, value in losses if kind == 0), None),
            'constructed_MSE': next((float(value.detach()) for kind, value in losses if kind == 1), None),
            'gradient_norm': norm, 'lr': opt.param_groups[0]['lr'], 'finite_gradients': True,
            'nonzero_gradient_tensors': {name: bool(g.count_nonzero()) for name, g in gradients.items()},
            'physical_candidates': 32 * len(active)}


def checked_identity(identity):
    if not isinstance(identity, dict) or set(identity) != {'code', 'config', 'sources', 'input', 'draw_plan', 'normalizer'}:
        raise ValueError('complete code/config/source/input/plan/normalizer SHA binding required')
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in identity.values()):
        raise ValueError('SHA256 identity fields required')
    return copy.deepcopy(identity)


def normalizer_tensors(normalizer):
    value = {key: torch.as_tensor(np.asarray(normalizer[key])).clone().to(torch.float64) for key in ['mean', 'std']}
    if any(v.shape != (98,) or not bool(torch.isfinite(v).all()) for v in value.values()) or bool((value['std'] < 1e-6).any()):
        raise ValueError('frozen finite 98-dimensional FIT normalizer required')
    return value


def checkpoint(model, opt, update, identity, normalizer):
    if not optimizer_binding(model, opt):
        raise ValueError('checkpoint optimizer binding changed')
    validate_optimizer_state(model, opt.state_dict(), update)
    numpy_rng = list(np.random.get_state())
    numpy_rng[1] = numpy_rng[1].tolist()
    return {'model': {k: v.detach().clone() for k, v in model.state_dict().items()},
            'optimizer': copy.deepcopy(opt.state_dict()), 'update': int(update), 'arm': model.arm,
            'seed': model.seed, 'schedule_total': 6000, 'identity': checked_identity(identity),
            'normalizer': normalizer_tensors(normalizer),
            'rng': {'torch_CPU': torch.get_rng_state(), 'numpy': numpy_rng, 'python': random.getstate()}}


def restore(model, opt, value, identity, normalizer):
    if (value['identity'] != checked_identity(identity) or value['arm'] != model.arm
            or value['seed'] != model.seed or value['schedule_total'] != 6000 or not optimizer_binding(model, opt)):
        raise ValueError('head/source/config/input/plan identity changed')
    expected_norm = normalizer_tensors(normalizer)
    if set(value['normalizer']) != set(expected_norm) or any(not torch.equal(value['normalizer'][k], v) for k, v in expected_norm.items()):
        raise ValueError('FIT normalizer identity changed')
    expected = model.state_dict()
    if set(value['model']) != set(expected) or any(value['model'][k].shape != v.shape or value['model'][k].dtype != v.dtype or not bool(torch.isfinite(value['model'][k]).all()) for k, v in expected.items()):
        raise ValueError('model complete keys/shape/dtype/finite contract changed')
    validate_optimizer_state(model, value['optimizer'], value['update'])
    model.load_state_dict(value['model'], strict=True)
    opt.load_state_dict(value['optimizer'])
    torch.set_rng_state(value['rng']['torch_CPU'])
    numpy_rng = list(value['rng']['numpy'])
    numpy_rng[1] = np.asarray(numpy_rng[1], dtype=np.uint32)
    np.random.set_state(tuple(numpy_rng))
    random.setstate(value['rng']['python'])
    if not optimizer_binding(model, opt):
        raise ValueError('restored optimizer lost exact live binding')
    return int(value['update'])


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('xb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


class UpdateJournal:
    """A pending update or uncheckpointed physical prefix stops resume.

    Latest checkpoints are committed every100 updates by the runner. A failure
    between those boundaries is retained as charged work, never silently replayed.
    """
    def __init__(self, folder, identity):
        self.folder, self.identity = Path(folder), checked_identity(identity)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.pending, self.trace = self.folder / 'pending.json', self.folder / 'physical_updates.jsonl'
        source = self.folder / 'identity.json'
        if source.exists():
            if json.loads(source.read_text()) != self.identity:
                raise ValueError('journal complete identity changed')
        else:
            with source.open('x') as handle:
                json.dump(self.identity, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
        if self.pending.exists():
            raise ValueError('ambiguous pending physical update; no automatic replay')
        self.cursor = 0
        for line in self.trace.read_text().splitlines() if self.trace.exists() else []:
            row = json.loads(line)
            if row['u'] != self.cursor or row['identity'] != self.identity:
                raise ValueError('physical trace gap/identity changed')
            self.cursor += 1

    def resume_cursor(self):
        if self.pending.exists():
            raise ValueError('ambiguous pending update')
        latest = self.folder / 'latest.pt'
        checkpoint_cursor = torch.load(latest, weights_only=True)['update'] if latest.exists() else 0
        if checkpoint_cursor != self.cursor:
            raise ValueError('physical cursor differs from latest checkpoint; no silent replay')
        return self.cursor

    def begin(self, update):
        if update != self.cursor or self.pending.exists():
            raise ValueError('out-of-order or pending physical update')
        with self.pending.open('x') as handle:
            json.dump({'u': int(update), 'identity': self.identity}, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())

    def finish(self, row, checkpoint_value=None):
        if not self.pending.exists() or json.loads(self.pending.read_text()) != {'u': self.cursor, 'identity': self.identity} or row['u'] != self.cursor:
            raise ValueError('pending/trace cursor identity changed')
        if checkpoint_value is not None:
            if checkpoint_value['update'] != self.cursor + 1 or checkpoint_value['identity'] != self.identity:
                raise ValueError('checkpoint commit cursor/identity changed')
            atomic_save(checkpoint_value, self.folder / 'latest.pt')
        with self.trace.open('a') as handle:
            handle.write(json.dumps({**row, 'identity': self.identity}, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.pending.unlink()
        self.cursor += 1
