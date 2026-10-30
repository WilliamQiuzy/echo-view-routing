"""PW-P1 independent scalar metrics, data/updates/control/state audit."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/pw_small_pilot_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def scalar_metrics(values):
    scores = [float(v) for v in values]
    n = len(scores); pos = scores[1::2]; neg = scores[0::2]
    correct = sum((score > 0) == (i % 2 == 1) for i, score in enumerate(scores))
    wins = sum(1 if p > q else .5 if p == q else 0 for p in pos for q in neg)
    pair = sum(1 if p > q else .5 if p == q else 0 for p, q in zip(pos, neg))
    return {'correct': correct, 'total': n, 'accuracy': correct / n, 'AUROC': wins / (len(pos) * len(neg)),
            'pair_order_points': float(pair), 'pairs': n // 2}


def audit():
    sys.path.insert(0, str(ROOT))
    import torch
    from echo_routing.temporal.pw_pilot import make_model
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    f = json.loads((OUT / 'input_freeze.json').read_text())
    manifest = json.loads((OUT / 'models_manifest.json').read_text())
    readout = json.loads((OUT / 'readout.json').read_text())
    assert sha(OUT / 'inputs.private.npz') == f['input_file_sha256']
    assert sha(OUT / 'readout_scores.private.npz') == readout['score_cache_sha256']
    data = np.load(OUT / 'inputs.private.npz', allow_pickle=False)
    cache = np.load(OUT / 'readout_scores.private.npz', allow_pickle=False)
    membership = json.loads((OUT / 'input_membership.private.json').read_text())
    assert sha(OUT / 'input_membership.private.json') == f['membership_sha256']
    fit_pairs = sorted({r['pair'] for r in membership if r['fold'] != 0})
    held_pairs = sorted({r['pair'] for r in membership if r['fold'] == 0})
    assert len(fit_pairs) == 19 and len(held_pairs) == 6 and not set(fit_pairs) & set(held_pairs)
    for name, pairs in [('fit_images', fit_pairs), ('held_images', held_pairs)]:
        assert np.array_equal(data[name], np.array([[2 * i, 2 * i + 1] for i in pairs]).ravel())
        assert np.array_equal(cache[name], data[name])
    assert len(membership) == 50
    for row in membership:
        a = np.load(ROOT / row['sample_path'] / 'cartesian_no_overlay.npy', allow_pickle=False)
        assert hashlib.sha256(a.tobytes()).hexdigest() == row['input_pixel_sha256']
    rng = np.random.RandomState(4702); stream = []
    while len(stream) < 1200: stream.extend(rng.permutation(19).tolist())
    plan = np.asarray(stream[:1200]).reshape(600, 2)
    assert np.array_equal(plan, data['plan'])
    l = np.tile([0., 1.], (19, 1)); s = l.copy()
    s[np.random.RandomState(4701).permutation(19)[:9]] = [1., 0.]
    assert np.array_equal(l, data['L']) and np.array_equal(s, data['S'])
    geometry = data['geometry']; x = geometry[data['fit_images']]
    mu, scale = x.mean(0), x.std(0); scale[scale == 0] = 1
    design = np.column_stack(((x - mu) / scale, np.ones(38)))
    penalty = np.diag([.01] * 6 + [0.])
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ np.tile([-1., 1.], 19))
    gs = np.column_stack(((geometry - mu) / scale, np.ones(50))) @ coef
    assert np.max(np.abs(gs - cache['geometry_scores'])) < 1e-10
    base_path = ROOT / 'runs/a48_valve_source_intake_v1/backbone_G.pt'
    assert sha(base_path) == f['base_sha256']
    base = torch.load(base_path, map_location='cpu', weights_only=True)
    recomputed, own_label_fit, model_checks = {'fit': {}, 'heldout': {}}, {}, {}
    functional_error, fresh_error, gradient_error = 0., 0., 0.
    for arm in ['L', 'S']:
        assert sha(OUT / f'{arm}.pt') == manifest['models'][arm]['weight_file_sha256']
        assert sha(OUT / f'{arm}.train.private.jsonl') == manifest['models'][arm]['train_log_sha256']
        logs = [json.loads(v) for v in (OUT / f'{arm}.train.private.jsonl').read_text().splitlines()]
        assert len(logs) == 600 and [r['step'] for r in logs] == list(range(1, 601))
        assert np.array_equal([r['fit_pair_positions'] for r in logs], plan)
        assert np.isfinite([r['loss'] for r in logs]).all()
        ck = torch.load(OUT / f'{arm}.pt', map_location='cpu', weights_only=True)
        assert ck['steps'] == 600 and ck['input_file_sha256'] == f['input_file_sha256']
        model = make_model(base)
        initial = model.state_dict()
        selected = {n for n, p in model.named_parameters() if n.startswith(('backbone.body.layer4.', 'backbone.fpn.', 'head.'))}
        assert selected == {n for n, p in model.named_parameters() if p.requires_grad}
        changed = {n for n in initial if not torch.equal(initial[n], ck['model'][n])}
        assert changed == selected
        assert all(torch.isfinite(v).all() for v in ck['model'].values())
        assert len(ck['optimizer']['param_groups']) == 2 and len(ck['optimizer']['state']) == len(selected)
        assert ck['optimizer']['param_groups'][0]['lr'] == 1e-4 and ck['optimizer']['param_groups'][1]['lr'] == 1e-3
        functional = cache[arm + '_features'].astype(np.float64) @ ck['model']['head.weight'].numpy().astype(np.float64).T + ck['model']['head.bias'].numpy()
        functional_error = max(functional_error, float(np.max(np.abs(functional.ravel() - cache[arm + '_scores']))))
        assert functional_error < 1e-4
        # Independent exact head-gradient formula on the frozen final fit subset.
        z = torch.tensor(cache[arm + '_features'][data['fit_images'][:4]], dtype=torch.float64)
        w = ck['model']['head.weight'].double().clone().requires_grad_(); b = ck['model']['head.bias'].double().clone().requires_grad_()
        y = torch.tensor(data[arm].ravel()[:4], dtype=torch.float64)
        loss = torch.nn.functional.binary_cross_entropy_with_logits((z @ w.T + b).flatten(), y)
        gw, gb = torch.autograd.grad(loss, [w, b])
        scores = (z.detach().numpy() @ w.detach().numpy().T + b.detach().numpy()).ravel()
        coefficients = (np.exp(-np.logaddexp(0., -scores)) - y.numpy()) / 4
        error = max(float(np.max(np.abs(coefficients @ z.numpy() - gw.numpy().ravel()))), abs(float(coefficients.sum()) - float(gb)))
        gradient_error = max(gradient_error, error); assert error < 1e-10
        model.load_state_dict(ck['model'], strict=True); model.cuda()
        with torch.no_grad():
            fresh = model(torch.from_numpy(data['images'][data['fit_images'][:2]]).cuda()).cpu().numpy()
        fresh_error = max(fresh_error, float(np.max(np.abs(fresh - cache[arm + '_scores'][data['fit_images'][:2]]))))
        assert fresh_error < 1e-4
        own_label_fit[arm] = int(np.sum((cache[arm + '_scores'][data['fit_images']] > 0) == data[arm].ravel()))
        model_checks[arm] = {'actual_updates_recounted': 600, 'live_tensors_changed': len(selected), 'frozen_tensors_unchanged': True}
        del model
    for name in ['L', 'S', 'geometry', 'constant']:
        for split, indices in [('fit', data['fit_images']), ('heldout', data['held_images'])]:
            result = scalar_metrics(cache[name + '_scores'][indices]); assert result == readout[split][name]
            recomputed[split][name] = result
    a, b, g = (recomputed['heldout'][n] for n in ['L', 'S', 'geometry'])
    signal = bool(a['AUROC'] >= .75 and a['AUROC'] - b['AUROC'] >= .10 and a['AUROC'] - g['AUROC'] >= .10 and a['correct'] >= 8 and a['pair_order_points'] >= 4)
    assert signal == readout['descriptive_directional_signal']
    return {'all_independent_checks_pass': True, 'models': model_checks, 'fit_own_training_labels_correct_of38': own_label_fit,
        'functional_head_max_score_error': functional_error, 'fresh_fixed_fit_subset_max_score_error': fresh_error,
        'independent_head_gradient_max_abs_error': gradient_error, 'metrics': recomputed,
        'descriptive_directional_signal': signal, 'EV9V_promotion_authorized': False,
        'score_cache_sha256': sha(OUT / 'readout_scores.private.npz'), 'model_forwards_on_heldout_during_audit': 0,
        'scope': 'Small source-task diagnosis only; six heldout exams, patient counts unavailable; no A5C/A4C/final-segmentation result.'}


if __name__ == '__main__':
    start, cpu = time.time(), time.process_time()
    value = audit(); value.update(wall_seconds=time.time() - start, process_CPU_seconds=time.process_time() - cpu,
        GPU_wall_upper_hours=(time.time() - start) / 3600, finished_unix=time.time(), audit_code_sha256=sha(__file__))
    (OUT / 'independent_audit.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2), flush=True)
