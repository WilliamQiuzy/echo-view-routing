"""PW-P1 matched small-source diagnostic. No EV9V inputs or promotion path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from echo_routing.temporal.pw_pilot import pair_plan, fit_labels, metrics, fit_geometry, geometry_scores, make_model

OUT = ROOT / 'runs/pw_small_pilot_v1'
SOURCE = ROOT / 'runs/a50_clinical_acquisition_v1'
BASE = ROOT / 'runs/a48_valve_source_intake_v1/backbone_G.pt'
BASE_SHA = '4ae0006ff349897f5b73d9c20da9ff33919f2d23db2cc1b7a5e00e9c08b1d42a'
PAIR_SHA = 'addb3daaa0f896edf25406012f8097be61bd9df939108fff577b66b28d0cfd14'
CODE = ['scripts/train_pw_small_pilot.py', 'echo_routing/temporal/pw_pilot.py']
CARD = 'docs/superpowers/plans/2026-10-30-pw-small-pilot.md'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def dump(name, value):
    p = OUT / name; temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(p)


def state_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def prepare():
    import torch
    from torch.nn import functional as F
    assert not (OUT / 'input_freeze.json').exists()
    assert sha(BASE) == BASE_SHA and sha(SOURCE / 'cohort_pairs.private.json') == PAIR_SHA
    pairs = json.loads((SOURCE / 'cohort_pairs.private.json').read_text())
    assert len(pairs) == 25
    frames, geometry, private, fit, heldout = [], [], [], [], []
    for i, pair in enumerate(pairs):
        (heldout if pair['provider_fold'] == 0 else fit).append(i)
        for purpose in ['MITRAL_INFLOW', 'LVOT']:
            image = next(r for r in pair['images'] if r['purpose'] == purpose)
            p = SOURCE / image['sample_relative_path']
            a = np.load(p / 'cartesian_no_overlay.npy', allow_pickle=False)
            assert a.dtype == np.uint8 and hashlib.sha256(a.tobytes()).hexdigest() == image['cartesian_sha256']
            tensor = torch.from_numpy(a.copy()).float()[None, None]
            tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)[0].repeat(3, 1, 1) / 255
            tensor = (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
            frames.append(tensor.numpy())
            attrs = json.loads((p / '.zattrs').read_text()); assert attrs['exam_id'] == pair['exam']
            g = attrs['recording_manifest']['sectors'][0]['geometry']
            geometry.append([g['depth_start_m'], g['depth_end_m'], g.get('tilt_rad', 0), g['width_rad'], *g['grid_size']])
            private.append({'pair': i, 'purpose': purpose, 'exam': pair['exam'], 'fold': pair['provider_fold'],
                            'sample_path': str(p.relative_to(ROOT)), 'input_pixel_sha256': image['cartesian_sha256']})
    assert len(fit) == 19 and len(heldout) == 6
    fit_images = np.asarray([[2 * i, 2 * i + 1] for i in fit], np.int64).ravel()
    held_images = np.asarray([[2 * i, 2 * i + 1] for i in heldout], np.int64).ravel()
    plan = pair_plan(); labels = fit_labels(); geometry = np.asarray(geometry, np.float64)
    np.savez_compressed(OUT / 'inputs.private.npz', images=np.stack(frames), geometry=geometry,
        fit_images=fit_images, held_images=held_images, plan=plan, L=labels['L'], S=labels['S'])
    control = fit_geometry(geometry[fit_images])
    dump('geometry_control.json', {k: v.tolist() for k, v in control.items()})
    dump('input_membership.private.json', private)
    result = {'diagnostic': 'PW-P1; original A50-PW quantity failure unchanged', 'fit_pairs': 19, 'heldout_pairs': 6,
        'fit_images': 38, 'heldout_images': 12, 'steps_per_arm': 600, 'batch_images': 4,
        'base_sha256': BASE_SHA, 'original_pair_sha256': PAIR_SHA, 'input_file_sha256': sha(OUT / 'inputs.private.npz'),
        'membership_sha256': sha(OUT / 'input_membership.private.json'), 'geometry_control_sha256': sha(OUT / 'geometry_control.json'),
        'plan_sha256': hashlib.sha256(plan.tobytes()).hexdigest(), 'shuffled_fit_pairs': 9,
        'code_sha256': {p: sha(ROOT / p) for p in CODE}, 'card_sha256': sha(ROOT / CARD),
        'model_forwards': 0, 'GPU_hours': 0, 'EV9V_promotion_authorized': False}
    dump('input_freeze.json', result); return result


def runtime():
    import torch
    torch.set_num_threads(1); torch.manual_seed(4703)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    f = json.loads((OUT / 'input_freeze.json').read_text())
    assert sha(BASE) == BASE_SHA and sha(OUT / 'inputs.private.npz') == f['input_file_sha256']
    assert sha(ROOT / CARD) == f['card_sha256']
    for p, digest in f['code_sha256'].items(): assert sha(ROOT / p) == digest
    state = torch.load(BASE, map_location='cpu', weights_only=True)
    assert len(state) == 281
    data = np.load(OUT / 'inputs.private.npz', allow_pickle=False)
    return torch, state, data, f


def optimizer(torch, model):
    body = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    return torch.optim.SGD([{'params': body, 'lr': 1e-4}, {'params': model.head.parameters(), 'lr': 1e-3}],
        momentum=.9, weight_decay=1e-4, foreach=False)


def check(commit):
    torch, state, data, f = runtime()
    from torch.nn import functional as F
    x = torch.from_numpy(data['images'][data['fit_images']]).cuda()
    plan = data['plan']; labels = torch.from_numpy(data['L']).cuda()
    outcomes, starts = [], []
    for _ in range(2):
        model = make_model(state).cuda(); starts.append(state_sha(model.state_dict()))
        opt = optimizer(torch, model)
        for row in plan[:2]:
            ids = np.column_stack([2 * row, 2 * row + 1]).ravel()
            opt.zero_grad(set_to_none=True); loss = F.binary_cross_entropy_with_logits(model(x[ids]), labels[row].flatten())
            loss.backward(); opt.step()
        outcomes.append(state_sha(model.state_dict()))
        del model, opt
    assert starts[0] == starts[1] and outcomes[0] == outcomes[1] and starts[0] != outcomes[0]
    result = {'source_commit': commit, 'repeat2_updates_bitwise_equal': True, 'initial_state_sha256': starts[0],
        'changed_after2_updates': True, 'actual_check_update_calls': 4, 'fit_only_check': True,
        'device': torch.cuda.get_device_name(0), 'pid': os.getpid(), 'input_file_sha256': f['input_file_sha256']}
    dump('checks.json', result); return result


def train(commit):
    torch, state, data, f = runtime()
    from torch.nn import functional as F
    checks = json.loads((OUT / 'checks.json').read_text())
    assert not (OUT / 'models_manifest.json').exists()
    x = torch.from_numpy(data['images'][data['fit_images']]).cuda()
    summaries = {}
    for arm in ['L', 'S']:
        assert not (OUT / f'{arm}.pt').exists()
        model = make_model(state).cuda(); initial = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        assert state_sha(initial) == checks['initial_state_sha256']
        opt = optimizer(torch, model); labels = torch.from_numpy(data[arm]).cuda()
        live = {n: False for n, p in model.named_parameters() if p.requires_grad}
        with (OUT / f'{arm}.train.private.jsonl').open('x') as log:
            for step, row in enumerate(data['plan'], 1):
                ids = np.column_stack([2 * row, 2 * row + 1]).ravel()
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(x[ids]), labels[row].flatten())
                assert torch.isfinite(loss)
                loss.backward()
                if step in [2, 600]:
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            assert p.grad is not None and torch.isfinite(p.grad).all()
                            live[n] |= bool(torch.count_nonzero(p.grad))
                opt.step()
                log.write(json.dumps({'step': step, 'fit_pair_positions': row.tolist(), 'loss': float(loss.detach())}) + '\n')
                if step == 1 or step % 25 == 0:
                    torch.cuda.synchronize()
                    dump('progress.json', {'arm': arm, 'step': step, 'total_steps': 600, 'pid': os.getpid(),
                        'device': torch.cuda.get_device_name(0), 'training_loss': float(loss.detach()), 'heldout_forwards': 0})
        final = {n: t.detach().cpu() for n, t in model.state_dict().items()}
        frozen = [n for n in final if n not in live]
        assert all(torch.equal(final[n], initial[n]) for n in frozen)
        drift = {n: float((final[n] - initial[n]).abs().max()) for n in live}
        assert all(live.values()) and all(v > 0 for v in drift.values())
        torch.save({'model': final, 'optimizer': opt.state_dict(), 'steps': 600,
            'initial_state_sha256': checks['initial_state_sha256'], 'input_file_sha256': f['input_file_sha256']}, OUT / f'{arm}.pt')
        summaries[arm] = {'steps': 600, 'weight_file_sha256': sha(OUT / f'{arm}.pt'), 'model_state_sha256': state_sha(final),
            'initial_state_sha256': checks['initial_state_sha256'], 'train_log_sha256': sha(OUT / f'{arm}.train.private.jsonl'),
            'live_tensor_count': len(live), 'all_selected_gradients_nonzero': True, 'all_selected_tensors_changed': True,
            'frozen_tensors_unchanged': True, 'drift_by_selected_tensor': drift}
        del model, opt, final
    result = {'source_commit': commit, 'models': summaries, 'input_freeze_sha256': sha(OUT / 'input_freeze.json'),
        'geometry_control_sha256': sha(OUT / 'geometry_control.json'), 'code_sha256': f['code_sha256'],
        'card_sha256': f['card_sha256'], 'heldout_forwards': 0, 'new_EV9V_readout': False, 'promotable': False}
    dump('models_manifest.json', result); return result


def readout(commit):
    assert not (OUT / 'readout_started.json').exists(), 'one full heldout readout only'
    torch, state, data, f = runtime()
    manifest = json.loads((OUT / 'models_manifest.json').read_text())
    dump('readout_started.json', {'frozen_manifest_commit': commit, 'started_unix': time.time(), 'pid': os.getpid()})
    cache = {'fit_images': data['fit_images'], 'held_images': data['held_images']}
    x = torch.from_numpy(data['images']).cuda()
    for arm in ['L', 'S']:
        assert sha(OUT / f'{arm}.pt') == manifest['models'][arm]['weight_file_sha256']
        ck = torch.load(OUT / f'{arm}.pt', map_location='cpu', weights_only=True)
        model = make_model(state).cuda(); model.load_state_dict(ck['model'], strict=True)
        vectors, logits = [], []
        with torch.no_grad():
            for start in range(0, len(x), 4):
                z = model.features(x[start:start + 4]); vectors.append(z.cpu().numpy())
                logits.append(model.head(z).flatten().cpu().numpy())
        cache[arm + '_features'] = np.concatenate(vectors); cache[arm + '_scores'] = np.concatenate(logits)
        cache[arm + '_head_weight'] = ck['model']['head.weight'].numpy(); cache[arm + '_head_bias'] = ck['model']['head.bias'].numpy()
        del model
    control = {k: np.asarray(v) for k, v in json.loads((OUT / 'geometry_control.json').read_text()).items()}
    cache['geometry_scores'] = geometry_scores(data['geometry'], control)
    cache['constant_scores'] = np.zeros(len(x))
    np.savez_compressed(OUT / 'readout_scores.private.npz', **cache)
    result = {'frozen_manifest_commit': commit, 'fit': {}, 'heldout': {}, 'new_EV9V_readout': False, 'promotable': False,
        'score_cache_sha256': sha(OUT / 'readout_scores.private.npz'), 'heldout_pairs': 6, 'heldout_images': 12}
    for name in ['L', 'S', 'geometry', 'constant']:
        scores = cache[name + '_scores']
        result['fit'][name] = metrics(scores[data['fit_images']])
        result['heldout'][name] = metrics(scores[data['held_images']])
    l, s, g = (result['heldout'][a] for a in ['L', 'S', 'geometry'])
    result['descriptive_directional_signal'] = bool(l['AUROC'] >= .75 and l['AUROC'] - s['AUROC'] >= .10 and
        l['AUROC'] - g['AUROC'] >= .10 and l['correct'] >= 8 and l['pair_order_points'] >= 4)
    truth = np.tile([0, 1], 6)
    lc = (cache['L_scores'][data['held_images']] > 0) == truth
    sc = (cache['S_scores'][data['held_images']] > 0) == truth
    result['heldout_correct_overlap'] = {'shared': int(np.sum(lc & sc)), 'L_only': int(np.sum(lc & ~sc)), 'S_only': int(np.sum(sc & ~lc))}
    dump('readout.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['prepare', 'check', 'train', 'readout'], required=True)
    parser.add_argument('--commit', default=''); args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    wall, cpu, ok = time.time(), time.process_time(), False
    try:
        if args.stage != 'prepare': assert len(args.commit) == 40
        result = prepare() if args.stage == 'prepare' else {'check': check, 'train': train, 'readout': readout}[args.stage](args.commit)
        ok = True; print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        dump(args.stage + '_failure.private.json', {'type': type(error).__name__, 'message': str(error)}); raise
    finally:
        dump(args.stage + '_receipt.json', {'ok': ok, 'started_unix': wall, 'finished_unix': time.time(),
            'wall_seconds': time.time() - wall, 'process_CPU_seconds': time.process_time() - cpu,
            'GPU_wall_upper_hours': (time.time() - wall) / 3600 if args.stage != 'prepare' else 0,
            'source_code_sha256': sha(__file__), 'namespace_bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())})
