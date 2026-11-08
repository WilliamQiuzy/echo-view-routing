"""A63 pinned teacher-only source functions; exact A61 semantic function copies."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np,torch
from scripts.selected_risk_common import ROOT,read,sha,verify_pin
from echo_routing.vit import target_mvit as tm
from echo_routing.temporal import jepa21_source
from echo_routing.temporal.dense_sequence import DenseSequence,chunk_starts,chunk_positions,stitch_probabilities
from echo_routing.temporal.structured_inputs import windows
from echo_routing.evaluate.window_target import nearest_centre,window_frames
from echo_routing.temporal.anchor_features import probabilities


def determinism():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_source(name, cfg, device='cpu'):
    pin = cfg['source'][name]
    path = verify_pin(pin)
    if name == 'a08':
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        model = tm.build_model(None, tm.initial_new_state(0))
        expected = 400
    elif name == 'jepa':
        source = ROOT / pin['source_manifest']
        if sha(source) != pin['source_manifest_sha256']:
            raise ValueError('JEPA source manifest changed')
        source = read(source)
        for path, expected_sha in source['loaded_model_subset_SHA'].items():
            if sha(jepa21_source.UPSTREAM / path) != expected_sha:
                raise ValueError('pinned official JEPA source subset changed')
        model = DenseSequence(jepa21_source.build_encoder(), 'J-high', seed=2026110257,
                              inactive_names=source['inactive_video_names'])
        state = torch.load(verify_pin(pin), map_location='cpu', weights_only=True, mmap=True)['model']
        expected = 214
    else:
        raise ValueError('declared frozen producer required')
    if len(state) != expected:
        raise ValueError('complete producer state-key count changed')
    model.load_state_dict(state, strict=True)
    if not all(torch.equal(model.state_dict()[k], value) for k, value in state.items()):
        raise ValueError('strict loaded source differs from pinned weights')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)

@torch.no_grad()
def a08_windows(model, q, pixels, centres, device='cuda', batch=32):
    out = []
    centres = np.asarray(centres, int)
    for begin in range(0, len(centres), batch):
        cs = centres[begin:begin + batch]
        samples = [window_frames(int(c), len(q['refs']), 16, 16) for c in cs]
        frames = np.stack([[pixels.frame(*q['refs'][int(i)]) for i in row] for row in samples])
        markers = torch.tensor([tm.marker_slot(row, int(c)) for row, c in zip(samples, cs)], device=device)
        out.append(model(tm.to_rgb(torch.from_numpy(frames).to(device)), tm.marker_planes(markers)).float().cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def a08_sequence(model, q, pixels, device='cuda', batch=32):
    n = len(q['refs'])
    grid = windows(n, 5)
    logits = a08_windows(model, q, pixels, grid['centre'], device, batch)
    result = torch.softmax(torch.from_numpy(logits), 1).numpy()[nearest_centre(n, grid['centre'])]
    probabilities(result)
    return result

@torch.no_grad()
def jepa_sequence(model, q, pixels, device='cuda'):
    n = len(q['refs'])
    starts, chunks = chunk_starts(n), []
    for begin in range(0, len(starts), 2):
        frames, masks = [], []
        for start in starts[begin:begin + 2]:
            indices, mask = chunk_positions(n, start)
            frames.append(np.stack([pixels.frame(*q['refs'][int(i)]) for i in indices]))
            masks.append(mask)
        chunks.extend(model.probabilities(torch.from_numpy(np.stack(frames)).to(device),
                                           torch.from_numpy(np.stack(masks)).to(device)).cpu().numpy())
    result = stitch_probabilities(chunks, n).astype(np.float32)
    probabilities(result)
    return result
