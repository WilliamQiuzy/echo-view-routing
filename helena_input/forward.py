"""Inference helpers. They reuse A08's window plan unchanged; only the frame source differs."""
import numpy as np
import torch
from scipy.special import logsumexp

from echo_routing.evaluate.external_native import window_plan
from echo_routing.evaluate.window_target import nearest_centre, window_frames
from echo_routing.temporal.structured_inputs import windows
from echo_routing.vit import target_mvit as tm
from exam_level import io

BATCH = 32
STRIDE = 5


@torch.no_grad()
def embed_batches(model, device, batches):
    """batches yields (uint8 frames, marker slots). Returns window logits and embeddings."""
    logits, embeddings = [], []
    for frames, slots in batches:
        rgb = tm.to_rgb(torch.from_numpy(frames).to(device))
        z = model.embed(rgb, tm.marker_planes(torch.as_tensor(slots, device=device)))
        logits.append(model.classifier(z).float().cpu().numpy())
        embeddings.append(z.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(embeddings)


def sequence_batches(q, pixels):
    n = len(q['refs'])
    centres = windows(n, STRIDE)['centre']
    memo = {}

    def frame_at(i):
        if i not in memo:
            memo[i] = pixels.frame(*q['refs'][i])
        return memo[i]

    for begin in range(0, len(centres), BATCH):
        cs = centres[begin:begin + BATCH]
        samples = [window_frames(int(c), n, 16, 16) for c in cs]
        frames = np.stack([[frame_at(int(i)) for i in row] for row in samples])
        yield frames, [tm.marker_slot(row, int(c)) for row, c in zip(samples, cs)]


def media_batches(frames):
    _, window_index, slots = window_plan(len(frames))
    for begin in range(0, len(window_index), BATCH):
        yield frames[window_index[begin:begin + BATCH]], slots[begin:begin + BATCH]


def stream_positions(model, q, pixels, device):
    """Per-position probabilities of one stream: the nearest window centre, as in A08's evaluation."""
    logits, _ = embed_batches(model, device, sequence_batches(q, pixels))
    n = len(q['refs'])
    return torch.softmax(torch.from_numpy(logits), 1).numpy()[nearest_centre(n, windows(n, STRIDE)['centre'])]


def summary(ids, logits_list, embeddings_list, extra=None):
    """One row per recording, in the format of EX2's stored groups."""
    return {'ids': np.array(ids), 'window_counts': np.array([len(x) for x in logits_list]),
            'prob_mean': np.stack([io.softmax_mean(x) for x in logits_list]),
            'logit_mean': np.stack([x.mean(0) for x in logits_list]),
            'energy': np.array([float(logsumexp(x.astype(float), axis=1).mean()) for x in logits_list]),
            'embed_mean': np.stack([x.mean(0) for x in embeddings_list]), **(extra or {})}


def build_model(state, device):
    model = tm.build_model(None, tm.initial_new_state(0))
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


def determinism(device_fraction):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(device_fraction)
    return torch.device('cuda')
