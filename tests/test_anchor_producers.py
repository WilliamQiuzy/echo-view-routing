import importlib

import numpy as np
import torch

from echo_routing.evaluate.window_target import nearest_centre
from echo_routing.temporal.structured_inputs import windows
from echo_routing.vit import target_mvit as tm


class Pixels:
    def frame(self, video, index, gamma):
        return np.full((224, 224, 3), index, np.uint8)


class A08:
    def __init__(self):
        self.calls = []
    def __call__(self, rgb, marker):
        self.calls.append((rgb.clone(), marker.clone()))
        value = torch.zeros(len(rgb), 5)
        value[:, 0] = rgb[:, 0].mean((1, 2, 3))
        return value


class JEPA:
    def probabilities(self, u8, mask):
        value = torch.zeros(len(u8), 64, 5)
        value[:, :, 0] = u8[:, :, :, 0, 0].float().mean(2) / 100
        return value.softmax(-1)


def test_new_A08_wrapper_uses_original5_centres_pixels_markers_and_assignments_only():
    m = importlib.import_module('scripts.cache_anchor_edits')
    q = {'refs': [('v', i, 1.) for i in range(19)]}
    model = A08()
    result = m.a08_sequence(model, q, Pixels(), device='cpu')
    grid = windows(19, 5)
    assert grid['centre'].tolist() == [2, 7, 12, 17]
    frames = np.stack([[Pixels().frame('v', int(i), 1.) for i in row] for row in grid['samples']])
    torch.testing.assert_close(model.calls[0][0], tm.to_rgb(torch.from_numpy(frames)))
    assert torch.equal(model.calls[0][1], tm.marker_planes(torch.from_numpy(grid['marker'])))
    logits = torch.zeros(4, 5)
    logits[:, 0] = model.calls[0][0][:, 0].mean((1, 2, 3))
    expected = logits.softmax(1).numpy()[nearest_centre(19, grid['centre'])]
    np.testing.assert_array_equal(result, expected)


def test_new_JEPA_wrapper_preserves64_chunk_padding_and_positive_probability_stitch():
    m = importlib.import_module('scripts.cache_anchor_edits')
    q = {'refs': [('v', i, 1.) for i in range(73)]}
    value = m.jepa_sequence(JEPA(), q, Pixels(), device='cpu')
    assert value.shape == (73, 5) and np.isfinite(value).all()
    np.testing.assert_allclose(value.sum(1), 1, atol=1e-6)
    assert value[-1, 0] > value[0, 0]
