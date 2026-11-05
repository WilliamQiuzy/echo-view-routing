"""Three interchangeable frame sources for one (video, frame, gamma) reference. All return uint8 (224, 224, 3)."""
import collections
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from echo_routing.temporal.joint_visual import gamma_lut, load_frame_u8
from helena_input import geometry as gm

SPLITS = ('train', 'validation')
MASK_MEMO = 256


class RawPixels:
    """A08's own input."""

    name = 'A08'

    def __init__(self, frame_paths):
        self.frame_paths = frame_paths

    def frame(self, video, frame, gamma):
        paths = self.frame_paths(str(video))
        return load_frame_u8(paths[min(int(frame), len(paths) - 1)], float(gamma))


class HelenaPixels:
    """Arm H: Helena's rendered frames from the prepared cache, gamma afterwards; A08's crop where no output exists."""

    name = 'H'

    def __init__(self, cache_root, frame_paths):
        self.root = Path(cache_root)
        self.index = json.loads((self.root / 'index.json').read_text())['videos']
        self.raw = RawPixels(frame_paths)
        self.arrays = collections.OrderedDict()

    def has_output(self, video):
        """A recording the preparation never saw is an error, not a silent fallback."""
        return self.index[str(video)].get('fallback') is None

    def _array(self, video):
        if video in self.arrays:
            self.arrays.move_to_end(video)
            return self.arrays[video]
        entry = self.index[video]
        self.arrays[video] = (np.load(self.root / entry['npy'], mmap_mode='r'), {int(f): i for i, f in enumerate(entry['frames'])})
        while len(self.arrays) > 64:
            self.arrays.popitem(last=False)
        return self.arrays[video]

    def frame(self, video, frame, gamma):
        video = str(video)
        if not self.has_output(video):
            return self.raw.frame(video, frame, gamma)
        array, position = self._array(video)
        image = np.ascontiguousarray(array[position[min(int(frame), int(self.index[video]['n_frames']) - 1)]])
        if abs(float(gamma) - 1.) > 1e-9:
            image = np.asarray(Image.fromarray(image).point(gamma_lut(float(gamma))), np.uint8)
        return image


class ZoomPixels:
    """Arm Z: original JPEG, gamma, Helena's mask, then Helena's centre at A08's magnification; A08's crop where no output exists."""

    name = 'Z'

    def __init__(self, index_path, frame_paths):
        with Path(index_path).open('rb') as handle:
            self.index = pickle.load(handle)
        self.frame_paths = frame_paths
        self.raw = RawPixels(frame_paths)

    def has_output(self, video):
        """A recording the preparation never saw is an error, not a silent fallback."""
        return self.index[str(video)].get('fallback') is None

    def frame(self, video, frame, gamma):
        video = str(video)
        if not self.has_output(video):
            return self.raw.frame(video, frame, gamma)
        paths = self.frame_paths(video)
        position = min(int(frame), len(paths) - 1)
        with Image.open(paths[position]) as opened:
            image = opened.convert('RGB')
            if abs(float(gamma) - 1.) > 1e-9:
                image = image.point(gamma_lut(float(gamma)))
            array = np.asarray(image, np.uint8)
        entry = self.index[video]
        if entry.get('fallback') is not None:
            raise RuntimeError('a fallback entry reached the Z geometry')
        side = gm.a08_side(array.shape[1], array.shape[0])
        return gm.z_frame(array, entry['segments'], entry['cx'], entry['cy'], side, position)


def frames_for_refs(refs, pixels):
    out = np.empty((len(refs), gm.OUTPUT, gm.OUTPUT, 3), np.uint8)
    for i, (video, frame, gamma) in enumerate(refs):
        out[i] = pixels.frame(str(video), int(frame), float(gamma))
    return out
