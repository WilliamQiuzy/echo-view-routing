"""Mechanical data checks: perceptual hash, correlation and three quality flags. No label or model enters here."""
import numpy as np
from PIL import Image

HASH_WIDTH, HASH_HEIGHT = 9, 8
MAX_HAMMING = 4
BLOCK = 2000
MIN_NCC = .95
NCC_SIZE = (64, 48)
STATIC_LIMIT = 1.
COLOUR_CHROMA = 40
COLOUR_SHARE = .02
SHORT_FRAMES = 16
BIT_COUNT = np.array([bin(i).count('1') for i in range(1 << 16)], np.uint8)
LOW_16 = np.uint64(0xFFFF)


def dhash(gray):
    """A 64-bit difference hash of one grayscale image, as a Python int."""
    small = np.asarray(Image.fromarray(np.asarray(gray, np.uint8)).resize((HASH_WIDTH, HASH_HEIGHT), Image.BILINEAR), int)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a, b):
    """All-pairs bit distance between two uint64 arrays."""
    x = np.asarray(a, np.uint64)[:, None] ^ np.asarray(b, np.uint64)[None, :]
    total = np.zeros(x.shape, np.uint8)
    for shift in (0, 16, 32, 48):
        total += BIT_COUNT[((x >> np.uint64(shift)) & LOW_16).astype(np.int64)]
    return total


def ncc(a, b):
    """Normalised correlation between two images of the same size."""
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator else 0.


def small_gray(gray):
    return np.asarray(Image.fromarray(np.asarray(gray, np.uint8)).resize(NCC_SIZE, Image.BILINEAR), float)


def is_static(frames):
    """The median over frame pairs of the mean absolute difference stays under one grey level."""
    f = np.asarray(frames, float)
    return bool(np.median(np.abs(f[1:] - f[:-1]).mean(axis=tuple(range(1, f.ndim)))) < STATIC_LIMIT)


def static_from_differences(differences):
    """The same rule when only some neighbouring pairs were read."""
    return bool(np.median(differences) < STATIC_LIMIT)


def near_duplicate_candidates(values, owners, sessions, block=BLOCK):
    """Recording pairs from different sessions with at least one frame pair within the hash limit, mapped to those frame pairs."""
    frame_session = np.asarray(sessions)[np.asarray(owners)]
    found = {}
    for begin in range(0, len(values), block):
        rows, cols = np.nonzero(hamming(values[begin:begin + block], values) <= MAX_HAMMING)
        rows = rows + begin
        keep = (cols > rows) & (frame_session[rows] != frame_session[cols])
        for r, c in zip(rows[keep].tolist(), cols[keep].tolist()):
            found.setdefault(tuple(sorted((int(owners[r]), int(owners[c])))), []).append((r, c))
    return found


def family_conflict(a, b):
    """Two recordings carry different five-class labels. Different codes in one family, or an excluded code, are not a conflict."""
    return bool(a['family'] and b['family'] and a['family'] != b['family'])


def describe_pairs(pairs, owners):
    """Aggregate counts for pairs that begin with two recording indices."""
    apical = {'A4C', 'A5C'}
    both = lambda a, b, key: bool(owners[a][key] and owners[b][key])
    return {'pairs': len(pairs), 'recordings': len({i for p in pairs for i in p[:2]}),
            'cross_split_train_validation': sum(owners[a]['split'] != owners[b]['split'] for a, b, *_ in pairs),
            'cross_half': sum(both(a, b, 'half') and owners[a]['half'] != owners[b]['half'] for a, b, *_ in pairs),
            'same_timestamp': sum(owners[a]['stamp'] == owners[b]['stamp'] for a, b, *_ in pairs),
            'raw_code_conflict': sum(owners[a]['code'] != owners[b]['code'] for a, b, *_ in pairs),
            'family_conflict': sum(family_conflict(owners[a], owners[b]) for a, b, *_ in pairs),
            'a4c_a5c_conflict': sum({owners[a]['family'], owners[b]['family']} == apical for a, b, *_ in pairs)}


def colour_share(frames):
    """Mean share of pixels whose largest and smallest channel differ by more than the chroma limit."""
    f = np.asarray(frames, int)
    return float(((f.max(-1) - f.min(-1)) > COLOUR_CHROMA).mean())


def is_colour(share):
    return share >= COLOUR_SHARE


def is_short(n_frames):
    return n_frames < SHORT_FRAMES
