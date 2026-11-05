"""EX1 data access. Only train/validation metadata, cached development outputs and non-EV9V external outputs."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('EX1_ROOT', '/home/william/echo-view-routing'))
RUN = ROOT / 'runs/ex1_exam_level_v1'
A59_INPUTS = ROOT / 'docs/results/2026-11-03_anchor_edits/input_manifest.json'
A59_CONFIG = ROOT / 'configs/experiments/anchor_edits_v1.json'
A08_EVAL = ROOT / 'runs/video_within_view_v1/evaluation'
A08_MODEL = 'models_frozen/video_within_view_v1/all_s{seed}/final.pt'
FOLDS = ROOT / 'runs/session_fold_library_v1/folds.json'
FOLD_MODELS = ROOT / 'docs/results/2026-10-07_session_fold_library/models.json'
EXTERNAL = ROOT / 'runs/external_original_test_v1/private'
MANIFEST = ROOT / 'data/manifests/ev9v_native.csv'
EXTERNAL_DATASETS = ('dynamic', 'lvh', 'camus')
SEEDS = (0, 1, 2)
HALVES = ('selection', 'calibration')
ALLOWED_SPLITS = ('train', 'validation')


def read_json(path):
    return json.loads(Path(path).read_text())


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=_plain) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_rows(split):
    """Rows of one non-test split. The test split is refused."""
    if split not in ALLOWED_SPLITS:
        raise ValueError('EX1 reads train and validation metadata only')
    with MANIFEST.open() as handle:
        return [r for r in csv.DictReader(handle) if r['split'] == split]


def role_sequences(role):
    if role not in HALVES:
        raise ValueError('EX1 reads only the two development halves from the A59 inputs')
    pin = read_json(A59_INPUTS)['roles'][role]
    path = ROOT / pin['path']
    if sha256(path) != pin['sha256']:
        raise ValueError('frozen A59 input changed: ' + pin['path'])
    return np.load(path, allow_pickle=True).tolist()


def natives(half):
    clips = [q for q in role_sequences(half) if q['kind'] == 'native']
    allowed = {r['video_id'] for r in manifest_rows('validation')}
    if not {q['id'] for q in clips} <= allowed:
        raise ValueError('a development clip is not in the validation split')
    return clips


def b3_params(half, seed):
    outer = read_json(ROOT / read_json(A59_CONFIG)['data']['outer_inputs'])
    return outer['A08_cache_pins'][half][str(seed)]['original_params']


class A08Cache:
    """Original A08 evaluation outputs for one half and seed: per-position and per-window."""

    def __init__(self, half, seed):
        if half not in HALVES or seed not in SEEDS:
            raise ValueError('declared development half and seed required')
        self.path = A08_EVAL / half / f'all_s{seed}.npz'
        with np.load(self.path, allow_pickle=True) as z:
            self.ids = z['sequence_ids'].tolist()
            self.offsets, self.sample_probs = z['sample_offsets'], z['sample_probs']
            self.seq, self.logits = z['seq'], z['logits']
        self.index = {v: i for i, v in enumerate(self.ids)}

    def positions(self, sequence_id):
        i = self.index[sequence_id]
        return self.sample_probs[self.offsets[i]:self.offsets[i + 1]]

    def window_mean(self, sequence_id):
        rows = self.logits[self.seq == self.index[sequence_id]]
        x = rows - rows.max(1, keepdims=True)
        p = np.exp(x)
        return (p / p.sum(1, keepdims=True)).mean(0)


def a08_weight_path(seed):
    """The frozen A08 weights for one seed, verified against their immutable manifest."""
    path = ROOT / A08_MODEL.format(seed=seed)
    listed = [f for f in read_json(path.parent / 'MANIFEST.json')['files'] if f['file'] == path.name]
    if len(listed) != 1 or sha256(path) != listed[0]['sha256']:
        raise ValueError(f'frozen A08 weights for seed {seed} do not match their manifest')
    return path


def external_records(dataset):
    """A50 cached outputs on one external dataset. The EV9V test folder is never opened."""
    if dataset not in EXTERNAL_DATASETS:
        raise ValueError('EX1 reads Dynamic, LVH and CAMUS outputs only')
    sections = read_json(EXTERNAL / 'manifest.json')['datasets']
    rows = {k: v for k, v in sections.items() if k in EXTERNAL_DATASETS}[dataset]
    folder = EXTERNAL / 'predictions' / dataset
    out = []
    for i, row in enumerate(rows):
        with np.load(folder / f'{i:05d}.npz', allow_pickle=True) as z:
            if str(z['id']) != str(row['id']):
                raise ValueError('external cached output and manifest order differ')
            out.append({'true': int(row['true']), 'group': str(row.get('group', row['id'])),
                        **{f'logits_s{s}': z[f'logits_s{s}'] for s in SEEDS},
                        **{f'frozen_B3_s{s}': z[f'frozen_B3_s{s}'] for s in SEEDS}})
    return out


def softmax_mean(logits):
    x = np.asarray(logits, float)
    x = x - x.max(1, keepdims=True)
    p = np.exp(x)
    return (p / p.sum(1, keepdims=True)).mean(0)
