"""EX2 paths and the recording groups of the open-set test. EV9V test rows are never returned."""
from pathlib import Path

import numpy as np

from echo_routing.features.sampling import stride_indices
from exam_level import io

RUN = io.ROOT / 'runs/ex2_open_routing_v1'
EX1_RUN = io.RUN
CAMUS = io.ROOT / 'data/camus/CAMUS_public/database_nifti'
CAMUS_PATIENTS = 500
OUT_OF_SCOPE = 'PMPALA'
TRAIN_PER_CLASS = 150
DYNAMIC_SAMPLE = 400
SAMPLE_SEED = 2026110401
N_CLASSES = 5


def native_from_row(row):
    """One recording as a sequence: all positions at 10 Hz, gamma 1."""
    refs = [(row['video_id'], int(i), 1.0) for i in stride_indices(int(row['n_frames']), float(row['fps_playback']), 10.)]
    return {'id': row['video_id'], 'kind': 'native', 'labels': np.zeros(0), 'refs': refs}


def train_reference_rows():
    """A fixed sample per class from the training split."""
    rows = [r for r in io.manifest_rows('train') if r['family5_label']]
    rng = np.random.default_rng(SAMPLE_SEED)
    out = []
    for c in range(N_CLASSES):
        pool = sorted((r for r in rows if int(float(r['family5_index'])) == c), key=lambda r: r['video_id'])
        out += [pool[int(i)] for i in sorted(rng.choice(len(pool), TRAIN_PER_CLASS, replace=False))]
    return out


def out_of_scope_rows():
    return [r for r in io.manifest_rows('validation') if r['raw_label'] == OUT_OF_SCOPE]


def external_rows(dataset):
    """A50 private manifest rows of one non-EV9V dataset, in their stored order."""
    if dataset not in io.EXTERNAL_DATASETS:
        raise ValueError('only Dynamic, LVH and CAMUS may be read')
    sections = io.read_json(io.EXTERNAL / 'manifest.json')['datasets']
    return [dict(r) for r in {k: v for k, v in sections.items() if k in io.EXTERNAL_DATASETS}[dataset]]


def dynamic_sample():
    """Fixed sample of Dynamic, with its original manifest index for the parity check."""
    rows = external_rows('dynamic')
    picks = sorted(np.random.default_rng(SAMPLE_SEED + 1).choice(len(rows), DYNAMIC_SAMPLE, replace=False).tolist())
    return [(int(i), rows[int(i)]) for i in picks]


def camus_row(number, view):
    """One half-sequence with its source frame rate. View is 2CH or 4CH."""
    ident = f'patient{number:04d}'
    folder = CAMUS / ident
    path = folder / f'{ident}_{view}_half_sequence.nii.gz'
    info = folder / f'{ident}_Info_{view}.cfg'
    if not info.exists():
        info = folder / f'Info_{view}.cfg'
    fields = {k.strip(): v.strip() for line in info.read_text().splitlines() if ':' in line for k, v in [line.split(':', 1)]}
    if not path.exists() or 'FrameRate' not in fields:
        raise ValueError('CAMUS half-sequence or source frame rate absent')
    return {'id': f'{ident}_{view}', 'group': ident, 'kind': 'nifti', 'path': str(path), 'fps': float(fields['FrameRate']), 'n_frames': None}


def save_group(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez(temporary, **values)
    temporary.replace(path)
