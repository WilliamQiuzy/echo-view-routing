"""EX3 paths and read-only access to Helena's outputs. Provider rows are filtered by the allowed paths before any other field is used."""
import csv
import hashlib
import json
import pickle
from pathlib import Path

from exam_level import io
from helena_input import pixels as px

RUN = io.ROOT / 'runs/ex3_helena_input_v1'
PUBLIC = RUN / 'public'
HELENA = Path('/home/william/hli-echo-segmentation/ev9v_processed_oct16')
HELENA_CONFIG = HELENA / 'configs/20ead5b2fb6522ca2ad983eccfaa62959b23fbc27552e19ea567c8aa021263b8.json'
HELENA_COMMIT = '53a2246'
CACHE = RUN / 'cache'
FOV_INDEX = RUN / 'fov_index.pkl'
GEOMETRY = RUN / 'external_geometry'
RATIO = PUBLIC / 'side_ratio.json'
A08_CONFIG = 'configs/experiments/video_within_view_v1.yaml'
V01_CONFIG = 'configs/experiments/raw_video_context_v1.yaml'
ARMS = ('H', 'Z')
GPU_FRACTION = .4
GPU_HOURS_ORIGINAL = 10.
GPU_HOURS_MAX = 13.
CACHE_BYTES_MAX = 45e9
PIXEL_CODE = ('geometry.py', 'pixels.py', 'external.py', 'forward.py')


def allowed_rows():
    """Our manifest rows of the two non-test splits, by recording."""
    rows = {}
    for split in px.SPLITS:
        for row in io.manifest_rows(split):
            rows[row['video_id']] = row
    return rows


def provider_rows(allowed):
    """Helena's manifest rows of the allowed recordings only, keyed by our recording id."""
    by_path = {row['video_path']: video for video, row in allowed.items()}
    out = {}
    with (HELENA / 'manifests/ev9v_oct16_model_ready.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            video = by_path.get(row['input_path'])
            if video is not None:
                out[video] = row
    return out


def provider_status(row, n_frames):
    """None when the provider's output can be used; otherwise the reason."""
    if row is None:
        return 'no_provider_row'
    if not all(row[k] == 'complete' for k in ('analysis_status', 'render_status', 'audit_status')):
        return 'provider_not_complete'
    if int(row['frame_count']) != int(n_frames):
        return 'frame_count_mismatch'
    if not Path(row['output_path']).is_file():
        return 'output_missing'
    return None


def pixels_for(arm, frame_paths):
    if arm == 'A08':
        return px.RawPixels(frame_paths)
    if arm == 'H':
        return px.HelenaPixels(CACHE, frame_paths)
    if arm == 'Z':
        return px.ZoomPixels(FOV_INDEX, frame_paths)
    raise ValueError(f'unknown input {arm}')


def side_ratio():
    return float(json.loads(RATIO.read_text())['ratio'])


def external_geometry(group):
    with (GEOMETRY / f'{group}.pkl').open('rb') as handle:
        return pickle.load(handle)


def pixel_code_digest():
    digest = hashlib.sha256()
    for name in PIXEL_CODE:
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def input_fingerprint(arm):
    """What the frames of this input depend on. Saved outputs are reused only under the same fingerprint."""
    if arm == 'A08':
        sources = ()
    elif arm == 'H':
        sources = (CACHE / 'index.json',)
    elif arm == 'Z':
        sources = (FOV_INDEX, RATIO)
    else:
        raise ValueError(f'unknown input {arm}')
    return '|'.join([arm, pixel_code_digest()[:16], *(io.sha256(path)[:16] for path in sources)])


def identifiers():
    """Every recording and session identifier of the allowed splits, for the guard on public files."""
    from exam_level import streams
    videos = set(allowed_rows())
    return videos | {streams.session_of(v) for v in videos} | {streams.timestamp_of(v) for v in videos}


def find_identifier(value, names):
    text = json.dumps(value, default=str)
    return next((name for name in names if name and name in text), None)


def write_public(name, value):
    """All public files go through this guard. A recording or session identifier stops the write."""
    found = find_identifier(value, identifiers())
    if found is not None:
        raise SystemExit(f'{name}: a private identifier reached a public file')
    io.write_json(PUBLIC / name, value)


def run_key(arm, seed):
    if arm not in ARMS or seed not in io.SEEDS:
        raise ValueError('declared arm and seed required')
    return f'{arm}_s{seed}'


def final_weights(arm, seed, stage='a08'):
    return RUN / 'checkpoints' / run_key(arm, seed) / stage / 'final.pt'


def arm_prefix(arms):
    """Empty when every arm is covered, so the original file names stay valid."""
    return '' if set(arms) == set(ARMS) else f"{'_'.join(arms)}_"


def dev_result_name(stage, arms, seeds):
    return f"dev_{stage}_{arm_prefix(arms)}{'_'.join(f's{s}' for s in seeds)}.json"
