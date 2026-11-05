"""External recordings under the three inputs, and Helena's analysis without its file writing."""
import hashlib
import inspect
import json

import numpy as np
from PIL import Image

from echo_routing.temporal.joint_visual import crop_uint8
from helena_input import geometry as gm


def read_recording(q):
    """All frames of one external recording, RGB uint8, in source order and at source size."""
    import cv2
    if q['kind'] == 'video':
        capture = cv2.VideoCapture(str(q['path']))
        frames = []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        capture.release()
        if not frames:
            raise ValueError('no decodable frame')
        return frames
    if q['kind'] == 'nifti':
        import nibabel as nib
        array = np.asanyarray(nib.load(str(q['path'])).dataobj)
        if array.ndim != 3:
            raise ValueError('CAMUS sequence is not a 3-D x/y/time array')
        if array.min() < 0 or array.max() > 255:
            raise ValueError('CAMUS values need an unspecified rescaling')
        array = array.transpose(2, 1, 0).astype(np.uint8)
        return [np.repeat(a[:, :, None], 3, axis=2) for a in array]
    raise ValueError(f'unsupported kind {q["kind"]}')


def analyze(frames_bgr, indices, frame_count, cfg, generator):
    """Helena's analyze_video (ev9v_oct16/analysis.py at 53a2246) with every file write removed."""
    from echo_fov import fastsam_geometry as v2
    from echo_fov.ev9v_oct16.candidates import score_candidates
    from echo_fov.ev9v_oct16.crop import plan_crop
    from echo_fov.ev9v_oct16.dynamic import evidence
    from echo_fov.ev9v_oct16.geometry import choose_geometry
    from echo_fov.ev9v_oct16.masking import dynamic_envelope, final_mask
    from echo_fov.ev9v_oct16.prompts import make_prompts, ui_rois
    from echo_fov.ev9v_oct16.temporal import select_path, stable_segments
    segments, masks = [], []
    for start, end in stable_segments(frames_bgr, cfg):
        subset = frames_bgr[start:end]
        ev = evidence(subset)
        envelope = False
        try:
            prior, _ = v2.fit_fan_geometry(ev['support'], seed=cfg['seed'], peak=ev['peak'])
        except ValueError:
            prior = dynamic_envelope(ev['support'], ev['ui'])
            envelope = True
        prompts = [make_prompts(ev['support'], prior, f, ev['ui'], cfg) for f in subset]
        rois = ui_rois(subset, prior, cfg)
        sets = []
        for raw, prompt in zip(generator.generate(subset, prompts), prompts):
            ranked = score_candidates(raw, ev, prior, prompt, rois, cfg)
            sets.append([r for r in ranked if r[0]['accepted']][:cfg['temporal_top_k']])
        if len(sets) != end - start:
            raise ValueError('Incomplete FastSAM results')
        path = select_path(sets, prior, cfg)
        selected = [item[1] if item else None for item in path]
        if not envelope:
            try:
                prior, _ = choose_geometry(ev, selected, cfg)
            except ValueError:
                envelope = True
        if envelope:
            prior = dynamic_envelope(ev['support'], ev['ui'])
        final, _, consensus = final_mask(prior, selected, ev, cfg)
        fallback = 'dynamic_envelope_fallback' if envelope else ('geometry_fallback' if consensus['geometry_only_fallback'] else '')
        begin = 0 if start == 0 else (indices[start - 1] + indices[start]) // 2 + 1
        finish = int(frame_count) if end == len(indices) else (indices[end - 1] + indices[end]) // 2 + 1
        segments.append({'start': int(begin), 'end': int(finish), 'mask': gm.pack_mask(final), 'fallback': fallback})
        masks.append(final)
    crop = plan_crop(masks, cfg['crop_margin_percentage'], cfg['minimum_crop_margin_pixels'])
    return {'segments': segments, 'plan': crop, 'cx': float(crop['crop_center_x']), 'cy': float(crop['crop_center_y']), 'size': float(crop['crop_size_original_pixels'])}


def geometry_of(frames_rgb, cfg, generator):
    """The frozen 30-frame sample, then the analysis. A geometry failure is recorded, never a dropped recording. Model errors stop the run."""
    import cv2
    from echo_fov.ev9v_oct16.sampling import sample_indices
    indices = sample_indices(len(frames_rgb))
    subset = [frames_rgb[i][:, :, ::-1].copy() for i in indices]
    try:
        return {**analyze(subset, indices, len(frames_rgb), cfg, generator), 'failed': None}
    except (ValueError, IndexError, ZeroDivisionError, cv2.error) as error:
        return {'failed': f'{type(error).__name__}: {str(error)[:100]}'}


def analysis_fingerprint(cfg):
    """What the external geometry depends on: the configuration, Helena's commit and the analysis code."""
    source = inspect.getsource(analyze) + inspect.getsource(geometry_of)
    return hashlib.sha256(json.dumps({'cfg': cfg, 'source': source}, sort_keys=True, default=str).encode()).hexdigest()


def a08_input(image, *_):
    return crop_uint8(Image.fromarray(image))


def h_input(image, geometry, frame, _ratio):
    if geometry.get('failed'):
        return gm.full_frame_square(image)
    from echo_fov.ev9v_oct16.crop import transform_frame
    return transform_frame(image, gm.unpack_mask(gm.segment_at(geometry['segments'], frame)['mask']), geometry['plan'], gm.OUTPUT)


def z_input(image, geometry, frame, ratio):
    if geometry.get('failed'):
        return a08_input(image)
    return gm.z_frame(image, geometry['segments'], geometry['cx'], geometry['cy'], ratio * geometry['size'], frame)


INPUTS = {'A08': a08_input, 'H': h_input, 'Z': z_input}


def media_frames(q, arm, geometry, ratio):
    """The recording's 10 Hz samples under one input, ready for media_batches."""
    from scripts.external_original_tests import sample_indices
    frames = read_recording(q)
    chosen = sample_indices(len(frames), float(q['fps']))
    transform = INPUTS[arm]
    out = np.stack([transform(frames[int(i)], geometry, int(i), ratio) for i in chosen])
    if out.dtype != np.uint8 or out.shape[1:] != (gm.OUTPUT, gm.OUTPUT, 3):
        raise ValueError(f'unexpected input shape {out.shape} or type {out.dtype}')
    return out
