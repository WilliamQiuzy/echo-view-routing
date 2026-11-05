#!/usr/bin/env python3
"""EX2 post-hoc, added after the colour flag marked almost every recording: the distribution of its measure, in two regions. CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image

import ex2_quality as q
from echo_routing.ingest.frames import list_frame_paths
from echo_routing.temporal.joint_visual import crop_uint8
from exam_level import io
from open_routing import data, quality as ql

CUTS = (.02, .05, .10, .20, .40)
LEVELS = (.01, .05, .25, .5, .75, .95, .99)


def shares(frames_dir):
    files = list_frame_paths(q.resolve(frames_dir))
    picks = q.picks(len(files), q.COLOUR_FRAMES)
    images = [Image.open(files[i]).convert('RGB') for i in picks]
    return ql.colour_share(np.stack([np.asarray(im) for im in images])), ql.colour_share(np.stack([crop_uint8(im) for im in images]))


def summary(values):
    return {'quantiles': dict(zip(map(str, LEVELS), np.quantile(values, LEVELS).tolist())), 'at_least': {str(c): int((values >= c).sum()) for c in CUTS}}


def main():
    rows = [r for s in io.ALLOWED_SPLITS for r in io.manifest_rows(s)]
    values = np.array([shares(r['frames_dir']) for r in rows])
    io.write_json(data.RUN / 'public/posthoc_colour.json', {'recordings': len(rows), 'whole_frame': summary(values[:, 0]), 'a08_centre_crop': summary(values[:, 1])})
    print('colour written', len(rows))


if __name__ == '__main__':
    main()
