"""Joint G/U label+score journal: one model forward, one durable commit per axis."""
import json
import math
import os

import numpy as np

from echo_routing.temporal.anchor_cache import ProbabilityCache, atomic_json, fsync_directory, sha
from echo_routing.temporal.anchor_edits import aligned, components, decode


class PolicyPairCache(ProbabilityCache):
    def __init__(self, folder, identity, bases, raws, readonly=False):
        if not len(bases) == len(raws) or not bases:
            raise ValueError('complete matching source-prediction axes required')
        pairs = [aligned(base, raw) for base, raw in zip(bases, raws)]
        self.bases = [base.astype(np.int8, copy=True) for base, _ in pairs]
        self.raws = [raw.astype(np.int8, copy=True) for _, raw in pairs]
        super().__init__(folder, identity, [len(base) for base in self.bases], readonly=readonly)

    def _check(self, value, index):
        spans = components(self.bases[index], self.raws[index])
        if [tuple(span) for span in value['spans']] != spans:
            raise ValueError('cached candidate spans differ from raw-supported action domain')
        scores = np.asarray(value['scores'], dtype=np.float32)
        if scores.shape != (len(spans),) or not np.isfinite(scores).all():
            raise ValueError('one finite FP32 score per unchanged candidate required')
        expected = {policy: decode(self.bases[index], self.raws[index], scores, policy == 'G') for policy in ['U', 'G']}
        for policy in ['U', 'G']:
            labels = np.asarray(value[policy]['prediction'])
            if labels.dtype.kind not in 'iu' or not np.array_equal(labels, expected[policy]['prediction']):
                raise ValueError('cached full labels do not follow declared G/U source-only policy')
            if value[policy]['accepted'] != expected[policy]['accepted'] or value[policy]['vetoed'] != expected[policy]['vetoed']:
                raise ValueError('cached accepted/vetoed actions differ from fixed score ranking')
        seconds = float(value['cached_feature_head_policy_seconds'])
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('finite cached-only timing required')
        return {**expected, 'scores': scores, 'spans': spans,
                'cached_feature_head_policy_seconds': seconds}

    def load(self, index):
        if not 0 <= index < self.cursor or sha(self.file(index)) != self.rows[index]['sha256']:
            raise ValueError('uncommitted or changed joint policy output')
        with np.load(self.file(index)) as z:
            if set(z.files) != {'labels_U', 'labels_G', 'scores', 'spans', 'accepted_U', 'accepted_G', 'vetoed_G', 'head_seconds'}:
                raise ValueError('complete joint policy payload keys changed')
            span = z['spans'].copy()
            accepted = {p: None if np.array_equal(z['accepted_' + p], [-1, -1]) else tuple(z['accepted_' + p].tolist()) for p in ['U', 'G']}
            value = {p: {'prediction': z['labels_' + p].copy(), 'accepted': accepted[p],
                         'vetoed': [tuple(s) for s in z['vetoed_G'].tolist()] if p == 'G' else []} for p in ['U', 'G']}
            value.update({'scores': z['scores'].copy(), 'spans': span.tolist(),
                          'cached_feature_head_policy_seconds': float(z['head_seconds'])})
        if self.rows[index]['positions'] != len(self.bases[index]):
            raise ValueError('joint journal original axis changed')
        return self._check(value, index)

    def run_one(self, index, predict):
        if self.readonly or index != self.cursor or index >= self.total or self.pending.exists():
            raise ValueError('exactly-next unambiguous joint policy claim required')
        atomic_json(self.pending, {'index': index, 'identity': self.identity})
        value = self._check(predict(), index)
        path = self.file(index)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('xb') as handle:
            np.savez(handle, labels_U=value['U']['prediction'].astype(np.int8), labels_G=value['G']['prediction'].astype(np.int8),
                     scores=value['scores'], spans=np.asarray(value['spans'], np.int64).reshape(-1, 2),
                     accepted_U=np.asarray(value['U']['accepted'] or [-1, -1], np.int64),
                     accepted_G=np.asarray(value['G']['accepted'] or [-1, -1], np.int64),
                     vetoed_G=np.asarray(value['G']['vetoed'], np.int64).reshape(-1, 2),
                     head_seconds=value['cached_feature_head_policy_seconds'])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(self.path)
        row = {'index': index, 'identity': self.identity, 'positions': len(self.bases[index]), 'sha256': sha(path)}
        with self.journal.open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(self.path)
        self.pending.unlink()
        fsync_directory(self.path)
        self.rows.append(row)
        self.cursor += 1
