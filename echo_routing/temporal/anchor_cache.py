"""A59 complete-axis cache with directory-durable claims and read-only reuse."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from echo_routing.temporal.anchor_features import probabilities


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path):
    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_directory(path):
    path = Path(path)
    missing, cursor = [], path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        fsync_directory(created.parent)
        fsync_directory(created)


def atomic_json(path, value):
    path = Path(path)
    durable_directory(path.parent)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


class ProbabilityCache:
    def __init__(self, folder, identity, lengths, readonly=False):
        self.path, self.identity = Path(folder), identity
        self.lengths, self.readonly = tuple(map(int, lengths)), bool(readonly)
        self.total, self.cursor = len(self.lengths), 0
        if not self.total or any(n < 1 for n in self.lengths):
            raise ValueError('one positive complete original-axis length per sequence required')
        if readonly:
            if not self.path.is_dir():
                raise ValueError('read-only prior cache does not exist')
        else:
            durable_directory(self.path)
        self.pending, self.journal = self.path / 'pending.json', self.path / 'committed.jsonl'
        binding = self.path / 'identity.json'
        expected = {'identity': identity, 'total': self.total}
        if binding.exists():
            if json.loads(binding.read_text()) != expected:
                raise ValueError('source/code/input/axis cache identity changed')
        elif readonly:
            raise ValueError('read-only prior identity is absent')
        else:
            atomic_json(binding, expected)
        if self.pending.exists():
            raise ValueError('ambiguous pending neural/readout operation; no automatic replay')
        self.rows = [json.loads(line) for line in self.journal.read_text().splitlines()] if self.journal.exists() else []
        self.cursor = len(self.rows)
        if self.cursor > self.total or readonly and self.cursor != self.total:
            raise ValueError('cache prefix exceeds total or read-only prefix is incomplete')
        if {p.name for p in self.path.glob('sequence*.private.npz')} != {self.file(i).name for i in range(self.cursor)}:
            raise ValueError('uncommitted output file cannot be silently reused or replayed')
        for i, row in enumerate(self.rows):
            if row['index'] != i or row['identity'] != identity or row['positions'] != self.lengths[i]:
                raise ValueError('journal order/identity/original axis changed')
            self.load(i)

    def file(self, index):
        return self.path / f'sequence{index:06d}.private.npz'

    def load(self, index):
        if not 0 <= index < self.cursor:
            raise ValueError('cache output is not committed')
        path = self.file(index)
        if sha(path) != self.rows[index]['sha256']:
            raise ValueError('committed posterior checksum changed')
        with np.load(path) as value:
            prob = value['prob'].copy()
        probabilities(prob)
        if prob.shape != (self.lengths[index], 5):
            raise ValueError('complete expected original probability axis changed')
        return prob

    def run_one(self, index, predict):
        if self.readonly or index != self.cursor or index >= self.total or self.pending.exists():
            raise ValueError('writable exactly-next unambiguous cache claim required')
        atomic_json(self.pending, {'index': index, 'identity': self.identity})
        prob = probabilities(predict()).astype(np.float32)
        if prob.shape != (self.lengths[index], 5):
            raise ValueError('new probability output has wrong original axis')
        path = self.file(index)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('xb') as handle:
            np.savez(handle, prob=prob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(self.path)
        row = {'index': index, 'identity': self.identity, 'sha256': sha(path), 'positions': len(prob)}
        with self.journal.open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(self.path)
        self.pending.unlink()
        fsync_directory(self.path)
        self.rows.append(row)
        self.cursor += 1
