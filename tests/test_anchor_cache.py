import importlib
import json
import os
import stat

import numpy as np
import pytest


def api():
    return importlib.import_module('echo_routing.temporal.anchor_cache')


def test_full_axis_clean_prefix_resume_never_forwards_committed_cells(tmp_path):
    m = api()
    count = []
    def forward(length):
        count.append(length)
        return np.full((length, 5), .2, np.float32)
    c = m.ProbabilityCache(tmp_path, 'fixed', [3, 4, 5])
    for i in range(2):
        c.run_one(i, lambda i=i: forward(i + 3))
    reopened = m.ProbabilityCache(tmp_path, 'fixed', [3, 4, 5])
    for i in range(reopened.cursor, 3):
        reopened.run_one(i, lambda i=i: forward(i + 3))
    assert count == [3, 4, 5] and reopened.cursor == 3
    with pytest.raises(ValueError):
        reopened.run_one(0, lambda: forward(3))
    assert count == [3, 4, 5]
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    old = m.ProbabilityCache(tmp_path, 'fixed', [3, 4, 5], readonly=True)
    assert old.cursor == 3 and old.load(2).shape == (5, 5)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        old.run_one(3, lambda: forward(6))


def test_pending_bad_shape_wrong_axis_or_corruption_stop_without_new_forward(tmp_path):
    m = api()
    c = m.ProbabilityCache(tmp_path, 'fixed', [3])
    with pytest.raises(ValueError):
        c.run_one(0, lambda: np.full((2, 5), .2))
    assert (tmp_path / 'pending.json').exists()
    with pytest.raises(ValueError):
        m.ProbabilityCache(tmp_path, 'fixed', [3])
    other = tmp_path / 'other'
    c = m.ProbabilityCache(other, 'fixed', [3])
    c.run_one(0, lambda: np.full((3, 5), .2))
    with pytest.raises(ValueError):
        m.ProbabilityCache(other, 'changed', [3])
    with pytest.raises(ValueError):
        m.ProbabilityCache(other, 'fixed', [4])
    with (other / 'sequence000000.private.npz').open('wb') as handle:
        np.savez(handle, prob=np.full((3, 5), .1))
    with pytest.raises(ValueError):
        m.ProbabilityCache(other, 'fixed', [3], readonly=True)


def test_source_claim_is_directory_durable_before_forward_and_output_unclaim(tmp_path, monkeypatch):
    m = api()
    events = []
    original = os.fsync
    def observe(fd):
        events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
        original(fd)
    monkeypatch.setattr(os, 'fsync', observe)
    c = m.ProbabilityCache(tmp_path, 'fixed', [3])
    events.clear()
    def forward():
        assert events[-1] == 'directory'
        assert json.loads((tmp_path / 'pending.json').read_text())['index'] == 0
        return np.full((3, 5), .2)
    c.run_one(0, forward)
    assert events[-1] == 'directory' and not c.pending.exists()
    with pytest.raises(ValueError):
        m.ProbabilityCache(tmp_path / 'missing', 'none', [3], readonly=True)
    assert not (tmp_path / 'missing').exists()
