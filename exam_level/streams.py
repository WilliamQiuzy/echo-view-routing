"""Exam-ordered stream construction. Pure logic: no file, label-table or model access."""
import numpy as np

MODES = ('exam', 'shuffled', 'rewired')
HALVES = ('selection', 'calibration')
SEED = 2026110399
MAX_REPAIR_STEPS = 200000
BANKS = {'exam': ('exam', SEED), 'shuffled': ('shuffled', SEED), 'rewired': ('rewired', SEED),
         'rewired_d1': ('rewired', SEED + 1), 'rewired_d2': ('rewired', SEED + 2)}


def session_of(video_id):
    return str(video_id).rsplit('_', 1)[0]


def clip_index(video_id):
    return int(str(video_id).rsplit('_', 1)[1])


def timestamp_of(video_id):
    return str(video_id)[:19]


def group_sessions(natives):
    """Sessions in sorted order, each with its clips in ascending numeric k."""
    groups = {}
    for q in natives:
        groups.setdefault(session_of(q['id']), []).append(q)
    return {s: sorted(v, key=lambda q: clip_index(q['id'])) for s, v in sorted(groups.items())}


def _assemble(name, clips):
    refs, labels, joins, spans = [], [], [], []
    for q in clips:
        if refs:
            joins.append(len(refs))
        spans.append((q['id'], len(refs), len(refs) + len(q['refs'])))
        refs.extend(tuple(r) for r in q['refs'])
        labels.append(np.asarray(q['labels'], np.int64))
    return {'id': name, 'kind': 'multi', 'refs': refs, 'labels': np.concatenate(labels),
            'joins': np.asarray(joins, np.int64), 'clips': spans}


def _same_session_joins(clips, position):
    """Joins touching this position whose two clips share a session."""
    count = 0
    for a, b in ((position - 1, position), (position, position + 1)):
        if 0 <= a and b < len(clips) and session_of(clips[a]['id']) == session_of(clips[b]['id']):
            count += 1
    return count


def _rewire(sizes, clips, rng):
    """Random reassignment keeping stream sizes; swaps repair any same-session join. Sizes must fit the pool."""
    order = [clips[i] for i in rng.permutation(len(clips))]
    edges = np.cumsum([0, *sizes])
    streams = [order[a:b] for a, b in zip(edges, edges[1:])]
    slots = [(s, i) for s, stream in enumerate(streams) for i in range(len(stream))]
    for _ in range(MAX_REPAIR_STEPS):
        bad = [(s, i) for s, i in slots if _same_session_joins(streams[s], i)]
        if not bad:
            return streams
        s, i = bad[int(rng.integers(len(bad)))]
        t, j = slots[int(rng.integers(len(slots)))]
        before = _same_session_joins(streams[s], i) + _same_session_joins(streams[t], j)
        streams[s][i], streams[t][j] = streams[t][j], streams[s][i]
        if _same_session_joins(streams[s], i) + _same_session_joins(streams[t], j) > before:
            streams[s][i], streams[t][j] = streams[t][j], streams[s][i]
    raise ValueError('could not remove every same-session join; do not use this bank')


def expand_clip_labels(spans, labels):
    """One label per clip to one label per position, using each clip's (id, start, end) span."""
    if len(spans) != len(labels):
        raise ValueError('one label per clip span required')
    return np.concatenate([np.full(end - start, y, np.int64) for (_, start, end), y in zip(spans, labels)])


def exam_streams(natives, mode, half, seed=SEED):
    """One stream per session. Every mode uses exactly the same whole clips and the same stream sizes."""
    if mode not in MODES or half not in HALVES:
        raise ValueError('declared stream mode and development half required')
    groups = group_sessions(natives)
    rng = np.random.default_rng([seed, MODES.index(mode), HALVES.index(half)])
    if mode == 'exam':
        lists = list(groups.values())
    elif mode == 'shuffled':
        lists = [[v[i] for i in rng.permutation(len(v))] for v in groups.values()]
    else:
        pool = [q for v in groups.values() for q in v]
        lists = _rewire([len(v) for v in groups.values()], pool, rng)
    return [_assemble(f'ex1_{mode}_{half}_{i:03d}', clips) for i, clips in enumerate(lists)]
