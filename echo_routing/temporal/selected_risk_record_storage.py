"""Lossless private A63 selected-record tables with a fixed byte bound.

All 28 effects.record fields survive, including scalar types, float bits, order
and duplicate rows. Only canonical sorted unique harm pairs are representable by
the 5x5 mask. Files are uncompressed NPZ, never pickle; their length depends on
row count/dictionary, not observed values or an assumed compression ratio.
Tables are immutable: a same-directory temporary file is atomically published
with a no-clobber hardlink. This does not atomically update a caller manifest.
All source IDs, records and files remain private on the server.
"""
import hashlib
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np

MAX_RECORDS = 4*(3000+265)
INTEGER_FIELDS = ('H', 'R', 'dE', 'dFP', 'delta_M', 'delta_P', 'changed_K', 'neutral',
                  'short16_lost', 'short16_recovered', 'Jexclusive_short16_recovered',
                  'new_FP16_segments', 'replacement_class')
FLOAT_FIELDS = ('u', 'v', 'min_overlap')
BOOL_FIELDS = ('D', 'X', 'safe', 'unsafe', 'old', 'jepa')
SCALAR_FIELDS = INTEGER_FIELDS+FLOAT_FIELDS+BOOL_FIELDS
INT_TYPES = (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
FLOAT_TYPES = (float, np.float16, np.float32, np.float64)
BOOL_TYPES = (bool, np.bool_)
HASH_FIELDS = ('canonical_id', 'draw_signature', 'event_signature')
FIELD_NAMES = ('canonical_id', 'kind', 'D', 'X', 'u', 'H', 'R', 'dE', 'dFP', 'safe', 'v',
               'delta_M', 'delta_P', 'unsafe', 'changed_K', 'neutral', 'short16_lost',
               'short16_recovered', 'Jexclusive_short16_recovered', 'event_signature',
               'draw_signature', 'responsible_sessions', 'harm_transitions', 'new_FP16_segments',
               'min_overlap', 'replacement_class', 'old', 'jepa')
FIELDS = set(FIELD_NAMES)
DTYPE = np.dtype([(k, 'u1', (32,)) for k in HASH_FIELDS] +
                 [('kind_code', 'u1'), ('event_present', 'u1')] +
                 [(k, '<i8') for k in INTEGER_FIELDS] + [(k, '<f8') for k in FLOAT_FIELDS] +
                 [(k, 'u1') for k in BOOL_FIELDS] +
                 [('type_tags', 'u1', (len(SCALAR_FIELDS),)), ('field_order', 'u1', (28,)), ('session_indices', '<u2', (8,)),
                  ('session_count', 'u1'), ('harm_mask', '<u4')], align=False)
ARCHIVE_KEYS = {'rows', 'session_blob', 'session_offsets', 'format_version'}


def _dictionary(sessions):
    if (not isinstance(sessions, (list, tuple)) or len(sessions) != 402 or
            any(type(s) is not str or not s for s in sessions) or len(set(sessions)) != 402):
        raise ValueError('the complete independently frozen402-session dictionary is required')
    try: parts = [s.encode('utf-8', errors='strict') for s in sessions]
    except UnicodeError as exc: raise ValueError('canonical UTF8 source sessions required') from exc
    lengths = [len(v) for v in parts]
    if sum(lengths) > np.iinfo(np.uint32).max:
        raise ValueError('session dictionary byte offsets exceed fixed uint32 format')
    offsets = np.asarray([0]+list(np.cumsum(lengths, dtype=np.uint64)), dtype='<u4')
    return b''.join(parts), offsets


def _hash(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('canonical lowercase hex64 signature required')
    return np.frombuffer(bytes.fromhex(value), dtype='u1')


def _scalar(value, field):
    types = INT_TYPES if field in INTEGER_FIELDS else FLOAT_TYPES if field in FLOAT_FIELDS else BOOL_TYPES
    if type(value) not in types:
        raise ValueError('unknown or lossy scalar type for '+field)
    if field in INTEGER_FIELDS:
        v = int(value)
        if not np.iinfo(np.int64).min <= v <= np.iinfo(np.int64).max:
            raise ValueError('integer exceeds fixed signed64 storage before casting')
        if field not in ('dE', 'dFP', 'delta_M', 'delta_P') and v < 0:
            raise ValueError('negative count field '+field)
        if field == 'replacement_class' and not 0 <= v < 5:
            raise ValueError('five-class replacement required')
    elif field in FLOAT_FIELDS:
        v = float(value)
        if not np.isfinite(v) or field == 'min_overlap' and not 0 <= v <= 1:
            raise ValueError('finite scalar/unit overlap required')
    else: v = int(value)
    return v, types.index(type(value))


def encode_records(rows, sessions):
    _dictionary(sessions)
    if not isinstance(rows, (list, tuple)) or len(rows) > MAX_RECORDS:
        raise ValueError('bounded complete selected-record sequence required')
    index = {s: i for i, s in enumerate(sessions)}; out = np.zeros(len(rows), dtype=DTYPE)
    out['session_indices'].fill(65535)
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError('all and only28 approved record fields required, never drop fields')
        out['field_order'][i] = [FIELD_NAMES.index(k) for k in row]
        for field in HASH_FIELDS:
            if field == 'event_signature' and row[field] is None: continue
            out[field][i] = _hash(row[field])
        out['event_present'][i] = row['event_signature'] is not None
        if type(row['kind']) is not str or row['kind'] not in ('native', 'multi'):
            raise ValueError('only native/multi FIT selected records allowed')
        out['kind_code'][i] = ('native', 'multi').index(row['kind'])
        for j, field in enumerate(SCALAR_FIELDS):
            value, tag = _scalar(row[field], field); out[field][i] = value; out['type_tags'][i, j] = tag
        ss = row['responsible_sessions']
        if not isinstance(ss, list) or len(ss) > 8 or any(type(s) is not str or s not in index for s in ss):
            raise ValueError('at most8 responsible sessions, all inside frozen402 dictionary required')
        out['session_count'][i] = len(ss)
        out['session_indices'][i, :len(ss)] = [index[s] for s in ss]
        pairs = row['harm_transitions']
        if (not isinstance(pairs, list) or any(not isinstance(p, list) or len(p) != 2 or
                any(type(v) is not int or not 0 <= v < 5 for v in p) for p in pairs) or
                pairs != [list(p) for p in sorted({tuple(p) for p in pairs})]):
            raise ValueError('canonical sorted unique5x5 harm pairs required for exact bitmask encoding')
        out['harm_mask'][i] = sum(1 << (5*a+b) for a, b in pairs)
    return out


def decode_records(array, sessions):
    _dictionary(sessions)
    if (not isinstance(array, np.ndarray) or array.dtype != DTYPE or array.ndim != 1 or len(array) > MAX_RECORDS):
        raise ValueError('exact bounded nonobject fixed record dtype required')
    records = []
    for encoded in array:
        if encoded['kind_code'] not in (0, 1) or encoded['event_present'] not in (0, 1):
            raise ValueError('invalid kind or event-None encoding')
        row = {k: encoded[k].tobytes().hex() for k in HASH_FIELDS}
        if not encoded['event_present']: row['event_signature'] = None
        row['kind'] = ('native', 'multi')[int(encoded['kind_code'])]
        for j, field in enumerate(SCALAR_FIELDS):
            types = INT_TYPES if field in INTEGER_FIELDS else FLOAT_TYPES if field in FLOAT_FIELDS else BOOL_TYPES
            tag = int(encoded['type_tags'][j])
            if not 0 <= tag < len(types): raise ValueError('unknown original scalar dtype tag')
            target = types[tag]; value = encoded[field]
            if field in INTEGER_FIELDS:
                if target is not int:
                    limits = np.iinfo(target)
                    if not limits.min <= int(value) <= limits.max: raise ValueError('scalar cannot roundtrip to original integer dtype')
                row[field] = target(int(value))
            elif field in BOOL_FIELDS:
                if value not in (0, 1): raise ValueError('invalid original boolean encoding')
                row[field] = target(bool(value))
            else: row[field] = target(float(value))
        count = int(encoded['session_count'])
        if count > 8 or any(int(v) >= 402 for v in encoded['session_indices'][:count]):
            raise ValueError('invalid responsible-session index/count')
        row['responsible_sessions'] = [sessions[int(v)] for v in encoded['session_indices'][:count]]
        mask = int(encoded['harm_mask'])
        if mask >= 1 << 25: raise ValueError('harm bitmask outside5x5 classes')
        row['harm_transitions'] = [[a, b] for a in range(5) for b in range(5) if mask & (1 << (5*a+b))]
        order = encoded['field_order'].tolist()
        if sorted(order) != list(range(28)): raise ValueError('exact28-key insertion-order permutation required')
        records.append({FIELD_NAMES[k]: row[FIELD_NAMES[k]] for k in order})
    # Bytewise canonical roundtrip also rejects unused payload, invalid type tags
    # and narrowing a stored double under a falsely declared float16/32 tag.
    if encode_records(records, sessions).tobytes() != array.tobytes():
        raise ValueError('noncanonical or lossy fixed-record representation')
    return records


def _serialize(array, sessions):
    blob, offsets = _dictionary(sessions); stream = io.BytesIO()
    np.savez(stream, rows=array, session_blob=np.frombuffer(blob, dtype='u1'),
             session_offsets=offsets, format_version=np.asarray([1], dtype='<u2'))
    return stream.getvalue()


def storage_bound(n, sessions):
    if type(n) is not int or not 0 <= n <= MAX_RECORDS: raise ValueError('fixed bounded row count required')
    # ZIP_STORED makes byte length independent of values. Real headers/dictionary
    # are serialized, so no guessed per-string overhead or compression ratio.
    size = len(_serialize(np.zeros(n, dtype=DTYPE), sessions))
    return {'records': n, 'maximum_records': MAX_RECORDS, 'row_bytes': DTYPE.itemsize,
            'table_bytes': size, 'atomic_coexistence_upper_bytes': 2*size,
            'format': 'uncompressed nonobject NPZ v1', 'bound_independent_of_record_values': True,
            'dictionary_and_all_NPY_ZIP_headers_included': True}


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY|getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)


def write_records(path, rows, sessions):
    """Publish a single immutable table, retaining a partial temp on IO failure.

    Caller must enforce the new RUN namespace and actual storage admission.
    Existing files, symlinks and other writers are never overwritten or removed.
    """
    array = encode_records(rows, sessions); raw = _serialize(array, sessions)
    if len(raw) != storage_bound(len(rows), sessions)['table_bytes']:
        raise ValueError('unexpected data-dependent NPZ byte length')
    path = Path(path)
    if path.exists() or path.is_symlink(): raise FileExistsError('immutable selected table already exists')
    if any(p.is_symlink() for p in path.parents): raise ValueError('table ancestor must not be a symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(temp)
    try:
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(temporary, path)  # atomic no-clobber publication on the same FS
        _fsync_directory(path.parent)
        temporary.unlink(); _fsync_directory(path.parent)
    except BaseException as exc:
        raise RuntimeError('selected-table IO failed; keep owned partial evidence at '+str(temporary)) from exc
    return {**storage_bound(len(rows), sessions), 'path': str(path), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(), 'all28_fields_lossless': True,
            'immutable_no_clobber_atomic_file_publication': True,
            'caller_manifest_update_not_cross_file_atomic': True}


def read_records(path, *, expected_sha256, expected_sessions):
    """Require external immutable file SHA and the independently frozen dictionary."""
    _hash(expected_sha256); blob, offsets = _dictionary(expected_sessions); path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('regular pinned record table required')
    if path.stat().st_size > storage_bound(MAX_RECORDS, expected_sessions)['table_bytes']:
        raise ValueError('table exceeds fixed task byte bound')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256: raise ValueError('record table external SHA differs')
    try:
        # Check headers before any payload allocation, and never enable pickle.
        shapes = {'rows': None, 'session_blob': (len(blob),), 'session_offsets': (403,), 'format_version': (1,)}
        dtypes = {'rows': DTYPE, 'session_blob': np.dtype('u1'), 'session_offsets': np.dtype('<u4'), 'format_version': np.dtype('<u2')}
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if len(archive.infolist()) != 4 or set(archive.namelist()) != {k+'.npy' for k in ARCHIVE_KEYS}:
                raise ValueError('unknown, missing or duplicate NPZ payload field')
            for name in ARCHIVE_KEYS:
                info = archive.getinfo(name+'.npy')
                if info.compress_type != zipfile.ZIP_STORED: raise ValueError('fixed uncompressed format required')
                with archive.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    if version != (1, 0): raise ValueError('fixed NPY v1 header required')
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                if fortran or dtype != dtypes[name] or (name != 'rows' and shape != shapes[name]):
                    raise ValueError('fixed NPZ shape/dtype schema differs')
                if name == 'rows' and (len(shape) != 1 or not 0 <= shape[0] <= MAX_RECORDS):
                    raise ValueError('bounded one-dimensional record table required')
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            if (set(saved.files) != ARCHIVE_KEYS or saved['format_version'].tolist() != [1] or
                    saved['session_blob'].tobytes() != blob or not np.array_equal(saved['session_offsets'], offsets)):
                raise ValueError('format version or independently frozen session dictionary differs')
            return decode_records(saved['rows'], expected_sessions)
    except (OSError, EOFError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError('invalid complete fixed NPZ record table') from exc
