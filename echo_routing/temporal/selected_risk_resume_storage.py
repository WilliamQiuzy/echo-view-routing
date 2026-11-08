"""No-pickle, deterministic typed JSON/NPY private artifact container.

Supported values/dtypes/float bits and mapping order round-trip; object aliasing, strides
and writeability do not. NaN/Inf remain data, never clipped or silently dropped.
The caller restricts the namespace, pins source, meters IO and checks all caps.
Array/scalar subclasses and recursive dtype metadata are outside this format.
NPY headers are limited to10000 bytes on both writing and reading. Two-file
logical coexistence is an own-artifact bound, not a disk/RAM/host peak.
"""
import hashlib
import ast
import io
import json
import math
import os
from pathlib import Path
import secrets
import stat
import struct
import zipfile

import numpy as np


def _metadata(dtype):
    if dtype.metadata is not None:
        return True
    if dtype.subdtype is not None and _metadata(dtype.subdtype[0]):
        return True
    return bool(dtype.fields) and any(_metadata(field[0]) for field in dtype.fields.values())


def _encode(value, arrays, active):
    if isinstance(value, (np.ndarray, np.generic)):
        if isinstance(value,np.ndarray) and type(value) is not np.ndarray:
            raise ValueError('array subclasses, including masks, are outside the lossless format')
        if isinstance(value,np.generic) and type(value) is not value.dtype.type:
            raise ValueError('scalar subclasses are outside the lossless format')
        if _metadata(value.dtype):
            raise ValueError('dtype metadata is outside the lossless NPY format')
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            raise ValueError('object/pickle arrays are forbidden')
        buf = io.BytesIO(); np.save(buf, arr, allow_pickle=False)
        normalized_dtype = _npy_header(io.BytesIO(buf.getvalue()))
        if normalized_dtype.type is not arr.dtype.type:
            raise ValueError('dtype scalar class does not round-trip through NPY')
        index = len(arrays); arrays.append(buf.getvalue())
        return ['scalar' if isinstance(value, np.generic) else 'array', index]
    if value is None:
        return ['none']
    if type(value) is bool:
        return ['bool', value]
    if type(value) is int:
        return ['int', str(value)]
    if type(value) is float:
        return ['float64', struct.pack('>d', value).hex()]
    if type(value) is str:
        return ['str', value]
    if type(value) not in (dict, list, tuple, set, frozenset):
        raise ValueError('unsupported private artifact value')
    if id(value) in active:
        raise ValueError('cyclic private artifacts are forbidden')
    active.add(id(value))
    try:
        if type(value) is dict:
            if any(type(k) is not str for k in value):
                raise ValueError('ordered string-key mappings required')
            return ['dict', [[k, _encode(v, arrays, active)] for k, v in value.items()]]
        if type(value) in (set, frozenset):
            def key(item):
                leaves = []; node = _encode(item, leaves, set())
                return json.dumps(node, separators=(',', ':')), tuple(leaves)
            items = sorted(value, key=key)
        else:
            items = value
        return [type(value).__name__, [_encode(v, arrays, active) for v in items]]
    finally:
        active.remove(id(value))


def _archive(value):
    arrays = []; tree = _encode(value, arrays, set())
    meta = json.dumps(dict(version=1, tree=tree, arrays=len(arrays)),
                      ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, data in [('metadata.json', meta), *[(f'arrays/{i:06d}.npy', a) for i, a in enumerate(arrays)]]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3; info.external_attr = (stat.S_IFREG | 0o600) << 16
            z.writestr(info, data)
    data = buf.getvalue()
    return data, dict(array_leaves=len(arrays), metadata_bytes=len(meta),
                      npy_bytes=sum(map(len, arrays)), archive_bytes=len(data), pickle=False)


def storage_bound(value):
    data, layout = _archive(value)
    return dict(artifact_bytes=len(data), own_atomic_coexistence_bytes=2*len(data), layout=layout,
                scope='own regular logical file bytes only; excludes other artifacts, FS overhead and RAM')


def _parent(path, create):
    path = Path(path).absolute()
    if '..' in path.parts or path.name in ('', '.', '..'):
        raise ValueError('explicit canonical private file path required')
    fd = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.parts[1:-1]:
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=fd)
                    os.fsync(fd)
                except FileExistsError: pass
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        return path, fd
    except OSError as exc:
        os.close(fd)
        raise ValueError('private artifact parents must be real directories, never symlinks') from exc


def save_value(path, value):
    """Create once; never overwrite or recover an unrelated/partial artifact.

    A crash after link/fsync may leave the complete target and an owned temp.
    Recovery is explicit at the caller; no cross-crash transaction is claimed.
    """
    data, layout = _archive(value)
    path, parent = _parent(path, True)
    temp = '.'+path.name+'.tmp-'+secrets.token_hex(12)
    made = False
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        made = True
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(temp, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.fsync(parent)
        os.unlink(temp, dir_fd=parent); made = False
        os.fsync(parent)
        return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), layout=layout,
                    own_atomic_coexistence_bytes=2*len(data), existing_target_overwritten=False)
    finally:
        if made:
            os.unlink(temp, dir_fd=parent)
            os.fsync(parent)
        os.close(parent)


def _object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError('duplicate metadata field')
        out[k] = v
    return out


def _decode(node, arrays, used):
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError('typed artifact node required')
    tag = node[0]
    if tag == 'none' and len(node) == 1:
        return None
    if len(node) != 2:
        raise ValueError('exact typed artifact node arity required')
    v = node[1]
    if tag == 'bool' and type(v) is bool: return v
    if tag == 'str' and type(v) is str: return v
    if tag == 'int' and type(v) is str:
        n = int(v)
        if str(n) != v: raise ValueError('canonical integer text required')
        return n
    if tag == 'float64' and type(v) is str and len(v) == 16:
        return struct.unpack('>d', bytes.fromhex(v))[0]
    if tag in ('array', 'scalar') and type(v) is int and 0 <= v < len(arrays):
        if v in used: raise ValueError('each typed array leaf must be referenced exactly once')
        used.add(v); a = arrays[v]
        if tag == 'scalar':
            if a.shape != (): raise ValueError('numpy scalar requires zero dimensions')
            return a[()]
        return a
    if tag == 'dict' and type(v) is list:
        out = {}
        for row in v:
            if type(row) is not list or len(row) != 2 or type(row[0]) is not str or row[0] in out:
                raise ValueError('unique ordered string-key rows required')
            out[row[0]] = _decode(row[1], arrays, used)
        return out
    if tag in ('list', 'tuple', 'set', 'frozenset') and type(v) is list:
        items = [_decode(x, arrays, used) for x in v]
        out = {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset}[tag](items)
        if len(out) != len(items): raise ValueError('duplicate typed set members forbidden')
        return out
    raise ValueError('unsupported typed artifact node')


def _npy_header(raw):
    """Parse the published NPY1/2/3 header without allocating its data shape."""
    version = np.lib.format.read_magic(raw)
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError('unsupported NPY version')
    width = 2 if version == (1, 0) else 4
    length_raw = raw.read(width)
    if len(length_raw) != width: raise ValueError('truncated NPY header')
    size = int.from_bytes(length_raw, 'little')
    if size > 10000: raise ValueError('NPY header exceeds safe header-size limit')
    body = raw.read(size)
    if len(body) != size: raise ValueError('truncated NPY header')
    header = ast.literal_eval(body.decode('utf-8' if version == (3, 0) else 'latin1'))
    if (not isinstance(header, dict) or set(header) != {'descr', 'shape', 'fortran_order'} or
            type(header['shape']) is not tuple or type(header['fortran_order']) is not bool or
            any(type(n) is not int or n < 0 for n in header['shape'])):
        raise ValueError('exact typed NPY shape/order/dtype header required')
    dtype = np.lib.format.descr_to_dtype(header['descr'])
    if dtype.hasobject or math.prod(header['shape'])*dtype.itemsize != len(raw.getvalue())-raw.tell():
        raise ValueError('typed NPY shape/dtype must match exact available byte length')
    return dtype


def load_value(path, expected_sha256):
    """Read a verified byte snapshot, rejecting links, pickle and schema drift."""
    if (type(expected_sha256) is not str or len(expected_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('independently pinned file SHA256 required')
    path, parent = _parent(path, False)
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        except OSError as exc:
            raise ValueError('regular non-symlink private artifact required') from exc
        with os.fdopen(fd, 'rb') as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                raise ValueError('regular private artifact required')
            data = f.read()
    finally:
        os.close(parent)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('private artifact bytes differ from external SHA')
    try:
        with zipfile.ZipFile(io.BytesIO(data), 'r') as z:
            infos = z.infolist(); names = [i.filename for i in infos]
            if len(names) != len(set(names)) or any(i.compress_type != zipfile.ZIP_STORED or i.flag_bits & 1 or
                    stat.S_IFMT(i.external_attr >> 16) != stat.S_IFREG for i in infos):
                raise ValueError('unique uncompressed regular typed archive entries required')
            meta = json.loads(z.read('metadata.json'), object_pairs_hook=_object)
            if (set(meta) != {'version', 'tree', 'arrays'} or type(meta['version']) is not int or
                    meta['version'] != 1 or type(meta['arrays']) is not int or
                    not 0 <= meta['arrays'] == len(names)-1):
                raise ValueError('exact typed artifact version and leaf count required')
            expected = {'metadata.json', *[f'arrays/{i:06d}.npy' for i in range(meta['arrays'])]}
            if set(names) != expected: raise ValueError('exact declared archive entry set required')
            arrays = []
            for i in range(meta['arrays']):
                raw = io.BytesIO(z.read(f'arrays/{i:06d}.npy'))
                # A tiny corrupted leaf must not allocate its claimed shape.
                _npy_header(raw)
                raw.seek(0)
                arr = np.load(raw, allow_pickle=False)
                if not isinstance(arr, np.ndarray) or arr.dtype.hasobject or raw.tell() != len(raw.getvalue()):
                    raise ValueError('one exact nonobject NPY leaf required')
                arrays.append(arr)
            used = set(); value = _decode(meta['tree'], arrays, used)
            if used != set(range(len(arrays))): raise ValueError('unreferenced array leaves forbidden')
            return value
    except (KeyError, TypeError, OSError, zipfile.BadZipFile, UnicodeError,
            OverflowError, struct.error, SyntaxError, RecursionError) as exc:
        raise ValueError('invalid no-pickle typed artifact') from exc
