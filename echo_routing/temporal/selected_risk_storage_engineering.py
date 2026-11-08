"""Prospective A63 lossless storage codecs, never alters old executed artifacts.

Shared Git objects expand byte-for-byte before the unchanged qualified verifier.
Only a newly COMPLETE future trace may be archived: every original JSONL byte,
6000 row order and identity remain reproducible and independently hash-bound.
An ambiguous archive/pending state fails closed; it never replays an update.
"""
import base64,copy,gzip,hashlib,json,os
from pathlib import Path
from echo_routing.temporal.anchor_cache import sha,fsync_directory


def pack_proof(proof,pool):
    out=copy.deepcopy(proof)
    for oid,value in out['objects'].items():
        if pool['objects'].get(oid)==value:out['objects'][oid]='@shared:'+oid
    return out

def expand_proof(proof,pool):
    from echo_routing.temporal.safety_git import unpack
    out=copy.deepcopy(proof)
    for oid,value in out['objects'].items():
        if isinstance(value,str) and value.startswith('@shared:'):
            if value!='@shared:'+oid or oid not in pool['objects']:raise ValueError('exact shared object OID required')
            raw=base64.b64decode(pool['objects'][oid],validate=True);unpack(raw,oid);out['objects'][oid]=pool['objects'][oid]
    return out


def trace_bytes(path,receipt):
    path=Path(path)
    if sha(path)!=receipt['archive_sha256']:raise ValueError('immutable archived trace bytes changed')
    with gzip.open(path,'rb') as f:raw=f.read()
    if len(raw)!=receipt['uncompressed_bytes'] or hashlib.sha256(raw).hexdigest()!=receipt['uncompressed_sha256']:raise ValueError('lossless original trace SHA/length mismatch')
    return raw

def load_trace(path,receipt):return [json.loads(line) for line in trace_bytes(path,receipt).splitlines()]


def archive_trace(path,identity,*,expected_rows=6000):
    path=Path(path);archive=path.with_suffix(path.suffix+'.gz');rp=path.with_suffix(path.suffix+'.archive.json');tmp=archive.with_suffix(archive.suffix+'.tmp')
    if archive.exists() or rp.exists() or tmp.exists():raise FileExistsError('archive already exists/ambiguous; retain original, no implicit reconciliation')
    if (path.parent/'pending.json').exists():raise ValueError('never archive ambiguous physical updates')
    if path.is_symlink() or not path.is_file():raise ValueError('new complete regular physical trace required')
    raw=path.read_bytes();rows=[json.loads(line) for line in raw.splitlines()]
    if isinstance(expected_rows,bool) or not isinstance(expected_rows,int) or expected_rows<1 or len(rows)!=expected_rows or any(r.get('u')!=i or r.get('identity')!=identity for i,r in enumerate(rows)) or not raw.endswith(b'\n'):raise ValueError('all exact original ordered rows/identity required before archive')
    rawsha=hashlib.sha256(raw).hexdigest()
    with tmp.open('xb') as out:
        with gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=out,mtime=0) as z:z.write(raw)
        out.flush();os.fsync(out.fileno())
    with gzip.open(tmp,'rb') as f:
        if f.read()!=raw:raise ValueError('archive roundtrip must retain EVERY original byte')
    os.replace(tmp,archive);fsync_directory(path.parent)
    result={'archive_name':archive.name,'archive_sha256':sha(archive),'uncompressed_sha256':rawsha,'uncompressed_bytes':len(raw),'archive_bytes':archive.stat().st_size,'rows':len(rows),'lossless_original_bytes_available':True,'identity':identity,'schema':'A63_future_complete_trace_gzip_v1','original_physical_update_count_unchanged':True}
    with rp.open('x') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fsync_directory(path.parent)
    if sha(path)!=rawsha or (path.parent/'pending.json').exists():raise ValueError('trace changed before lossless archive publication; retain both')
    path.unlink();fsync_directory(path.parent);return result
