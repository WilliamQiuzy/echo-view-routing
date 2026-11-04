"""Minimal actual Git commit/tree/blob proof, verified without a remote .git.

Only the three public seal inputs are collected. Raw tree/commit objects remain
inside the server-only proof, not the public report or a copy of repository history.
"""
import base64,hashlib,subprocess
from pathlib import PurePosixPath


def oid(value):
    if not isinstance(value,str) or len(value)!=40 or any(c not in '0123456789abcdef' for c in value):raise ValueError('canonical Git SHA1 object ID required')
    return value


def unpack(raw,expected):
    if hashlib.sha1(raw).hexdigest()!=oid(expected):raise ValueError('actual Git object hash mismatch')
    header,separator,data=raw.partition(b'\0');kind,space,size=header.partition(b' ')
    if not separator or not space or size!=str(len(data)).encode() or kind not in [b'commit',b'tree',b'blob']:raise ValueError('canonical Git object header/length/type required')
    return kind.decode(),data


def tree_entry(data,name):
    position=0;found=None;names=set()
    while position<len(data):
        stop=data.find(b'\0',position)
        if stop<0 or stop+21>len(data):raise ValueError('invalid raw Git tree')
        mode,space,entry=data[position:stop].partition(b' ')
        if not space or entry in names:raise ValueError('ambiguous Git tree entry')
        names.add(entry);child=data[stop+1:stop+21].hex();position=stop+21
        if entry==name:found=(mode,child)
    if found is None:raise ValueError('sealed public path missing from actual Git tree')
    return found


def root_tree(data):
    first=data.splitlines()[0].split()
    if len(first)!=2 or first[0]!=b'tree':raise ValueError('actual commit root tree absent')
    return oid(first[1].decode())


def verify(proof,commit,expected):
    if proof.get('object_format')!='sha1' or proof.get('commit')!=oid(commit) or set(proof.get('files',{}))!=set(expected):raise ValueError('exact recorded commit/public input coverage required')
    used=set()
    def get(key,kind):
        used.add(key)
        try:raw=base64.b64decode(proof['objects'][key],validate=True)
        except (KeyError,ValueError) as e:raise ValueError('actual Git proof object missing/invalid') from e
        actual,data=unpack(raw,key)
        if actual!=kind:raise ValueError('Git tree/blob/commit type differs')
        return data
    root=root_tree(get(commit,'commit'))
    for path,h in expected.items():
        pp=PurePosixPath(path)
        if pp.is_absolute() or not pp.parts or any(p in ['.','..'] for p in pp.parts):raise ValueError('canonical relative public path required')
        node=root
        for index,part in enumerate(pp.parts):
            mode,node=tree_entry(get(node,'tree'),part.encode())
            final=index==len(pp.parts)-1
            if mode not in ([b'100644',b'100755'] if final else [b'40000']):raise ValueError('sealed path must traverse trees to regular blob, not symlink')
        data=get(node,'blob');row=proof['files'][path]
        if hashlib.sha256(data).hexdigest()!=h or row!={'blob_oid':node,'sha256':h,'bytes':len(data)}:raise ValueError('actual Git blob differs from current sealed public bytes')
    if set(proof['objects'])!=used:raise ValueError('unrelated objects must not be exported in minimal Git proof')
    return {'passed':True,'git_commit':commit,'object_count':len(used),'public_inputs':len(expected),'actual_commit_tree_and_blob_hashes_verified':True}


def build(repo,commit,paths):
    commit=oid(commit);objects={};files={}
    def get(key,kind):
        if key not in objects:
            actual=subprocess.check_output(['git','-C',str(repo),'cat-file','-t',key],text=True).strip()
            if actual!=kind:raise ValueError('local actual Git object type differs')
            data=subprocess.check_output(['git','-C',str(repo),'cat-file',kind,key]);raw=kind.encode()+b' '+str(len(data)).encode()+b'\0'+data
            unpack(raw,key);objects[key]=base64.b64encode(raw).decode()
        return unpack(base64.b64decode(objects[key]),key)[1]
    root=root_tree(get(commit,'commit'))
    for path in paths:
        parts=PurePosixPath(path).parts;node=root
        for index,part in enumerate(parts):
            mode,node=tree_entry(get(node,'tree'),part.encode())
            if mode not in ([b'100644',b'100755'] if index==len(parts)-1 else [b'40000']):raise ValueError('regular committed public path required')
        data=get(node,'blob');files[path]={'blob_oid':node,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    proof={'object_format':'sha1','commit':commit,'objects':objects,'files':files}
    verify(proof,commit,{p:r['sha256'] for p,r in files.items()});return proof
