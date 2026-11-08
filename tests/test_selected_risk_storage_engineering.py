import importlib,json,hashlib,base64
import pytest

def module():
    try:m=importlib.import_module('echo_routing.temporal.selected_risk_storage_engineering')
    except ModuleNotFoundError:m=None
    assert m is not None,'prospective proof sharing and lossless trace archive absent'
    return m

def test_shared_proof_expands_byte_identically_and_rejects_wrong_reference():
    m=module();raw=b'blob 3\0abc';oid=hashlib.sha1(raw).hexdigest();value=base64.b64encode(raw).decode();p={'object_format':'sha1','commit':'a'*40,'files':{},'objects':{oid:value,'b'*40:'unshared'}};pool={'objects':{oid:value}}
    packed=m.pack_proof(p,pool);assert packed['objects'][oid]=='@shared:'+oid and m.expand_proof(packed,pool)==p
    packed['objects'][oid]='@shared:'+'c'*40
    with pytest.raises(ValueError):m.expand_proof(packed,pool)

def test_future_trace_archive_preserves_every_original_byte_and_refuses_pending(tmp_path):
    m=module();identity={'x':'a'*64};p=tmp_path/'physical_updates.jsonl';raw=b''.join((json.dumps({'u':i,'identity':identity,'loss':float(i)/3},sort_keys=True)+'\n').encode() for i in range(3));p.write_bytes(raw)
    pending=tmp_path/'pending.json';pending.write_text('{}')
    with pytest.raises(ValueError):m.archive_trace(p,identity,expected_rows=3)
    pending.unlink();v=m.archive_trace(p,identity,expected_rows=3)
    assert not p.exists() and v['uncompressed_sha256']==hashlib.sha256(raw).hexdigest()
    assert m.trace_bytes(tmp_path/v['archive_name'],v)==raw and m.load_trace(tmp_path/v['archive_name'],v)==[json.loads(x) for x in raw.splitlines()]
    with pytest.raises(FileExistsError):m.archive_trace(p,identity,expected_rows=3)

def test_archive_malformed_sequence_fails_before_any_removal(tmp_path):
    m=module();p=tmp_path/'physical_updates.jsonl';p.write_text('{"u":1,"identity":{}}\n')
    with pytest.raises(ValueError):m.archive_trace(p,{},expected_rows=1)
    assert p.exists() and not (tmp_path/'physical_updates.jsonl.gz').exists()


def test_actual_current_Git_tree_proof_survives_shared_object_encoding(tmp_path):
    import subprocess
    from echo_routing.temporal.safety_git import build,verify
    m=module();r=tmp_path/'repo';r.mkdir();subprocess.run(['git','init','-q',str(r)],check=True)
    def commit(files,parent=None):
        entries=[]
        for name,text in sorted(files.items()):
            (r/name).write_text(text);oid=subprocess.check_output(['git','-C',str(r),'hash-object','-w',name]).decode().strip();entries.append(f'100644 blob {oid}\t{name}\n')
        tree=subprocess.check_output(['git','-C',str(r),'mktree'],input=''.join(entries).encode()).decode().strip();args=['git','-C',str(r),'-c','user.name=test','-c','user.email=test@example.invalid','commit-tree',tree]
        if parent:args.extend(['-p',parent])
        return subprocess.check_output(args,input=b'fixture\n').decode().strip()
    a=commit({'legacy.txt':'exact old bytes'*100});pool=build(r,a,['legacy.txt']);b=commit({'legacy.txt':'exact old bytes'*100,'new.txt':'new stage'},a);full=build(r,b,['legacy.txt','new.txt']);packed=m.pack_proof(full,pool)
    assert len(json.dumps(packed))<len(json.dumps(full));expanded=m.expand_proof(packed,pool);assert expanded==full
    verify(expanded,b,{p:v['sha256'] for p,v in full['files'].items()})
