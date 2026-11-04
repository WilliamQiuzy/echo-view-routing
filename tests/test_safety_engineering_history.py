import hashlib,json,subprocess
from pathlib import Path
import pytest
from echo_routing.temporal.safety_git import build
from echo_routing.temporal.safety_engineering_history import validate


def h(x):return hashlib.sha256(x).hexdigest()


def test_exact_original_cost_contracts_only_allow_pinned_document_revision(tmp_path):
    def git(*args):return subprocess.check_output(['git','-C',str(tmp_path),*args],text=True).strip()
    git('init','-q');git('config','user.email','toy@example.invalid');git('config','user.name','toy')
    doc='cost.md';code='code.py';(tmp_path/code).write_text('unchanged-code\n');bundle={'contracts':{}};raw={};expected={}
    for name,text in [('component','first'),('rle','second')]:
        (tmp_path/doc).write_text(text);git('add','--',doc,code);git('commit','-qm',name);commit=git('rev-parse','HEAD');objects={p:h((tmp_path/p).read_bytes()) for p in [doc,code]};r=(json.dumps({'git_commit':commit,'objects':objects},sort_keys=True)+'\n').encode();raw[name]=r;expected[name]={'commit':commit,'objects':2,'identity_sha256':h(r),'old_doc_sha256':objects[doc]};bundle['contracts'][name]={'proof':build(tmp_path,commit,objects)}
    (tmp_path/doc).write_text('third-current');current=h((tmp_path/doc).read_bytes());out=validate(bundle,raw,tmp_path,expected=expected,doc_path=doc,current_doc_sha=current)
    assert out['passed'] and out['historical_object_bindings']==4 and all(x['current_divergence']==[doc] for x in out['contracts'])
    bad=dict(bundle);bad['contracts']={k:v for k,v in bundle['contracts'].items() if k!='rle'}
    with pytest.raises(ValueError):validate(bad,raw,tmp_path,expected=expected,doc_path=doc,current_doc_sha=current)
    (tmp_path/code).write_text('changed')
    with pytest.raises(ValueError):validate(bundle,raw,tmp_path,expected=expected,doc_path=doc,current_doc_sha=current)
