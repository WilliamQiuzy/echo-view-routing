import importlib,json,hashlib
import numpy as np
import pytest
from tests.test_structured_producers import A08,Pixels

def test_same_weights_but_different_stride_role_or_code_never_share_cache_identity():
    m=importlib.import_module('scripts.produce_structured_cache');p={'weights':'same','code':'code','stride':2};a=m.cache_identity('a08_dense','fit','input',p)
    assert len({a,m.cache_identity('a08_original','fit','input',{**p,'stride':5}),m.cache_identity('a08_dense','inner','input',p),m.cache_identity('a08_dense','fit','input',{**p,'code':'changed'})})==4
    q={'refs':[('v',i,1.) for i in range(19)]};p=m.a08_original_sequence(A08(),q,Pixels(),device='cpu');assert p.shape==(19,5) and np.array_equal(p[0],p[1])

def test_old_JEPA_prefix_is_readonly_and_corruption_or_axis_mismatch_fails(tmp_path):
    m=importlib.import_module('scripts.produce_structured_cache');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=tmp_path/'sequence000000.private.npz';np.savez(p,prob=np.full((3,5),.2,np.float32));binding=tmp_path/'identity.json';binding.write_text(json.dumps({'identity':'fixed','total':1}));j=tmp_path/'committed.jsonl';j.write_text(json.dumps({'index':0,'identity':'fixed','positions':3,'sha256':sha(p)})+'\n');seqs=[{'id':'q','refs':[('v',i,1.) for i in range(3)]}]
    assert m.load_readonly_prefix(tmp_path,'fixed',seqs,sha(j),sha(binding))['q'].shape==(3,5)
    with pytest.raises(ValueError):m.load_readonly_prefix(tmp_path,'fixed',[{'id':'q','refs':[('v',0,1.)]}],sha(j),sha(binding))
    np.savez(p,prob=np.full((3,5),.1,np.float32))
    with pytest.raises(ValueError):m.load_readonly_prefix(tmp_path,'fixed',seqs,sha(j),sha(binding))
