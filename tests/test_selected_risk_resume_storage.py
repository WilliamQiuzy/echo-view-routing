"""No-pickle private artifact codec; synthetic values and filesystem only."""
import importlib
import hashlib
from pathlib import Path
import runpy

import numpy as np
import pytest


def module(): return importlib.import_module('echo_routing.temporal.selected_risk_resume_storage')


def sample():
    return {'ordered-z':(None,True,False,2**85,-11,'text'), 'ordered-a':[
        np.arange(12,dtype='>i4').reshape(3,4), np.array([np.nan,np.inf,-np.inf,-0.],dtype='<f8'),
        np.float32(-0.),np.int16(-2),np.bool_(True),float('nan'),float('inf'),-0.],
        'sets':{('a',1),('b',2)},'frozen':frozenset([3,4]),
        'structured':np.array([(2,3.)],dtype=[('i','<i2'),('f','>f8')])}


def test_roundtrip_preserves_order_types_array_dtype_and_float_bits(tmp_path):
    m=module();v=sample();path=tmp_path/'private.npz';receipt=m.save_value(path,v)
    out=m.load_value(path,receipt['sha256'])
    assert list(out)==list(v) and isinstance(out['ordered-z'],tuple)
    assert isinstance(out['frozen'],frozenset) and out['sets']==v['sets']
    for a,b in zip(v['ordered-a'],out['ordered-a']):
        assert type(a) is type(b)
        if isinstance(a,np.ndarray): assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        elif isinstance(a,(float,np.floating)): assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
        else: assert a==b
    assert out['structured'].dtype==v['structured'].dtype
    assert out['structured'].tobytes()==v['structured'].tobytes()
    assert m.storage_bound(v)['artifact_bytes']==path.stat().st_size
    assert m.storage_bound(v)['own_atomic_coexistence_bytes']==2*path.stat().st_size


def test_set_order_and_saves_have_deterministic_bytes_and_no_clobber(tmp_path):
    m=module();a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    first=m.save_value(a,{'set':set([3,'a',('x',np.int8(2))])})
    second=m.save_value(b,{'set':set([('x',np.int8(2)),'a',3])})
    assert a.read_bytes()==b.read_bytes() and first['sha256']==second['sha256']
    with pytest.raises((ValueError,FileExistsError)): m.save_value(a,{'changed':1})
    assert hashlib.sha256(a.read_bytes()).hexdigest()==first['sha256']


@pytest.mark.parametrize('bad',[np.array([object()],dtype=object),{1:'bad-key'},object()])
def test_unsupported_or_pickle_values_are_rejected_without_output(tmp_path,bad):
    with pytest.raises(ValueError): module().save_value(tmp_path/'bad.npz',bad)
    assert not (tmp_path/'bad.npz').exists()


def test_cycles_and_external_SHA_mismatch_are_rejected(tmp_path):
    m=module();v=[];v.append(v)
    with pytest.raises(ValueError): m.storage_bound(v)
    path=tmp_path/'good.npz';m.save_value(path,sample())
    with pytest.raises(ValueError): m.load_value(path,'0'*64)


def test_symlink_leaf_and_parent_are_rejected(tmp_path):
    m=module();real=tmp_path/'real';real.mkdir();link=tmp_path/'link';link.symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError): m.save_value(link/'artifact.npz',sample())
    target=real/'artifact.npz';pin=m.save_value(target,sample());leaf=tmp_path/'leaf.npz';leaf.symlink_to(target)
    with pytest.raises(ValueError): m.load_value(leaf,pin['sha256'])


def test_metadata_schema_tampering_is_rejected_even_with_correct_file_SHA(tmp_path):
    import zipfile,json
    path=tmp_path/'bad.npz'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',json.dumps({'version':1,'tree':['execute','bad']}))
    with pytest.raises(ValueError): module().load_value(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_new_parent_directory_is_fsynced_before_descending(tmp_path,monkeypatch):
    m=module();events=[];mkdir=m.os.mkdir;fsync=m.os.fsync;opening=m.os.open
    def make(name,*args,**kwargs):
        out=mkdir(name,*args,**kwargs);events.append(('mkdir',name,kwargs.get('dir_fd')));return out
    def sync(fd): events.append(('fsync',None,fd));return fsync(fd)
    def op(name,*args,**kwargs): events.append(('open',name,kwargs.get('dir_fd')));return opening(name,*args,**kwargs)
    monkeypatch.setattr(m.os,'mkdir',make);monkeypatch.setattr(m.os,'fsync',sync);monkeypatch.setattr(m.os,'open',op)
    m.save_value(tmp_path/'new-parent'/'private.npz',sample())
    n=next(i for i,e in enumerate(events) if e[0:2]==('mkdir','new-parent'))
    end=next(i for i in range(n+1,len(events)) if events[i][0:2]==('open','new-parent'))
    assert ('fsync',None,events[n][2]) in events[n+1:end]


def test_impossible_NPY_shape_is_rejected_before_numpy_allocation(tmp_path,monkeypatch):
    import zipfile,json,io,stat
    m=module();raw=io.BytesIO()
    np.lib.format.write_array_header_1_0(raw,{'descr':'|u1','fortran_order':False,'shape':(10**12,)})
    path=tmp_path/'impossible.npz'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        for name,data in [('metadata.json',json.dumps({'version':1,'tree':['array',0],'arrays':1}).encode()),
                           ('arrays/000000.npy',raw.getvalue())]:
            info=zipfile.ZipInfo(name);info.external_attr=(stat.S_IFREG|0o600)<<16;z.writestr(info,data)
    def forbidden(*args,**kwargs): raise AssertionError('numpy allocation before exact leaf size check')
    monkeypatch.setattr(m.np,'load',forbidden)
    with pytest.raises(ValueError): m.load_value(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_actual_qualified_pair_bundle_roundtrip_keeps_its_identity(tmp_path):
    fixture=runpy.run_path(str(Path(__file__).with_name('test_selected_risk_resume_pairing.py')))
    out,expected=fixture['produce']()
    from echo_routing.temporal.selected_risk_resume_pairing import validate_pair_bundle
    m=module();p=tmp_path/'counts.private.npz';pin=m.save_value(p,out['pair_bundle'])
    loaded=m.load_value(p,pin['sha256'])
    assert validate_pair_bundle(loaded,out['row'],expected)


def test_writer_rejects_its_reader_header_domain_before_creating_any_output(tmp_path):
    m=module();array=np.zeros(1,dtype=[('field'+str(i),'u1') for i in range(700)])
    with pytest.raises(ValueError,match='header'):
        m.storage_bound(array)
    path=tmp_path/'must-not-create'/'artifact.npz'
    with pytest.raises(ValueError,match='header'):
        m.save_value(path,array)
    assert not path.parent.exists()


@pytest.mark.parametrize('kind',['masked','metadata','nested_metadata','scalar_subclass'])
def test_unsupported_lossy_numpy_values_are_rejected_before_publication(tmp_path,kind):
    class ExtraFloat(np.float64): pass
    values={
        'masked':np.ma.array([1,2],mask=[False,True]),
        'metadata':np.array([1],dtype=np.dtype('i4',metadata={'unit':'synthetic'})),
        'nested_metadata':np.zeros(1,dtype=[('x',np.dtype('i2',metadata={'unit':'synthetic'}))]),
        'scalar_subclass':ExtraFloat(1.)}
    path=tmp_path/'must-not-create'/'artifact.npz'
    with pytest.raises(ValueError): module().save_value(path,values[kind])
    assert not path.parent.exists()
