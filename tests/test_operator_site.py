"""A48 metadata admission and physical target contract, no real media."""
import importlib
import importlib.util
import copy
import hashlib
import numpy as np
import pytest


def core():
    name='echo_routing.temporal.operator_site'
    assert importlib.util.find_spec(name) is not None, 'operator-site geometry is not implemented'
    return importlib.import_module(name)


def metadata():
    gate={'gate_center_depth_m':.08,'gate_tilt_rad':.1,'gate_sample_volume_m':.004}
    geo={'depth_start_m':.01,'depth_end_m':.2,'tilt_rad':0.,'width_rad':1.2,'grid_size':[101,61]}
    sec={'semantic_id':'bmode','frames':{'zarr_path':'data/2d_brightness_mode'},'geometry':geo,'sampling_gate_metadata':gate.copy()}
    man={'sampling_gate_metadata':gate.copy(),'sectors':[sec],'annotations':[{'kind':'measurement','links':{'measurement_kind':'aortic_lvot_trace_manual','measure_id':'Cardiac/SD/Aortic/LVOT Trace/Manual'},'target':{'semantic_id':'pw_doppler'},'target_data':{'zarr_path':'data/1d_pulsed_wave_doppler'}}]}
    return {'metadata':{'.zattrs':{'recording_manifest':man,'arrays':[{'name':'data/2d_brightness_mode','data_sha256':'a'*64}]},'data/2d_brightness_mode/.zarray':{'shape':[1,101,61],'chunks':[1,101,61],'dtype':'|u1','order':'C','filters':None,'compressor':{'id':'blosc'}}}}


def test_exact_static_manual_site_admission_rejects_ambiguous_and_multiframe():
    m=core();q=metadata();r=m.admit(q);assert r['site']=='LVOT' and r['shape']==[1,101,61]
    for kind in ['multi','ambiguous','different_gate','wrong_site','wrong_target']:
        z=copy.deepcopy(q);d=z['metadata'];man=d['.zattrs']['recording_manifest']
        if kind=='multi':d['data/2d_brightness_mode/.zarray']['shape'][0]=2
        elif kind=='ambiguous':man['sectors'].append(copy.deepcopy(man['sectors'][0]))
        elif kind=='different_gate':man['sampling_gate_metadata']['gate_tilt_rad']=.2
        elif kind=='wrong_site':man['annotations'][0]['links']['measurement_kind']='automated_unknown'
        else:man['annotations'][0]['target_data']['zarr_path']='data/other'
        with pytest.raises(ValueError):m.admit(z)


def test_physical_beam_and_cartesian_roundtrip_without_clipping():
    m=core();r=m.admit(metadata());xy=m.beam_xy(r['gate'],r['geometry'],(101,61))
    assert np.allclose(xy,[35.,(.08-.01)/.19*100])
    box=(-.12,.12,0.,.2);pix=m.cartesian_xy(r['gate'],box,(201,241))
    assert np.allclose(pix,[(.08*np.sin(.1)+.12)/.24*240,.08*np.cos(.1)/.2*200])
    depth,angle=m.invert_cartesian(pix,box,(201,241));assert abs(depth-.08)<1e-12 and abs(angle-.1)<1e-12
    for g in [dict(r['gate'],gate_center_depth_m=0.),dict(r['gate'],gate_tilt_rad=2.),dict(r['gate'],gate_center_depth_m=float('nan'))]:
        with pytest.raises(ValueError):m.beam_xy(g,r['geometry'],(101,61))


def test_rendered_target_outside_grid_is_rejected_not_clipped():
    m=core();g=metadata()['metadata']['.zattrs']['recording_manifest']['sampling_gate_metadata']
    with pytest.raises(ValueError):m.cartesian_xy(g,(.1,.2,0.,.2),(224,224))


def test_point_spread_requires_both_axes_and_multiple_occupied_bins():
    m=core();same=np.tile([.5,.5],(80,1));assert not m.spatial_spread(same)['passed']
    good=np.random.default_rng(48).uniform(.2,.8,(80,2));assert m.spatial_spread(good)['passed']


def test_raw_array_hash_size_and_dtype_are_verified_after_decompression():
    m=core();assert hasattr(m,'decode_static'),'released raw-array verification missing'
    raw=np.arange(12,dtype=np.uint8).reshape(1,3,4);r={'shape':[1,3,4],'array_sha256':hashlib.sha256(raw.tobytes()).hexdigest()}
    assert np.array_equal(m.decode_static(raw.tobytes(),r,lambda x:x),raw)
    with pytest.raises(ValueError):m.decode_static(raw.tobytes()[:-1],r,lambda x:x)
    with pytest.raises(ValueError):m.decode_static(b'X'+raw.tobytes()[1:],r,lambda x:x)
