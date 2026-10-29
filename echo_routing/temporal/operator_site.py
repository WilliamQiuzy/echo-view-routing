"""Strict admission of weak operator-site targets, not expert anatomy labels."""
import math
import hashlib
import numpy as np


def beam_xy(gate,geometry,shape):
    depth=float(gate['gate_center_depth_m']);angle=float(gate['gate_tilt_rad'])
    width=float(geometry['width_rad']);tilt=float(geometry.get('tilt_rad',0.))
    d0=float(geometry['depth_start_m']);d1=float(geometry['depth_end_m'])
    size=float(gate['gate_sample_volume_m']);a0=tilt-width/2;a1=tilt+width/2
    if not all(math.isfinite(v) for v in [depth,angle,width,tilt,d0,d1,size]):
        raise ValueError('nonfinite native geometry')
    if not d0>=0 or not d1>d0 or not 0<width<math.pi or not size>0:
        raise ValueError('invalid native geometry')
    if not d0<depth-size/2<depth+size/2<d1 or not a0<angle<a1:
        raise ValueError('operator sample volume outside sector; never clip')
    height,cols=shape
    if height<2 or cols<2:raise ValueError('degenerate image')
    return np.array([(angle-a0)/(a1-a0)*(cols-1),(depth-d0)/(d1-d0)*(height-1)],np.float64)


def cartesian_xy(gate,bounds,shape):
    depth=float(gate['gate_center_depth_m']);angle=float(gate['gate_tilt_rad'])
    x=depth*math.sin(angle);y=depth*math.cos(angle);x0,x1,y0,y1=bounds
    if not all(math.isfinite(v) for v in [x,y,x0,x1,y0,y1]) or not x0<x<x1 or not y0<y<y1:
        raise ValueError('point outside rendered grid; never clip')
    height,width=shape
    return np.array([(x-x0)/(x1-x0)*(width-1),(y-y0)/(y1-y0)*(height-1)],np.float64)


def invert_cartesian(xy,bounds,shape):
    x0,x1,y0,y1=bounds;height,width=shape
    x=x0+float(xy[0])/(width-1)*(x1-x0);y=y0+float(xy[1])/(height-1)*(y1-y0)
    return math.hypot(x,y),math.atan2(x,y)


def admit(q):
    meta=q['metadata'];attrs=meta['.zattrs'];man=attrs['recording_manifest']
    path='data/2d_brightness_mode';z=meta[path+'/.zarray'];shape=z['shape']
    if len(shape)!=3 or shape[0]!=1 or z['chunks']!=shape or np.dtype(z['dtype'])!=np.dtype('uint8') or z['order']!='C' or z.get('filters'):
        raise ValueError('one exact static native tissue frame/chunk is required')
    sectors=[s for s in man.get('sectors',[]) if s.get('semantic_id')=='bmode' or s.get('frames',{}).get('zarr_path')==path]
    if len(sectors)!=1 or sectors[0].get('frames',{}).get('zarr_path')!=path:
        raise ValueError('ambiguous B-mode sector identity')
    sector=sectors[0];gate=sector.get('sampling_gate_metadata');rootgate=man.get('sampling_gate_metadata')
    if gate is None or rootgate is None or any(gate.get(k)!=rootgate.get(k) for k in ['gate_center_depth_m','gate_tilt_rad','gate_sample_volume_m']):
        raise ValueError('root/sector sample-volume pose identity is not established')
    geometry=sector['geometry']
    if list(geometry['grid_size'])!=shape[1:]:raise ValueError('native grid identity mismatch')
    labels=[]
    for item in man.get('annotations',[]):
        if item.get('kind')!='measurement':continue
        links=item.get('links',{})
        if item.get('target',{}).get('semantic_id')=='pw_doppler':
            if item.get('target_data',{}).get('zarr_path')!='data/1d_pulsed_wave_doppler':raise ValueError('different clinical target stream')
            labels.append((links.get('measurement_kind'),links.get('measure_id')))
    if not labels or set(labels)!={('aortic_lvot_trace_manual','Cardiac/SD/Aortic/LVOT Trace/Manual')}:
        raise ValueError('exact manual LVOT site identity is not established')
    beam_xy(gate,geometry,shape[1:])
    hashes=[a['data_sha256'] for a in attrs.get('arrays',[]) if a.get('name')==path]
    if len(hashes)!=1 or len(hashes[0])!=64:raise ValueError('released native array hash missing/ambiguous')
    return {'site':'LVOT','gate':dict(gate),'geometry':dict(geometry),'shape':list(shape),'array_sha256':hashes[0],'zarray':z,'target_semantics':'weak operator sampling-location; not anatomy/visibility/A5C truth'}


def spatial_spread(points):
    xy=np.asarray(points,np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all() or not ((xy>0)&(xy<1)).all():
        raise ValueError('strict normalized interior target coordinates required')
    std=xy.std(0);occupied=len(np.unique(np.floor(xy*14).astype(np.int64),axis=0))
    return {'std_xy':std.tolist(),'occupied_14x14_bins':occupied,'passed':bool((std>=.025).all() and occupied>=12)}


def decode_static(blob,record,decoder):
    raw=bytes(decoder(blob));shape=record['shape']
    if len(raw)!=math.prod(shape):raise ValueError('native uint8 chunk decoded size mismatch')
    if hashlib.sha256(raw).hexdigest()!=record['array_sha256']:raise ValueError('released raw array SHA mismatch')
    return np.frombuffer(raw,dtype=np.uint8).reshape(shape).copy()
