"""Bounded A48 source contract experiment. No EV9V/model read or training.

Public recording identities, pixel arrays and targets remain in the server-only
namespace. Reuses existing A47 pages without modifying those artifacts.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import time
import types
import urllib.request

ROOT=Path('/home/william/echo-view-routing');sys.path.insert(0,str(ROOT))
OLD=ROOT/'runs/a47_public_evidence_intake_v1'
OUT=ROOT/'runs/a48_anatomical_qualification_v1'
sys.path.insert(0,str(OUT/'vendor'))
import numpy as np
from numcodecs import get_codec
from echo_routing.temporal import operator_site as op

REV='7a36922880061791519a8f01553c15cad5df564f'


class Reader:
    def __init__(self,exam,size):
        self.url=f'https://huggingface.co/datasets/Ahus-AIM/EchoXFlow/resolve/{REV}/exams/{exam}.tar'
        self.size=size;self.bytes=0;self.requests=0;self.old={};self.cache={};self.started=time.monotonic()
        for d in ['ranges_attempt2','ranges_indexed','ranges_target']:
            for p in (OLD/d).glob('*.bin'):
                if p.stat().st_size==65536:self.old[int(p.stem)]=p
        self.dir=OUT/'ranges';self.dir.mkdir(exist_ok=True)

    def read(self,pos,count):
        if not 0<=pos<self.size or not 0<count<=5_000_000 or pos+count>self.size:raise ValueError('unbounded/outside archive read')
        chunks=[]
        while count:
            start=pos//65536*65536
            if start in self.old:
                page=self.old[start].read_bytes();off=pos-start;n=min(count,len(page)-off);chunks.append(page[off:off+n]);pos+=n;count-=n;continue
            start=pos//16384*16384;end=min(start+16384,self.size)-1
            if start not in self.cache:
                if self.bytes+end-start+1>230_000_000 or self.requests>=3000 or time.monotonic()-self.started>1200:raise RuntimeError('qualification transport subcap')
                req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=25) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('exact HTTP range required')
                    page=r.read(end-start+2)
                if len(page)!=end-start+1:raise ValueError('wrong range byte count')
                self.bytes+=len(page);self.requests+=1;self.cache[start]=page;(self.dir/f'{start:012d}.bin').write_bytes(page)
            page=self.cache[start];off=pos-start;n=min(count,len(page)-off);chunks.append(page[off:off+n]);pos+=n;count-=n
        return b''.join(chunks)

    def member(self,pos):
        block=self.read(pos,512);info=tarfile.TarInfo.frombuf(block,'utf-8','strict')
        return info,pos+512+math.ceil(info.size/512)*512


def upstream():
    source=OUT/'upstream/src/echoxflow'
    for name,path in [('echoxflow',source),('echoxflow.scan',source/'scan')]:
        pkg=types.ModuleType(name);pkg.__path__=[str(path)];sys.modules[name]=pkg
    from echoxflow.scan.geometry import sector_geometry_from_mapping,CartesianGrid
    from echoxflow.scan.coordinates import BeamspacePixelGrid,CartesianPixelGrid
    from echoxflow.scan.conversion import sector_to_cartesian
    return sector_geometry_from_mapping,CartesianGrid,BeamspacePixelGrid,CartesianPixelGrid,sector_to_cartesian


def run():
    sel=json.loads((OLD/'target_selection.private.json').read_text());sq=json.loads((OLD/'sample_selection.private.json').read_text());q=json.loads((OLD/f'sample_{sel["sha256"]}.private.json').read_text());assert hashlib.sha256((OLD/f'sample_{sel["sha256"]}.private.json').read_bytes()).hexdigest()==sel['sha256']
    admitted=op.admit(q);pack=json.loads((OLD/'package_manifest.json').read_text());reader=Reader(sq['exam'],pack['tar_bytes'][sq['exam']]);pos=sel['offset'];wanted=sel['target']+'/data/2d_brightness_mode/0.0.0';blob=None
    try:
        for visited in range(2000):
            info,nxt=reader.member(pos)
            if info.name==wanted:
                if not info.isfile() or info.size>5_000_000:raise ValueError('wrong native chunk type/size')
                blob=reader.read(pos+512,info.size);break
            if pos>sel['offset'] and '.zarr/' in info.name and not info.name.startswith(sel['target']+'/'):raise ValueError('left exact selected recording without finding raw chunk')
            pos=nxt
        if blob is None:raise RuntimeError('bounded native chunk lookup failed')
        codec=get_codec(admitted['zarray']['compressor']);raw=op.decode_static(blob,admitted,codec.decode);np.save(OUT/'sample_native.private.npy',raw)
        sg,CG,BG,PG,convert=upstream();geometry=sg(admitted['geometry']);grid=CG.from_sector_height(geometry,224)
        gate=admitted['gate'];physical=np.array([[gate['gate_center_depth_m']*math.sin(gate['gate_tilt_rad']),gate['gate_center_depth_m']*math.cos(gate['gate_tilt_rad'])]])
        beam=op.beam_xy(gate,admitted['geometry'],raw.shape[1:]);official_beam=BG(geometry,raw.shape[1:]).physical_to_pixel_xy(physical)[0];be=float(np.max(abs(beam-official_beam)/np.array(raw.shape[:0:-1])))
        bounds=(*grid.x_range_m,*grid.y_range_m);point=op.cartesian_xy(gate,bounds,grid.shape);official=PG(grid).physical_to_pixel_xy(physical)[0];pe=float(np.max(abs(point-official)/np.array(grid.shape[::-1])))
        dd,aa=op.invert_cartesian(point,bounds,grid.shape);roundtrip=max(abs(dd-gate['gate_center_depth_m']),abs(aa-gate['gate_tilt_rad']));assert be<1e-5 and pe<1e-5 and roundtrip<1e-5
        converted=convert(raw[0],geometry,grid=grid);assert converted.mask[int(round(point[1])),int(round(point[0]))]
        # A separate direct inverse-grid construction checks the render, no annotation overlay.
        from scipy.ndimage import map_coordinates
        xx,yy=np.meshgrid(np.linspace(grid.x_range_m[0],grid.x_range_m[1],grid.shape[1]),np.linspace(grid.y_range_m[0],grid.y_range_m[1],grid.shape[0]));depth=np.hypot(xx,yy);theta=np.arctan2(xx,yy);rr=(depth-geometry.depth_start_m)/geometry.depth_span_m*(raw.shape[1]-1);cc=(theta-geometry.angle_start_rad)/geometry.angle_span_rad*(raw.shape[2]-1);direct=map_coordinates(raw[0].astype(np.float64),[rr,cc],order=1,mode='nearest');direct[~converted.mask]=0;direct=np.rint(direct).clip(0,255).astype(np.uint8);pixel_error=int(abs(direct.astype(int)-converted.data.astype(int)).max());assert pixel_error<=1
        np.save(OUT/'sample_cartesian.private.npy',converted.data);(OUT/'sample_target.private.json').write_text(json.dumps({'point':point.tolist(),'bounds':bounds,'admitted':admitted,'metadata_sha256':sel['sha256']}))
        result={'sample_contract_ok':True,'manual_LVOT_measurement_and_single_Bmode_frame':True,'root_sector_gate_identical':True,'released_raw_array_sha256_verified':True,'native_raw_shape':list(raw.shape),'raw_variance':float(raw.var()),'normalized_official_beam_error':be,'normalized_official_cartesian_error':pe,'physical_roundtrip_error':roundtrip,'independent_render_max_uint8_error':pixel_error,'no_annotation_overlay_fed':True,'new_bytes_fetched':reader.bytes,'requests':reader.requests,'headers_visited':visited+1,'public_point_semantics':'operator sampling-location only, no expert anatomy or A4C absence labels','A48_training_activated':False,'reason':'one source-contract image is not the required80-image patient-fold admitted manifest','written_unix':time.time()};(OUT/'sample_contract.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=1))
    finally:
        (OUT/'transport_receipt.json').write_text(json.dumps({'new_bytes':reader.bytes,'requests':reader.requests,'wall_seconds':time.monotonic()-reader.started,'gpu_hours':0},indent=2)+'\n')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);st=time.time();cpu=time.process_time();ok=False
    try:run();ok=True
    finally:(OUT/'sample_receipt.json').write_text(json.dumps({'ok':ok,'wall_seconds':time.time()-st,'process_cpu_seconds':time.process_time()-cpu,'gpu_hours':0,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'finished_unix':time.time()},indent=2)+'\n')
