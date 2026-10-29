"""Decode the fixed fully verified source subset; no learned model."""
import pathlib,json,sys,hashlib,itertools,collections,time
sys.path.insert(0,'.');sys.path.insert(0,'runs/a48_anatomical_qualification_v1/upstream/src')
import numpy as np,zarr,numcodecs
from scripts.lvot_caliper_contract import bind_main_lvot_caliper
from echoxflow.scan.geometry import sector_geometry_from_mapping,CartesianGrid
from echoxflow.scan.coordinates import BeamspacePixelGrid,CartesianPixelGrid
P=pathlib.Path('runs/a49_lvot_evidence_qualification_v1'); t=time.time()
def decode(path):
 meta=json.loads((path/'.zarray').read_text()); assert meta['zarr_format']==2 and not meta.get('filters')
 standard=np.asarray(zarr.open_array(str(path),mode='r')[:]); own=np.zeros(meta['shape'],dtype=meta['dtype'])
 for idx in itertools.product(*[range((s+c-1)//c) for s,c in zip(meta['shape'],meta['chunks'])]):
  body=(path/'.'.join(map(str,idx))).read_bytes(); raw=numcodecs.get_codec(meta['compressor']).decode(body) if meta.get('compressor') else body
  block=np.frombuffer(raw,dtype=meta['dtype']).reshape(meta['chunks'],order=meta['order']); sl=tuple(slice(i*c,min((i+1)*c,s)) for i,c,s in zip(idx,meta['chunks'],meta['shape'])); own[sl]=block[tuple(slice(0,x.stop-x.start) for x in sl)]
 assert np.array_equal(standard,own); return standard
plan=json.loads((P/'preparation_selection.private.json').read_text()); admitted=[]; excluded=[]; labels=collections.Counter(); arrayhash={'image_matches':0,'image_checked':0,'points_matches':0,'points_checked':0}; worst=0.; images={}; output=P/'admitted';output.mkdir(exist_ok=True)
for row in plan['rows']:
 receipt=P/'archive_receipts'/(row['exam']+'.private.json')
 if not receipt.exists() or not json.loads(receipt.read_text())['publisher_verified']:
  excluded.append({'exam':row['exam'],'reason':'archive_not_publisher_verified'});continue
 for rid in sorted({v['record'] for v in row['candidates']}):
  root=P/'archive_quarantine'/row['exam']; rel=next(v['recording_zarr_path'] for v in row['candidates'] if v['record']==rid); path=root/rel
  try:
   attrs=json.loads((path/'.zattrs').read_text()); ann=attrs['recording_manifest'].get('annotations',[])
   for a in ann:
    if a.get('kind')=='measurement':labels[a.get('links',{}).get('measure_id','unknown')]+=1
   if not any(a.get('kind')=='measurement' and a.get('links',{}).get('measurement_kind')=='dimension_lvot_diam' for a in ann):
    excluded.append({'exam':row['exam'],'record':rid,'reason':'not_named_LVOT_diameter'});continue
   decoded={x['point_array']:decode(path/x['point_array']) for x in row['candidates'] if x['record']==rid}
   binding=bind_main_lvot_caliper(attrs,decoded); pts=decoded[binding['target_path']]; image=decode(path/'data/2d_brightness_mode'); assert image.ndim==3 and image.shape[0]==1 and image.dtype==np.uint8
   sector=attrs['recording_manifest']['sectors'][0]; geom=sector['geometry']; assert tuple(geom['grid_size'])==image.shape[1:]
   g=sector_geometry_from_mapping(geom,grid_shape=image.shape[1:]); off=BeamspacePixelGrid(g,image.shape[1:]).physical_to_row_col(pts)/(np.array(image.shape[1:])-1); err=float(np.max(np.abs(off-binding['geometry']['row_col'])));assert err<=1e-5
   cart=CartesianGrid.from_sector_height(g,image.shape[1]); xy=CartesianPixelGrid(cart).physical_to_pixel_xy(pts); assert ((xy>0)&(xy<(np.array(cart.shape[::-1])-1))).all();worst=max(worst,err)
   header=str((str(image.dtype),image.shape)).encode(); h=hashlib.sha256(header+b'\0'+image.tobytes()).hexdigest()
   if h in images:
    prior=images[h]
    if prior['fold']!=row['fold']: raise RuntimeError('crossfold_duplicate_image')
    excluded.append({'exam':row['exam'],'record':rid,'reason':'same_fold_duplicate_image','same_image_as':prior['record']});continue
   f=output/(rid+'.npy');np.save(f,image,allow_pickle=False)
   d={'exam':row['exam'],'record':rid,'fold':row['fold'],'raw_root':str(path),'image_file':str(f),'image_sha256':h,'npy_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'point_array_sha256':hashlib.sha256(pts.tobytes()).hexdigest(),'shape':list(image.shape),'binding':binding,'geometry':geom,'official_max_error':err,'source_archive_sha256':row['publisher_object']['lfs']['oid']}; images[h]=d;admitted.append(d)
   metadata={v['name']:v for v in attrs['arrays']}
   for name,x,key in [('data/2d_brightness_mode',image,'image'),(binding['target_path'],pts,'points')]:
    arrayhash[key+'_checked']+=1;arrayhash[key+'_matches']+=hashlib.sha256(x.tobytes()).hexdigest()==metadata[name].get('data_sha256')
  except RuntimeError as e:
   if str(e)=='crossfold_duplicate_image':raise
   excluded.append({'exam':row['exam'],'record':rid,'reason':type(e).__name__+': '+str(e)[:150]})
  except Exception as e:excluded.append({'exam':row['exam'],'record':rid,'reason':type(e).__name__+': '+str(e)[:150]})
fit=[r for r in admitted if r['fold']!=0]; held=[r for r in admitted if r['fold']==0]; points=np.array([r['binding']['geometry']['midpoint_row_col'] for r in fit]); std=points.std(0).tolist() if len(points) else [0,0]; bins=len({tuple(np.floor(p*14).astype(int)) for p in points}); gates={'images_at_least_80':len(admitted)>=80,'exams_at_least_40':len({r['exam'] for r in admitted})>=40,'fit_images_at_least_64':len(fit)>=64,'heldout_exams_at_least_16':len({r['exam'] for r in held})>=16,'fit_midpoint_std_both_at_least_0025':min(std)>=.025,'fit_midpoint_bins_at_least_12':bins>=12,'crossfold_duplicates_absent':True}
summary={'admitted_images':len(admitted),'admitted_exams':len({r['exam'] for r in admitted}),'fit_images':len(fit),'fit_exams':len({r['exam'] for r in fit}),'heldout_images':len(held),'heldout_exams':len({r['exam'] for r in held}),'fit_midpoint_std_row_col':std,'fit_occupied_14x14_bins':bins,'per_fold_images':{str(f):sum(r['fold']==f for r in admitted) for f in range(5)},'exclusion_counts':dict(collections.Counter(r['reason'] for r in excluded)),'named_measurement_counts':dict(labels),'official_normalized_max_error':worst,'standard_zarr_independent_decode_all_equal':True,'raw_digest_comparison_not_defined_publisher_contract':arrayhash,'gates':gates,'passed':all(gates.values()),'elapsed_seconds':time.time()-t,'gpu_hours':0,'selection_sha256':hashlib.sha256((P/'preparation_selection.private.json').read_bytes()).hexdigest()}
(P/'admitted_manifest.private.json').write_text(json.dumps({'admitted':admitted,'excluded':excluded},indent=2));summary['manifest_sha256']=hashlib.sha256((P/'admitted_manifest.private.json').read_bytes()).hexdigest();(P/'admission_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
