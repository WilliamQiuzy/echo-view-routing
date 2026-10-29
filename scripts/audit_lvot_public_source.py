"""Independent A49 source failure audit: explicit metadata/array math, no admission helper."""
import pathlib,json,hashlib,itertools,collections,time,math
import numpy as np,zarr,numcodecs
P=pathlib.Path('runs/a49_lvot_evidence_qualification_v1');start=time.time(); plan=json.loads((P/'preparation_selection.private.json').read_text()); named=[]; proofcount=0; membercount=0; verified_archive_bytes=0; reasons=collections.Counter(); dtypecount=collections.Counter(); pointdecode=0; fieldmatches=0; finite=0; near=[]; inside=0; centerfirst=[]; imagesopened=0
for row in plan['rows']:
 receipt=json.loads((P/'archive_receipts'/(row['exam']+'.private.json')).read_text()); assert receipt['exam']==row['exam'] and receipt['fold']==row['fold']
 if not receipt['publisher_verified']:
  for a in receipt['attempts']:
   if not a['ok']:reasons[(a['error_type'],a['error'])]+=1
  continue
 passed=[a['proof'] for a in receipt['attempts'] if a['ok']];assert len(passed)==1;proof=passed[0]; assert proof['publisher_verified'] and proof['sha256']==row['publisher_object']['lfs']['oid'] and proof['bytes']==row['publisher_object']['size'];proofcount+=1;verified_archive_bytes+=proof['bytes'];root=P/'archive_quarantine'/row['exam']
 for member in proof['members']:
  file=root/member['name'];assert file.is_file() and file.stat().st_size==member['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==member['sha256'];membercount+=1
 for rid in sorted({x['record'] for x in row['candidates']}):
  rel=next(x['recording_zarr_path'] for x in row['candidates'] if x['record']==rid);r=root/rel;attrs=json.loads((r/'.zattrs').read_text());m=attrs['recording_manifest']; anns=[x for x in m.get('annotations',[]) if x.get('kind')=='measurement' and x.get('links',{}).get('measure_id')=='Cardiac/2D/Dimension/LVOT Diam']
  if not anns:continue
  assert len(anns)==1;ann=anns[0]; targetpath=ann['value']['zarr_path'];mp=json.loads((r/targetpath/'.zarray').read_text());arr=np.asarray(zarr.open_array(str(r/targetpath),mode='r')[:]);assert arr.shape==(2,2);assert not mp.get('filters');dtypecount[str(arr.dtype)]+=1
  own=np.empty(mp['shape'],dtype=mp['dtype'])
  for idx in itertools.product(*[range(math.ceil(s/c)) for s,c in zip(mp['shape'],mp['chunks'])]):
   b=(r/targetpath/'.'.join(map(str,idx))).read_bytes();b=numcodecs.get_codec(mp['compressor']).decode(b) if mp['compressor'] else b;block=np.frombuffer(b,dtype=mp['dtype']).reshape(mp['chunks'],order=mp['order']);sl=tuple(slice(i*c,min((i+1)*c,s)) for i,c,s in zip(idx,mp['chunks'],mp['shape']));own[sl]=block[tuple(slice(0,s.stop-s.start) for s in sl)]
  assert np.array_equal(own,arr);pointdecode+=1;finite+=np.isfinite(arr).all()
  arrays={x['name']:x for x in attrs['arrays']};fieldmatches+=hashlib.sha256(arr.tobytes()).hexdigest()==arrays[targetpath].get('data_sha256')
  sectors=m.get('sectors',[]);assert len(sectors)==1 and sectors[0]['semantic_id']=='bmode';s=sectors[0];g=s['geometry'];d0,d1=g['depth_start_m'],g['depth_end_m'];a0=g['tilt_rad']-g['width_rad']/2;a1=g['tilt_rad']+g['width_rad']/2
  radii=np.sqrt(np.sum(arr.astype(np.float64)**2,axis=1));theta=np.arctan2(arr[:,0],arr[:,1]);rc=np.stack([(radii-d0)/(d1-d0),(theta-a0)/(a1-a0)],1);strict=bool(((rc>0)&(rc<1)).all());inside+=strict
  overlays=[np.array(x['points'],dtype=float) for x in s.get('overlays',{}).get('physical_points',[]) if np.array(x.get('points',[])).shape==(2,2)]
  errs=[min(np.max(abs(arr-v)),np.max(abs(arr[::-1]-v))) for v in overlays];err=float(min(errs)) if errs else None;near.append(err)
  # Metadata's own sector overlay endpoints: independently count geometry sanity.
  overlay_inside=[]
  for v in overlays:
   rho=np.linalg.norm(v,axis=1);ang=np.arctan2(v[:,0],v[:,1]);uv=np.stack([(rho-d0)/(d1-d0),(ang-a0)/(a1-a0)],1);overlay_inside.append(bool(((uv>0)&(uv<1)).all()))
  named.append({'exam':row['exam'],'record':rid,'fold':row['fold'],'array_data_sha256':hashlib.sha256(arr.tobytes()).hexdigest(),'nearest_sector_overlay_max_abs_m':err,'array_inside_sector':strict,'sector_overlay_count':len(overlays),'sector_overlays_inside':overlay_inside,'raw_field_matches':hashlib.sha256(arr.tobytes()).hexdigest()==arrays[targetpath].get('data_sha256'),'original_target':ann['target']})
finite_err=[x for x in near if x is not None];source_matches=sum(x is not None and x<=5e-7 for x in near);prior=json.loads((P/'admission_summary.json').read_text());assert len(named)==109 and pointdecode==109 and source_matches==0;assert prior['admitted_images']==0 and not prior['passed'];assert sum(not x['array_inside_sector'] for x in named)==25
out={'ok':True,'publisher_verified_archives':proofcount,'publisher_verified_archive_bytes':verified_archive_bytes,'retained_member_hashes_recomputed':membercount,'failed_archive_attempts':{str(k):v for k,v in reasons.items()},'named_LVOT_measurements':len(named),'canonical_and_independent_target_arrays_equal':pointdecode,'target_array_dtype_counts':dict(dtypecount),'finite_target_arrays':int(finite),'target_arrays_inside_declared_sector':inside,'target_arrays_outside_declared_sector':len(named)-inside,'unique_sector_overlay_matches_at_5e_7_m':source_matches,'nearest_overlay_error_m_min_median_max':[float(np.min(finite_err)),float(np.median(finite_err)),float(np.max(finite_err))],'arrays_raw_sha_matches_undocumented_data_sha256':int(fieldmatches),'nonmatching_arrays_from_distinct_records':len({r['array_data_sha256'] for r in named}),'weak_named_target_array_binding_passes':0,'official_geometry_parity_on_admitted_main_images':None,'main_bmode_image_decodes':imagesopened,'training_admitted':False,'model_forwards':0,'gpu_hours':0,'elapsed_seconds':time.time()-start,'source_manifest_sha256':hashlib.sha256((P/'admitted_manifest.private.json').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((P/'preparation_selection.private.json').read_bytes()).hexdigest(),'scope':'Independent persisted member-hash checks, canonical versus separate codec point decoding and explicit geometry/binding recount. Full archives were cryptographically verified once while streaming; discarded bytes are not independently downloaded a second time. Undocumented array digest semantics remain unresolved. No transform/axis/offset guessed to rescue targets.'}
(P/'independent_binding_audit.private.json').write_text(json.dumps(named,indent=2));(P/'independent_source_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
