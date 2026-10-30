#!/usr/bin/env python3
"""Prospective source-timing qualification, before any LVH model forward."""
from pathlib import Path
import copy,json,sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.external_original_tests as e


def amended_rows(old,qualified):
    at={r['id']:r for r in qualified}
    if len(at)!=len(qualified) or set(at)!={q['id'] for q in old}:raise ValueError('complete unique membership required')
    result=[]
    for q in old:
        r=at[q['id']]
        if r['decoded_frames']!=q['n_frames'] or r['csv_field_fps']!=q['fps'] or r['container_fps']<=0 or r['pts_delta_max_error']>1e-6:
            raise ValueError('source timing qualification failed')
        v=copy.deepcopy(q);v['reported_csv_fps']=q['fps'];v['fps']=r['container_fps'];v['fps_authority']='decoded_PTS_and_container_CFR';result.append(v)
    return result


def main():
    run,private=e.RUN,e.PRIVATE
    if (run/'protocol_bindings.json').exists():raise RuntimeError('timing amendment already applied')
    if list((private/'predictions/lvh').glob('*.npz')) or list((private/'predictions/camus').glob('*.npz')):
        raise RuntimeError('cannot amend after remaining-target inference')
    pf,mf=run/'protocol.json',private/'manifest.json'
    p=json.loads(pf.read_text());m=json.loads(mf.read_text());original_sha=e.sha(pf)
    if e.sha(mf)!=p['manifest_sha256']:raise RuntimeError('original manifest differs')
    qf=private/'lvh_timing_qualification.json';qualified=json.loads(qf.read_text())
    updated=amended_rows(m['datasets']['lvh'],qualified['rows'])
    (run/'protocol_v1.json').write_bytes(pf.read_bytes());(private/'manifest_v1.json').write_bytes(mf.read_bytes())
    if (e.RESULT/'checks.json').exists():(e.RESULT/'checks_v1.json').write_bytes((e.RESULT/'checks.json').read_bytes())
    m['datasets']['lvh']=updated;m['timing_amendment']={'created_utc':e.now(),'qualification_sha256':e.sha(qf),
        'rule':'Use decoded PTS-confirmed container CFR; retain original CSV field without guessing its semantics.'}
    e.write(mf,m);p['created_utc']=e.now();p['manifest_sha256']=e.sha(mf)
    p['code_sha256'][str(Path(__file__).relative_to(ROOT))]=e.sha(__file__)
    p['timing_amendment']={'predecessor_protocol_sha256':original_sha,'qualification_sha256':e.sha(qf),
        'lvh_records':len(updated),'all_before_first_LVH_forward':True,'max_pts_delta_error_s':qualified['max_error'],
        'unchanged':['weights','B3 numbers','RGB transform','window/marker rules','source split/membership','all previous predictions']}
    e.write(pf,p);e.write(e.RESULT/'preparation_v2.json',p)
    e.write(run/'protocol_bindings.json',{'ev9v':'protocol_v1.json','dynamic':'protocol_v1.json','lvh':'protocol.json','camus':'protocol.json'})
    summary={'created_utc':e.now(),'records':len(updated),'frames_match':len(updated),'all_CFR':True,'max_pts_delta_error_s':qualified['max_error'],
        'container_fps_range':[min(q['fps'] for q in updated),max(q['fps'] for q in updated)],
        'predecessor_protocol_sha256':original_sha,'active_protocol_sha256':e.sha(pf),'qualification_sha256':e.sha(qf),
        'binding_file_sha256':e.sha(run/'protocol_bindings.json'),'qualification_wall_seconds':qualified['seconds'],
        'previous_job_failure':{'reason':'LVH CSV rate mismatches decoded container rate; zero LVH forwards',
            'wall_seconds':1022.36,'process_user_seconds':960.83,'process_system_seconds':28.19,'gpu_forward_seconds':766.6388087272644},
        'scope':'Source-timing correction before first LVH prediction. No model/output/threshold selection; EV9V and Dynamic retain v1 caches.'}
    e.write(e.RESULT/'timing_amendment.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
