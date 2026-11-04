#!/usr/bin/env python3
"""Cache-only24 native cubes+2 Base references+16 declared development records."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.event_risk_common import *
from scripts.event_risk_execution import bank,old_records,score_values,decoded
from echo_routing.temporal.event_risk_metrics import plan,SEEDS,HALVES,POLICIES,system
from echo_routing.temporal.event_risk_forensics import native_cube,native_reference,regression_records,regression_rle


def main():
    p=execution_parser(__doc__);a=p.parse_args()
    if not authorize(a):return 0
    freeze=verify_freeze('forensics_freeze.json');summary=read(PUB/'summary.json');fit=read(PUB/'training_inputs.json')['F_fit'];cal=read(PUB/'calibration.json');cubes=[];references=[];regression=[];private=[]
    for half in HALVES:
        qs,pa,pj,b=bank(half);idx=[i for i,q in enumerate(qs) if q['kind']=='native'];nq=[qs[i] for i in idx];nb=[b[i] for i in idx];ref=native_reference(nq,nb,fit)
        base=summary['systems']['main'][half]['A08_s0_original_B3']
        if ref['true_prediction_frame_confusion5']!=base['decoded_position_confusion']['native']:raise ValueError('Base native cube mismatch')
        references.append({'table':'main','half':half,**ref})
        for seed in SEEDS:
            raw,_=score_values(seed,half,readonly=True);taus={r['policy']:r['tau'] for r in cal['thresholds']+cal['legacy_thresholds'] if r['seed']==seed};v=decoded(raw,b,taus)
            for policy in POLICIES:
                row=native_cube(nq,nb,[v[i] for i in idx],policy,fit);sealed=summary['systems']['main'][half][system(seed,policy)];gross=sealed['gross_reference_position_effects']['native']
                if row['true_prediction_frame_confusion5']!=sealed['decoded_position_confusion']['native'] or (row['Base_correct_to_wrong'],row['Base_wrong_to_correct'],row['frame_error_delta'])!=(gross['H'],gross['R'],gross['dE']):raise ValueError('native cube/gross/CM mismatch sealed row')
                cubes.append({'table':'main','half':half,'seed':seed,'system':system(seed,policy),**row})
    cfg=read(CFG);evidence_path=next(p for p in cfg['prior_source_pins'] if p.endswith('/case_evidence.anonymous.json'))
    from echo_routing.temporal.event_risk_context import verify as verify_context
    binding=read(PUB/'prior_context_binding.json');paths=verify_context(ROOT,cfg,binding,{commit:read(verify_pin(p)) for commit,p in binding['actual_Git_proofs'].items()})
    evidence=read(paths[evidence_path]);old=old_records()['outer_metrics.json']
    for case in [r for r in evidence['cases'] if r['seed']==SEEDS[0]]:
        axis=case['case'].replace(' ','_');role=case['source_role'];oldrow=next(r for r in old['rows'] if r.get('seed')==SEEDS[0] and r.get('arm')=='S-L' and r.get('policy')=='C' and r.get('source_role')==role);folder=ROOT/oldrow['policy_cache']['path'];journal=folder/'committed.jsonl'
        if sha(journal)!=oldrow['policy_cache']['journal_sha256']:raise ValueError('declared regression old journal changed')
        hits=[r['index'] for r in [json.loads(x) for x in journal.read_text().splitlines()] if r['sha256']==case['evidence']['primary_policy_payload_sha256']]
        if len(hits)!=1 or read(PUB/'input_manifest.json')['roles'][role]['sha256']!=case['evidence']['role_metadata_sha256']:raise ValueError('exact anonymous regression axis not uniquely rebound to original payload/input')
        index=hits[0];qs,pa,pj,b=bank(role)
        for seed in SEEDS:
            raw,_=score_values(seed,role,readonly=True);taus={r['policy']:r['tau'] for r in cal['thresholds']+cal['legacy_thresholds'] if r['seed']==seed};v=decoded([raw[index]],[b[index]],taus)[0];records=regression_records(axis,qs[index],b[index],pa[index],pj[index],v,seed,fit);regression.extend(records);private.append({'axis':axis,'seed':seed,'original_index':index,'original_id':qs[index]['id'],'RLE':regression_rle(qs[index],b[index],v)})
    if len(cubes)!=24 or len(references)!=2 or len(regression)!=16 or {(r['axis'],r['seed'],r['policy']) for r in regression}!={(r['axis'],r['seed'],r['policy']) for r in plan()['regression_records']}:raise ValueError('complete24+2/16 declared diagnostics required')
    p=RUN/'development_regression_RLE.private.npy';save_array(p,np.asarray(private,dtype=object));write(PUB/'native_attribution.json',{'cells':cubes,'cell_count':24,'base_references':references,'base_reference_count':2,'summary_sha256':sha(PUB/'summary.json'),'reuses_main_native_predictions_across_tables_not_independent_samples':True,'new_forward_or_optimizer_updates':0,'freeze_git':freeze['git_commit']});write(PUB/'regression_summary.json',{'records':regression,'record_count':16,'development_regression_not_independent_confirmation':True,'original_case_evidence_sha256':cfg['prior_source_pins'][evidence_path],'Case_A_original_single_action_H40_R0_cell_elsewhere_R37':True,'Case_B_one_axis_two_seeds_original_H9_R2_lost_A4C11_A5C9':True,'private_RLE':pin(p),'new_forward_or_optimizer_updates':0,'freeze_git':freeze['git_commit']});print({'native':24,'Base_refs':2,'development_regression':16,'newNN':0},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
