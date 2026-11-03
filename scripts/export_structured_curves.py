#!/usr/bin/env python3
"""Seal actual private training journals, then export aggregate fixed curves only."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.structured_common import ROOT,RUN,PUB,read,write,sha,verify_freeze
from echo_routing.temporal.structured_curves import binned_curve

def main(mode):
    models=read(PUB/'models_manifest.json');assert models['passed'] and models['formal_optimizer_updates']==48000;records=[];curves=[]
    if mode=='export':verify_freeze('package_freeze.json');pins=read(PUB/'training_journal_manifest.json');assert pins['models_manifest_sha256']==sha(PUB/'models_manifest.json')
    else:assert mode=='seal';verify_freeze('curve_seal_freeze.json')
    for model in models['models']:
        seed,arm=model['seed'],model['arm'];path=RUN/f'seed{seed}'/arm/'physical_updates.jsonl';digest=sha(path);rows=[json.loads(s) for s in path.read_text().splitlines()];assert len(rows)==6000 and all(r['identity']==model['head_identity'] and r['seed']==seed and r['arm']==arm for r in rows);assert rows[-1]['model_digest_after_update']==model['physical']['final_model_digest'];bins=binned_curve(rows);assert sum(b['valid_original_positions'] for b in bins)==model['physical']['valid_original_positions'];record={'seed':seed,'arm':arm,'journal_sha256':digest,'updates':len(rows),'valid_original_positions':model['physical']['valid_original_positions']};records.append(record)
        if mode=='export':assert record==next(r for r in pins['records'] if r['seed']==seed and r['arm']==arm);curves.extend({'seed':seed,'arm':arm,**b} for b in bins)
    if mode=='seal':write(PUB/'training_journal_manifest.json',{'records':records,'models_manifest_sha256':sha(PUB/'models_manifest.json'),'formal_optimizer_updates':48000,'no_private_indices_or_identifiers_exported':True,'epoch':time.time()})
    else:
        assert len(curves)==480;pd.DataFrame(curves).to_csv(PUB/'training_curves.csv',index=False);inner=read(PUB/'inner_results.json');assert inner['count']==24;keys=['seed','arm','step','native_macro_f1','segment_f1_50_multi','boundary_precision_multi','boundary_recall_multi','short_true_segments_10samples','short_true_unmatched_10samples','short_spurious_segments_10samples','short_true_segments_16samples','short_true_unmatched_16samples','short_spurious_segments_16samples'];pd.DataFrame([{k:r[k] for k in keys} for r in inner['rows']]).to_csv(PUB/'inner_curves.csv',index=False);write(PUB/'curve_receipt.json',{'training_journal_manifest_sha256':sha(PUB/'training_journal_manifest.json'),'training_curve_sha256':sha(PUB/'training_curves.csv'),'inner_curve_sha256':sha(PUB/'inner_curves.csv'),'training_bins':480,'bin_optimizer_updates':100,'inner_rows':24,'training_NLL_scope':'FITequalupdateaveragesoftwolengthnormalizeddomainNLLs;notheldoutloss','timing_scope':'optimizerstepwallbeforejournal;notCPUcoretimeorfullpipeline','private_fields_exported':False,'epoch':time.time()})
    print(json.dumps({'curve_stage':mode,'models':len(records),'training_bins':len(curves)}),flush=True)
if __name__=='__main__':main(sys.argv[1])
