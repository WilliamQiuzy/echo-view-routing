#!/usr/bin/env python3
"""A56 pre-readout audit of both actual input journals; zero model forwards."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.source_retention_audit import matched_journals
from echo_routing.temporal.direct_view_training import write_json
OUT=ROOT/'runs/a56_source_retention_v1';PUB=ROOT/'docs/results/2026-11-02_source_retention'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
 manifest=read(PUB/'models_manifest.json');checks=read(PUB/'checks.json');inputs=read(PUB/'input_manifest.json');plan=np.load(OUT/'plan.private.npz');assert sha(OUT/'plan.private.npz')==inputs['plan_sha256'];journals={}
 for arm in ('Z','R'):
  path=OUT/'models'/arm/'updates.private.jsonl';assert sha(path)==manifest['models'][arm]['journal_sha256'];assert manifest['models'][arm]['initial_full_state_sha256']==checks['initial_full_state_sha256'];journals[arm]=[json.loads(row) for row in path.read_text().splitlines()]
  for i,row in enumerate(journals[arm]):assert row['source_plan_sha256']==array_sha(plan['source'][i]) and row['replay_plan_sha256']==array_sha(plan['replay'][i])
 result=matched_journals(journals['Z'],journals['R'],1200);result.update({'passed':True,'models_manifest_sha256':sha(PUB/'models_manifest.json'),'input_manifest_sha256':sha(PUB/'input_manifest.json'),'both_initial_states_equal_checked_original_A08':True,'all_plan_rows_match_frozen_npz':True,'model_forwards':0,'source_diagnostic_forwards':0,'epoch':time.time()});write_json(PUB/'matched_inputs.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
