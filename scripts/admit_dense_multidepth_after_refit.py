#!/usr/bin/env python3
"""Cost-only MD admission after adaptivepair, retaining separately metered CPU work."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import admit_dense_multidepth as old,train_dense_multidepth as run,train_dense_sequence as base
from echo_routing.temporal.multidepth_budget import select_tier
def main():
    assert run.read(base.PUB/'full_refit/summary.json')['raw_metrics_independently_reproduced'];old.main();value=run.read(run.PUB/'admission.json');past=[json.loads(x) for x in (base.RUN/'processes.jsonl').read_text().splitlines()];aux=[json.loads(x) for x in (base.RUN/'auxiliary_CPU_receipts.jsonl').read_text().splitlines()];assert not {r['model_pid'] for r in past}&{r['model_pid'] for r in aux};extra=sum(r['process_CPU_seconds'] for r in aux)/3600
    for row in value['candidates']:row['already_measured_CPUcore_h']+=extra;row['CPU_total']+=extra;row['auxiliary_CPUcore_h_retained']=extra
    chosen=select_tier(value['candidates']);value.update({'selected_seed_count':chosen,'training_seeds':[] if not chosen else [2026110257,2026110258][:chosen],'all_shared_caps_fit':bool(chosen),'all_adaptivepair_training_readout_obligations_complete':True,'auxiliary_CPU_receipts_count':len(aux),'auxiliary_CPUcore_h_retained':extra});run.write(run.PUB/'admission.json',value);print(json.dumps({'MD_selected_seed_count_final':chosen,'auxiliary_CPUcore_h_retained':extra}),flush=True)
if __name__=='__main__':main()
