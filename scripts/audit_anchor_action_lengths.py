#!/usr/bin/env python3
"""Card44 reporting supplement from immutable score/label journals; zero forwards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scripts.anchor_common import RUN,PUB,read,write,sha,verify_pin,verify_freeze,sequences
from echo_routing.temporal.anchor_action_lengths import summarize


def role_for(table,half):return half if table=='main' else ('old_stress_' if table=='old_stress' else 'confirmation_')+half


def main():
    freeze=verify_freeze('action_lengths_freeze.json');verify_freeze('readout_freeze.json')
    write(RUN/'runtimecoordination.json',{'phase':'cached_policy_length_reporting_supplement','model_pid':os.getpid(),
         'new_forwards':0,'epoch':time.time(),'next_action':'complete card44 descriptors then reconcile additional CPU and close'})
    metric=read(PUB/'outer_metrics.json');summary=read(PUB/'summary.json')
    if metric['metric_rows']!=70 or summary['paired_comparison_count']!=96:raise ValueError('complete frozen matrix required')
    preserved=[PUB/p for p in ['outer_metrics.json','summary.json','models_manifest.json','action_space_diagnostic.json','resources.json','terminal_exit_receipt.json','local_cpu_receipts.json']]
    before={str(p.relative_to(ROOT)):sha(p) for p in preserved}
    source_rows=[r for r in metric['rows'] if r['role']=='policy'];loaded={};units={};metadata={};output=[]
    for row in source_rows:
        role=role_for(row['table'],row['half']);key=(row['weight_sha256'],role)
        if key not in loaded:
            qs=sequences(role);metadata[role]=qs;info=row['policy_cache'];folder=ROOT/info['path'];journal=folder/'committed.jsonl'
            if sha(journal)!=info['journal_sha256']:raise ValueError('sealed policy journal changed')
            records=[json.loads(line) for line in journal.read_text().splitlines()]
            if len(records)!=len(qs) or (folder/'pending.json').exists():raise ValueError('complete unambiguous policy prefix required')
            collected={'U':[],'G':[]}
            for i,(q,record) in enumerate(zip(qs,records)):
                n=len(q['labels']);path=folder/f'sequence{i:06d}.private.npz'
                if record['index']!=i or record['identity']!=info['identity'] or record['positions']!=n or sha(path)!=record['sha256']:
                    raise ValueError('journal full original axis/hash/identity changed')
                with np.load(path) as payload:
                    spans=[tuple(s) for s in payload['spans'].tolist()]
                    for policy in ['U','G']:
                        if payload['labels_'+policy].shape!=(n,):raise ValueError('full label axis changed')
                        raw=payload['accepted_'+policy].tolist();action=None if raw==[-1,-1] else tuple(raw)
                        collected[policy].append({'T':n,'spans':spans,'accepted':action})
            loaded[key]=collected
        units[(row['seed'],row['arm'],row['policy'],role)]=loaded[key][row['policy']]
    if len(loaded)!=24 or len(source_rows)!=48:raise ValueError('all24 physical paired caches/48 policy rows required')
    for row in source_rows:
        role=role_for(row['table'],row['half']);actual=units[(row['seed'],row['arm'],row['policy'],role)]
        source=summarize(actual)
        if source['accepted_count']!=row['accepted_actions'] or source['candidate_count']!=row['candidate_count']:
            raise ValueError('descriptor counts do not match sealed metric row')
        if row['table']=='main':context=actual
        else:
            main_units=units[(row['seed'],row['arm'],row['policy'],row['half'])]
            context=[u for q,u in zip(metadata[row['half']],main_units) if q['kind']!='multi']+actual
        output.append({'table':row['table'],'half':row['half'],'system':row['system'],'seed':row['seed'],'arm':row['arm'],
             'policy':row['policy'],'source_role':role,'weight_sha256':row['weight_sha256'],
             'source_role_statistics':source,'full_evaluation_context_statistics':summarize(context),
             'sealed_accepted_count_matched':True,'sealed_candidate_count_matched':True,
             'source_scope':'main complete task' if row['table']=='main' else 'new multi streams only;source-role accepted count is sealed row count',
             'full_context_scope':'full native+pair+multi original axes;stress native/pairs reuse main units without any rescoring',
             'native_pair_reused_from_main_no_new_forward_or_rescore':row['table']!='main'})
    if any(sha(ROOT/p)!=v for p,v in before.items()):raise ValueError('original model/metric/resource bytes changed during reporting')
    value={'passed':True,'rows':output,'policy_cells':48,'physical_paired_caches':24,
           'all_sealed_source_role_candidate_and_accepted_counts_reproduced':True,
           'original70rows_96comparisons_12oracle_and_weights_unchanged':True,'unchanged_input_sha256':before,
           'new_model_or_source_forwards':0,'scores_thresholds_and_labels_modified':False,
           'GT_content_used_for_descriptor':False,'T_from_complete_original_label_array_shape_only':True,
           'reporting_gap':'card44 accepted-length/T and whole-axis descriptors added after initial worker closure;initial receipts preserved',
           'freeze_git':freeze['git_commit'],'epoch':time.time()}
    write(PUB/'action_length_audit.json',value);pd.json_normalize(output).to_csv(PUB/'action_length_audit.csv',index=False)
    write(RUN/'runtimecoordination.json',{'phase':'action_length_reporting_complete_pending_resource_reconciliation','model_pid':None,
          'new_forwards':0,'epoch':time.time(),'next_action':'reconcile additional CPU without changing initial receipts'})
    print({'reporting_supplement_complete':True,'policy_cells':48,'paired_caches':24,'new_forwards':0},flush=True)

if __name__=='__main__':main()
