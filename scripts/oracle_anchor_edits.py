#!/usr/bin/env python3
"""Frozen post-readout label-assisted single-action domain; never a policy."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.anchor_common import RUN,PUB,CFG,read,write,sha,verify_freeze
from scripts.readout_anchor_edits import contexts,role_for
from scripts.train_anchor_edits import role_values
from echo_routing.temporal.anchor_oracle import action_options,fractional_oracle,missed_coverage
from echo_routing.temporal.anchor_edits import counts,components,apply_action,majority
from echo_routing.temporal.anchor_readout_plan import TABLES,HALVES


def main():
    freeze=verify_freeze('oracle_freeze.json');summary=read(PUB/'summary.json')
    if summary['metric_rows']!=70 or summary['paired_comparison_count']!=96:raise ValueError('all frozen system readouts required before oracle')
    cfg=read(CFG);ctx=contexts(cfg);rows=[]
    for table in TABLES:
        for half in HALVES:
            q,a,j,b,_=role_values(role_for(table,half));values={v['id']:(v,p,base) for v,p,base in zip(q,a,b)}
            if table!='main':
                mq,ma,mj,mb,_=role_values(half)
                values.update({v['id']:(v,p,base) for v,p,base in zip(mq,ma,mb) if v['kind']!='multi'})
            seqs=ctx[table][half]['sequences'];multi=[v for v in seqs if v['kind']=='multi']
            for guarded in [False,True]:
                options=[];coverage={'baseline_missed_true_units':0,'candidate_union_recovered_true_units':0}
                for seq in multi:
                    q,p,base=values[seq['id']];raw=p.argmax(1)
                    options.append(action_options(base,raw,q['labels'],guarded))
                    cover=missed_coverage(base,raw,q['labels'],guarded)
                    for key in coverage:coverage[key]+=cover[key]
                oracle=fractional_oracle(options)
                row={'table':table,'half':half,'domain':'G' if guarded else 'U','multi_streams':len(multi),
                     **{k:v for k,v in oracle.items() if k!='choices'},**coverage,
                     'union_is_jointly_reachable_claim':False,'per_sequence_action_choices_exported':False}
                rows.append(row)
    diagnostics=[]
    for table in TABLES:
        for half in HALVES:
            contexts_q=ctx[table][half]['sequences'];values={}
            for role in [half] if table=='main' else [half,role_for(table,half)]:
                qs,a,j,b,_=role_values(role)
                values.update({q['id']:(q,p,base) for q,p,base in zip(qs,a,b)})
            for kind in ['native','multi','pair']:
                group=[q for q in contexts_q if ('native' if q['kind']=='native' else 'multi' if q['kind']=='multi' else 'pair')==kind]
                pooled=np.asarray([counts(values[q['id']][2],q['labels']) for q in group]).sum(0)
                f1=2*int(pooled[0])/int(pooled[1]+pooled[2]);totals={k:0 for k in ['sequences','candidates','guard_vetoed_actions','positive_countutility_actions',
                    'guard_vetoed_positive_countutility_actions','guard_vetoed_increased_matched_true_segments_actions',
                    'native_frame_harm_actions','native_frame_harm_guard_admissible_actions','native_cut_increase_actions',
                    'native_Base_wrong_class_correcting_actions','native_Base_wrong_class_correcting_guard_vetoed_actions']}
                for q in group:
                    _,p,base=values[q['id']];raw=p.argmax(1);before=counts(base,q['labels']);totals['sequences']+=1
                    for span in components(base,raw):
                        edited=apply_action(base,raw,span);after=counts(edited,q['labels']);veto=majority(edited)!=majority(base)
                        useful=2*(after[0]-before[0])-f1*(after[1]-before[1])>0
                        totals['candidates']+=1;totals['guard_vetoed_actions']+=veto
                        totals['positive_countutility_actions']+=useful;totals['guard_vetoed_positive_countutility_actions']+=veto and useful
                        totals['guard_vetoed_increased_matched_true_segments_actions']+=veto and after[0]>before[0]
                        if kind=='native':
                            frame_harm=np.count_nonzero(edited!=q['labels'])>np.count_nonzero(base!=q['labels'])
                            totals['native_frame_harm_actions']+=frame_harm
                            totals['native_frame_harm_guard_admissible_actions']+=frame_harm and not veto
                            totals['native_cut_increase_actions']+=np.count_nonzero(edited[1:]!=edited[:-1])>np.count_nonzero(base[1:]!=base[:-1])
                            corrected=majority(base)!=int(q['labels'][0]) and majority(edited)==int(q['labels'][0])
                            totals['native_Base_wrong_class_correcting_actions']+=corrected
                            totals['native_Base_wrong_class_correcting_guard_vetoed_actions']+=corrected and veto
                diagnostics.append({'table':table,'half':half,'kind':kind,'operating_pooled_count_F1':f1,**{k:int(v) for k,v in totals.items()},
                    'actions_are_alternatives_not_jointly_executable':True,'positive_countutility_is_not_native_macroF1_surrogate':True})
    if len(rows)!=12 or len(diagnostics)!=18:raise ValueError('all12fixed oracle domains/18 actiondiagnosticgroups required')
    write(PUB/'action_space_diagnostic.json',{'oracle_domains':rows,'domain_count':12,'action_diagnostics':diagnostics,
          'label_assisted_not_a_system_or_gate_or_rescue':True,'no_oracle_choices_used_for_learning_or_selection':True,
          'upper_bound_scope':'exact pooledmultiF1 only within stated maxONE raw-supported action per fullstream;not arbitrary segmenter',
          'coverage_union_not_jointly_reachable':True,'summary_sha256':sha(PUB/'summary.json'),
          'oracle_freeze_git':freeze['git_commit'],'epoch':time.time()})

if __name__=='__main__':main()
