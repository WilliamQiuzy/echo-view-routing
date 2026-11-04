#!/usr/bin/env python3
"""Post124/144 frozen label-assisted32 one-action ceilings;never model selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.proposal_common import *
from scripts.readout_jepa_proposals import contexts,role_for
from scripts.train_jepa_proposals import role_values
from echo_routing.temporal.proposal_oracle import action_options,solve,missed_coverage,actions
from echo_routing.temporal.proposal_features import targets
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.proposal_readout_plan import TABLES,HALVES


def main():
    freeze=verify_freeze('oracle_freeze.json');summary=read(PUB/'summary.json');inputs=read(PUB/'training_inputs.json')
    if summary['metric_rows']!=124 or summary['paired_comparison_count']!=144:raise ValueError('all124/144 must freeze before label-assisted oracle')
    reference=read(PUB/'oracle_reference_identity.json');old=read(verify_pin(reference['A59_oracle']))
    expected={(r['table'],r['half'],r['domain']):r for r in old['oracle_domains']};aliases={'main':'main','a58_stress':'old_stress','a59_confirmation':'fresh_confirmation'}
    ctx=contexts(read(CFG));rows=[];diagnostics=[];reproduced=0
    for table in TABLES:
        for half in HALVES:
            roles=[half] if table=='main' else [half,role_for(table,half)];lookup={}
            for role in roles:
                qs,aa,jj,bb,_=role_values(role);lookup.update({q['id']:(q,a,j,b) for q,a,j,b in zip(qs,aa,jj,bb)})
            multi=[q for q in ctx[table][half]['sequences'] if q['kind']=='multi']
            for domain in ['old','expanded']:
                for guard in ['U','G']:
                    options=[];coverage={k:0 for k in ['baseline_missed_true_units','candidate_union_recovered_true_units','candidate_union_recovered_short10_true_units','candidate_union_recovered_short16_true_units','baseline_missed_short10_true_units','baseline_missed_short16_true_units']}
                    for q in multi:
                        _,a,j,b=lookup[q['id']];ra,rj=a.argmax(1),j.argmax(1);options.append(action_options(b,ra,rj,q['labels'],domain,guard=='G'))
                        cover=missed_coverage(b,ra,rj,q['labels'],domain,guard=='G')
                        for k in coverage:coverage[k]+=cover[k]
                    solved=solve(options,domain,guard);row={'table':table,'half':half,**{k:v for k,v in solved.items() if k!='choices'},'multi_streams':len(multi),**coverage,'union_is_jointly_reachable_claim':False,'per_sequence_choices_exported':False}
                    if domain=='old' and table in aliases:
                        prior=expected[(aliases[table],half,guard)]
                        for k in ['M','P','T','numerator','denominator','F1','baseline_missed_true_units','candidate_union_recovered_true_units']:
                            if row[k]!=prior[k]:raise ValueError('old-domain overlappingA59 oracle did not reproduce exactly')
                        row['overlapping_A59_domain_reproduced']=True;reproduced+=1
                    rows.append(row)
                for kind in ['native','multi','pair']:
                    totals={k:0 for k in ['sequences','actions','G_rejected','positive_FIT_countutility_actions','G_rejected_positive_FIT_countutility_actions','wrong_position_harm_actions','G_eligible_wrong_position_harm_actions','short16_FP_harm_actions','G_eligible_short16_FP_harm_actions','native_wrong_Base_majority_correcting_actions','native_wrong_Base_majority_correcting_G_rejected']}
                    for q in ctx[table][half]['sequences']:
                        qkind='native' if q['kind']=='native' else 'multi' if q['kind']=='multi' else 'pair'
                        if qkind!=kind:continue
                        _,a,j,b=lookup[q['id']];totals['sequences']+=1
                        for action in actions(b,a.argmax(1),j.argmax(1),domain,False):
                            value=targets(b,action,q['labels'],inputs['F_fit']);edited=apply(b,action);veto=majority(edited)!=majority(b)
                            totals['actions']+=1;totals['G_rejected']+=veto;totals['positive_FIT_countutility_actions']+=value[0]>0;totals['G_rejected_positive_FIT_countutility_actions']+=veto and value[0]>0
                            totals['wrong_position_harm_actions']+=value[1]>0;totals['G_eligible_wrong_position_harm_actions']+=not veto and value[1]>0;totals['short16_FP_harm_actions']+=value[2]>0;totals['G_eligible_short16_FP_harm_actions']+=not veto and value[2]>0
                            if kind=='native':
                                corrected=majority(b)!=int(q['labels'][0]) and majority(edited)==int(q['labels'][0]);totals['native_wrong_Base_majority_correcting_actions']+=corrected;totals['native_wrong_Base_majority_correcting_G_rejected']+=corrected and veto
                    diagnostics.append({'table':table,'half':half,'kind':kind,'domain':domain,**{k:int(v) for k,v in totals.items()},'positive_FIT_countutility_not_native_macroF1_surrogate':True,'actions_alternatives_not_jointly_executable':True})
    if len(rows)!=32 or reproduced!=12 or len(diagnostics)!=48:raise ValueError('complete32oracle/old12reproduction/48descriptivekindgroups required')
    write(PUB/'action_space_diagnostic.json',{'oracle_domains':rows,'domain_count':32,'overlapping_old_A59_oracles_exactly_reproduced':reproduced,'action_diagnostics':diagnostics,'label_assisted_not_system_gate_or_rescue':True,'scope':'only exact pooledmultiIoU50 F1 within old/expanded maxONEaction+noop and U/G;not arbitrary segmenter or allJEPAinformation','native_pair_not_in_oracle_objective':True,'summary_sha256':sha(PUB/'summary.json'),'no_oracle_used_for_model_or_threshold_selection':True,'oracle_freeze_git':freeze['git_commit'],'epoch':time.time()})
if __name__=='__main__':main()
