"""New-window joined teacher producer; predecessor source functions unchanged.

Only a future guardian-held shared queue may invoke this driver. Actual teacher
work is never performed by preparation fixtures. Original metadata/recipes and
all original producer bytes remain read-only; new journals live in new RUN.
"""
import os
import time
from pathlib import Path
from echo_routing.temporal.selected_risk_resume_contract import digest

SOURCE='echo_routing/temporal/selected_risk_resume_source.py'


def phase_rule(phase,run,coverage,aux_audit):
    from echo_routing.temporal.selected_risk_policy import production_phase
    run=Path(run)
    if phase not in ['FIT','CONFIRM'] or (run/'models').exists() or (run/'models').is_symlink():raise ValueError('only pre-formal exact FIT/CONFIRM source phases allowed')
    if phase=='FIT' and ((run/'aux').exists() or (run/'aux').is_symlink()):raise ValueError('FIT teacher production must precede all actual auxiliary preparation/training')
    return production_phase(phase,coverage,aux_audit)


def source_identity(source,role,input_sha,owner_sha,objects,source_cfg,phase):
    return digest({'source':source,'role':role,'input':input_sha,'runtime_owner_raw_sha256':owner_sha,'actual_stage_objects':objects,'source_contract':source_cfg,'phase':phase,'original_phase_counters':{'aux':0 if phase=='FIT' else 48000,'formal':0}})


class Source:
    def __init__(self,context,inputs,freeze_pin,descriptors):
        self.ctx=context;self.inputs=inputs;self.freeze_pin=freeze_pin;self.descriptors=descriptors

    def produce(self,phase):
        stage='source_'+phase
        freeze,e=self.ctx.enter(stage,self.freeze_pin,self.descriptors,required={SOURCE,'echo_routing/temporal/selected_risk_source.py','echo_routing/temporal/selected_risk_resume_context.py','configs/experiments/selected_action_risk_v1.json'})
        rule=phase_rule(phase,self.ctx.output('producer_namespace_guard').parent,e.get('coverage',{}).get('payload'),e.get('seal_aux',{}).get('payload'))
        admission=e['admit_FIT' if phase=='FIT' else 'admit_formal']['payload']
        if admission.get('input_manifest_sha256')!=self.inputs.metadata_pin['sha256'] or admission.get('source_phase')!=phase:raise ValueError('complete current source phase admission must bind exact unchanged metadata')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('sole pinned GPU must be explicit before Torch import')
        # All code and independent actual admission gates precede any model.
        from scripts.selected_risk_common import CFG,read
        from echo_routing.temporal.selected_risk_source import determinism,build_source,a08_sequence,jepa_sequence
        from echo_routing.temporal.dense_readout import ReadoutPixels
        from echo_routing.temporal.anchor_cache import ProbabilityCache,sha
        from echo_routing.temporal.anchor_inputs import workload
        import torch
        if torch.cuda.is_initialized():raise ValueError('unexpected pre-existing CUDA context at producer entry')
        determinism();cfg=read(CFG);targets={};banks={r:self.inputs.sequences(r) for r in rule['roles']}
        for name in ['a08','jepa']:
            for role,qs in banks.items():
                identity=source_identity(name,role,self.inputs.manifest['roles'][role]['sha256'],self.ctx.owner_sha,freeze['objects'],cfg['source'][name],phase)
                folder=self.ctx.output(f'producer_{phase}/'+name+'/'+role)
                targets[(name,role)]=ProbabilityCache(folder,identity,[len(q['refs']) for q in qs])
        records=[]
        for name in ['a08','jepa']:
            model=build_source(name,cfg,'cuda')
            for role,qs in banks.items():
                target=targets[(name,role)];pixels=ReadoutPixels(qs);start=time.time();units=0;prefix=target.cursor
                for i in range(prefix,len(qs)):
                    self.ctx.revalidate()
                    target.run_one(i,lambda i=i:(a08_sequence if name=='a08' else jepa_sequence)(model,qs[i],pixels))
                    units+=workload([qs[i]])['a08_stride5_windows' if name=='a08' else 'jepa_chunks']
                logical=workload(qs)
                records.append({'source':name,'role':role,'path':str(target.path.relative_to(self.ctx.root)),'identity':target.identity,'journal_sha256':sha(target.journal),'committed_sequences':target.cursor,'input_sha256':self.inputs.manifest['roles'][role]['sha256'],'actual_new_neural_units_this_invocation':units,'committed_logical_neural_units':logical['a08_stride5_windows' if name=='a08' else 'jepa_chunks'],'neural_unit':'A08 stride5 window' if name=='a08' else 'JEPA chunk','resumed_committed_prefix_sequences':prefix,'wall_seconds':time.time()-start,'complete':target.cursor==len(qs)})
            del model;torch.cuda.empty_cache()
        if len(records)!=4 or sum(r['committed_sequences'] for r in records)!=rule['payloads']:raise ValueError('every complete joined-context source payload required')
        value={'passed':True,'complete':True,'phase':phase,'records':records,'aux_updates_at_production':rule['aux_updates_at_production'],'formal_updates_at_production':0,'new_event_head_instances_or_forwards':0,'source_production_stage':stage,'source_freeze_git':freeze['git_commit'],'input_manifest_sha256':self.inputs.metadata_pin['sha256'],'runtime_authorization_sha256':self.ctx.owner_sha,'genuine_joined_context_forward':True,'teacher_streams':rule['payloads']//2,'committed_payloads':rule['payloads'],'actual_forward_units_this_invocation':sum(r['actual_new_neural_units_this_invocation'] for r in records),'actual_all_attempt_forward_units':None,'actual_all_attempt_units_require_independent_process_ledger_recount':True,'protected_media_forwards':0,'constructor_metadata_label_parse_disclosed':True,'labels_used_for_new_teacher_predictions':False,'old_namespaces_written':False}
        return self.ctx.write_json('producer_'+phase+'_manifest.json',value,public=True)
