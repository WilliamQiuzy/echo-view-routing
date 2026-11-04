"""Synthetic contract fixtures; no optimizer step, forward or server execution."""
import copy,hashlib,json,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
try:
    from echo_routing.temporal.event_risk_audit import audit_trace,validate_state,validate_fullstates,validate_barrier,FIXED_A61
except ModuleNotFoundError as exc:
    if exc.name!='echo_routing.temporal.event_risk_audit':raise
    audit_trace=validate_state=validate_fullstates=validate_barrier=FIXED_A61=None


class EventRiskAuditTests(unittest.TestCase):
    def setUp(self):self.assertTrue(callable(audit_trace),'A62 independent audit not implemented yet')

    def identity(self):return {k:'a'*64 for k in ['code','config','sources','input','draw_plan','normalizer','target_scales']}

    def trace(self):
        from echo_routing.temporal.safety_training import draw_plan
        from echo_routing.temporal.proposal_training import lr_factor
        p=draw_plan(265,3918,2026110461,'risk');k={'native':np.arange(265),'constructed':np.arange(265,4183)};y=(np.arange(4183)%3==0).astype(np.float32);rows=[]
        for u in range(6000):
            idx=[]
            for block,kind in enumerate(p['source_kinds']):idx.extend(k[['native','constructed'][kind]][p['indices'][u,32*block:32*(block+1)]].tolist())
            rows.append({'u':u,'identity':self.identity(),'physical_candidates':64,'candidate_indices':idx,'source_kind_blocks':p['source_kinds'],'unsafe_targets':int(y[idx].sum()),'candidate_BCE':.5,'loss':.5,'lr':.001*lr_factor(u),'finite_gradients':True,'gradient_norm':.1,'nonzero_gradient_tensors':{'network.0.weight':u>0,'network.0.bias':u>0,'network.2.weight':True,'network.2.bias':True},'optimizer_update_wall_seconds':.001})
        return rows,p,k,y

    def test_exact_original_draw_exposures_and_any_trace_identity_or_label_drift_rejected(self):
        rows,p,k,y=self.trace();r=audit_trace(rows,p,k,y,2026110461,self.identity())
        self.assertEqual(r['physical_updates'],6000);self.assertEqual(r['candidate_exposures'],384000)
        self.assertEqual(r['native_candidate_exposures'],192000);self.assertEqual(r['constructed_candidate_exposures'],192000)
        self.assertEqual(r['unsafe_candidate_exposures'],sum(x['unsafe_targets'] for x in rows))
        self.assertTrue(all(r['all_tensor_gradient_liveness_over_history'].values()))
        for field,value in [('candidate_indices',list(reversed(rows[0]['candidate_indices']))),('unsafe_targets',-1),('u',True),('identity',{**self.identity(),'input':'b'*64})]:
            bad=[{**rows[0],field:value},*rows[1:]]
            with self.assertRaises(ValueError):audit_trace(bad,p,k,y,2026110461,self.identity())
        with self.assertRaises(ValueError):audit_trace(rows[:-1],p,k,y,2026110461)
        with self.assertRaises(ValueError):audit_trace(rows,p,k,y.astype(float)+.1,2026110461)
        with self.assertRaises(ValueError):audit_trace(rows,p,k,y,2026110462)

    def states(self):
        import torch
        from echo_routing.temporal.event_risk_model import EventRisk,optimizer,checkpoint
        from echo_routing.temporal.proposal_training import lr_factor
        model=EventRisk('R1',2026110461);opt=optimizer(model);norms={'candidate':{'mean':np.zeros(109),'std':np.ones(109)},'context':{'mean':np.zeros(96),'std':np.ones(96)}}
        initial=copy.deepcopy(model.state_dict());zero=checkpoint(model,opt,0,self.identity(),norms);states={}
        # Artificial state schema fixture: never reported as optimizer execution.
        for cursor in [2000,4000,6000]:
            value=copy.deepcopy(zero);value['update']=cursor
            value['optimizer']['param_groups'][0]['lr']=.001*lr_factor(cursor-1)
            for key,(_,parameter) in zip(value['optimizer']['param_groups'][0]['params'],model.named_parameters()):
                value['optimizer']['state'][key]={'step':torch.tensor(float(cursor)),'exp_avg':torch.ones_like(parameter)*.1,'exp_avg_sq':torch.ones_like(parameter)*.01}
            value['model']={k:v+.25 for k,v in initial.items()};states[cursor]=value
        states['latest']=copy.deepcopy(states[6000])
        return model,opt,norms,initial,zero,states

    def test_complete_schema_state_validation_without_forward_or_optimizer_work(self):
        model,opt,norms,initial,zero,states=self.states()
        self.assertEqual(validate_state(model,opt,zero,self.identity(),norms,0)['cursor'],0)
        self.assertEqual(validate_fullstates(model,opt,states,self.identity(),norms,initial)['fullstates'],4)
        bad=copy.deepcopy(states);bad['latest']['rng']['python']=(0,(),None)
        with self.assertRaises(ValueError):validate_fullstates(model,opt,bad,self.identity(),norms,initial)
        bad=copy.deepcopy(states[6000]);bad['optimizer']['state'][0]['step'].fill_(5999)
        with self.assertRaises(ValueError):validate_state(model,opt,bad,self.identity(),norms,6000)
        bad=copy.deepcopy(states[6000]);bad['normalizers']['candidate']['std'][0]=2
        with self.assertRaises(ValueError):validate_state(model,opt,bad,self.identity(),norms,6000)
        with self.assertRaises(ValueError):validate_fullstates(model,opt,{k:v for k,v in states.items() if k!=2000},self.identity(),norms,initial)

    def barrier(self,repo):
        from echo_routing.temporal.event_risk_metrics import plan
        from echo_routing.temporal.safety_git import build
        root=Path(repo);pub='docs/results/2026-11-04_anchor_context_risk/';(root/pub).mkdir(parents=True)
        models=[];states=[]
        for seed in [2026110461,2026110462]:
            for arm in ['R1','R2']:
                path=f'runs/a62_anchor_context_risk_v1/models/seed{seed}/{arm}/step6000.pt';sha=hashlib.sha256(path.encode()).hexdigest()
                models.append({'seed':seed,'arm':arm,'step':6000,'path':path,'sha256':sha,'parameters':6625})
                for step in [2000,4000,6000]:
                    p=path.replace('step6000',f'step{step}');states.append({'seed':seed,'arm':arm,'step':step,'kind':'milestone','path':p,'sha256':hashlib.sha256(p.encode()).hexdigest()})
                states.append({'seed':seed,'arm':arm,'step':6000,'kind':'latest','path':path.replace('step6000','latest'),'sha256':sha})
        thresholds=[{'seed':s,'policy':p,'tau':.3} for s in [2026110461,2026110462] for p in ['M0','R1','R2']]
        legacy=[{'seed':s,'policy':'L0','tau':v} for s,v in [(2026110461,.007963286712765694),(2026110462,.006616456434130669)]]
        cal={'thresholds':thresholds,'legacy_thresholds':legacy,'inner_records':plan()['inner_rows']}
        raw=lambda x:(json.dumps(x,sort_keys=True,indent=2)+'\n').encode();calsha=hashlib.sha256(raw(cal)).hexdigest();plansha=hashlib.sha256(raw(plan())).hexdigest()
        manifest={'models':models,'fixed_A61_models':[dict(v) for v in FIXED_A61],'fullstates':states,'thresholds':thresholds,'legacy_thresholds':legacy,'inner_records':cal['inner_records'],'formal_optimizer_updates':24000,'physical_models':4,'calibration_sha256':calsha,'readout_plan_sha256':plansha,'audited_readout_plan_sha256':plansha}
        modelsha=hashlib.sha256(raw(manifest)).hexdigest()
        for name,value in [('models_manifest.json',manifest),('calibration.json',cal),('readout_plan.json',plan())]:(root/pub/name).write_bytes(raw(value))
        def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
        git('init','-q','-b','main');git('config','user.name','Synthetic');git('config','user.email','synthetic@example.invalid');git('add',pub);git('commit','-qm','synthetic barrier fixture');commit=git('rev-parse','HEAD')
        proof=build(root,commit,[pub+n for n in ['models_manifest.json','calibration.json','readout_plan.json']])
        seal={'models_manifest_sha256':modelsha,'calibration_sha256':calsha,'readout_plan_sha256':plansha,'git_commit':commit,'all_final_checkpoints':[{'path':r['path'],'sha256':r['sha256']} for r in [*models,*FIXED_A61]]}
        return manifest,seal,modelsha,calsha,plansha,proof

    def test_actual_git_three_blob_barrier_exact_model_state_tau_and_endpoint_coverage(self):
        with tempfile.TemporaryDirectory(prefix='a62-audit-toy-') as d:
            args=self.barrier(d);self.assertTrue(validate_barrier(*args))
            for field,value in [('formal_optimizer_updates',23999),('audited_readout_plan_sha256','b'*64),('inner_records',args[0]['inner_records'][:-1]),('fullstates',args[0]['fullstates'][:-1])]:
                changed=copy.deepcopy(args);changed[0][field]=value
                with self.assertRaises(ValueError):validate_barrier(*changed)
            changed=copy.deepcopy(args);changed[0]['fixed_A61_models'][0]['sha256']='b'*64
            with self.assertRaises(ValueError):validate_barrier(*changed)
            changed=list(copy.deepcopy(args));changed[-1]={'passed':True}
            with self.assertRaises(ValueError):validate_barrier(*changed)
            changed=copy.deepcopy(args);changed[1]['all_final_checkpoints']=changed[1]['all_final_checkpoints'][:-1]
            with self.assertRaises(ValueError):validate_barrier(*changed)

if __name__=='__main__':unittest.main()
