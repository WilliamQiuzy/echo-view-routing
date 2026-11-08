import numpy as np
import pytest
from echo_routing.temporal.proposal_actions import Action,apply


def test_native_actual_cube_preserves_gross_harm_even_majority_unchanged():
    from echo_routing.temporal.selected_risk_resume_diagnostics import native_result
    b=np.zeros(12,np.int64);a=Action(2,3,(1,),True,False);v={'C1':{'prediction':apply(b,a),'accepted_index':0},'actions':[a]}
    q={'kind':'native','labels':b.copy()};r=native_result([q],[b],[v],'C1')
    assert r['Base_correct_to_wrong']==1 and r['majority_changed_native_sequences']==0
    assert np.asarray(r['true_Base_prediction_joint5cube'])[0,0,1]==1
    v['C1']['prediction']=b.copy()
    with pytest.raises(ValueError):native_result([q],[b],[v],'C1')


def test_regression_absent_top_keeps_null_potential_and_zero_actual():
    from echo_routing.temporal.selected_risk_resume_diagnostics import case_records
    b=np.zeros(12,np.int64);a=np.eye(5)[b];v={'actions':[],'old_scores':np.empty((0,6)),'old_q':np.empty((0,2)),'q':np.empty((0,2))}
    from echo_routing.temporal.selected_risk_resume_readout import decoded
    v=decoded(b,v,v['q'],dict(L0=1.,M0=1.,R1=1.,R2=1.),dict(C0=1.,C1=1.))
    r=case_records('Case_A',{'labels':b},b,a,a,v,2026110461)
    assert len(r)==8 and all(x['H'] is None and x['actual_H']==0 and x['candidate_key_sha256'] is None for x in r)


def test_support_reference_adapter_is_exact48_and_preserves_payload():
    from echo_routing.temporal.selected_risk_resume_diagnostics import old_support
    from echo_routing.temporal.selected_risk_metrics import readout_plan
    original=[{**s,'multi':{'sentinel':1}} for s in readout_plan()['support_cells'] if s['table']!='a63_confirmation']
    rows=old_support({'support_count':48,'fixed_N_top_support':original})
    assert len(rows)==48 and all(r['readonly_A62_reference'] for r in rows)
    assert rows[0]['result']['multi'] is original[0]['multi']
    with pytest.raises(ValueError):old_support({'support_count':48,'fixed_N_top_support':original[:-1]+original[:1]})



def _package_fixture(tmp_path,monkeypatch):
    import gzip,hashlib,json,runpy
    from pathlib import Path
    from echo_routing.temporal.selected_risk_resume_diagnostics import Diagnostics
    from echo_routing.temporal.selected_risk_resume_context import Context
    from echo_routing.temporal.selected_risk_resume_contract import PUB
    from echo_routing.temporal.selected_risk_metrics import acceptance,readout_plan,SEEDS
    class C(Context):
        def revalidate(self):return True
    ctx=object.__new__(C);ctx.root=tmp_path.resolve();ctx.owner={}
    def put(path,value,raw=False):
        b=value if raw else json.dumps(value).encode();p=ctx.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        return {'path':path,'sha256':hashlib.sha256(b).hexdigest()}
    ctx._bound_runtime_spec_pin=put('toy_spec.json',{'stage':'package'})
    rows,comparisons=runpy.run_path(str(Path(__file__).with_name('test_selected_risk_scope.py')))['complete']()
    bad=next(r for r in rows if r['policy']=='C1');bad['H']['native']=1;bad['class_H'][0]=1
    verdict=acceptance(rows,comparisons);accept=put(PUB+'/acceptance.json',verdict)
    plan=readout_plan();models=[];archives=[];physical_records=[]
    for seed in SEEDS:
        for arm in ['C0','C1']:
            folder=f'runs/a63_selected_action_risk_resume_v1/models/seed{seed}/{arm}';identity={'fixture':'not a scientific trace'}
            state=put(folder+'/step6000.pt',b'toy, not a model',True)
            models.append(dict(seed=seed,arm=arm,step=6000,identity=identity,**state))
            trace=[dict(u=u,loss=.125,identity=identity) for u in range(6000)]
            raw=b''.join((json.dumps(r)+'\n').encode() for r in trace);z=gzip.compress(raw)
            archive=put(folder+'/physical_updates.jsonl.gz',z,True)
            completion=put(folder+'/completion.json',dict(passed=True,identity=identity,physical_updates=6000,fullstate_count=4,raw_trace_sha256=hashlib.sha256(raw).hexdigest()))
            physical_records.append(dict(seed=seed,arm=arm,completion=completion,archive_sha256=archive['sha256'],reused_complete=False))
            receipt=dict(identity=identity,archive_sha256=archive['sha256'],uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest())
            archives.append(put(folder+'/physical_updates.jsonl.archive.json',receipt))
    fit_scores={'passed':True,'complete':True,'seeds':list(SEEDS),'records':[{'seed':s} for s in SEEDS],'FIT_q_completed_before_CAL':True,'FIT_scores_used_to_select_model_or_tau':False,'actual_new_MLP_calls_this_process':12,'actual_new_candidate_outputs_this_process':48,'actual_all_attempt_MLP_calls':None,'informational_consumer_CPU_seconds_to_snapshot':.75,'new_teacher_or_S_L_or_aux_forward_calls':0,'optimizer_updates':0}
    fitpin=put(PUB+'/FIT_risk_scores_manifest.json',fit_scores)
    diagpin=put(PUB+'/FIT_risk_diagnostics.json',{'passed':True,'complete':True,'FIT_q_scores_manifest':fitpin})
    innerpin=put(PUB+'/inner_risk_diagnostics.json',{'FIT_risk_diagnostics':diagpin})
    payloads={
        'aggregate':dict(passed=True,rows=rows,paired_comparisons=comparisons,comparison_count=140,acceptance_sha256=accept['sha256']),
        'supports':dict(passed=True,record_count=56,records=[{} for _ in range(56)]),
        'forensics':dict(passed=True,native_count=34,regression_count=32,native=put(PUB+'/native_attribution.json',{'records':[{} for _ in range(34)]}),regression=put(PUB+'/regression_summary.json',{'records':[{} for _ in range(32)]})),
        'calibrate':dict(passed=True,inner_records=[{} for _ in range(34)],FIT_risk_scores_manifest=fitpin,FIT_risk_diagnostics=diagpin,risk_diagnostics=innerpin),
        'fit_risk_scores':fit_scores,
        'audit_formal':dict(passed=True,formal_optimizer_updates=24000,physical_models=4,fullstate_count=16,models=models),
        'audit_aux':dict(passed=True,aux_optimizer_updates=48000,physical_models=8,fullstate_count=32),
        'coverage':dict(passed=True),
        'train_formal':dict(passed=True,formal_optimizer_updates=24000,physical_models=4,fullstate_count=16,records=physical_records)}
    evidence={};descriptors={}
    for key,v in payloads.items():
        pin=fitpin if key=='fit_risk_scores' else put(PUB+'/fixture_'+key+'.json',v);descriptors[key]={'payload_pin':pin};evidence[key]={'payload':v,'payload_sha256':pin['sha256']}
    ctx.enter=lambda *a,**k:({},evidence)
    d=object.__new__(Diagnostics);d.ctx=ctx;d.freeze_pin={};d.descriptors=descriptors
    d._setup=lambda *a:(_ for _ in ()).throw(AssertionError('package must not instantiate inference'))
    class ResourceCutoff:
        def __init__(self,context,spec,pin):assert context is ctx and spec['stage']=='package' and pin==ctx._bound_runtime_spec_pin
        def spent(self):return {'new_measured_CPU_seconds':10.,'new_CPU_accounting_seconds':12.,'new_GPU_wall_seconds':3.,'cumulative_GPU_hours':1.,'cumulative_CPU_accounting_hours':2.}
        def storage(self):return 1000
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_runtime.RuntimeState',ResourceCutoff)
    seen={}
    def render(pub,r,c,i,s,n,reg,curves,resources,**kw):
        assert (len(r),len(c),len(i),len(s),len(n),len(reg),len(curves))==(238,140,34,56,34,32,24000)
        assert not kw.get('synthetic',False);assert resources['actual_caps_fit'] is None
        assert {(v['seed'],v['arm']) for v in curves}=={(seed,arm) for seed in SEEDS for arm in ['C0','C1']}
        assert all(np.isfinite(v['loss']) and v['loss']>=0 for v in curves);seen['render']=True;seen['curves']=curves
        value={'synthetic':False,'actual_visual_QA_passed':False,'overall_complete':False,'counts':dict(outer=238,comparisons=140,inner=34,support=56,native=34,regression=32,curves=24000),'files':[]}
        put(PUB+'/full_figures/render_receipt.json',value);return value
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_present.render_full',render)
    return d,ctx,put,verdict,archives,seen


def test_package_preserves_negative_verdict_and_pins_finished_reports(tmp_path,monkeypatch):
    from echo_routing.temporal.selected_risk_resume_contract import PUB
    d,ctx,put,verdict,archives,seen=_package_fixture(tmp_path,monkeypatch)
    pin=d.package();receipt=ctx.read(pin)
    assert seen['render'] and receipt['passed'] is True
    assert all(v['loss']==.125 for v in seen['curves'])
    assert receipt['acceptance']==verdict and receipt['acceptance']['H63_claim_supported'] is False
    assert receipt['actual_visual_QA_passed'] is False and receipt['overall_complete'] is False
    for name in ['README','FINDINGS','render_receipt']:ctx.raw(receipt['artifacts'][name])
    assert receipt['figure_receipt']==receipt['artifacts']['render_receipt']
    assert {v['path'] for v in receipt['file_pins']}=={v['path'] for v in receipt['artifacts'].values()}
    for pin in receipt['file_pins']:ctx.raw(pin)
    findings=ctx.raw(receipt['artifacts']['FINDINGS']).decode()
    assert 'hard_protection_pass' in findings and 'false' in findings
    assert receipt['FIT_risk_inference_costs']['actual_new_MLP_calls_this_process']==12
    assert receipt['FIT_risk_inference_costs']['actual_all_attempt_MLP_calls'] is None
    assert receipt['FIT_risk_inference_costs']['extra_canonical_CPU_charge_seconds']==0
    assert receipt['resources']['known_CPU_seconds']==10. and receipt['resources']['actual_caps_fit'] is None
    assert not (ctx.root/'docs/results/2026-11-05_selected_action_risk').exists()


@pytest.mark.parametrize('failure',['curve_gap','render_failure'])
def test_package_has_no_passed_receipt_on_missing_trace_or_failed_render(tmp_path,monkeypatch,failure):
    from echo_routing.temporal.selected_risk_resume_contract import PUB
    d,ctx,put,verdict,archives,seen=_package_fixture(tmp_path,monkeypatch)
    if failure=='curve_gap':
        import gzip,hashlib,json
        pin=archives[0];r=ctx.read(pin);path=pin['path'].replace('.archive.json','.gz');raw=b''.join(gzip.decompress((ctx.root/path).read_bytes()).splitlines(keepends=True)[:-1]);z=gzip.compress(raw)
        put(path,z,True);r.update(archive_sha256=hashlib.sha256(z).hexdigest(),uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest());put(pin['path'],r)
    else:
        monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_present.render_full',lambda *a,**k:(_ for _ in ()).throw(ValueError('render failed')))
    with pytest.raises(ValueError):d.package()
    assert not (ctx.root/PUB/'package_receipt.json').exists()



def test_package_rejects_joint_trace_archive_completion_rewrite_against_original_records(tmp_path,monkeypatch):
    import gzip,hashlib,json
    from echo_routing.temporal.selected_risk_resume_contract import PUB
    d,ctx,put,verdict,archives,seen=_package_fixture(tmp_path,monkeypatch)
    pin=archives[0];archive=ctx.read(pin);gz_path=pin['path'].replace('.archive.json','.gz')
    trace=[json.loads(v) for v in gzip.decompress((ctx.root/gz_path).read_bytes()).splitlines()]
    trace[0]['loss']=.75
    raw=b''.join((json.dumps(v)+'\n').encode() for v in trace);z=gzip.compress(raw);put(gz_path,z,True)
    digest=hashlib.sha256(raw).hexdigest();archive.update(archive_sha256=hashlib.sha256(z).hexdigest(),uncompressed_sha256=digest,uncompressed_bytes=len(raw));put(pin['path'],archive)
    completion_path=pin['path'].replace('physical_updates.jsonl.archive.json','completion.json')
    original=json.loads((ctx.root/completion_path).read_bytes());original['raw_trace_sha256']=digest;put(completion_path,original)
    # These three files remain mutually consistent. The external train_formal
    # evidence and its original record pins deliberately remain unchanged.
    with pytest.raises(ValueError):d.package()
    assert not seen and not (ctx.root/PUB/'package_receipt.json').exists()


def test_fit_cost_reader_uses_shared_pins_and_preserves_unknown_attempt_total():
    from echo_routing.temporal.selected_risk_resume_diagnostics import fit_inference_costs
    from echo_routing.temporal.selected_risk_resume_contract import PUB
    pins={k:{'path':PUB+'/'+name,'sha256':str(i)*64} for i,(k,name) in enumerate([('FIT_risk_scores_manifest','FIT_risk_scores_manifest.json'),('FIT_risk_diagnostics','FIT_risk_diagnostics.json'),('risk_diagnostics','inner_risk_diagnostics.json')],1)}
    scores={'passed':True,'complete':True,'seeds':[2026110461,2026110462],'records':[{'seed':s,'successful_axis_receipt_model_load_forward_CPU_seconds':.25} for s in [2026110461,2026110462]],'FIT_q_completed_before_CAL':True,'FIT_scores_used_to_select_model_or_tau':False,'actual_new_MLP_calls_this_process':12,'actual_new_candidate_outputs_this_process':48,'actual_all_attempt_MLP_calls':None,'informational_consumer_CPU_seconds_to_snapshot':.75,'new_teacher_or_S_L_or_aux_forward_calls':0,'optimizer_updates':0}
    fit={'passed':True,'complete':True,'FIT_q_scores_manifest':pins['FIT_risk_scores_manifest']}
    inner={'FIT_risk_diagnostics':pins['FIT_risk_diagnostics']}
    class C:
        def read(self,p):
            assert p in pins.values()
            return {'FIT_risk_scores_manifest.json':scores,'FIT_risk_diagnostics.json':fit,'inner_risk_diagnostics.json':inner}[p['path'].rsplit('/',1)[1]]
    result=fit_inference_costs(C(),pins,{'payload_pin':pins['FIT_risk_scores_manifest']})
    assert result['actual_new_MLP_calls_this_process']==12 and result['actual_new_candidate_outputs_this_process']==48
    assert result['actual_all_attempt_MLP_calls'] is None and result['extra_canonical_CPU_charge_seconds']==0
    assert result['nested_informational_CPU_seconds_to_snapshot']==.75
    with pytest.raises(ValueError):fit_inference_costs(C(),pins,{'payload_pin':{**pins['FIT_risk_scores_manifest'],'sha256':'f'*64}})
    scores['actual_new_MLP_calls_this_process']=True
    with pytest.raises(ValueError):fit_inference_costs(C(),pins,{'payload_pin':pins['FIT_risk_scores_manifest']})
