import importlib
from types import SimpleNamespace
import numpy as np
from echo_routing.temporal.dense_runtime import metric_stream
from echo_routing.temporal.structured_metrics import evaluate

def test_paired_whole_units_zero_identity_and_exact_native_segment_disagreements():
    m=importlib.import_module('echo_routing.temporal.structured_uncertainty')
    def q(name,kind,y):return {'id':name,'kind':kind,'labels':np.array(y),'refs':[('v',i,1.) for i in range(len(y))],'joins':np.array([2]) if kind!='native' else np.array([],int)}
    qs=[q('a_0','native',[0,0,0,0]),q('b_0','native',[1,1,1,1]),q('mixed','multi',[0,0,1,1]),q('pair','pair',[0,0,1,1])];context={'sequences':qs,'eval_set':SimpleNamespace(natives=[SimpleNamespace(seq_id=x['id'],labels=x['labels']) for x in qs[:2]],multi=[metric_stream(qs[2])],pairs=[metric_stream(qs[3])])};left={q['id']:q['labels'].copy() for q in qs};right={**left,'a_0':np.ones(4,int),'mixed':np.zeros(4,int)};_,lt=evaluate(left,context);_,rt=evaluate(right,context);a=m.prepare_system(left,qs,lt);b=m.prepare_system(right,qs,rt);same=m.paired_bootstrap(a,a,n_boot=32,seed=58);assert all(v is None or v==[0.,0.] for v in same['delta95'].values());assert same['delta95']['same_class_cut_rate_multi'] is None and same['undefined_bootstrap_replicates']['same_class_cut_rate_multi']==32;result=m.paired_bootstrap(a,b,n_boot=32,seed=58);assert result['exact_disagreements']['native_left_only_correct']==1 and result['exact_disagreements']['native_both_correct']==1 and result['exact_disagreements']['multi_true_segments_recovered']==1 and result['bootstrap']['native_sessions']==2 and result['bootstrap']['multi_streams']==1 and result['bootstrap']['pair_streams']==1;assert result['delta95']['boundary_precision_multi'] is None
