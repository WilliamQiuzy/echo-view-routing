import importlib
import numpy as np

def test_legacy16_and_declared10_short_definitions_share_exact_cached_matching():
 a=importlib.import_module('echo_routing.temporal.dense_analysis');q={'id':'stream','kind':'multi','labels':np.array([2]*14+[3]*16)};perfect={'stream':q['labels'].copy()};constant={'stream':np.full(30,3)}
 r=a.short_metrics([q],perfect,16);s=a.short_metrics([q],perfect,10)
 assert r=={'short_true_segments_16samples':2,'short_true_unmatched_16samples':0,'short_predicted_segments_16samples':2,'short_spurious_segments_16samples':0}
 assert s['short_true_segments_10samples']==0
 x=a.short_metrics([q],constant,16)
 assert x['short_true_unmatched_16samples']==1 and x['short_true_segments_16samples']==2 and x['short_predicted_segments_16samples']==0

def test_paired_bootstrap_zero_identical_and_exact_recovered_segment_counts():
 import importlib
 from echo_routing.temporal import dense_runtime as d
 a=importlib.import_module('scripts.aggregate_dense_sequence')
 q=[{'id':'s0_k0','kind':'native','labels':np.full(20,2),'refs':[('s0_k0',i,1.) for i in range(20)],'joins':np.array([],int)},{'id':'s1_k0','kind':'native','labels':np.full(20,3),'refs':[('s1_k0',i,1.) for i in range(20)],'joins':np.array([],int)},{'id':'m','kind':'multi','labels':np.array([2]*14+[3]*16),'refs':[('s0_k0',i%20,1.) for i in range(30)],'joins':np.array([14])}]
 p={x['id']:x['labels'].copy() for x in q};_,t=d.metrics(q,p)
 result=a.boot_pair(p,p,q,t,t,n_boot=30)
 assert result['paired_conditional_stream_bootstrap_segmentF1_delta95']==[0,0] and result['paired_session_bootstrap_nativeMacroF1_delta95']==[0,0]
 assert result['bootstrap']['multi_streams']==1
 assert result['exact_disagreements']['native_both_correct']==2 and result['exact_disagreements']['multi_true_segments_recovered']==0
 bad={**p,'m':np.full(30,3)};_,b=d.metrics(q,bad);result=a.boot_pair(p,bad,q,t,b,n_boot=30)
 assert result['exact_disagreements']['multi_true_segments_recovered']==1 and result['exact_disagreements']['short_true_segments_recovered_16samples']==1 and result['exact_disagreements']['short_true_segments_recovered_10samples']==0


def test_inner_probability_cache_admission_and_independent_raw_recount():
 import pytest
 from echo_routing.temporal import dense_runtime as d
 a=importlib.import_module('echo_routing.temporal.dense_analysis')
 q=[{'id':'n','kind':'native','labels':np.full(20,2),'refs':[('v',i,1.) for i in range(20)],'joins':np.array([],int)},{'id':'m','kind':'mixed','labels':np.array([2]*10+[3]*10),'refs':[('v',i,1.) for i in range(20)],'joins':np.array([10])}]
 probs={x['id']:np.eye(5,dtype=np.float32)[x['labels']] for x in q};preds={k:v.argmax(1) for k,v in probs.items()};record=d.metrics(q,preds)[0];record['input_sequence_sha256']='fixedinput'
 assert a.validate_inner_cache(probs,q,'fixedinput',record)['segment_f1_50_multi']==1
 for bad in [{**probs,'n':probs['n'][:1]},{**probs,'n':probs['n']*.5},{**probs,'n':np.full((20,5),np.nan)},{'wrong':probs['n'],'m':probs['m']},{**probs,'n':np.eye(5)[np.full(20,0)]}]:
  with pytest.raises(ValueError):a.validate_inner_cache(bad,q,'fixedinput',record)
 with pytest.raises(ValueError):a.validate_inner_cache(probs,q,'differentinput',record)
