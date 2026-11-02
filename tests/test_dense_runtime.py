import importlib
from types import SimpleNamespace
import numpy as np
import pytest

def api():return importlib.import_module('echo_routing.temporal.dense_runtime')
def seq(ident='a',refs=None,labels=None,kind='native'):
    return {'id':ident,'kind':kind,'refs':refs or [('v1',i,1.) for i in range(65)],'labels':np.array(labels if labels is not None else [2]*65),'joins':np.array([],int)}
def test_full_footprint_allowlist_isolation_and_repeat_padding():
    a=api();q=seq();a.verify_partition([q],{'v1'})
    q['refs'][-1]=('forbidden',0,1.)
    with pytest.raises(ValueError):a.verify_partition([q],{'v1'})
    q=seq();indices,mask=a.chunk_arrays(q,64)
    assert len(indices)==64 and np.all(indices==64) and mask.sum()==1
    with pytest.raises(ValueError):a.verify_partition([seq('same'),seq('same')],{'v1'})
def test_metric_adapter_matches_existing_exact_counts_and_bidirectional_errors():
    a=api();native=[seq('n1',labels=[2]*65),seq('n2',labels=[3]*65)]
    mixed=seq('m',refs=[('v1',i,1.) for i in range(20)],labels=[2]*10+[3]*10,kind='mixed');mixed['joins']=np.array([2,10,18])
    pred={'n1':np.full(65,3),'n2':np.full(65,2),'m':mixed['labels'].copy()}
    r,t=a.metrics(native+[mixed],pred)
    assert r['segment_f1_50_multi']==1 and r['boundary_recall_multi']==1 and r['same_class_cut_rate_multi']==0
    assert r['native_A4C_to_A5C']==1 and r['native_A5C_to_A4C']==1 and r['counts']['multi_same_joins']==2
    assert r['short_true_segments_10samples']==2 and r['short_true_unmatched_10samples']==0
    assert len(t['native'])==2
    with pytest.raises(ValueError):a.metrics(native+[mixed],{**pred,'m':np.zeros(19,int)})
def test_checkpoint_lexicographic_selection_is_inner_only_and_deterministic():
    a=api();rows=[{'arm':'J-high','step':2000,'segment_f1_50_multi':.7,'native_macro_f1':.9,'native_A4C_A5C_errors':3,'backbone_lr':3e-5},{'arm':'J-low','step':2000,'segment_f1_50_multi':.7,'native_macro_f1':.9,'native_A4C_A5C_errors':3,'backbone_lr':1e-5}]
    assert a.choose_inner(rows)['arm']=='J-low'
    rows[0]['segment_f1_50_multi']=.71
    assert a.choose_inner(rows)['arm']=='J-high'
