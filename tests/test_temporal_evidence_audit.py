import numpy as np
from echo_routing.temporal.temporal_evidence_audit import singleton_view, informative, auc_partition, dense_gate


def test_origin_endpoint_and_native_axis_disagreements_remain_unknown():
    records=[{'start':10,'end':20,'true_label':'A4C'},{'start':20,'end':30,'true_label':'A5C'}]
    assert singleton_view(records,15,15)==0
    assert singleton_view(records,25,25)==1
    assert singleton_view(records,20,20) is None
    assert singleton_view(records,15,25) is None


def test_clipped_duplicate_slots_do_not_create_dense_evidence():
    assert not informative([0,0,1,1,None,None,None,None],[0,0,1,1,2,3,4,5],0)
    assert informative([0,0,1,1,None,None,None,None],list(range(8)),0)
    assert not informative([0,0,0,1,None,None,None,None],list(range(8)),0)


def test_auc_partition_reconstructs_exact_pair_masses():
    a=auc_partition([0,1,0,1],[0.,1.,2.,3.],['a','a','b','b'])
    assert a['global_AUC']==.75 and a['within_AUC']==1 and a['between_AUC']==.5
    assert a['within_pairs']==a['between_pairs']==2
    assert a['reconstructed_AUC']==a['global_AUC']
    b=auc_partition([0,1],[2.,1.],['a','b'])
    assert b['within_AUC'] is None and b['within_eligible_groups']==0
    assert b['between_AUC']==0


def test_dense_gate_cannot_be_satisfied_by_large_single_group():
    r={role:{'requests':100,'patients':1,'videos':1,'marker_classes':2} for role in ['fit','diagnostic']}
    assert not dense_gate(r)['passed']
