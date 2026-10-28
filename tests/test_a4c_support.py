import importlib
import importlib.util
import copy
import numpy as np
import torch

def api():
    assert importlib.util.find_spec('echo_routing.temporal.a4c_support') is not None, 'A45 support implementation missing'
    return importlib.import_module('echo_routing.temporal.a4c_support')

def test_full_footprint_partition_excludes_cross_session_inputs():
    m=api();fp=[{'a'},{'a','b'},{'c'},{'b','c'}]
    assert m.partition_mask(fp,{'a','b'}).tolist()==[True,True,False,False]

def test_run_embedding_keeps_multiplicity_and_nonzero_offset():
    m=api();pk={'offsets':np.array([0,2,3]),'rows':np.array([10,11,11]),'weights':np.array([.25,.75,1.])}
    x=np.array([[0.,4.],[8.,0.]])
    np.testing.assert_array_equal(m.run_embeddings(pk,x,10),[[6.,1.],[8.,0.]])

def test_rank_threshold_ties_and_pvalues():
    m=api();cal=np.arange(199,dtype=float);threshold=m.calibrate(cal)
    assert threshold==197 and m.outlier(np.array([197.,198.,199.]),threshold).tolist()==[False,True,True]
    np.testing.assert_array_equal(m.pvalues(cal,np.array([197.,198.,199.])),[.015,.01,.005])

def test_insufficient_calibration_is_infinite_noop():
    m=api();assert np.isinf(m.calibrate(np.arange(98.)))
    assert not m.outlier(np.array([1e6]),m.calibrate(np.arange(98.)))[0]

def test_equal_calibration_scores_conservative_at_tie():
    m=api();cal=np.ones(199);assert m.calibrate(cal)==1
    assert m.pvalues(cal,np.array([1.,2.])).tolist()==[1.,.005]

def test_proposal_support_intersection_and_original_base_unchanged():
    m=api();b=np.array([0,1,2,2,2,3,4]);r=np.array([9,9,1,0,1,9,9]);e=np.array([9,9,2,9,1,9,9])
    np.testing.assert_array_equal(m.deploy(b,r,e,1),[0,1,3,2,2,3,4]);np.testing.assert_array_equal(b,[0,1,2,2,2,3,4])

def test_standardizer_uses_only_given_fit_rows_and_floor():
    m=api();x=np.array([[1.,0.],[3.,0.],[1000.,999.]])
    mu,sd=m.normalizer(x[:2]);np.testing.assert_array_equal(mu,[2.,0.]);np.testing.assert_array_equal(sd,[1.,.001])

def test_exact_autoencoder_capacity_and_real_reconstruction_update():
    m=api();torch.manual_seed(4);net=m.SupportAE();assert sum(p.numel() for p in net.parameters())==140064
    x=torch.randn(8,512);before=copy.deepcopy(net.state_dict()); opt=torch.optim.AdamW(net.parameters(),lr=.001)
    loss=((net(x)-x)**2).mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters());opt.step()
    assert all(not torch.equal(v,before[k]) for k,v in net.state_dict().items())

def test_fixed_plan_same_common_half_and_valid_class_exposure():
    m=api();a,u=m.plans(np.array([2,4,8]),np.array([3,9]),0,6)
    assert a.shape==u.shape==(6,32);np.testing.assert_array_equal(a[:,:16],u[:,:16]);assert set(a.ravel())<= {2,4,8};assert set(u[:,16:].ravel())<= {3,9}
    aa,uu=m.plans(np.array([2,4,8]),np.array([3,9]),0,6);np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(u,uu)

def test_reconstruction_error_independent_coordinate_loop():
    m=api();torch.manual_seed(1);net=m.SupportAE().eval();x=np.zeros((3,512),np.float32);mu=np.ones(512);sd=np.full(512,2.)
    e=m.errors(net,x,mu,sd);z=torch.full((3,512),-.5);p=net(z).detach().numpy()
    expected=np.array([sum((float(v)+.5)**2 for v in row)/512 for row in p]);np.testing.assert_allclose(e,expected,rtol=1e-6,atol=1e-8)
