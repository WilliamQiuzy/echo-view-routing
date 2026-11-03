import importlib,copy
import pytest

def test_fixed_bins_keep_equal_update_domain_means_and_exposure_without_private_fields():
    m=importlib.import_module('echo_routing.temporal.structured_curves');rows=[{'u':i,'loss':i+1.,'native_NLL':2.*i,'constructed_NLL':2.,'gradient_norm':i/2,'lr':.001/(i+1),'valid_original_positions':10+i,'step_wall_seconds_before_journal':.1,'plan_row':[i,5],'identity':'PRIVATE','model_digest_after_update':'PRIVATE'} for i in range(4)];bins=m.binned_curve(rows,block=2,expected_updates=4);assert len(bins)==2 and bins[0]['training_NLL_equal_update_mean']==1.5 and bins[0]['native_NLL_equal_update_mean']==1 and sum(b['valid_original_positions'] for b in bins)==46 and bins[1]['unclipped_gradient_norm_gt1_fraction']==.5;assert all('PRIVATE' not in str(b) and 'plan_row' not in b for b in bins)
    bad=copy.deepcopy(rows);bad[2]['u']=1
    with pytest.raises(ValueError):m.binned_curve(bad,block=2,expected_updates=4)
