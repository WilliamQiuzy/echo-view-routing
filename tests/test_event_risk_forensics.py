"""Anonymous native and development-regression recounts; synthetic axes only."""
import unittest
import numpy as np
from echo_routing.temporal.proposal_actions import propose,partition
from echo_routing.temporal.event_risk_policy import decode
try:
 from echo_routing.temporal.event_risk_forensics import native_cube,native_reference,regression_records,regression_rle
except ModuleNotFoundError as e:
 if e.name!='echo_routing.temporal.event_risk_forensics':raise
 native_cube=native_reference=regression_records=regression_rle=None

class EventForensicsTests(unittest.TestCase):
 def setUp(self):self.assertTrue(callable(native_cube),'A62 forensics not implemented yet')
 def value(self,base,raw,origin='old',positive=True):
  aa=raw if origin=='old' else base;jj=raw if origin=='jepa' else base
  all_actions=propose(base,aa,jj);actions,rejected=partition(base,all_actions);K=int(np.count_nonzero(base!=raw))
  scores=np.array([[1. if positive else -1.,.01,.02,K-.03,.05,.05]])
  q=np.array([[.03,.2]]);taus={'L0':.08,'M0':.04,'R1':.04,'R2':.1}
  return {'actions':actions,'all_actions':all_actions,'G_rejected':rejected,'old_scores':scores,'q':q,**decode(base,actions,scores,q[:,0],q[:,1],taus)}
 def test_native_cube_actual_H_not_net_majority_and_Base_reference(self):
  base=np.zeros(100,dtype=int);raw=base.copy();raw[10:20]=1;q={'id':'private_synthetic_id_never_export','kind':'native','labels':base.copy()};v=self.value(base,raw)
  a=native_cube([q],[base],[v],'L0',.9);b=native_cube([q],[base],[v],'R2',.9);ref=native_reference([q],[base],.9)
  cube=np.asarray(a['true_Base_prediction_joint5cube']);self.assertEqual(cube.shape,(5,5,5));self.assertEqual(cube[0,0,1],10)
  self.assertEqual(a['Base_correct_to_wrong'],10);self.assertEqual(a['accepted_native_actions'],1);self.assertEqual(a['majority_changed_native_sequences'],0)
  self.assertEqual(b['Base_correct_to_wrong'],0);self.assertEqual(b['accepted_native_actions'],0)
  self.assertEqual(ref['positions'],100);self.assertEqual(ref['Base_frame_errors'],0);self.assertEqual(np.asarray(ref['true_Base_prediction_joint5cube'])[0,0,0],100)
  self.assertNotIn(q['id'],str(a));self.assertEqual(a['accepted_groups'][0]['provenance'],'old_only')
  bad={**v,'L0':{**v['L0'],'prediction':base.copy()}}
  with self.assertRaises(ValueError):native_cube([q],[base],[bad],'L0',.9)
 def test_regression_candidate_H9_R2_and_actual_refusal_are_distinct(self):
  base=np.zeros(100,dtype=int);raw=base.copy();raw[:11]=1;truth=base.copy();truth[:2]=1;q={'id':'private_case_source','kind':'multi','labels':truth};v=self.value(base,raw,'jepa');a=np.eye(5)[base];j=np.eye(5)[raw]
  rows=regression_records('Case_B',q,base,a,j,v,2026110461,.9)
  self.assertEqual(len(rows),4);r=next(r for r in rows if r['policy']=='R2');old=next(r for r in rows if r['policy']=='L0')
  self.assertEqual((r['H'],r['R']),(9,2));self.assertFalse(r['accepted']);self.assertEqual((r['actual_H'],r['actual_R']),(0,0))
  self.assertEqual((old['actual_H'],old['actual_R']),(9,2));self.assertTrue(old['accepted'])
  self.assertAlmostEqual(r['context_summary']['minimum_overlap_over_full_run'],.11);self.assertEqual(r['context_summary']['minimum_full_Base_run_A08_margin'],1.)
  changed=regression_records('Case_B',{**q,'labels':base},base,a,j,v,2026110461,.9)
  self.assertEqual(rows[0]['context_summary'],changed[0]['context_summary']);self.assertNotIn(q['id'],str(rows));self.assertNotIn('truth_RLE',str(rows))
  private=regression_rle(q,base,v);self.assertEqual(set(private),{'truth','Base','predictions','scope'});self.assertEqual(len(private['predictions']),6)
  with self.assertRaises(ValueError):regression_records(q['id'],q,base,a,j,v,2026110461,.9)
 def test_no_positive_top_retains_four_explicit_missing_candidate_records(self):
  base=np.zeros(100,dtype=int);raw=base.copy();raw[:11]=1;q={'id':'not_public','kind':'multi','labels':base};v=self.value(base,raw,'jepa',positive=False)
  rows=regression_records('Case_A',q,base,np.eye(5)[base],np.eye(5)[raw],v,2026110462,.9)
  self.assertEqual(len(rows),4)
  for r in rows:
   self.assertFalse(r['top_exists']);self.assertFalse(r['accepted']);self.assertIsNone(r['H']);self.assertIsNone(r['R']);self.assertIsNone(r['gate_score']);self.assertIsNone(r['context_summary']);self.assertEqual(r['actual_H'],0)

if __name__=='__main__':unittest.main()
