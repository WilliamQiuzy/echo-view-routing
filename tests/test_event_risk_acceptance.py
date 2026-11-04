"""Declared-goal checker tests; fabricated aggregates, not A62 efficacy."""
import copy,unittest
from echo_routing.temporal.event_risk_metrics import plan,BASE
try:
 from echo_routing.temporal.event_risk_acceptance import assess
except ModuleNotFoundError as e:
 if e.name!='echo_routing.temporal.event_risk_acceptance':raise
 assess=None

class EventRiskAcceptanceTests(unittest.TestCase):
 def setUp(self):self.assertTrue(callable(assess),'A62 scientific acceptance checker not implemented yet')
 def fixture(self):
  p=plan();rows=[]
  for r in p['rows']:
   policy=r.get('policy','Base');point={'R1':.91,'R2':.92}.get(policy,.9)
   recovered=2
   if r['table']=='a61_confirmation' and policy=='R2':recovered=([6,5] if r['half']=='selection' else [4,3])[[2026110461,2026110462].index(r['seed'])]
   rows.append({**r,'segment_f1_50_multi':point,'gross_reference_position_effects':{k:{'H':0,'R':0,'dE':0} for k in ['native','multi','pair']},'action_statistics_full_task':{'accepted_effect_totals':{'dFP':0},'accepted_short16_true_units_lost':0},'action_statistics_source_role':{'accepted_new_JEPA_short16_true_units_recovered':recovered}})
  ci=[{**r,'delta95':{'segment_f1_50_multi':[.001,.02]},**({'delta99_375':{'segment_f1_50_multi':[.0001,.025]}} if r['primary_family'] else {})} for r in p['comparisons']]
  info={str(s):{'N_admitted':30,'unsafe_N_top':8} for s in [2026110461,2026110462]}
  return rows,ci,info
 def test_all_prespecified_goals_and_both_matched_primary_contrasts_are_separate(self):
  out=assess(*self.fixture())
  self.assertTrue(out['endpoint_coverage_passed']);self.assertEqual(out['R2_joint_empirical_target']['evaluated_cells'],24)
  self.assertTrue(out['R2_joint_empirical_target']['passed'])
  self.assertTrue(out['attribution']['H1']['claim_supported']);self.assertTrue(out['attribution']['H2']['claim_supported'])
  self.assertEqual(out['primary_system'],'R2');self.assertFalse(out['choose_R1_after_R2_failure']);self.assertFalse(out['clinical_or_finite_sample_safety_guarantee'])
 def test_H9_even_net_repair_and_gross_short_loss_are_preserved_as_failures(self):
  rows,ci,info=self.fixture();r=next(r for r in rows if r.get('policy')=='R2' and r['table']=='a62_confirmation' and r['half']=='calibration')
  r['gross_reference_position_effects']['multi']={'H':9,'R':35,'dE':-26};r['action_statistics_full_task']['accepted_short16_true_units_lost']=2
  out=assess(rows,ci,info);self.assertFalse(out['R2_joint_empirical_target']['passed']);self.assertFalse(out['attribution']['H2']['claim_supported'])
  bad=next(r for r in out['R2_joint_empirical_target']['cells'] if r['table']=='a62_confirmation' and r['half']=='calibration' and r['seed']==2026110461)
  self.assertIn('H_zero',bad['failed']);self.assertIn('short16_gross_lost_zero',bad['failed']);self.assertEqual(bad['H'],9)
 def test_all_noop_cannot_win_and_R1_is_never_promoted(self):
  rows,ci,info=self.fixture()
  for r in rows:
   if r.get('policy')=='R2':r['action_statistics_source_role']['accepted_new_JEPA_short16_true_units_recovered']=0
  out=assess(rows,ci,info);self.assertFalse(out['R2_joint_empirical_target']['passed']);self.assertEqual(out['primary_system'],'R2')
  self.assertTrue(any('retained_A61_JEPA_short16_recovery' in c['failed'] for c in out['R2_joint_empirical_target']['cells']))
  self.assertTrue(any('new_A62_JEPA_short16_recovery' in c['failed'] for c in out['R2_joint_empirical_target']['cells']))
 def test_intervals_containing_zero_are_not_equivalence_and_CAL_limited_blocks_H1_identification(self):
  rows,ci,info=self.fixture()
  for r in ci:
   if r['primary_family']:r['delta99_375']['segment_f1_50_multi']=[-.01,.02]
  out=assess(rows,ci,info);self.assertFalse(out['attribution']['H1']['F1_increment_evidence_passed']);self.assertFalse(out['attribution']['H2']['claim_supported'])
  rows,ci,info=self.fixture();info['2026110461']={'N_admitted':19,'unsafe_N_top':4}
  out=assess(rows,ci,info);self.assertTrue(out['attribution']['H1']['F1_increment_evidence_passed']);self.assertFalse(out['attribution']['H1']['claim_supported']);self.assertTrue(out['calibration_limited'])
 def test_missing_or_duplicate_rows_and_CIs_fail_closed_instead_of_selective_reporting(self):
  rows,ci,info=self.fixture()
  for args in [(rows[:-1],ci,info),(rows,ci[:-1],info),(rows[:-1]+[rows[0]],ci,info)]:
   with self.assertRaises(ValueError):assess(*args)
  ci=copy.deepcopy(ci);r=next(r for r in ci if r['primary_family']);r.pop('delta99_375')
  with self.assertRaises(ValueError):assess(rows,ci,info)

if __name__=='__main__':unittest.main()
