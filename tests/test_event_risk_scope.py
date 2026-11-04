"""A62 fixed endpoint and post-seal support contracts; synthetic labels only."""
import itertools
import unittest
import numpy as np
from echo_routing.temporal.proposal_actions import canonical
try:
    from echo_routing.temporal.event_risk_metrics import plan, support, risk_summary, primary_intervals
except ModuleNotFoundError as exc:
    if exc.name != 'echo_routing.temporal.event_risk_metrics': raise
    plan = support = risk_summary = primary_intervals = None


class EventRiskScopeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(plan), 'A62 metric/support graph not implemented yet')

    def value(self, actions, top=0, admitted=True):
        return {'actions': actions, 'N': {'top_index': top, 'accepted_index': top if admitted else None}}

    def test_complete_unique_endpoints_and_exact_matched_primary_family(self):
        p = plan()
        self.assertEqual(len(p['rows']),156)
        self.assertEqual(len(p['inner_rows']),26)
        self.assertEqual(len(p['comparisons']),240)
        for group, fields in [('rows',('table','half','system')),('inner_rows',('population','system')),('comparisons',('table','half','seed','contrast')),('native_cells',('half','seed','policy')),('new_oracle_cells',('half','safety')),('reused_oracle_cells',('table','half','safety')),('support_cells',('table','half','seed','safety')),('regression_records',('axis','seed','policy'))]:
            rows=p[group]
            self.assertEqual(len(rows),len({tuple(r[k] for k in fields) for r in rows}),group)
        self.assertEqual([len(p[k]) for k in ['native_cells','base_native_references','new_oracle_cells','reused_oracle_cells','support_cells','regression_records']],[24,2,6,30,48,16])
        primary=[r for r in p['comparisons'] if r['primary_family']]
        self.assertEqual(len(primary),8)
        self.assertEqual({r['contrast'] for r in primary},{'R1-M0','R2-R1'})
        self.assertTrue(all(r['table']=='a62_confirmation' and r['bootstrap_replicates']==5000 for r in primary))
        self.assertEqual({r['axis'] for r in p['regression_records']},{'Case_A','Case_B'})
        for table in p['tables']:
            for half in ['selection','calibration']:
                rs=[r for r in p['rows'] if r['table']==table and r['half']==half]
                self.assertEqual(len(rs),13)
                self.assertEqual(sum(r['role']=='reference' for r in rs),1)
        self.assertFalse(p['checkpoint_or_seed_selection'])

    def test_support_cannot_choose_beneficial_second_candidate_and_separates_short_loss(self):
        truth=np.array([0]*10+[1]*10+[0]*20)
        base=np.array([0]*10+[2,3]*5+[0]*20)
        unsafe_short=canonical(base,10,20,[0]*10,jepa=True)
        useful_second=canonical(base,10,20,[1]*10,jepa=True)
        q={'id':'synthetic_multi','kind':'multi','labels':truth}
        v=self.value([unsafe_short,useful_second])
        event=support([q],[base],[v],'event-safe',.5,private_details=True)
        strict=support([q],[base],[v],'acceptance-safe',.5,private_details=True)
        self.assertEqual(event['multi']['N_admitted_tops'],1)
        self.assertEqual(event['multi']['available_tops'],1)
        self.assertEqual(event['multi']['all_available_top_effects']['H'],0)
        self.assertEqual(event['multi']['all_available_top_effects']['R'],0)
        self.assertEqual(event['multi']['all_available_top_effects']['short16_lost'],1)
        self.assertEqual(event['multi']['oracle']['F1'],.5)
        self.assertEqual(event['multi']['joint_chosen_effects']['short16_lost'],1)
        self.assertEqual(event['private_details']['multi_chosen_accepted_indices'],[0])
        self.assertEqual(strict['multi']['available_tops'],0)
        self.assertEqual(strict['multi']['oracle']['M'],2)
        self.assertEqual(strict['multi']['joint_chosen_effects']['short16_lost'],0)
        self.assertEqual(strict['private_details']['multi_chosen_accepted_indices'],[None])
        self.assertEqual(strict['multi']['union_recovered_short16_true_units'],0)
        self.assertNotIn('private_details',support([q],[base],[v],'event-safe',.5))

    def test_exact_noop_tie_and_no_N_or_positive_top_never_fallback(self):
        truth=np.zeros(20,dtype=int);base=np.array([0]*10+[1]*5+[0]*5)
        tie=canonical(base,10,15,[2]*5,jepa=True)
        q={'id':'synthetic_tie','kind':'multi','labels':truth}
        r=support([q],[base],[self.value([tie])],'event-safe',.5,private_details=True)
        self.assertEqual(r['multi']['available_tops'],1)
        self.assertEqual(r['multi']['oracle']['F1'],.5)
        self.assertEqual(r['private_details']['multi_chosen_accepted_indices'],[None])
        for v in [self.value([tie],admitted=False),self.value([tie],top=None,admitted=False)]:
            r=support([q],[base],[v],'event-safe',.5)
            self.assertEqual(r['multi']['available_tops'],0)
        with self.assertRaises(ValueError):support([q],[base],[{'actions':[tie],'N':{'top_index':0,'accepted_index':1}}],'event-safe',.5)
        with self.assertRaises(ValueError):support([q],[base],[self.value([tie])],'gross-safe',.5)

    def test_pooled_objective_excludes_native_pair_and_reports_unknown_empty_ratio(self):
        truth=np.array([0]*10+[1]*10+[0]*20);base=np.array([0]*10+[2,3]*5+[0]*20)
        action=canonical(base,10,20,[1]*10,jepa=True)
        qs=[{'id':'synthetic_'+k,'kind':k,'labels':truth} for k in ['multi','native','pair']]
        r=support(qs,[base]*3,[self.value([action])]*3,'acceptance-safe',.5,private_details=True)
        self.assertEqual([r[k]['available_tops'] for k in ['multi','native','pair']],[1,1,1])
        self.assertEqual(r['multi']['oracle']['T'],3)
        self.assertEqual(r['multi']['oracle']['F1'],1.)
        self.assertEqual(r['multi']['joint_chosen_effects']['R'],10)
        self.assertEqual(r['multi']['joint_chosen_effects']['JEPA_exclusive_short16_recovered'],1)
        self.assertEqual(r['native']['all_available_top_effects']['R'],10)
        empty=support(qs[1:],[base]*2,[self.value([action])]*2,'event-safe',.5)
        self.assertIsNone(empty['multi']['oracle'])
        self.assertEqual(empty['multi']['axes'],0)
        with self.assertRaises(ValueError):support(qs,[base]*2,[self.value([action])]*3,'event-safe',.5)

    def test_event_risk_bins_include_one_and_saturated_infinite_BCE_is_explicit(self):
        r=risk_summary([0.,.2,.8,1.],[False,True,False,True])
        self.assertEqual([b['count'] for b in r['bins']],[1,1,0,0,2])
        self.assertEqual([b['unsafe'] for b in r['bins']],[0,1,0,0,1])
        self.assertAlmostEqual(r['Brier'],(.8**2+.8**2)/4)
        self.assertIsNotNone(r['BCE'])
        bad=risk_summary([0.,1.],[True,False])
        self.assertIsNone(bad['BCE'])
        self.assertEqual(bad['infinite_BCE_rows'],2)
        self.assertIsNone(risk_summary([],[])['Brier'])
        with self.assertRaises(ValueError):risk_summary([float('nan')],[False])
        with self.assertRaises(ValueError):risk_summary([.5],[.25])

    def test_same_bootstrap_samples_primary_quantiles_and_null_ratios(self):
        samples=np.arange(5000,dtype=float)
        out=primary_intervals({'F1':samples,'ratio':np.r_[np.nan,np.arange(4999,dtype=float)]},primary=True)
        self.assertEqual(out['bootstrap_replicates'],5000)
        np.testing.assert_array_equal(out['delta95']['F1'],np.quantile(samples,[.025,.975]))
        np.testing.assert_array_equal(out['delta99_375']['F1'],np.quantile(samples,[.003125,.996875]))
        np.testing.assert_array_equal(out['delta95']['ratio'],np.quantile(np.arange(4999,dtype=float),[.025,.975]))
        self.assertEqual(out['undefined_bootstrap_replicates']['ratio'],1)
        self.assertNotIn('delta99_375',primary_intervals({'F1':samples},primary=False))
        with self.assertRaises(ValueError):primary_intervals({'F1':samples[:4999]},primary=True)


class PairedSampleTests(unittest.TestCase):
    def test_once_generated_whole_units_match_literal_pooled_resampling(self):
        from echo_routing.temporal.event_risk_metrics import paired
        streams={k:[(k+'0',np.array([2,3,3,40,30,1,2,3,0,0,2,1,3,1,2,1,3,1])),(k+'1',np.array([1,2,2,20,15,0,0,0,0,0,1,0,2,0,1,0,2,0]))] for k in ['multi','pair']}
        rightstreams={k:[(key,np.array([1,*vec[1:]])) for key,vec in rows] for k,rows in streams.items()}
        common={'session_order':['s0','s1'],'session_confusion':np.stack([np.eye(5,dtype=int)]*2),'session_extra':np.array([[2,40,30,1],[1,20,15,1]]),'native':{'n0':(0,0),'n1':(1,1)},'true_lengths':{k+str(i):[10,20,30] for k in ['multi','pair'] for i in range(2)},'matched':{k:{k+str(i):{0,1} for i in range(2)} for k in ['multi','pair']},'session_gross':np.array([[0,4],[0,2]]),'stream_gross':{k:[(k+'0',np.array([0,4])),(k+'1',np.array([0,2]))] for k in ['multi','pair']}}
        left={**common,'streams':streams};right={**common,'streams':rightstreams}
        out=paired(left,right,primary=True,seed=19)
        rng=np.random.default_rng(19);rng.integers(0,2,(5000,2));indices=rng.integers(0,2,(5000,2))
        aa=np.stack([v for _,v in streams['multi']])[indices].sum(1);bb=np.stack([v for _,v in rightstreams['multi']])[indices].sum(1)
        delta=2*aa[:,0]/(aa[:,1]+aa[:,2])-2*bb[:,0]/(bb[:,1]+bb[:,2])
        np.testing.assert_array_equal(out['delta95']['segment_f1_50_multi'],np.quantile(delta,[.025,.975]))
        np.testing.assert_array_equal(out['delta99_375']['segment_f1_50_multi'],np.quantile(delta,[.003125,.996875]))
        self.assertEqual(out['undefined_bootstrap_replicates']['same_class_cut_rate_multi'],5000)
        self.assertIsNone(out['delta95']['same_class_cut_rate_multi'])
        self.assertEqual(out['bootstrap']['multi_streams'],2)
        self.assertEqual(out['exact_disagreements']['multi_true_segments_recovered'],0)
        with self.assertRaises(ValueError):paired(left,{**right,'session_order':['s1','s0']},primary=True)

if __name__=='__main__':unittest.main()
