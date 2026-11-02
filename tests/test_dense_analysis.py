import importlib
import numpy as np

def test_legacy16_and_declared10_short_definitions_share_exact_cached_matching():
 a=importlib.import_module('echo_routing.temporal.dense_analysis');q={'id':'stream','kind':'multi','labels':np.array([2]*14+[3]*16)};perfect={'stream':q['labels'].copy()};constant={'stream':np.full(30,3)}
 r=a.short_metrics([q],perfect,16);s=a.short_metrics([q],perfect,10)
 assert r=={'short_true_segments_16samples':2,'short_true_unmatched_16samples':0,'short_predicted_segments_16samples':2,'short_spurious_segments_16samples':0}
 assert s['short_true_segments_10samples']==0
 x=a.short_metrics([q],constant,16)
 assert x['short_true_unmatched_16samples']==1 and x['short_true_segments_16samples']==2 and x['short_predicted_segments_16samples']==0
