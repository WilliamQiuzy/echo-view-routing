import importlib
from types import SimpleNamespace
import numpy as np
import pytest

def test_duration_stress_reuses_identical_native_and_pair_predictions_only_replaces_multi():
    m=importlib.import_module('echo_routing.temporal.structured_metrics')
    def q(name,kind):return {'id':name,'kind':kind,'labels':np.array([0,1]),'refs':[('v',0,1.),('v',1,1.)],'joins':np.array([1])}
    qs=[q('n','native'),q('m','multi'),q('p','pair')];stress=[q('s','multi')];main={'sequences':qs,'eval_set':SimpleNamespace(natives=['n'],multi=['m'],pairs=['p'])};context=m.stress_context(main,stress,expected_streams=1);assert [x['id'] for x in context['sequences']]==['n','p','s'] and context['eval_set'].natives is main['eval_set'].natives and context['eval_set'].pairs is main['eval_set'].pairs
    original={x['id']:np.array([0,1],np.int8) for x in qs};other={'s':np.array([1,1],np.int8)};p=m.compose_stress_predictions(original,other,qs,stress);assert p['n'] is original['n'] and p['p'] is original['p'] and 'm' not in p
    with pytest.raises(ValueError):m.compose_stress_predictions(original,{},qs,stress)
