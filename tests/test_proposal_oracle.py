import itertools
from fractions import Fraction
import numpy as np
from echo_routing.temporal.proposal_oracle import action_options,solve,missed_coverage


def test_expanded_single_action_oracle_exhaustive_and_old_domain_subset():
    base=np.zeros(10,int);ra=base.copy();rj=np.array([0]*4+[2]*3+[0]*3);truth=rj.copy()
    old=action_options(base,ra,rj,truth,'old',True);expanded=action_options(base,ra,rj,truth,'expanded',True)
    assert len(old)==1 and len(expanded)>1
    options=[expanded,expanded]
    exact=max(Fraction(2*sum(x['M'] for x in choice),sum(x['P']+x['T'] for x in choice)) for choice in itertools.product(*options))
    result=solve(options,'expanded','G')
    assert Fraction(result['numerator'],result['denominator'])==exact==1
    assert 'raw-supported declared domain only' not in result['scope'] and result['domain']=='expanded'
    assert solve([old],'old','G')['F1']==0
    coverage=missed_coverage(base,ra,rj,truth,'expanded',True)
    assert coverage['candidate_union_recovered_true_units']==3 and not coverage['union_is_jointly_reachable_claim']
