import importlib,itertools
import numpy as np
from echo_routing.temporal.proposal_oracle import action_options as old_options,solve as old_solve


def test_net_safe_can_trade_gross_harm_but_gross_safe_cannot_and_unrestricted_reproduces():
    m=importlib.import_module('echo_routing.temporal.safety_oracle');b=np.r_[np.zeros(5,int),np.ones(3,int),np.zeros(5,int)];y=np.zeros(13,int);a=b.copy();a[4]=1;a[5:8]=0;j=b.copy()
    un=m.options(b,a,j,y,'old','unrestricted');net=m.options(b,a,j,y,'old','net-safe');gross=m.options(b,a,j,y,'old','gross-safe')
    assert len(un)==len(net)==2 and len(gross)==1
    r=m.solve([net],'old','net-safe');g=m.solve([gross],'old','gross-safe');assert r['F1']>g['F1']
    old=old_solve([old_options(b,a,j,y,'old',True)],'old','G');new=m.solve([un],'old','unrestricted')
    assert all(old[k]==new[k] for k in ['M','P','T','numerator','denominator','F1'])
    assert m.coverage(b,a,j,y,'old','gross-safe')['candidate_union_recovered_short16_true_units']==0


def test_safe_fractional_solution_matches_exhaustive_one_action_per_stream():
    m=importlib.import_module('echo_routing.temporal.safety_oracle');b=np.zeros(6,int);y=np.array([0,0,1,1,0,0]);j=y.copy();a=b.copy()
    pools=[m.options(b,a,j,y,'expanded','gross-safe'),m.options(b,a,j,y,'expanded','gross-safe')]
    solved=m.solve(pools,'expanded','gross-safe');best=max(2*sum(r['M'] for r in pair)/(sum(r['P'] for r in pair)+sum(r['T'] for r in pair)) for pair in itertools.product(*pools))
    assert solved['F1']==best and solved['guard']=='G' and solved['safety']=='gross-safe'
