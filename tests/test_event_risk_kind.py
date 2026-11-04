import numpy as np,pytest

def test_explicit_frozen_legacy_pairs_alias_is_count_only_and_preserves_axis():
 from echo_routing.temporal.event_risk_kind import canonical_kind,support
 q={'id':'toy','kind':'pairs','labels':np.zeros(3,int)};v={'actions':[],'N':{'top_index':None,'accepted_index':None}}
 assert canonical_kind('pairs')=='pair'
 r=support([q],[np.zeros(3,int)],[v],'event-safe',.8)
 assert r['pair']['axes']==1 and r['multi']['oracle'] is None and q['kind']=='pairs'
 with pytest.raises(ValueError):canonical_kind('unqualified_other')
