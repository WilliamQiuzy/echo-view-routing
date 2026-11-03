import copy
import pytest
from echo_routing.temporal.anchor_source_binding import validate


def test_all12_source_journals_inputs_and_axes_must_match_pinned_manifest():
    roles=['selection','calibration','old_stress_selection','old_stress_calibration','confirmation_selection','confirmation_calibration']
    observed={(s,r):{'identity':'source-'+s+r,'journal_sha256':s+r,'committed_sequences':120,'input_sha256':r} for s in ['a08','jepa'] for r in roles}
    rows=[{'source':s,'role':r,'complete':True,**v} for (s,r),v in observed.items()]
    assert validate(rows,observed)
    for bad in [rows[:-1],rows+[rows[0]]]:
        with pytest.raises(ValueError):validate(bad,observed)
    for field in ['identity','journal_sha256','committed_sequences','input_sha256','complete']:
        bad=copy.deepcopy(rows);bad[0][field]='changed'
        with pytest.raises(ValueError):validate(bad,observed)
