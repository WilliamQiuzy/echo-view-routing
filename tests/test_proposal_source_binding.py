import copy,json
from pathlib import Path
import pytest
from echo_routing.temporal.proposal_source_binding import validate

def test_exact_prior_source_contract_identity_not_just_N5_shape():
    root=Path(__file__).resolve().parents[1];pub=root/'docs/results/2026-11-03_jepa_proposals/prior_A59_public'
    cfg=json.loads((root/'configs/experiments/anchor_edits_v1.json').read_text());producer=json.loads((pub/'producer_identity.json').read_text());inputs=json.loads((pub/'input_manifest.json').read_text());rows=[]
    for n in ['cache_fit_manifest.json','cache_outer_manifest.json']:rows+=json.loads((pub/n).read_text())['records']
    assert validate(cfg['source'],cfg,producer,inputs,rows)['source_role_contracts_recomputed']==16
    bad=copy.deepcopy(cfg['source']);bad['a08']['stride']=2
    with pytest.raises(ValueError):validate(bad,cfg,producer,inputs,rows)
    bad=copy.deepcopy(rows);bad[0]['identity']='0'*64
    with pytest.raises(ValueError):validate(cfg['source'],cfg,producer,inputs,bad)
