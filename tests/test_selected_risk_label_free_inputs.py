import importlib
import pytest

def api():
    try:return importlib.import_module('echo_routing.temporal.selected_risk_label_free_inputs')
    except ModuleNotFoundError:pytest.fail('label-free pinned native reader absent')

def test_native_alignment_never_observes_held_truth():
    class TruthBomb:
        def __array__(self,*a,**kw):raise AssertionError('held truth compared before selector seal')
        def __eq__(self,other):raise AssertionError('held truth compared before selector seal')
    rows=[{'id':'source_0','kind':'native','refs':[('source_0',0,1.)],'labels':TruthBomb()},
          {'id':'mixed','kind':'multi','refs':[],'labels':TruthBomb()}]
    assert api().native_indices(rows,[rows[0]])==[0]
    with pytest.raises(ValueError):api().native_indices(rows,[{**rows[0],'refs':[('source_0',1,1.)]}])
    with pytest.raises(ValueError):api().native_indices(rows,[rows[0],rows[0]])

def test_reader_rejects_unauthorized_role_and_purpose_before_IO():
    m=api()
    with pytest.raises(ValueError):m.load_bank('calibration',purpose='selector')
    with pytest.raises(ValueError):m.load_bank('FIT_native',purpose='effects')
