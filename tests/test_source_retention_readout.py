import importlib
import numpy as np
import pytest
from echo_routing.temporal import direct_view_training as d


def test_vector_reader_resumes_prefix_without_duplicate_model_forward(tmp_path):
    cells=[d.open_cell(tmp_path,str(c),{'column':c},3) for c in range(5)]
    for cell in cells:d.finish_row(cell,d.claim_row(cell),10.)
    for cell in cells:d.close_cell(cell)
    cells=[d.open_cell(tmp_path,str(c),{'column':c},3) for c in range(5)]
    calls=[]
    fn=importlib.import_module('echo_routing.temporal.source_retention_readout').score_rows
    fn(cells,lambda i:(calls.append(i) or np.arange(5)+i))
    assert calls==[1,2]
    fn(cells,lambda i:pytest.fail('complete cell must never forward'))
    assert all(np.array_equal(c['scores'],[10.,1.+j,2.+j]) for j,c in enumerate(cells))
    for cell in cells:d.close_cell(cell)


@pytest.mark.parametrize('pending',[False,True])
def test_vector_reader_rejects_mismatched_or_ambiguous_prefix_before_claim(tmp_path,pending):
    cells=[d.open_cell(tmp_path,str(c),{'column':c},3) for c in range(5)]
    if pending:d.claim_row(cells[0])
    else:d.finish_row(cells[0],d.claim_row(cells[0]),1.)
    before=[dict(c['record']) for c in cells]
    fn=importlib.import_module('echo_routing.temporal.source_retention_readout').score_rows
    with pytest.raises(RuntimeError):fn(cells,lambda i:pytest.fail('invalid prefix must never forward'))
    assert [c['record'] for c in cells]==before
    for cell in cells:d.close_cell(cell)
