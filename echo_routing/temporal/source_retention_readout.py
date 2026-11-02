"""A56 vector once-only inference with validated committed-prefix resumption."""
import numpy as np
from . import direct_view_training as d


def score_rows(cells,infer):
    if len(cells)!=5:raise ValueError('original five-logit columns required')
    records=[cell['record'] for cell in cells]
    if any(r['pending'] is not None for r in records):
        raise RuntimeError('possibly executed source row cannot be repeated')
    if len({(r['done'],r['n']) for r in records})!=1:
        raise RuntimeError('five-logit cells do not share a committed prefix')
    start,n=records[0]['done'],records[0]['n']
    for index in range(start,n):
        assert [d.claim_row(cell) for cell in cells]==[index]*5
        q=np.asarray(infer(index))
        if q.shape!=(5,) or not np.isfinite(q).all():raise ValueError('nonfinite/incomplete original five logits')
        for c,cell in enumerate(cells):d.finish_row(cell,index,float(q[c]))
    for cell in cells:d.seal_cell(cell)
    return n-start
