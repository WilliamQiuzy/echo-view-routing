"""Whole-clip chunks for the action-budget readout, and the pre-written real-order rule for A60."""
CHUNK_POSITIONS = 124
MIN_CLEAR_CELLS = 2


def chunk_edges(spans, minimum=CHUNK_POSITIONS):
    """Join consecutive (id, start, end) clips until a chunk has at least this many positions. A short tail joins the last chunk."""
    if not spans:
        raise ValueError('at least one clip span required')
    edges, start = [], spans[0][1]
    for _, _, end in spans:
        if end - start >= minimum:
            edges.append((start, end))
            start = end
    final = spans[-1][2]
    if start < final:
        if edges:
            edges[-1] = (edges[-1][0], final)
        else:
            edges.append((start, final))
    return edges


def survives(cells):
    """All cells above zero, at least two with a clear interval, and no cell with more frame errors."""
    return (all(c['delta'] > 0 for c in cells) and sum(c['ci_low'] > 0 for c in cells) >= MIN_CLEAR_CELLS
            and all(c['frame_errors'] <= 0 for c in cells))
