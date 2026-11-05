"""The declared B3 grid and selection rule, with real-order streams in place of the constructed bank."""
HALF_WINDOWS = (5, 7, 9, 11, 15)
QUANTILES = (.80, .85, .90, .95, .97, .99, .995, .999)
NMS_RADIUS = 5
MIN_NATIVE_MACRO_F1 = .98
MIN_CHANGE_RECALL = .95


def settings():
    """Grid entries in their declared order: half_window, then quantile, both ascending."""
    return [{'half_window': h, 'quantile': q, 'nms_radius': NMS_RADIUS} for h in HALF_WINDOWS for q in QUANTILES]


def select(rows):
    """Highest segment F1 among feasible rows; an equal score goes to the earlier row. Rows arrive in grid order."""
    feasible = [r for r in rows if r['native_macro_f1'] >= MIN_NATIVE_MACRO_F1 and r['change_recall'] >= MIN_CHANGE_RECALL]
    pool = feasible or rows
    best = max(r['segment_f1'] for r in pool)
    chosen = next(r for r in pool if r['segment_f1'] == best)
    return {**chosen, 'infeasible': not feasible}
