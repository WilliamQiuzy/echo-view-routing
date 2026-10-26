"""A35: the fixed EchoFine acceptance rule on synthetic probabilities."""
import numpy as np

import echo_routing.temporal.echofine_rule as er
from echo_routing.temporal.interval_recognition import runs


def onehot(col, n=47, w=0.6):
    q = np.full(n, (1 - w) / (n - 1)); q[col] = w; return q


def test_unmapped_top_keeps_base():
    for col in (er.COL_A5C_OUTFLOW, 8, 3, 29, 46):
        assert er.decide(onehot(col), er.A4C) == er.A4C and er.decide(onehot(col), er.A5C) == er.A5C


def test_full_classes_change_and_other_bases_never_change():
    assert er.decide(onehot(er.COL_A5C_FULL), er.A4C) == er.A5C and er.decide(onehot(er.COL_A4C_FULL), er.A5C) == er.A4C
    for base in (0, 1, 4):
        assert er.decide(onehot(er.COL_A5C_FULL), base) == base


def test_tie_keeps_base():
    q = np.zeros(47); q[er.COL_A4C_FULL] = q[er.COL_A5C_FULL] = 0.4; q[0] = 0.2
    assert er.decide(q, er.A4C) == er.A4C and er.decide(q, er.A5C) == er.A5C and er.top_category(q) == "tie"


def test_sample_multiplicity_and_merge():
    base = np.array([2, 2, 2, 3, 3, 0, 0])
    p = np.stack([onehot(er.COL_A5C_FULL), onehot(er.COL_A4C_FULL), onehot(er.COL_A5C_FULL),      # run 1: a5ch-full twice
                  onehot(er.COL_A5C_FULL), onehot(er.COL_A5C_FULL), onehot(er.COL_A4C_FULL), onehot(er.COL_A4C_FULL)])
    out = er.relabel(base, p, runs)
    assert out.tolist() == [3, 3, 3, 3, 3, 0, 0] and len(runs(out)) == 2                                  # merges with the next A5C run
    assert er.relabel(base, np.stack([onehot(er.COL_A4C_FULL)] * 7), runs).tolist() == [2, 2, 2, 2, 2, 0, 0]


def test_full_47_mean_not_two_columns():
    q = np.zeros(47); q[er.COL_A5C_FULL] = 0.3; q[er.COL_A4C_FULL] = 0.1; q[er.COL_A5C_OUTFLOW] = 0.6     # outflow dominates
    assert er.decide(q, er.A4C) == er.A4C and er.top_category(q) == "a5ch-outflow"
