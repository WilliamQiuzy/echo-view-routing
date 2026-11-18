"""The published development numbers, from the published weights on the real dataset.

Needs the weights, a CUDA GPU and EV9V:  ECHOVIEW_EV9V=/data/EV9V/Images pytest -m "weights and gpu and ev9v"
The run takes about 18 minutes on one L40S.
"""
import os

import pytest

from echoview import Ensemble
from echoview.evaluation import benchmark, ev9v, run
from echoview.reference import B3

pytestmark = [pytest.mark.weights, pytest.mark.gpu, pytest.mark.ev9v]

PUBLISHED = {  # system: (selection F1@50, calibration F1@50, half window of the rule's choice)
    "C2_s0": (0.9584, 0.9856, 7),
    "C2_s1": (0.9476, 0.9827, 5),
    "C2_s2": (0.9483, 0.9778, 5),
    "ensemble": (0.9594, 0.9837, 7),
}


@pytest.fixture(scope="module")
def report() -> dict:
    import torch

    images = os.environ.get("ECHOVIEW_EV9V")
    if not images:
        pytest.skip("set ECHOVIEW_EV9V to the Images folder of EV9V")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    halves = benchmark.load()
    tables = {half: ev9v.FrameTable.build(halves[half], images) for half in benchmark.HALVES}
    return run.evaluate_reference(Ensemble(device="cuda", batch=32), halves, tables)


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_a_system_reaches_its_published_numbers(report, name):
    selection, calibration, half_window = PUBLISHED[name]
    entry = report["systems"][name]
    assert entry["b3"]["status"] == "feasible" and entry["b3"]["params"]["half_window"] == half_window
    assert round(entry["selection"]["system"]["segment_f1_50_multi"], 4) == selection
    assert round(entry["calibration"]["system"]["segment_f1_50_multi"], 4) == calibration


def test_the_published_setting_gives_the_numbers_of_the_rule(report):
    rule, published = report["systems"]["ensemble"], report["systems"]["ensemble_published_setting"]
    assert published["b3"]["params"] == B3
    for half in benchmark.HALVES:
        assert published[half]["system"] == rule[half]["system"]
