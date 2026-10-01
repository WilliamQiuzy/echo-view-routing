"""EchoNet-Pediatric external protocol: name parsing, true-frame-rate sampling and the frozen stitch banks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from echo_routing.realvideo.pediatric import (digest_rows, exam_recipes, parse_name, recipe_group, recipes_digest,
                                              same_view_recipes, timestamp_indices)


def test_parse_name():
    assert parse_name("CR32a7555-CR32a7582-000039.avi") == ("CR32a7555", "CR32a7555-CR32a7582", "000039")
    with pytest.raises(ValueError):
        parse_name("bad_name.avi")


def test_timestamp_indices_follow_the_true_frame_rate():
    assert timestamp_indices(100, 50.0, 10.0).tolist() == list(range(0, 100, 5))
    idx = timestamp_indices(76, 76.0, 10.0)  # 1 s at 76 fps: nearest frames to 0.0, 0.1, ..., 0.9 s
    assert idx.tolist() == [0, 8, 15, 23, 30, 38, 46, 53, 61, 68]
    assert timestamp_indices(23, 23.0, 10.0).size == 10  # no integer-stride drift at low frame rates
    assert timestamp_indices(3, 50.0, 10.0).tolist() == [0]
    with pytest.raises(ValueError):
        timestamp_indices(0, 50.0, 10.0)


def _toy() -> pd.DataFrame:
    rows = [  # group a: one exam with PSAX, A4C, A4C; another exam with one A4C. group b: one exam, one clip
        ("a-e1-000003", "A4C", "a", "a-e1", "000003", 2), ("a-e1-000001", "PSAX", "a", "a-e1", "000001", 1),
        ("a-e1-000007", "A4C", "a", "a-e1", "000007", 2), ("a-e2-000002", "A4C", "a", "a-e2", "000002", 2),
        ("b-e9-000001", "PSAX", "b", "b-e9", "000001", 1)]
    return pd.DataFrame([{"video_id": v, "view": w, "group": g, "exam": e, "clip_no": c, "label": l, "n_samples": 12}
                         for v, w, g, e, c, l in rows])


def test_exam_recipes_follow_clip_order_and_use_each_video_once():
    rec = exam_recipes(_toy())
    assert [r.recipe_id for r in rec] == ["pedexam_a-e1"]
    assert [f.video_id for f in rec[0].fragments] == ["a-e1-000001", "a-e1-000003", "a-e1-000007"]
    assert [f.label for f in rec[0].fragments] == [1, 2, 2] and rec[0].fragments[0].end == 12
    assert recipe_group(rec[0]) == "a"


def test_same_view_pairs_stay_inside_one_group():
    rec = same_view_recipes(_toy())
    assert len(rec) == 1 and [f.video_id for f in rec[0].fragments] == ["a-e1-000003", "a-e1-000007"]
    used = [f.video_id for r in rec for f in r.fragments]
    assert len(used) == len(set(used))


def test_digests_are_order_sensitive_and_stable():
    df = _toy()
    assert digest_rows(df) == digest_rows(df.copy()) and digest_rows(df) != digest_rows(df.iloc[::-1])
    assert recipes_digest(exam_recipes(df)) == recipes_digest(exam_recipes(df))
    assert np.isfinite(len(recipes_digest([])))


def test_clip_numbers_are_ordered_as_integers():
    rows = [("g-e-10", "A4C", "10", 2), ("g-e-2", "PSAX", "2", 1), ("g-e-9", "A4C", "9", 2)]
    df = pd.DataFrame([{"video_id": v, "view": w, "group": "g", "exam": "g-e", "clip_no": c, "label": l, "n_samples": 5}
                       for v, w, c, l in rows])
    assert [f.video_id for f in exam_recipes(df)[0].fragments] == ["g-e-2", "g-e-9", "g-e-10"]
    assert [f.video_id for f in same_view_recipes(df)[0].fragments] == ["g-e-9", "g-e-10"]
