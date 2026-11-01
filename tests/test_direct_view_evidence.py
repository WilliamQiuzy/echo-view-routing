import copy

import pytest

from echo_routing.temporal import direct_view_evidence as dv


def source_rows():
    meta = [
        {"video_filename": "v4", "view": "A4C", "patient_id": "p1"},
        {"video_filename": "v5", "view": "A5C", "patient_id": "p2"},
    ]
    ann = [
        {"Video_FileName": "v4.avi", "View_Folder": "A4C", "Track": "view", "Start_Frame": "0", "End_Frame": "200", "Label": "A5C"},
        {"Video_FileName": "v5.avi", "View_Folder": "A5C", "Track": "view", "Start_Frame": "0", "End_Frame": "200", "Label": "A4C"},
    ]
    return meta, ann


def test_exact_join_rejects_ambiguous_owner_and_unmatched_folder():
    meta, ann = source_rows()
    records = dv.join_annotations(meta, ann)
    assert len(records) == 2
    assert records[0]["patient"] == "p1"
    assert records[0]["true_label"] == "A5C"
    assert records[0]["directory_label"] == "A4C"
    with pytest.raises(ValueError, match="owner"):
        dv.join_annotations(meta + [{**meta[0], "patient_id": "other"}], ann)
    bad = copy.deepcopy(ann)
    bad[0]["View_Folder"] = "A5C"
    with pytest.raises(ValueError, match="folder"):
        dv.join_annotations(meta, bad)


def test_conflicting_overlap_and_bad_frame_bounds_rejected():
    meta, ann = source_rows()
    extra = {**ann[0], "Start_Frame": "100", "Label": "A4C"}
    with pytest.raises(ValueError, match="overlap"):
        dv.join_annotations(meta, ann + [extra])
    with pytest.raises(ValueError, match="bounds"):
        dv.join_annotations(meta, [{**ann[0], "End_Frame": "0"}])


def test_common_interior_does_not_guess_index_origin():
    assert dv.common_interior(0, 10) == (1, 9)
    for raw in range(1, 9):
        for origin in (0, 1):
            for inclusive in (False, True):
                stop = 10 - origin + int(inclusive)
                assert 0 - origin <= raw < stop


def test_requests_keep_all_16_samples_in_constant_view_interior():
    meta, ann = source_rows()
    rec = dv.join_annotations(meta, ann)[0]
    q = dv.interval_requests(rec)
    assert 1 <= len(q) <= 8
    assert len({tuple(x["frames"]) for x in q}) == len(q)
    assert all(len(x["frames"]) == 16 for x in q)
    assert all(x["frames"][8] == x["centre"] for x in q)
    assert all(min(x["frames"]) >= 1 and max(x["frames"]) < 199 for x in q)
    assert all(x["true_label"] != x["directory_label"] for x in q)
    assert dv.interval_requests({**rec, "start": 1, "end": 30}) == []


def test_reservation_includes_boundary_ties_and_all_views():
    recs = [
        {"patient": p, "start": 0, "end": n, "true_label": c}
        for p, n, c in [("p1", 205, "NO_VIEW"), ("p2", 205, "PLAX"), ("p3", 25, "A4C"), ("p4", 25, "A5C")]
    ]
    out = dv.reserve_publisher_candidates(recs)
    assert {"p1", "p2"} <= set(out["reserved"])
    assert set(out["publisher_test_union"]) <= set(out["reserved"])
    assert out["exact_publisher_split_proved"] is False


def test_patient_split_is_order_independent_and_disjoint():
    patients = [f"p{x}" for x in range(40)]
    a = dv.patient_split(patients)
    b = dv.patient_split(list(reversed(patients)))
    assert a == b
    assert len(a["diagnostic"]) == 8
    assert not set(a["fit"]) & set(a["diagnostic"])
    assert set(a["fit"]) | set(a["diagnostic"]) == set(patients)


def test_source_qualification_cannot_promote_unchecked_partition():
    counts = {"fit_patients": 30, "diagnostic_patients": 8, "fit_videos_A4C": 20, "fit_videos_A5C": 20, "diagnostic_videos_A4C": 6, "diagnostic_videos_A5C": 6, "fit_disagreement_patients": 10, "diagnostic_disagreement_patients": 3}
    checks = {k: True for k in dv.REQUIRED_CHECKS}
    assert dv.source_qualified(counts, checks)["passed"]
    checks["source_partition"] = False
    result = dv.source_qualified(counts, checks)
    assert not result["passed"]
    assert "source_partition" in result["failed"]


def test_source_signal_gate_checks_damage_and_disagreement():
    a = {"balanced_accuracy": .90, "a4c_recall": .90, "a5c_recall": .90, "a4c_fpr": .10, "discordant_accuracy": .80}
    w = {**a, "balanced_accuracy": .80, "a4c_fpr": .12, "discordant_accuracy": .65}
    assert dv.source_signal_gate(a, w, .6, .5)["passed"]
    assert not dv.source_signal_gate({**a, "a4c_fpr": .13}, w, .6, .5)["passed"]
    assert not dv.source_signal_gate(a, {**w, "discordant_accuracy": .81}, .6, .5)["passed"]


def test_marker_requests_preserve_short_interval_labels_and_context():
    meta, ann = source_rows()
    rec = {**dv.join_annotations(meta, ann)[0], "start": 100, "end": 110}
    assert dv.interval_requests(rec) == []
    q = dv.marker_requests(rec, n_frames=200)
    assert len(q) == 1
    assert q[0]["centre"] == 101 and q[0]["frames"][8] == 101
    assert q[0]["true_label"] == "A5C"
    assert min(q[0]["frames"]) < rec["start"]
    assert max(q[0]["frames"]) > rec["end"]


def test_marker_clipping_and_bounds_never_moves_marker_or_invents_labels():
    meta, ann = source_rows()
    rec = {**dv.join_annotations(meta, ann)[0], "start": 0, "end": 10}
    q = dv.marker_requests(rec, n_frames=10)
    assert q[0]["frames"][0] == 0 and q[0]["frames"][-1] == 9
    assert q[0]["frames"][8] == q[0]["centre"] == 1
    assert q[0]["clipped_slots"] > 0
    assert dv.marker_requests({**rec, "true_label": "NO_VIEW"}, n_frames=10) == []
    with pytest.raises(ValueError, match="marker"):
        dv.marker_requests({**rec, "start": 10, "end": 20}, n_frames=10)


def test_marker_request_cap_is_even_and_deterministic():
    meta, ann = source_rows()
    rec = {**dv.join_annotations(meta, ann)[0], "start": 0, "end": 2000}
    q = dv.marker_requests(rec, n_frames=2000)
    assert len(q) == 8
    assert q == dv.marker_requests(rec, n_frames=2000)
    assert q[0]["centre"] == 1
    assert q[-1]["centre"] < 1999
    assert len({tuple(r["frames"]) for r in q}) == 8


def test_video_axis_uses_decoded_inventory_and_rejects_guessed_resampling():
    assert dv.validate_video_axis(30., 200, [(0, 200)])
    assert dv.validate_video_axis(30., 182, [(0, 183)])
    with pytest.raises(ValueError, match="fps"):
        dv.validate_video_axis(26., 200, [(0, 200)])
    with pytest.raises(ValueError, match="inventory"):
        dv.validate_video_axis(30., 100, [(0, 200)])


def test_duplicate_requests_across_patients_or_folds_are_rejected():
    rows = [{"pixel_sha256": "same", "patient": "p1", "role": "fit"},
            {"pixel_sha256": "other", "patient": "p2", "role": "diagnostic"}]
    assert dv.check_duplicate_isolation(rows)
    with pytest.raises(ValueError, match="duplicate"):
        dv.check_duplicate_isolation([rows[0], {**rows[1], "pixel_sha256": "same"}])


def test_marker_label_requires_consensus_for_ordinal_and_native_pts_axes():
    rows = [{"start": 0, "end": 100, "true_label": "A4C"},
            {"start": 100, "end": 150, "true_label": "A5C"}]
    assert dv.robust_marker_label(rows, 20, 22, "A4C")
    assert dv.robust_marker_label(rows, 120, 122, "A5C")
    assert not dv.robust_marker_label(rows, 98, 101, "A4C")
    assert not dv.robust_marker_label(rows, 99, 99, "A4C")
    assert not dv.robust_marker_label(rows, 160, 160, "A5C")
