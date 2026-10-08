"""Pure helpers of scripts/session_fold_library.py and the fixed-epoch trainer on synthetic data: the fold rule, the
symmetric native contrast, the recognition tallies, the candidate fields, the composition and a short CPU training."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

import scripts.session_fold_library as sfl
from echo_routing.features.train_encoder import TrainConfig, train_fixed_epochs
from echo_routing.ingest.frames import VideoRecord


def _clips():
    rng = np.random.default_rng(0); clips, labels = [], {}
    for s in range(30):
        for k in range(1 + s % 4):
            c = f"s{s:02d}_{k}"; clips.append(c); labels[c] = int(rng.integers(0, 5))
    return clips, labels


def test_folds_keep_sessions_whole_and_are_fixed():
    clips, labels = _clips()
    a = sfl.assign_folds(clips, labels, 3, 7); b = sfl.assign_folds(list(reversed(clips)), labels, 3, 7)
    assert a == b and set(a.values()) == {0, 1, 2} and set(a) == {c.rsplit("_", 1)[0] for c in clips}
    rows = sfl.fold_balance(sorted(clips), labels, a, 3)
    assert sum(r["clips"] for r in rows) == len(clips) and sum(r["sessions"] for r in rows) == 30
    assert sum(r["A4C clips"] for r in rows) == sum(v == 2 for v in labels.values())


def test_native_contrast_is_seeded_and_about_half():
    clips = [f"c{k}_1" for k in range(400)]
    a = sfl.native_variants(clips, 0.5, 3, "gamma090"); b = sfl.native_variants(list(reversed(clips)), 0.5, 3, "gamma090")
    assert a == b and set(a.values()) == {"orig", "gamma090"} and 160 < sum(v == "gamma090" for v in a.values()) < 240


def test_recognition_counts_the_focus_errors():
    items = [("a_1", 2, np.array([2, 2, 3])), ("a_2", 2, np.array([3, 3, 2])), ("b_1", 3, np.array([2, 2, 3, 3])), ("c_1", 0, np.array([0, 0]))]
    r = sfl.recognition_table(items)
    f = r["focus"]["A4C->A5C"]
    assert (f["samples"], f["of samples"], f["recordings with any"], f["sessions with any"], f["clips voted"], f["of clips"]) == (3, 6, 2, 1, 1, 2)
    g = r["focus"]["A5C->A4C"]                                              # a 2-2 tie votes the lower index, A4C
    assert (g["samples"], g["clips voted"], g["sessions of clips voted"]) == (2, 1, 1)
    assert r["clip"]["n"] == 4 and r["sample"]["n"] == 12 and r["sessions"] == 3


def test_candidates_record_source_join_and_contrast():
    p = np.full((30, 5), 0.01); p[:, 2] = 0.96; p[12:18, 2] = 0.01; p[12:18, 3] = 0.96          # an A4C clip read A4C, A5C x 6, A4C
    native = ("session:n", "native clips", p, np.full(30, 2), np.zeros(0, int), np.array([0, 30]), ["n_1"], ["gamma090"])
    q = np.full((30, 5), 0.01); q[:, 2] = 0.96; q[10:20, 2] = 0.01; q[10:20, 3] = 0.96          # A4C | A5C | A4C fragments
    stream = ("stream:r1", "multi streams", q, np.array([2] * 10 + [3] * 10 + [2] * 10), np.array([10, 20]), np.array([0, 10, 20, 30]),
              ["s_1", "t_1", "u_1"], ["orig", "gamma090", "orig"])
    rows = sfl.fold_candidates([native, stream], 8.0, 1e-6, 5, "gamma090")
    assert [(r["kind"], r["cat"], r["join"], r["contrast"], r["mid_videos"], r["len_mid"]) for r in rows] == [
        ("native clips", "removed", "none", "edited", ["n_1"], 6), ("multi streams", "kept", "view-change", "edited", ["t_1"], 10)]
    assert rows[0]["conf_mid"] == pytest.approx(0.96)


def test_describe_and_concentration():
    def row(cat, x, y, kind, join, contrast, n, conf, vids, reason=None):
        return {"cat": cat, "x": x, "y": y, "kind": kind, "join": join, "contrast": contrast, "len_mid": n, "conf_mid": conf, "mid_videos": vids, "reason": reason}
    rows = [row("removed", 2, 3, "native clips", "none", "orig", 12, 0.8, ["a_1"]), row("removed", 3, 2, "native clips", "none", "edited", 20, 0.6, ["a_1"]),
            row("removed", 2, 3, "multi streams", "same-view", "mixed", 8, 0.7, ["b_1", "c_1"]),
            row("kept", 0, 1, "multi streams", "view-change", "orig", 30, 0.9, ["d_1"]), row("other", 0, 1, "native clips", "none", "orig", 5, 0.5, ["e_1"], "sides not X")]
    d = sfl.describe(rows)
    w = d["A4C/A5C pairs"]["removed"]
    assert (w["candidates"], w["recordings"], w["sessions"], w["from native clips"], w["same-view join"], w["contrast mixed"]) == (3, 3, 3, 2, 1, 1)
    c = w["concentration"]                                                  # a_1 carries 2 of 3; b_1 and c_1 share the third
    assert c["largest recording share"] == pytest.approx(2 / 3) and c["most candidates on one recording"] == 2
    assert c["effective recordings"] == pytest.approx(1 / ((2 / 3) ** 2 + 2 * (1 / 6) ** 2))
    assert d["other pairs"]["kept"]["candidates"] == 1 and d["outside reasons"] == {"sides not X": 1}
    assert d["all pairs"]["removed"]["len_mid quartiles"] == [10.0, 12.0, 16.0] and d["class pairs"]["kept"] == {"PLAX>PSAX": 1}
    assert sfl.concentration([])["effective recordings"] is None


def test_only_this_rounds_results_directory_is_written():
    sfl.check_out("docs/results/2026-10-07_session_fold_library", {"results_dir": "docs/results/2026-10-07_session_fold_library"})
    with pytest.raises(SystemExit):
        sfl.check_out("docs/results/2026-10-07_short_segment_transfer", {"results_dir": "docs/results/2026-10-07_session_fold_library"})


def test_fixed_epoch_training_keeps_the_last_epoch_without_validation(tmp_path):
    recs = []
    for k, lab in enumerate([0, 1, 0, 1]):
        d = tmp_path / f"v{k}"; d.mkdir()
        for f in range(4):
            Image.fromarray(np.full((40, 48, 3), 40 * f + 60 * lab, np.uint8)).save(d / f"{f:04d}.png")
        recs.append(VideoRecord(f"v{k}_1", d, 4, lab, "train"))
    cfg = TrainConfig.from_dict({"task": "family5", "image_size": 32, "frames_per_video": 2, "batch_size": 4, "epochs": 2, "num_workers": 0, "amp": False, "seed": 0})
    out = train_fixed_epochs(cfg, recs, ["A", "B"], tmp_path / "ck", pretrained=False)
    log = [json.loads(x) for x in (tmp_path / "ck" / "train_log.jsonl").read_text().splitlines()]
    assert out.name == "last.pt" and [r["epoch"] for r in log] == [0, 1] and not any("val" in r for r in log)
    meta = torch.load(out, weights_only=False)
    assert meta["epoch"] == 1 and meta["selection"].startswith("none") and not (tmp_path / "ck" / "best.pt").exists()
    train_fixed_epochs(cfg, recs, ["A", "B"], tmp_path / "ck", resume=out, pretrained=False)           # a finished run trains nothing more
    assert len((tmp_path / "ck" / "train_log.jsonl").read_text().splitlines()) == 2
    train_fixed_epochs(cfg, recs, ["A", "B"], tmp_path / "ck", pretrained=False)                       # a fresh run starts a fresh log
    assert [json.loads(x)["epoch"] for x in (tmp_path / "ck" / "train_log.jsonl").read_text().splitlines()] == [0, 1]


def test_a_middle_over_two_fragments_has_mixed_contrast():
    q = np.full((30, 5), 0.01); q[:, 2] = 0.96; q[8:20, 2] = 0.01; q[8:20, 3] = 0.96                 # decoded middle 8-20 runs past the join at 10
    stream = ("stream:r2", "multi streams", q, np.array([2] * 10 + [3] * 10 + [2] * 10), np.array([10, 20]), np.array([0, 10, 20, 30]),
              ["s_1", "t_1", "u_1"], ["orig", "gamma090", "orig"])
    (row,) = sfl.fold_candidates([stream], 8.0, 1e-6, 5, "gamma090")
    assert (row["contrast"], row["mid_videos"], row["cat"]) == ("mixed", ["s_1", "t_1"], "kept")


def test_written_summaries_carry_no_ids():
    items = [("sessA_1", 2, np.array([3, 3, 2])), ("sessB_2", 3, np.array([2, 2]))]
    rows = [{"cat": "removed", "x": 2, "y": 3, "kind": "native clips", "join": "none", "contrast": "orig", "len_mid": 9, "conf_mid": 0.7,
             "mid_videos": ["sessA_1"], "reason": None, "group": "session:sessA"}]
    text = json.dumps({"recognition": sfl.recognition_table(items), "composition": sfl.describe(rows)})
    assert "sessA" not in text and "sessB" not in text                     # key names such as "sessions" are fine


def test_own_view_sources_leave_out_the_neighbours():
    rows = [{"cat": "kept", "y": 3, "mid_videos": ["a_1", "b_1"]}, {"cat": "kept", "y": 2, "mid_videos": ["c_1"]}, {"cat": "removed", "y": 3, "mid_videos": ["d_1"]}]
    labels = {"a_1": 2, "b_1": 3, "c_1": 2, "d_1": 2}                      # a_1 is the A4C neighbour of the A5C middle on b_1
    assert sfl.own_view_sources(rows, labels) == {"middles": 2, "recordings of the middle's view": 2, "their sessions": 2, "middles over two or more recordings": 1}
