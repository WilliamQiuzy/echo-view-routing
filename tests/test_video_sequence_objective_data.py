"""A07 data side (configs/experiments/video_sequence_objective_v1.yaml, task sheet section 3): four-centre groups on the
whole-sequence stride-5 grid, native repeats, eligibility on the four sampled labels only (X-Y-X kept, same-class joins
ineligible, no centre repeated in a constructed group), the transcript and the skipped reference runs, the frozen draw
order, whole-stream context, class/marker alignment, deterministic manifests in V01's layout that the V01 loader reads with
matching pixel digests, the counts, and labels read only for allowlisted training rows. CPU only; synthetic references."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import yaml

import echo_routing.temporal.video_sequence_features as vsf
import echo_routing.vit.target_mvit as tm
import scripts.prepare_video_sequence_objective as pr
import scripts.train_raw_video_context as rvc
from echo_routing.temporal.losses import segment_objective as so

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiments/video_sequence_objective_v1.yaml").read_text())["video_sequence_objective"]


# ----------------------------------------------------------------------------------------------- groups
def test_native_groups_are_four_consecutive_centres_or_repeat_the_last():
    assert pr.native_starts(7).tolist() == [0, 1, 2, 3] and pr.native_starts(4).tolist() == [0]
    assert pr.group_indices(7, 2).tolist() == [2, 3, 4, 5]
    assert pr.native_starts(2).tolist() == [0] and pr.group_indices(2, 0).tolist() == [0, 1, 1, 1]
    assert pr.native_starts(1).tolist() == [0] and pr.group_indices(1, 0).tolist() == [0, 0, 0, 0]
    assert pr.repeated_targets(pr.group_indices(2, 0)) == 2 and pr.repeated_targets(pr.group_indices(7, 0)) == 0


def test_constructed_eligibility_uses_the_four_sampled_labels_only():
    assert pr.eligible_starts(np.array([0, 0, 0, 0, 2, 2, 0])).tolist() == [1, 2, 3]
    assert pr.eligible_starts(np.array([1, 1, 1, 1, 1, 1])).size == 0          # one view only (same-class joins included)
    assert pr.eligible_starts(np.array([1, 3, 1])).size == 0                    # fewer than four centres: never a repeated centre
    assert pr.eligible_starts(np.array([2, 3, 3, 2])).tolist() == [0]
    assert so.reference_runs(np.array([2, 3, 3, 2])).transcript == (2, 3, 2)   # X-Y-X keeps three states


def test_a_class_only_between_sampled_centres_is_skipped_not_inserted():
    labels = np.array([0] * 9 + [4] * 2 + [0] * 9 + [1] * 20)                   # a two-sample class-4 run between centres 7 and 12
    centres = vsf.windows(len(labels))["centres"]
    assert centres.tolist() == [2, 7, 12, 17, 22, 27, 32, 37]
    idx = pr.group_indices(len(centres), 1); targets = labels[centres[idx]]
    assert targets.tolist() == [0, 0, 0, 1] and so.reference_runs(targets).transcript == (0, 1)
    assert pr.skipped_runs(labels, centres[idx]) == 1 and pr.skipped_runs(labels, centres[pr.group_indices(len(centres), 3)]) == 0
    assert pr.eligible_starts(labels[centres]).tolist() == [1, 2, 3]


def test_the_draw_order_is_native_clip_native_start_stream_stream_start():
    nat, elig = [3, 1, 4, 2, 5], [2, 6, 1]
    rng = np.random.default_rng([20261015, 1, 7]); i = int(rng.integers(5)); j = int(rng.integers(nat[i])); k = int(rng.integers(3)); l = int(rng.integers(elig[k]))
    assert pr.draw(1, 7, nat, elig) == (i, j, k, l) and pr.draw(1, 7, nat, elig) == pr.draw(1, 7, nat, elig)
    draws = {pr.draw(s, u, nat, elig) for s in range(3) for u in range(40)}
    assert len(draws) > 10 and all(0 <= a < 5 and 0 <= b < nat[a] and 0 <= c < 3 and 0 <= d < elig[c] for a, b, c, d in draws)
    assert pr.RNG_BASE == 20261015


# ----------------------------------------------------------------------------------------------- manifests
def synthetic_seqs() -> list[dict]:
    """Three native clips (one with fewer than four centres) and three streams (one same-view only), with references."""
    seqs = []
    for k, (n, lab) in enumerate(((40, 2), (12, 3), (33, 0))):
        seqs.append({"id": f"clip{k}", "kind": "native", "refs": [(f"clip{k}", 3 * t, 1.0) for t in range(n)], "labels": np.full(n, lab, np.int64)})
    frags = (((0, 0, 20, 2, 0.8), (2, 5, 30, 0, 1.25)), ((2, 0, 18, 0, 1.0), (0, 10, 30, 2, 1.0), (1, 0, 12, 3, 0.8)), ((0, 0, 25, 2, 1.0), (0, 5, 25, 2, 0.8)))
    for k, fr in enumerate(frags):
        refs, labels = [], []
        for c, a, b, lab, g in fr:
            refs += [(f"clip{c}", 3 * t, g) for t in range(a, b)]; labels += [lab] * (b - a)
        seqs.append({"id": f"stream{k}", "kind": "bank", "refs": refs, "labels": np.asarray(labels, np.int64)})
    return seqs


def test_streams_without_an_eligible_start_stay_in_the_accounting():
    info = pr.sequence_info(synthetic_seqs())
    assert [q["kind"] for q in info] == ["native"] * 3 + ["bank"] * 3
    assert [len(q["starts"]) > 0 for q in info[3:]] == [True, True, False]
    plan = pr.make_plan(info, seed=0, total=40)
    assert plan["native_seq"].shape == (40,) and plan["stream_seq"].shape == (40,) and set(plan["stream_seq"].tolist()) <= {3, 4}
    st = pr.plan_stats(info, plan)
    assert st["streams"] == {"eligible": 2, "ineligible": 1} and st["native_clips_with_repeats"] == 1


def test_manifest_windows_come_from_the_whole_sequence_grid_and_align_targets_and_markers():
    seqs = synthetic_seqs(); info = pr.sequence_info(seqs); plan = pr.make_plan(info, seed=1, total=30); m = pr.build_manifest(seqs, info, plan)
    assert m["vid"].shape == (30, 8, 16) and m["target"].shape == (30, 8) and np.array_equal(m["order_ordered"][0, 0], np.arange(16))
    for u in range(30):
        for w in range(8):
            q = seqs[int(m["seq"][u, w])]; W = vsf.windows(len(q["refs"])); ci = int(m["centre_index"][u, w])
            assert np.array_equal(m["s"][u, w], W["samples"][ci]) and int(m["c"][u, w]) == int(W["centres"][ci])
            assert int(m["marker"][u, w]) == int(W["markers"][ci]) == tm.marker_slot(m["s"][u, w], int(m["c"][u, w]))
            assert int(m["target"][u, w]) == int(q["labels"][int(m["c"][u, w])]) == int(q["labels"][m["s"][u, w][int(m["marker"][u, w])]])
            for j in range(16):
                ref = q["refs"][int(m["s"][u, w, j])]
                assert (str(m["videos"][m["vid"][u, w, j]]), int(m["frame"][u, w, j]), round(float(m["gamma"][u, w, j]), 6)) == (ref[0], ref[1], round(ref[2], 6))
        assert (m["kind"][u, :4] == 0).all() and (m["kind"][u, 4:] == 1).all()
        assert np.all(np.diff(m["c"][u, 4:]) > 0) and np.all(np.diff(m["c"][u, :4]) >= 0)
        assert len(so.reference_runs(m["target"][u, 4:]).transcript) >= 2
    first = vsf.windows(len(seqs[3]["refs"]))["samples"][0]            # context beyond a group's own span: the whole-stream window
    assert first.tolist() == list(range(16))


def test_manifests_are_identical_twice_and_change_with_the_seed():
    seqs = synthetic_seqs(); info = pr.sequence_info(seqs)
    a = pr.build_manifest(seqs, info, pr.make_plan(info, seed=2, total=25)); b = pr.build_manifest(seqs, info, pr.make_plan(info, seed=2, total=25))
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a) and pr.arrays_digest(a) == pr.arrays_digest(b)
    c = pr.build_manifest(seqs, info, pr.make_plan(info, seed=3, total=25))
    assert pr.arrays_digest(a) != pr.arrays_digest(c)


def test_the_v01_loader_reads_the_manifest_and_its_pixel_digests_match(monkeypatch):
    seqs = synthetic_seqs(); info = pr.sequence_info(seqs); m = pr.build_manifest(seqs, info, pr.make_plan(info, seed=0, total=3))

    def fake_frame(path, gamma):
        v = int(str(path).split("/")[-1]); g = int(round(float(gamma) * 100))
        return np.full((224, 224, 3), (v * 7 + g) % 251, np.uint8)
    paths = lambda vid: [f"{vid}/{i}" for i in range(200)]  # noqa: E731
    monkeypatch.setattr(rvc, "load_frame_u8", fake_frame)
    refs = {(str(m["videos"][v]), int(f), round(float(g), 6)) for v, f, g in zip(m["vid"].ravel(), m["frame"].ravel(), m["gamma"].ravel())}
    hashes = {r: tm.frame_hash(fake_frame(paths(r[0])[min(r[1], 199)], r[2])) for r in refs}
    m["digest_ordered"] = pr.window_digests(m, hashes)
    for u in range(3):
        _, slots, target, digests = rvc.window_inputs(m, m["order_ordered"], paths, [(u, w) for w in range(8)])
        rvc.check_digests(m, "Ordered context", u, digests)
        assert np.array_equal(slots.numpy(), m["marker"][u]) and np.array_equal(target.numpy(), m["target"][u])


def test_epoch_digests_cover_every_epoch_and_change_with_the_inputs():
    seqs = synthetic_seqs(); info = pr.sequence_info(seqs); m = pr.build_manifest(seqs, info, pr.make_plan(info, seed=0, total=20))
    m["digest_ordered"] = np.zeros((20, 8, 32), np.uint8)
    d = pr.epoch_digests(m, 5)
    assert len(d) == 4 and len(set(d)) == 4
    m2 = {k: v.copy() for k, v in m.items()}; m2["target"][7, 5] = (m2["target"][7, 5] + 1) % 5
    assert pr.epoch_digests(m2, 5)[1] != d[1] and pr.epoch_digests(m2, 5)[0] == d[0]


def test_counts_cover_classes_runs_lengths_exposure_repeats_and_skips():
    seqs = synthetic_seqs(); info = pr.sequence_info(seqs); plan = pr.make_plan(info, seed=0, total=50); m = pr.build_manifest(seqs, info, plan)
    st = pr.manifest_counts(m, info, plan)
    assert sum(st["target_classes"].values()) == 50 * 8 and st["windows"] == 400
    assert st["constructed_runs"]["min"] >= 2 and st["native_runs"]["max"] == 1
    assert st["sources"]["native_clips_drawn"] <= 3 and st["sources"]["streams_drawn"] <= 2
    assert st["repeated_native_targets"] >= 0 and "skipped_reference_runs" in st and "transcript_lengths" in st


# ----------------------------------------------------------------------------------------------- isolation
def test_training_labels_are_read_for_training_rows_only(tmp_path):
    p = tmp_path / "index.csv"
    p.write_text("video_id,split,n_samples,label_index,raw_label,path\na,train,10,2,x,y\nb,validation,12,NOT_A_LABEL,x,y\nc,train,9,4,x,y\n")
    idx = vsf.training_index(p)
    assert idx["video_id"].tolist() == ["a", "c"] and idx["label_index"].tolist() == [2, 4]


def test_preparation_reads_training_data_through_the_allowlisted_path_only():
    src = inspect.getsource(pr)
    assert "TrainingData(" in src and "read_csv" not in src
    banned = ("validation(", "EvalSet", "report_center_supervision", "pediatric", "evaluation/selection", "evaluation/calibration", "'test'",
              "features/development", "ordered_outputs", "window_plans", "half_sets")
    assert not any(b.lower() in src.lower() for b in banned)


def test_the_config_fixes_the_sampling_rules():
    g = CFG["groups"]
    assert (g["size"], g["stride"], g["windows_per_update"]) == (4, 5, 8) and "20261015" in g["rng"]
    assert CFG["training"]["total_updates"] == 3750 and CFG["training"]["warmup_updates"] == 375 and CFG["objectives"]["class_weights"] == [1.0] * 5
    assert pr.WINDOWS == 8 and pr.GROUP == 4
