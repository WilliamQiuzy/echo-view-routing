"""Checks of the FACT short-segment localisation (scripts/diagnose_fact_localisation.py and
scripts/fact_localisation_tables.py) on synthetic data: the training allowlist is applied before any label is parsed;
the half-open partitions of the intervention (ends, cuts on existing boundaries, length 1, k = L - 1, equal local counts
in B and C, single-label sequences); the official closed segments; on the pinned FACT checkout (CPU), adjacent
same-class intervals kept apart, interleaved passes without effect on the model and every equality check holding; the
per-reference-run measures; the source dedupe; the direction of fixed and new runs; and the summary and tables on
synthetic records."""
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.diagnose_fact_localisation as dl
import scripts.fact_localisation_tables as ft
from echo_routing.evaluate.metrics_segmentation import segment_f1
from echo_routing.temporal.dev_data import Seq
from echo_routing.temporal.segments import labels_to_segments

ROOT = Path(__file__).resolve().parents[1]
HAS_FACT = (ROOT / "third_party/fact/.git").exists()


# ----------------------------------------------------------------------------------------------- allowlist
def write_index(path: Path, rows: list) -> None:
    path.write_text("video_id,split,raw_label,label_index,n_samples,path\n" + "".join(f"{v},{s},view_{lab},{lab},{n},/x/{v}.avi\n" for v, s, lab, n in rows))


def test_allowlist_filters_rows_before_labels_are_parsed(tmp_path):
    p = tmp_path / "index.csv"
    write_index(p, [("a_1", "train", 0, 30), ("v_1", "validation", "garbage", 20), ("b_1", "train", 3, 25), ("t_1", "train", 99, 10)])
    rows, info = dl.allowlist_rows(p, ("b_1", "a_1"))
    assert rows["video_id"].tolist() == ["a_1", "b_1"] and rows["label_index"].tolist() == [0, 3]   # the index's own row order
    assert info["index_rows"] == 4 and info["training_rows_kept"] == 2 and info["rows_dropped_before_label_parsing"] == 2
    assert list(rows.columns) == ["video_id", "split", "n_samples", "label_index"]          # no raw view name, no path
    assert info["columns_read_for_every_row"] == ["video_id", "split", "n_samples"] and info["columns_read_for_kept_rows"] == ["video_id", "label_index"]


@pytest.mark.parametrize("rows, clips", [
    ([("a_1", "train", 0, 30)], ("a_1", "a_1")),                      # repeated in the allowlist
    ([("a_1", "train", 0, 30)], ("a_1", "c_1")),                      # missing from the index
    ([("a_1", "train", 0, 30), ("a_1", "train", 0, 30)], ("a_1",)),   # repeated in the index
    ([("a_1", "validation", 0, 30)], ("a_1",)),                       # not a training row
    ([("a_1", "train", 0, 0)], ("a_1",)),                             # no samples
    ([("a_1", "train", 7, 30)], ("a_1",)),                            # label out of range
    ([("a_1", "train", "x", 30)], ("a_1",)),                          # label not an integer
])
def test_allowlist_refuses_bad_rows(tmp_path, rows, clips):
    p = tmp_path / "index.csv"; write_index(p, rows)
    with pytest.raises(SystemExit):
        dl.allowlist_rows(p, clips)


def test_the_context_never_takes_the_full_label_path():
    src = inspect.getsource(dl.TrainRows) + inspect.getsource(dl.Diag); module = inspect.getsource(dl)
    assert "TrainOnly(" not in src and "read_index(" not in src and "check_index(" not in src and "tr.Ctx(" not in module
    assert "backward(" not in module and "optim." not in module   # inference only
    assert {"echo_routing/temporal/segments.py", "echo_routing/temporal/dev_run.py"} <= set(dl.CODE)   # the identity covers what builds A and B
    assert "inference_constants" in inspect.getsource(dl.identity)


# ----------------------------------------------------------------------------------------------- partitions
def test_reference_cut_on_an_existing_boundary_is_not_repeated():
    c = dl.conditions([(0, 5), (5, 10)], np.array([0] * 5 + [1] + [2] * 4))   # reference changes at 5 (a boundary of A) and 6
    assert c["B_cuts"] == [6] and c["B"] == [(0, 5), (5, 6), (6, 10)]
    assert c["C_cuts"] == [7] and c["C"] == [(0, 5), (5, 7), (7, 10)]         # 5 + floor(1 x 5 / 2)


def test_length_one_and_as_many_cuts_as_inner_positions():
    assert dl.uniform_cuts([(2, 3)], [0]) == [[]]
    with pytest.raises(ValueError):
        dl.uniform_cuts([(2, 3)], [1])
    with pytest.raises(ValueError):
        dl.uniform_cuts([(0, 5)], [5])
    c = dl.conditions([(0, 5)], np.array([0, 1, 0, 1, 0]))                     # k = L - 1 = 4
    assert c["B_cuts"] == c["C_cuts"] == [1, 2, 3, 4] and c["B"] == c["C"] == [(i, i + 1) for i in range(5)]
    c = dl.conditions([(0, 1), (1, 4)], np.array([0, 1, 1, 2]))                # a length-1 interval at the start
    assert c["B"] == [(0, 1), (1, 3), (3, 4)] and c["C"] == [(0, 1), (1, 2), (2, 4)]


def test_b_and_c_add_the_same_number_of_cuts_in_every_interval():
    rng = np.random.default_rng(0)
    for _ in range(300):
        ref = np.concatenate([np.full(int(rng.integers(1, 8)), int(rng.integers(0, 5))) for _ in range(int(rng.integers(1, 8)))]); n = ref.size
        pts = sorted(set(rng.integers(1, n, int(rng.integers(0, 5))).tolist())) if n > 1 else []
        a = list(zip([0, *pts], [*pts, n])); c = dl.conditions(a, ref)
        for s, e in a:
            k = sum(1 for x, y in c["B"] if s <= x and y <= e) - 1
            assert k == sum(1 for x, y in c["C"] if s <= x and y <= e) - 1
            assert [x for x, _ in c["C"] if s < x < e] == [s + (j * (e - s)) // (k + 1) for j in range(1, k + 1)]
        assert set(c["B_cuts"]) == {int(x) for x in np.flatnonzero(np.diff(ref)) + 1} - {s for s, _ in a}


def test_a_single_label_sequence_is_unchanged():
    c = dl.conditions([(0, 4), (4, 9), (9, 12)], np.full(12, 3))
    assert c["A"] == c["B"] == c["C"] and c["B_cuts"] == c["C_cuts"] == []
    with pytest.raises(ValueError):
        dl.conditions([(0, 4), (5, 12)], np.full(12, 3))                     # a gap is not a partition


# ----------------------------------------------------------------------------------------------- the pinned FACT on CPU
@pytest.fixture(scope="module")
def fact():
    if not HAS_FACT:
        pytest.skip("the pinned FACT checkout")
    from echo_routing.temporal.fact_head import FACTHead, TokenRule
    head = FACTHead(ROOT / "third_party/fact", 8, TokenRule(8, 2.654333333333333, 1.5, 12), base_config="gtea.yaml")
    torch.manual_seed(0); model = head.build(); model.eval(); dl.check_blocks(model)
    return head, model


def seq(labels, seed: int, native: bool = False) -> Seq:
    rng = np.random.default_rng(seed); labels = np.asarray(labels, np.int64)
    return Seq(f"s{seed}", native, rng.standard_normal((labels.size, 8)).astype(np.float16), labels, np.zeros(0, int),
               rng.dirichlet(np.ones(5), labels.size).astype(np.float32))


def feats(s: Seq) -> torch.Tensor:
    return torch.from_numpy(s.feat.astype(np.float32))


def test_half_open_intervals_become_the_official_closed_segments(fact):
    head, _ = fact; utils, basic = head.mods["blocks"].utils, head.mods["basic"]
    segs = dl.closed_segments([(0, 3), (3, 4), (4, 9)], np.array([2, 2, 2, 4, 1, 1, 1, 1, 1]), utils.Segment)
    assert [(g.action, g.start, g.end, g.len) for g in segs] == [(2, 0, 2, 3), (4, 3, 3, 1), (1, 4, 8, 5)]
    same = dl.closed_segments([(0, 5), (5, 9)], np.zeros(9, int), utils.Segment)   # one class on both sides: not merged
    assert len(same) == 2 and basic.TemporalDownsampleUpsample(same).num_seg == 2 and len(utils.parse_label(np.zeros(9, int))) == 1
    with pytest.raises(ValueError):
        dl.closed_segments([(0, 3), (3, 8)], np.zeros(9, int), utils.Segment)        # short of the end


def test_the_given_partition_is_the_one_the_u_block_uses(fact):
    head, model = fact; p0 = dl.fact_pass(model, feats(seq([0] * 20, 1)), head.mods); u = dl.block_readout(p0, 1)
    p = dl.fact_pass(model, feats(seq([0] * 20, 1)), head.mods, [(0, 7), (7, 8), (8, 20)]); last = model.block_list[-1]
    assert last.tdu.num_seg == 3 and p["attrs"][-1]["segs"] == [(0, 7), (7, 8), (8, 20)] and [g.len for g in last.tdu.segs] == [7, 1, 12]
    assert [g.action for g in last.tdu.segs] == [u[0], u[7], u[8]]                    # the u block's argmax, never a reference label
    assert dl.same_outs(p, p0, 2) and "temporal_downsample" not in vars(last)         # the inputs to U are untouched


def test_interleaved_passes_leave_the_model_unchanged(fact):
    head, model = fact; w0 = dl.weights(model); last = model.block_list[-1]
    x1, x2 = feats(seq([0] * 12 + [1] * 6 + [0] * 12, 2)), feats(seq([3] * 25, 3))
    a = dl.fact_pass(model, x1, head.mods, dl.runs(dl.block_readout(dl.fact_pass(model, x1, head.mods), 1)))
    dl.fact_pass(model, x1, head.mods, [(0, 12), (12, 18), (18, 30)]); dl.fact_pass(model, x2, head.mods, [(0, 25)])
    dl.fact_pass(model, x1, head.mods, [(0, 9), (9, 18), (18, 30)]); again = dl.fact_pass(model, x1, head.mods, a["attrs"][-1]["segs"])
    assert dl.same_pass(a, again) and dl.same_weights(model, w0) and "temporal_downsample" not in vars(last)
    with pytest.raises(ValueError):
        dl.fact_pass(model, x1, head.mods, [(0, 12), (13, 30)])                      # fails inside the forward
    assert "temporal_downsample" not in vars(last) and dl.same_pass(a, dl.fact_pass(model, x1, head.mods, a["attrs"][-1]["segs"]))


def test_every_check_holds_on_a_stream_and_a_native_window(fact):
    head, model = fact; s = seq([0] * 10 + [1] * 4 + [0] * 10 + [2] * 6, 4)
    rec, chk = dl.fact_sequence(head, model, s, "cpu", run_bc=True)
    assert all(chk[k] for k in dl.CHECKS) and rec["B"]["executed"] and rec["C"]["executed"] and len(rec["B"]["cuts"]) == len(rec["C"]["cuts"])
    other = Seq(s.seq_id, False, s.feat, np.array([3] * 10 + [4] * 4 + [3] * 10 + [1] * 6), s.joins, s.prob)   # same cut positions, other views
    rec2, _ = dl.fact_sequence(head, model, other, "cpu", run_bc=True)
    assert rec2["B"]["final"] == rec["B"]["final"] and rec2["C"]["final"] == rec["C"]["final"]   # only positions enter the network
    n = seq([2] * 20, 5, native=True)
    full, chk1 = dl.fact_sequence(head, model, n, "cpu", run_bc=True); run, chk2 = dl.fact_sequence(head, model, n, "cpu", run_bc=False)
    assert all(chk1[k] and chk2[k] for k in dl.CHECKS) and full["B"]["executed"] and not run["B"]["executed"] and full["B"]["cuts"] == []
    assert full["B"]["final"] == run["B"]["final"] == full["C"]["final"] == full["stages"]["FACT_final"]


# ----------------------------------------------------------------------------------------------- measures and summaries
def test_segment_measures_describe_misses_without_a_single_cause():
    ref = np.array([0] * 10 + [1] * 4 + [0] * 10)
    m = ft.segment_measures(ref, np.zeros(24, int), 5)                                  # the middle run absorbed
    assert [r["matched"] for r in m] == [False, False, False] and m[1]["iou_any"] == pytest.approx(4 / 24) and m[1]["iou_same"] == 0
    assert m[1]["inside"] == [4, 0, 0, 0, 0] and m[1]["start_found"] is False and m[0]["start_found"] is None
    assert m[1]["miss"] == {"no_run_at_iou_50": True, "other_view_at_iou_50": False, "same_view_run_taken": False, "shape": "absent"}
    assert m[0]["miss"]["shape"] == "longer"
    m = ft.segment_measures(ref, np.array([0] * 10 + [1] + [0] * 13), 5)                # a too short middle run
    assert m[1]["miss"]["shape"] == "shorter" and m[1]["start_found"] and m[1]["end_found"] and m[0]["matched"] and m[2]["matched"]
    assert ft.segment_measures(ref, np.array([0] * 10 + [1, 0, 1, 0] + [0] * 10), 5)[1]["miss"]["shape"] == "fragmented"
    assert ft.segment_measures(ref, np.array([0] * 12 + [1] * 4 + [0] * 8), 5)[1]["miss"]["shape"] == "shifted"
    pred = np.array([0] * 10 + [2] * 4 + [0] * 10); m = ft.segment_measures(ref, pred, 5)   # read as another view
    assert m[1]["miss"]["other_view_at_iou_50"] and not m[1]["miss"]["no_run_at_iou_50"] and m[0]["matched"] and m[2]["matched"]
    assert sum(r["matched"] for r in m) == segment_f1(labels_to_segments(pred), labels_to_segments(ref), 0.5)["matched"]


def test_units_count_distinct_recordings_and_sessions_and_the_direction_of_changes():
    rec = {"seq_id": "q", "native": False, "ref": [0] * 10 + [1] * 4 + [0] * 10, "tokens": {"U": [0, 1] + [5] * 10}, "partition_A": [[0, 24]],
           "frags": [["p1_1", 0, 6], ["p1_2", 6, 10], ["p1_3", 10, 14], ["p2_1", 14, 24]], "stages": {n: [0] * 24 for n in ft.STAGES},
           "B": {"final": [0] * 10 + [1] * 4 + [0] * 10, "U": [0] * 24}, "C": {"final": [0] * 24, "U": [0] * 24}}
    us = ft.units_of(rec, 5, 16)
    assert [u["recordings"] for u in us] == [["p1_1", "p1_2"], ["p1_3"], ["p2_1"]] and [u["sessions"] for u in us] == [["p1"], ["p1"], ["p2"]]
    assert us[1]["groups"] == ["all", "short", "xyx"] and ft.sources(us) == {"segments": 3, "recordings": 4, "sessions": 2}
    assert ft.changes(us, "B_final", "FACT_final") == {"fixed": {"segments": 3, "recordings": 4, "sessions": 2}, "new": {"segments": 0, "recordings": 0, "sessions": 0}}
    assert ft.changes(us, "FACT_final", "B_final")["new"]["segments"] == 3                # new = matched by the second only


def test_own_share_separates_absorbed_runs_from_slightly_long_intervals():
    t = labels_to_segments(np.array([0] * 10 + [1] * 4 + [0] * 10))[1]                  # the middle run [10, 14)
    assert ft.own_share(t, [(0, 24)]) == pytest.approx(4 / 24)                          # absorbed: minority everywhere
    assert ft.own_share(t, [(0, 10), (10, 16), (16, 24)]) == pytest.approx(4 / 6)       # own interval two samples too long
    assert ft.own_share(t, [(0, 12), (12, 24)]) == pytest.approx(2 / 12)                # split between two neighbours
    us = ft.units_of({"seq_id": "q", "native": False, "ref": [0] * 10 + [1] * 4 + [0] * 10, "tokens": {"U": [5] * 12}, "partition_A": [[0, 24]],
                      "frags": [["p1_1", 0, 24]], "stages": {n: [0] * 24 for n in ft.STAGES}, "B": {"final": [0] * 10 + [1] * 4 + [0] * 10, "U": [0] * 24},
                      "C": {"final": [0] * 24, "U": [0] * 24}}, 5, 16)
    s = ft.by_own_share(us)["xyx"]
    assert s["minority in every interval"]["segments"] == 1 and s["minority in every interval"]["matched"] == {"A": 0, "B": 1, "C": 0}
    rec = {"native": False, "partition_A": [[0, 3], [3, 6]], "stages": {"FACT_final": [1, 1, 2, 0, 0, 0], "FACT_U": [1, 1, 1, 0, 3, 0]},
           "tokens": {"U": [1, 3] + [5] * 10}, "assign": [0, 0, 0, 1, 1, 1]}
    c = ft.changes_inside_intervals([rec])["stream"]   # interval 2 is labelled 0 against its token's class 3 on every frame
    assert (c["intervals"], c["official_output_changes_inside"], c["U_frame_branch_changes_inside"]) == (2, 1, 1)
    assert c["label_differs_from_the_assigned_token_class_on_every_frame"] == 1 and c["label_differs_from_the_assigned_token_class_on_some_frames"] == 2
    assert us[0]["U_tokens_of_view"] == 0 and us[0]["runs_of_view"] == 2 and us[1]["runs_of_view"] == 1
    m = ft.partition_mixing([{"ref": [0] * 10 + [1] * 4 + [0] * 10, "partition_A": [[0, 10], [10, 16], [16, 24]], "frags": [["p1_1", 0, 24]]}], 16)
    assert m["reference_changes_on_a_boundary"]["all"] == [1, 2]                         # the change at 10 is a boundary, 14 is not


def test_fit_table_pools_over_sequences():
    recs = [{"ref": [0] * 5 + [1] * 5, "stages": {"E1": [0] * 10}}, {"ref": [2] * 4, "stages": {"E1": [2] * 4}}]
    t = ft.fit_table(recs, "E1", 5)
    assert t["segment_f1_50"] == pytest.approx(2 * 2 / (2 + 3)) and t["frame_error"] == pytest.approx(5 / 14)
    assert t["per_sequence_f1"]["mean"] == pytest.approx((2 / 3 + 1) / 2) and t["boundary_recall"] == 0 and t["boundary_precision"] is None


def synthetic(i: int, native: bool, rng) -> dict:
    if native:
        ref = np.full(int(rng.integers(8, 30)), int(rng.integers(0, 5))); frags = [[f"n{i}_1", 0, int(ref.size)]]
    else:
        lens = rng.integers(5, 20, 5); ref = np.repeat([2, 3, 2, 0, 1], lens); e = np.concatenate([[0], np.cumsum(lens)])
        frags = [[f"r{i}_{k}", int(a), int(b)] for k, (a, b) in enumerate(zip(e[:-1], e[1:]))]
    noisy = ref.copy(); noisy[rng.random(ref.size) < 0.1] = 4; c = dl.conditions(dl.runs(noisy), ref)
    stage = lambda: np.where(rng.random(ref.size) < 0.9, ref, 4).tolist()  # noqa: E731
    return {"seq_id": f"q{i}", "native": native, "ref": ref.tolist(), "frags": frags, "stages": {n: stage() for n in ft.STAGES},
            "tokens": {b: [2, 3] + [5] * 10 for b in ft.BLOCKS}, "partition_A": [list(x) for x in c["A"]], "assign": None, "logit_argmax_mismatches": 0,
            **{k: {"cuts": c[f"{k}_cuts"], "final": stage(), "U": stage(), "executed": not native} for k in ("B", "C")}}


def test_the_summary_and_tables_run_on_synthetic_records():
    rng = np.random.default_rng(1); recs = [synthetic(i, i % 2 == 1, rng) for i in range(12)]
    units = [u for r in recs for u in ft.units_of(r, 5, 16)]; s = ft.seed_summary(recs, units, 5, 16)
    assert s["intervention"]["cuts"]["C_counts_equal_B"] and s["intervention"]["native_windows"]["no_cut_added"] == 6
    assert s["groups"]["stream:xyx"]["E1"]["segments"] == 6 and s["fit"]["native"]["E1"]["sequences"] == 6
    md = ft.tables_md({"seeds": {0: s}, "across_seeds": ft.across_seeds({0: units, 1: units}), "cost": {}})
    assert "## Intervention at the last U block" in md and "q0" not in md
