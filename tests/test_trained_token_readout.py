"""CPU checks of the trained token readout (configs/experiments/trained_token_readout_v1.yaml, Task 1): the position-only
anchor selector on even and odd markers, clamped ends, fully repeated slots, several exact anchors, ties and marker
mismatches; the logit read from synthetic (N, 8, 5) arrays; the rejection of wrong identities, digests, class orders,
centres and sequence lists; and the allowlist metadata adapter, which parses labels only for the half's own clips and
never touches another half, the test split or any feature. Synthetic data only."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.trained_token_readout as tt
from echo_routing.evaluate.window_target import window_frames
from echo_routing.temporal.windowed import window_centres

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiments/trained_token_readout_v1.yaml").read_text())["trained_token_readout"]


def window(c: int, n: int) -> tuple[np.ndarray, int]:
    s = np.asarray(window_frames(c, n, 16, 16)); return s, tt.marker_of(s, c)


# ----------------------------------------------------------------------------------------------- selector
def test_anchors_and_rule_follow_the_config():
    assert list(tt.ANCHORS) == CFG["selector"]["anchors"] == [0, 2, 4, 6, 8, 10, 12, 14]


def test_the_selector_reads_positions_only():
    assert list(inspect.signature(tt.selected_anchor).parameters) == ["s", "c", "marker"]
    assert set(inspect.signature(tt.select_all).parameters) == {"S", "C", "M"}


def test_even_marker_picks_the_exact_anchor():
    s, m = window(30, 100)                      # s = 22..37, c at slot 8
    assert m == 8 and tt.selected_anchor(s, 30, m) == 4


def test_odd_marker_ties_go_to_the_nearer_slot_then_the_smaller_index():
    s = np.arange(21, 37); m = 9; c = int(s[m])  # c at slot 9: anchors 8 and 10 are one sample away
    assert tt.selected_anchor(s, c, m) == 4     # |8 - 9| = |10 - 9| = 1, so the smaller j wins


def test_clamped_ends():
    """For n >= 16 the window start is clamped, so no sample repeats: c = 0 sits at slot 0 (anchor 0, exact) and the last
    sample at slot 15, whose nearest anchor is slot 14 (offset -1)."""
    s, m = window(0, 100)
    assert m == 0 and tt.selected_anchor(s, 0, m) == 0
    s, m = window(99, 100)
    j = tt.selected_anchor(s, 99, m)
    assert m == 15 and j == 7 and int(s[tt.ANCHORS[j]]) - 99 == -1
    s, m = window(2, 9)                          # a short clip: samples repeat across the 16 slots
    j = tt.selected_anchor(s, 2, m)
    assert int(s[tt.ANCHORS[j]]) == 2


def test_fully_repeated_slots_pick_the_anchor_nearest_the_marker():
    s = np.zeros(16, int); m = 8
    assert tt.selected_anchor(s, 0, m) == 4
    assert tt.selected_anchor(s, 0, 7) == 3      # anchors 6 and 8 tie at distance 1; the smaller j wins


def test_several_exact_anchors():
    s = np.repeat(np.arange(8), 2); c = 3       # slots 6 and 7 hold sample 3; anchor 6 is the only exact even slot
    m = tt.marker_of(s, c)
    assert m == 7 and tt.selected_anchor(s, c, m) == 3
    s = np.repeat(np.arange(4), 4); c = 1       # slots 4-7 hold sample 1; anchors 4 and 6 are exact
    m = tt.marker_of(s, c)
    assert m == 6 and tt.selected_anchor(s, c, m) == 3


def test_marker_mismatches_stop():
    s = np.arange(10, 26)
    for bad in ((s, 18, 3), (s[:15], 18, 8), (s, 18, 16), (s, 18, -1)):
        with pytest.raises(ValueError):
            tt.selected_anchor(*bad)


def test_select_all_and_the_logit_read():
    S = np.stack([window(c, 60)[0] for c in (0, 7, 30, 59)]); C = np.array([0, 7, 30, 59]); M = np.array([tt.marker_of(s, c) for s, c in zip(S, C)])
    j = tt.select_all(S, C, M)
    assert j.tolist() == [tt.selected_anchor(s, c, m) for s, c, m in zip(S, C, M)]
    tl = np.arange(4 * 8 * 5, dtype=np.float32).reshape(4, 8, 5)
    out = tt.token_logits(tl, j)
    assert out.shape == (4, 5) and all(np.array_equal(out[i], tl[i, j[i]]) for i in range(4))
    off = tt.anchor_offsets(S, C, j)
    assert off.tolist() == [int(S[i, tt.ANCHORS[j[i]]]) - int(C[i]) for i in range(4)]


def test_the_real_window_rule_gives_offsets_of_at_most_one_sample():
    for n in (1, 3, 8, 16, 17, 40):
        for c in window_centres(n, 5):
            s, m = window(int(c), n)
            j = tt.selected_anchor(s, int(c), m)
            assert abs(int(s[tt.ANCHORS[j]]) - int(c)) <= 1


# ----------------------------------------------------------------------------------------------- source checks
def test_wrong_identities_digests_class_orders_centres_and_sequences_are_rejected(tmp_path):
    f = tmp_path / "x.npz"; f.write_bytes(b"abc")
    assert tt.pinned(tmp_path, {"path": "x.npz", "sha256": tt.sha256_file(f)}) == f
    with pytest.raises(SystemExit):
        tt.pinned(tmp_path, {"path": "x.npz", "sha256": "0" * 64})
    with pytest.raises(SystemExit):
        tt.check_identity("aaa", "bbb", "seed 0")
    tt.check_class_order(("PLAX", "PSAX", "A4C", "A5C", "SC4C"), "family5_v1")
    for bad in ((("PSAX", "PLAX", "A4C", "A5C", "SC4C"), "family5_v1"), (("PLAX", "PSAX", "A4C", "A5C", "SC4C"), "family4")):
        with pytest.raises(SystemExit):
            tt.check_class_order(*bad)
    W = {"seq": np.array([0, 0, 1]), "centre": np.array([2, 7, 2]), "marker": np.array([8, 8, 8])}
    saved = {"sequence_ids": np.array(["a", "b"]), "seq": W["seq"].copy(), "centre": W["centre"].copy(), "marker": W["marker"].copy()}
    tt.check_windows(W, ["a", "b"], saved, "s0")
    for key, value in (("centre", np.array([2, 6, 2])), ("seq", np.array([0, 1, 1])), ("marker", np.array([8, 9, 8]))):
        with pytest.raises(SystemExit):
            tt.check_windows(W, ["a", "b"], {**saved, key: value}, "s0")
    with pytest.raises(SystemExit):
        tt.check_windows(W, ["b", "a"], saved, "s0")


# ----------------------------------------------------------------------------------------------- metadata adapter
def write_index(path: Path) -> None:
    rows = ["video_id,split,raw_label,label_index,n_samples,path",
            "c1,validation,x,2,30,p", "c2,validation,y,0,25,p", "o1,validation,z,NOT_A_LABEL,20,p",
            "t1,train,u,NOT_A_LABEL,40,p", "x1,test,v,NOT_A_LABEL,33,p"]
    path.write_text("\n".join(rows) + "\n")


def test_the_index_is_read_for_the_half_clips_only(tmp_path):
    p = tmp_path / "index.csv"; write_index(p)
    df = tt.half_index(p, ["c1", "c2"])
    assert df["video_id"].tolist() == ["c1", "c2"] and df["label_index"].tolist() == [2, 0] and df["n_samples"].tolist() == [30, 25]
    for bad in (["c1", "t1"], ["c1", "x1"], ["c1", "zz"]):
        with pytest.raises(SystemExit):
            tt.half_index(p, bad)


def test_the_adapter_builds_one_half_and_never_reads_the_other(tmp_path):
    p = tmp_path / "index.csv"; write_index(p)
    rec = {"recipe_id": "r1", "cell": "multi", "fragments": [{"video_id": "c1", "label": 2, "variant": "orig", "start": 0, "end": 10},
                                                            {"video_id": "c1", "label": 2, "variant": "orig", "start": 12, "end": 16},
                                                            {"video_id": "c2", "label": 0, "variant": "orig", "start": 9, "end": 15}]}
    proto = {"mapping_version": "family5_v1",
             "validation": {"calibration": {"clips": ["c1", "c2"], "banks": {"pairs": {"recipes": []}, "multi": {"recipes": [rec]}}},
                            "selection": "POISON: never parsed"}, "test_split": "POISON", "train": "POISON"}
    refs = {"c1": [["c1", k, 1.0] for k in range(30)], "c2": [["c2", k, 1.0] for k in range(25)],
            "r1": [["c1", k, 1.0] for k in range(10)] + [["c1", k, 1.0] for k in range(12, 16)] + [["c2", k, 1.0] for k in range(9, 15)]}
    bank = {"c1": "native_calibration", "c2": "native_calibration", "r1": "multi_calibration"}
    plans = {"streams": {k: {"n": len(v), "refs": v, "bank": bank[k]} for k, v in refs.items()} | {"other": "POISON"}}
    ev, seqs = tt.half_sets(proto, "calibration", p, plans)
    assert [q["id"] for q in seqs] == ["c1", "c2", "r1"] and [q["kind"] for q in seqs] == ["native", "native", "multi"]
    assert [s.seq_id for s in ev.natives] == ["c1", "c2"] and ev.natives[0].labels.tolist() == [2] * 30 and ev.natives[0].feat is None
    st = ev.multi[0]
    assert st.labels.tolist() == [2] * 14 + [0] * 6 and st.joins.tolist() == [10, 14] and st.semantic_boundaries.tolist() == [14]
    assert st.prob is None and st.feat is None and seqs[2]["refs"][0] == ("c1", 0, 1.0)
    bad = json.loads(json.dumps(proto)); bad["validation"]["calibration"]["banks"]["multi"]["recipes"][0]["fragments"][0]["end"] = 31
    with pytest.raises(SystemExit):
        tt.half_sets(bad, "calibration", p, plans)          # a fragment beyond its clip's length
    bad = json.loads(json.dumps(proto)); bad["validation"]["calibration"]["banks"]["multi"]["recipes"][0]["fragments"][0]["video_id"] = "o1"
    with pytest.raises(SystemExit):
        tt.half_sets(bad, "calibration", p, plans)          # a fragment outside the half's clips
    bad = json.loads(json.dumps(proto)); bad["validation"]["calibration"]["banks"]["multi"]["recipes"][0]["fragments"][2]["label"] = 3
    with pytest.raises(SystemExit):
        tt.half_sets(bad, "calibration", p, plans)          # a fragment label that is not its clip's label
    other = json.loads(json.dumps(plans)); other["streams"]["r1"]["bank"] = "multi_selection"
    with pytest.raises(SystemExit):
        tt.half_sets(proto, "calibration", p, other)        # a plan of another half's bank
    bad = json.loads(json.dumps(proto)); bad["validation"]["calibration"]["banks"]["multi"]["recipes"][0]["fragments"][0]["video_id"] = "o1"
    with pytest.raises(SystemExit) as e:
        tt.half_sets(bad, "calibration", p, plans)
    assert not any(x in str(e.value) for x in ("r1", "c1", "c2", "o1"))   # error messages carry positions, not ids


def test_the_round_code_never_refers_to_validation_inputs_test_or_pediatric():
    src = inspect.getsource(tt).lower()
    for banned in ("validation(", "inputs(", "run_evaluate", "pediatric", "split_test", "evaluation/test", "decode_frames", "load_frame"):
        assert banned not in src, banned
