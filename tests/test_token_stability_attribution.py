"""CPU checks of the token stability attribution round's source side (configs/experiments/token_stability_attribution_v1.yaml,
Task 1), written before the module: a saved prediction with a wrong identity, reordered windows, other shapes or
non-finite values stops; a wrong class order or mapping stops; the selector and the metadata adapter are A03's own; the
probability mapping is A03's bit for bit; the stem geometry check stops on another kernel or a temporal pooling stride;
and the module never refers to a model load, a raw frame, the validation entry, the test split or Pediatric data.
Synthetic data only."""
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.report_raw_video_context as rr
import scripts.token_stability_attribution as ts
import scripts.trained_token_readout as tt
from echo_routing.evaluate.window_target import window_frames
from echo_routing.temporal.windowed import window_centres

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiments/token_stability_attribution_v1.yaml").read_text())["token_stability_attribution"]


def fake_windows(lengths=(20, 9)):
    seqs, S, C, M, Q = [], [], [], [], []
    for qi, n in enumerate(lengths):
        seqs.append({"id": f"q{qi}", "kind": "native", "labels": np.full(n, qi % 5), "refs": [(f"v{qi}", k, 1.0) for k in range(n)]})
        for c in window_centres(n, 5):
            s = np.asarray(window_frames(int(c), n, 16, 16)); S.append(s); C.append(int(c)); M.append(tt.marker_of(s, int(c))); Q.append(qi)
    W = {"seq": np.asarray(Q), "centre": np.asarray(C), "s": np.asarray(S), "marker": np.asarray(M)}
    W["labels16"] = np.stack([np.asarray(seqs[q]["labels"])[s] for q, s in zip(Q, S)]); W["label"] = np.asarray([seqs[q]["labels"][c] for q, c in zip(Q, C)])
    return W, seqs


def write_npz(path: Path, W: dict, ids: list, identity: str = "run-1", **over) -> dict:
    n = len(W["seq"]); rng = np.random.default_rng(0)
    d = {"identity": np.asarray(identity), "sequence_ids": np.asarray(ids), "seq": W["seq"], "centre": W["centre"], "marker": W["marker"],
         "logits": rng.normal(size=(n, 5)).astype(np.float32), "time_logits": rng.normal(size=(n, 8, 5)).astype(np.float32)} | over
    np.savez(path, **d)
    return {"path": path.name, "sha256": tt.sha256_file(path)}


# ----------------------------------------------------------------------------------------------- the saved predictions
def test_a_saved_prediction_is_checked_before_use(tmp_path):
    W, seqs = fake_windows(); ids = [q["id"] for q in seqs]
    pin = write_npz(tmp_path / "ok.npz", W, ids)
    d = ts.checked_prediction(tmp_path, pin, "run-1", W, ids)
    assert d["time_logits"].shape == (len(W["seq"]), 8, 5)
    with pytest.raises(SystemExit):
        ts.checked_prediction(tmp_path, pin, "run-2", W, ids)                           # another training identity
    with pytest.raises(SystemExit):
        ts.checked_prediction(tmp_path, {**pin, "sha256": "0" * 64}, "run-1", W, ids)   # another file
    bad = {"centre": W["centre"][::-1].copy(), "seq": W["seq"][::-1].copy(), "marker": (W["marker"] + 1) % 16}
    for k, v in bad.items():
        p = write_npz(tmp_path / f"{k}.npz", W, ids, **{k: v})
        with pytest.raises(SystemExit):
            ts.checked_prediction(tmp_path, p, "run-1", W, ids)                         # reordered or changed windows
    p = write_npz(tmp_path / "ids.npz", W, ids[::-1])
    with pytest.raises(SystemExit):
        ts.checked_prediction(tmp_path, p, "run-1", W, ids)                             # another sequence order
    n = len(W["seq"])
    for k, v in (("time_logits", np.zeros((n, 7, 5), np.float32)), ("logits", np.zeros((n, 4), np.float32)),
                 ("time_logits", np.full((n, 8, 5), np.nan, np.float32))):
        p = write_npz(tmp_path / f"shape_{k}.npz", W, ids, **{k: v})
        with pytest.raises(SystemExit):
            ts.checked_prediction(tmp_path, p, "run-1", W, ids)                         # other shapes or non-finite values


def test_a_wrong_class_order_or_mapping_stops():
    ts.check_mapping(("PLAX", "PSAX", "A4C", "A5C", "SC4C"), "family5_v1")
    for bad in ((("PLAX", "PSAX", "A5C", "A4C", "SC4C"), "family5_v1"), (("PLAX", "PSAX", "A4C", "A5C", "SC4C"), "family5_v2")):
        with pytest.raises(SystemExit):
            ts.check_mapping(*bad)


def test_the_selector_and_the_adapter_are_a03s():
    assert ts.select_all is tt.select_all and ts.half_sets is tt.half_sets and ts.ANCHORS == tt.ANCHORS
    W, _ = fake_windows((40, 3, 16))
    j = ts.select_all(W["s"], W["centre"], W["marker"])
    assert j.tolist() == [tt.selected_anchor(s, c, m) for s, c, m in zip(W["s"], W["centre"], W["marker"])]


def test_the_probability_mapping_is_a03s_bit_for_bit():
    W, seqs = fake_windows((40, 3, 16)); lg = np.random.default_rng(1).normal(size=(len(W["seq"]), 5)).astype(np.float32)
    want = rr.per_sample(W, lg, seqs); got = ts.map_probs(W, ts.softmax(lg), seqs)
    assert all(np.array_equal(got[k], want[k]) for k in want)


def test_mixed_windows_need_two_reference_classes():
    W, seqs = fake_windows((20,)); W["labels16"][1, :3] = 4; W["labels16"][2, :] = 4
    kinds = np.array(["multi"] * len(W["seq"]))
    assert ts.mixed_flags(W, kinds).tolist() == [False, True] + [False] * (len(W["seq"]) - 2)
    assert not ts.mixed_flags(W, np.array(["native"] * len(W["seq"]))).any()     # native windows are pure


# ----------------------------------------------------------------------------------------------- the stem geometry
def test_the_stem_geometry_is_checked_from_the_source():
    g = ts.stem_geometry()
    assert g == {"kernel_t": 3, "stride_t": 2, "padding_t": 1, "later_time_strides": [1] * 16, "marker_stem": "copies conv_proj", "marker_plane": "one slot"}
    with pytest.raises(SystemExit):
        ts.stem_geometry(defaults={"patch_embed_kernel": (5, 7, 7), "patch_embed_stride": (2, 4, 4), "patch_embed_padding": (1, 3, 3)})
    src = inspect.getsource(ts.mvit_module().mvit_v2_s).replace("[1, 2, 2],", "[2, 2, 2],", 1)
    with pytest.raises(SystemExit):
        ts.stem_geometry(v2s_source=src)
    with pytest.raises(SystemExit):
        ts.stem_geometry(target_source=inspect.getsource(ts.tm).replace("conv.kernel_size, conv.stride", "(3, 7, 7), (1, 4, 4)"))


# ----------------------------------------------------------------------------------------------- scope
def test_the_module_never_loads_a_model_or_reads_frames_or_other_splits():
    src = inspect.getsource(ts).lower()
    for banned in ("validation(", "inputs(", "run_evaluate", "pediatric", "split_test", "evaluation/test", "decode_frames", "load_frame",
                   "torch.load", "load_state_dict", "build_model", "mvit_v2_s(", "targetmvit(", ".cuda", "device("):
        assert banned not in src, banned


def test_the_config_pins_the_task_sheet_and_the_a03_sources():
    assert CFG["task_sheet"]["sha256"] == "87f6ac5e52f6e0666873984cb40687d5daa39d01ab67cd06b911d114d2f916b0"
    assert {k: CFG["source"][k]["sha256"][:12] for k in ("a03_config", "a03_summary", "a03_readme")} == {
        "a03_config": "c2e90d605316", "a03_summary": "1cc2936f05a5", "a03_readme": "4d7272929186"}
    assert CFG["budget"]["gpu_hours_max"] == 0.0 and CFG["budget"]["cpu_wall_hours_max"] == 2.0
