"""CPU checks of the video dense supervision round's training labels (configs/experiments/video_dense_supervision_v1.yaml,
Task 1): the time anchors under the RGB stem's geometry, the SC/RV window pairing (every reference, marker, order and
target), pinned digests, the training allowlist, the anchor labels, and the shuffled control's permutations (fixed
NumPy RNG per window, same label multiset, independent of the global RNGs). Synthetic manifests only; no server file."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import echo_routing.vit.target_mvit as tm
import scripts.prepare_video_dense_supervision as pd

ROOT = Path(__file__).resolve().parents[1]
VD = yaml.safe_load((ROOT / "configs/experiments/video_dense_supervision_v1.yaml").read_text())["video_dense_supervision"]


def manifests(u: int = 3, w: int = 4, seed: int = 0) -> tuple[dict, dict]:
    """A consistent synthetic SC/RV pair: windows of 16 time-ordered slots around c, native (kind 0) and mixed (kind 1)."""
    rng = np.random.default_rng(seed)
    s = np.empty((u, w, 16), np.int32); c = np.empty((u, w), np.int32); labels = np.empty((u, w, 16), np.int8)
    kind = np.zeros((u, w), np.int8); kind[:, w // 2:] = 1
    for i in range(u):
        for j in range(w):
            centre = int(rng.integers(10, 40)); s[i, j] = np.clip(np.arange(centre - 8, centre + 8), 0, None); c[i, j] = centre
            if kind[i, j] == 0:
                labels[i, j] = rng.integers(0, 5)
            else:
                cut = int(rng.integers(1, 15)); a, b = rng.choice(5, 2, replace=False); labels[i, j, :cut], labels[i, j, cut:] = a, b
    target = np.array([[labels[i, j][s[i, j] == c[i, j]][0] for j in range(w)] for i in range(u)], np.int8)
    common = {"vid": rng.integers(0, 6, (u, w, 16)).astype(np.int32), "frame": rng.integers(0, 90, (u, w, 16)).astype(np.int32),
              "gamma": rng.uniform(0.8, 1.2, (u, w, 16)), "s": s, "c": c, "target": target, "kind": kind,
              "seq": rng.integers(0, 50, (u, w)).astype(np.int32), "repeats": np.zeros((u, w), np.int8),
              "videos": np.array([f"rec{k}" for k in range(6)])}
    sc = {**{k: v.copy() for k, v in common.items()}, "labels": labels, "target_vid": np.zeros((u, w), np.int32), "rel": np.zeros((u, w, 16), np.int8)}
    marker = np.array([[tm.marker_slot(s[i, j], int(c[i, j])) for j in range(w)] for i in range(u)], np.int8)
    rv = {**{k: v.copy() for k, v in common.items()}, "marker": marker, "order_ordered": np.tile(np.arange(16, dtype=np.int8), (u, w, 1)),
          "digest_ordered": np.zeros((u, w, 32), np.uint8)}
    return sc, rv


def test_anchors_follow_the_stem_geometry():
    assert pd.stem_anchors(3, 2, 1, 1, 8) == [0, 2, 4, 6, 8, 10, 12, 14] == VD["model"]["anchors"]
    with pytest.raises(ValueError):
        pd.stem_anchors(2, 2, 1, 1, 8)            # half-integer centres
    with pytest.raises(ValueError):
        pd.stem_anchors(3, 3, 1, 1, 8)            # an anchor beyond the 16 input slots


def test_consistent_manifests_pair_and_give_counts():
    sc, rv = manifests()
    out = pd.check_pairing(sc, rv)
    assert out["windows"] == 12 and out["mismatches"] == 0


@pytest.mark.parametrize("field", ["vid", "frame", "gamma", "s", "c", "target", "kind", "seq", "repeats"])
def test_any_reference_difference_is_rejected(field):
    sc, rv = manifests()
    arr = rv[field]
    if arr.dtype.kind == "f":
        arr[1, 2, 3] += 1e-9
    elif arr.ndim == 3:
        arr[1, 2, 3] += 1
    else:
        arr[1, 2] += 1
    with pytest.raises(ValueError):
        pd.check_pairing(sc, rv)


def test_marker_order_video_list_and_target_labels_are_checked():
    for change in ("marker", "order", "videos", "label"):
        sc, rv = manifests()
        if change == "marker":
            rv["marker"][0, 0] = (rv["marker"][0, 0] + 1) % 16
        elif change == "order":
            rv["order_ordered"][2, 1, [0, 1]] = [1, 0]
        elif change == "videos":
            rv["videos"] = rv["videos"].copy(); rv["videos"][3] = "other"
        else:
            sl = np.flatnonzero(sc["s"][0, 0] == sc["c"][0, 0]); sc["labels"][0, 0, sl] = (sc["labels"][0, 0, sl] + 1) % 5
        with pytest.raises(ValueError):
            pd.check_pairing(sc, rv)


def test_pinned_files_must_match_their_digest(tmp_path):
    f = tmp_path / "x.npz"; f.write_bytes(b"abc")
    good = {"path": "x.npz", "sha256": pd.sha256_file(f)}
    assert pd.pinned(tmp_path, good) == f
    with pytest.raises(SystemExit):
        pd.pinned(tmp_path, {"path": "x.npz", "sha256": "0" * 64})


def test_recordings_outside_the_training_allowlist_are_rejected():
    pd.check_allowlist(["rec0", "rec1"], {"rec0", "rec1", "rec2"})
    with pytest.raises(SystemExit):
        pd.check_allowlist(["rec0", "dev7"], {"rec0", "rec1"})


def test_aux_labels_are_the_anchor_slot_labels():
    sc, _ = manifests()
    a = pd.aux_labels(sc["labels"], VD["model"]["anchors"])
    assert a.shape == (3, 4, 8) and a.dtype == np.int8 and np.array_equal(a, sc["labels"][:, :, ::2])


def test_permutations_use_the_fixed_window_rng_and_keep_the_multiset():
    sc, _ = manifests(u=5, w=8, seed=3); a = pd.aux_labels(sc["labels"], VD["model"]["anchors"])
    perm = pd.permutations(seed=2, n_updates=5, n_windows=8); sh = pd.shuffled_labels(a, perm)
    assert perm.shape == (5, 8, 8) and perm.dtype == np.int8
    for u in range(5):
        for w in range(8):
            assert perm[u, w].tolist() == np.random.default_rng([20261013, 2, u, w]).permutation(8).tolist()
            assert sorted(sh[u, w].tolist()) == sorted(a[u, w].tolist()) and sh[u, w].tolist() == a[u, w][perm[u, w]].tolist()


def test_native_windows_keep_their_labels_after_shuffling():
    sc, _ = manifests(u=4, w=6, seed=5); a = pd.aux_labels(sc["labels"], VD["model"]["anchors"])
    sh = pd.shuffled_labels(a, pd.permutations(seed=0, n_updates=4, n_windows=6)); native = sc["kind"] == 0
    assert native.any() and np.array_equal(sh[native], a[native])


def test_permutations_ignore_the_global_rngs():
    torch.manual_seed(1); np.random.seed(1); first = pd.permutations(seed=1, n_updates=3, n_windows=8)
    state = torch.get_rng_state(); np_state = np.random.get_state()[1].copy()
    torch.manual_seed(99); np.random.seed(99); second = pd.permutations(seed=1, n_updates=3, n_windows=8)
    assert np.array_equal(first, second)
    torch.manual_seed(1); np.random.seed(1); pd.permutations(seed=1, n_updates=3, n_windows=8)
    assert torch.equal(torch.get_rng_state(), state) and np.array_equal(np.random.get_state()[1], np_state)


def test_label_summary_counts_changes_and_repeats():
    sc, _ = manifests(u=4, w=6, seed=7); a = pd.aux_labels(sc["labels"], VD["model"]["anchors"])
    perm = pd.permutations(seed=0, n_updates=4, n_windows=6); sh = pd.shuffled_labels(a, perm)
    s = pd.label_summary(a, sh, perm, sc["kind"], sc["s"], VD["model"]["anchors"])
    mixed = sc["kind"] == 1
    assert s["windows"] == 24 and s["mixed_windows"] == int(mixed.sum())
    assert s["identity_permutations"] == int(sum((perm[u, w] == np.arange(8)).all() for u in range(4) for w in range(6)))
    assert s["mixed_windows_with_changed_labels"] == int(sum((sh[u, w] != a[u, w]).any() for u in range(4) for w in range(6) if mixed[u, w]))
    assert s["native_windows_with_changed_labels"] == 0
    assert 0.0 <= s["repeated_anchor_share"] <= 1.0
