"""scripts/train_fan224_visual.py: the batch plan, its digests, the frame reader, training-split access, the schedule
and (where the frozen E1 export exists) the trainable sets. Synthetic data only."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import scripts.train_fan224_visual as tv
from echo_routing.ingest import fan224
from echo_routing.temporal.dev_train import lr_factor
from echo_routing.temporal.joint_visual import load_frame_u8

ROOT = Path(__file__).resolve().parents[1]
FV = {"classes": ["PLAX", "PSAX", "A4C", "A5C", "SC4C"],
      "sampling": {"data_seed_base": 20261007, "updates": 40, "recordings_per_update": 16, "samples_per_recording": 4,
                   "gamma": {"orig": 0.5, "gamma090": 0.5}}}


def _table(n_per_class: int = 6) -> dict:
    vids, labels, frames = [], [], []
    for c in range(5):
        for i in range(n_per_class):
            vids.append(f"v{c}_{i}"); labels.append(c); n = 3 if i == 0 else 9 + 3 * i   # one short recording per class
            frames.append(np.arange(n, dtype=np.int64) * 3)
    return {"videos": vids, "labels": np.array(labels), "frames": frames, "frames_dir": {}}


def test_plan_is_deterministic_balanced_and_consistent():
    t = _table(); a = tv.make_plan(FV, t, 0); b = tv.make_plan(FV, t, 0); c = tv.make_plan(FV, t, 1)
    assert all(np.array_equal(a[k], b[k]) for k in a) and not np.array_equal(a["vid"], c["vid"])
    assert a["vid"].shape == (40, 16) and a["pos"].shape == a["frame"].shape == (40, 16, 4)
    assert np.array_equal(a["label"], t["labels"][a["vid"]])
    for u in range(40):
        for j in range(16):
            v = a["vid"][u, j]; n = len(t["frames"][v])
            assert np.all((a["pos"][u, j] >= 0) & (a["pos"][u, j] < n)) and np.all(np.diff(a["pos"][u, j]) >= 0)
            assert np.array_equal(a["frame"][u, j], t["frames"][v][a["pos"][u, j]])
            if n >= 4:   # spaced_indices: one sample per equal-width bin (neighbours can share a frame at fractional edges)
                edges = np.linspace(0, n, 5)
                assert all(np.floor(edges[i]) <= a["pos"][u, j][i] <= min(np.floor(edges[i + 1]), n - 1) for i in range(4))
    assert set(np.unique(a["gamma"]).tolist()) == {0.9, 1.0} and 0.4 < (a["gamma"] == 0.9).mean() < 0.6
    counts = np.bincount(a["label"].ravel(), minlength=5) / a["label"].size
    assert np.all(np.abs(counts - 0.2) < 0.06)


def test_short_recordings_are_sampled_with_replacement():
    t = _table(); p = tv.make_plan({**FV, "sampling": {**FV["sampling"], "updates": 200}}, t, 3)
    short = [(u, j) for u in range(200) for j in range(16) if len(t["frames"][p["vid"][u, j]]) < 4]
    assert short and all(np.all(p["pos"][u, j] < 3) for u, j in short)


def test_plan_refuses_other_gamma_shares():
    with pytest.raises(SystemExit):
        tv.make_plan({**FV, "sampling": {**FV["sampling"], "gamma": {"orig": 0.7, "gamma090": 0.3}}}, _table(), 0)


def test_update_digests_follow_frames_and_gamma():
    p = tv.make_plan(FV, _table(), 0); d = tv.update_digests(p)
    assert len(d) == 40 and len(set(d)) == 40
    q = {**p, "gamma": p["gamma"].copy()}; q["gamma"][5, 0] = 1.9 - q["gamma"][5, 0]
    e = tv.update_digests(q)
    assert e[5] != d[5] and e[:5] == d[:5] and e[6:] == d[6:]


def test_saved_plan_is_reused_or_refused(tmp_path):
    t = _table(); path, p = tv.save_or_check_plan(FV, t, 0, tmp_path)
    assert path.is_file()
    path2, _ = tv.save_or_check_plan(FV, t, 0, tmp_path)
    assert path2 == path
    with np.load(path) as z:
        arrs = {k: z[k] for k in z.files}
    arrs["gamma"][0, 0] = 1.9 - arrs["gamma"][0, 0]; np.savez(path, **arrs)
    with pytest.raises(SystemExit, match="differs"):
        tv.save_or_check_plan(FV, t, 0, tmp_path)


def _frames_on_disk(tmp_path: Path, videos: list[str], n: int = 12):
    rng = np.random.default_rng(0); dirs = {}
    for v in videos:
        d = tmp_path / "frames" / v; d.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)).save(d / f"{i}.jpg", quality=95)
        dirs[v] = str(d)
        for variant, g in (("orig", 1.0), ("gamma090", 0.9)):
            out = tmp_path / "fan" / variant / v; out.mkdir(parents=True)
            for i in range(0, n, 3):
                with Image.open(d / f"{i}.jpg") as im:
                    Image.fromarray(fan224.fan224(im, g)).save(out / f"{i:06d}.png")
    return dirs


def test_frame_reader_gives_each_input_its_own_pixels(tmp_path):
    from scripts.train_joint_visual_temporal import FramePaths
    dirs = _frames_on_disk(tmp_path, ["va", "vb"])
    plan = {"videos": np.array(["va", "vb"]), "vid": np.array([[0, 1]]), "frame": np.array([[[0, 3, 6, 9], [0, 3, 6, 9]]]),
            "gamma": np.array([[1.0, 0.9]]), "label": np.array([[2, 3]])}
    fp = FramePaths(dirs)
    legacy = tv.UpdateFrames(plan, "legacy", fp, tmp_path / "fan", [0])[0]
    fan = tv.UpdateFrames(plan, "fan224_v1", fp, tmp_path / "fan", [0])[0]
    assert legacy[0].shape == fan[0].shape == (8, 224, 224, 3) and legacy[1].tolist() == [2] * 4 + [3] * 4 and legacy[2] == 0
    assert np.array_equal(legacy[0][4].numpy(), load_frame_u8(Path(dirs["vb"]) / "0.jpg", 0.9))
    with Image.open(Path(dirs["vb"]) / "3.jpg") as im:
        assert np.array_equal(fan[0][5].numpy(), fan224.fan224(im, 0.9))
    assert not np.array_equal(legacy[0][0].numpy(), fan[0][0].numpy())
    with pytest.raises(ValueError):
        tv.UpdateFrames(plan, "other", fp, tmp_path, [0])


def test_png_contract_detects_a_tampered_frame(tmp_path):
    from scripts.train_joint_visual_temporal import FramePaths
    dirs = _frames_on_disk(tmp_path, ["va"])
    plan = {"videos": np.array(["va"]), "vid": np.array([[0]]), "frame": np.array([[[0, 3, 6, 9]]]), "gamma": np.array([[0.9]]), "label": np.array([[1]])}
    assert tv.png_contract(plan, FramePaths(dirs), tmp_path / "fan") == 4
    p = tmp_path / "fan" / "gamma090" / "va" / "000003.png"; arr = np.asarray(Image.open(p)).copy(); arr[100, 100] ^= 1
    Image.fromarray(arr).save(p)
    with pytest.raises(SystemExit, match="differs"):
        tv.png_contract(plan, FramePaths(dirs), tmp_path / "fan")


class _Paths:
    def __init__(self, root: Path) -> None:
        self.repo_root, self.cache_root, self.manifest_path = root, root / "cache", root / "manifest.csv"


def test_training_table_reads_the_training_split_only(tmp_path, monkeypatch):
    import json
    vids = ["t1", "t2", "v1"]
    (tmp_path / "protocol.json").write_text(json.dumps({"train": {"clips": ["t2", "t1"]}, "validation": {"clips": ["v1"]}}))
    idx = pd.DataFrame({"video_id": vids, "split": ["train", "train", "validation"], "label_index": [0, 3, 2]})
    man = pd.DataFrame({"video_id": vids, "split": ["train", "train", "validation"], "family5_label": ["PLAX", "A5C", "A4C"],
                        "frames_dir": ["d1", "d2", "d3"]})
    seen = []

    class Store:
        def __init__(self, *a):
            pass

        def get(self, v, variant):
            seen.append((v, variant)); return type("VF", (), {"frame_idx": np.arange(10) * 3})()
    monkeypatch.setattr(tv, "read_index", lambda *a: idx); monkeypatch.setattr(tv, "load_manifest", lambda *a: man)
    monkeypatch.setattr(tv, "FeatureStore", Store)
    fv = {"classes": FV["classes"], "inputs": {"protocol": {"path": "protocol.json"}, "cache_key": "k"}}
    t = tv.train_table(fv, _Paths(tmp_path))
    assert t["videos"] == ["t1", "t2"] and t["labels"].tolist() == [0, 3] and {v for v, _ in seen} == {"t1", "t2"}
    man.loc[1, "family5_label"] = "A4C"
    with pytest.raises(SystemExit, match="disagree"):
        tv.train_table(fv, _Paths(tmp_path))
    (tmp_path / "protocol.json").write_text(json.dumps({"train": {"clips": ["t1", "v1"]}}))
    with pytest.raises(SystemExit, match="not in the training split"):
        tv.train_table(fv, _Paths(tmp_path))


def test_schedule_is_linear_warmup_then_cosine_to_zero():
    assert lr_factor(0, 100, 1500) == pytest.approx(0.01) and lr_factor(99, 100, 1500) == pytest.approx(1.0)
    assert lr_factor(100, 100, 1500) == pytest.approx(1.0) and lr_factor(1499, 100, 1500) < 1e-5


def test_build_trainable_sets_on_a_fake_vit(monkeypatch):
    import types

    import torch
    from tests.test_joint_visual_temporal import FakeViT
    monkeypatch.setattr(tv, "load_vision_transformer", lambda repo, export: (FakeViT(d=16, layers=12, out=512), {}))
    probe = {"weight": torch.randn(5, 512), "bias": torch.randn(5)}
    monkeypatch.setattr(tv, "load_probe_state", lambda fv, root: {k: v.clone() for k, v in probe.items()})
    paths = types.SimpleNamespace(repo_root=ROOT); fv = {"classes": FV["classes"], "inputs": {"visual_export": {"path": "unused"}}}
    for arm in ("L0", "F0"):
        enc, head = tv.build(fv, paths, arm, torch.device("cpu"))
        assert not [n for n, p in enc.vt.named_parameters() if p.requires_grad]
        assert torch.equal(head.weight, probe["weight"]) and torch.equal(head.bias, probe["bias"])
    for arm in ("L1", "F1"):
        enc, _ = tv.build(fv, paths, arm, torch.device("cpu"))
        assert {n for n, p in enc.vt.named_parameters() if p.requires_grad} == {
            "transformer.resblocks.11.ln.weight", "transformer.resblocks.11.ln.bias", "transformer.resblocks.11.fc.weight",
            "transformer.resblocks.11.fc.bias", "ln_post.weight", "ln_post.bias", "proj"}
    monkeypatch.setattr(tv.VisualEncoder, "is_trainable", lambda self, name: name.startswith(("transformer.resblocks.10.", "transformer.resblocks.11.", "ln_post.", "proj")))
    with pytest.raises(SystemExit, match="block 11, ln_post and proj"):
        tv.build(fv, paths, "L1", torch.device("cpu"))


EXPORT = ROOT / "models_frozen/e1_echoviewclip_probe/v1/visual_encoder_fp32.pt"


@pytest.mark.skipif(not EXPORT.is_file() or not (ROOT / "third_party/EchoViewCLIP").is_dir(), reason="needs the frozen E1 export")
def test_trainable_sets_and_probe_start():
    import torch
    from echo_routing.config import load_config, load_paths
    fv = load_config(ROOT / "configs/experiments/fan224_visual_adaptation_v1.yaml")["fan224"]; paths = load_paths()
    probe = tv.load_probe_state(fv, ROOT)
    for arm in ("L0", "F0"):
        enc, head = tv.build(fv, paths, arm, torch.device("cpu"))
        assert not [n for n, p in enc.vt.named_parameters() if p.requires_grad]
        assert torch.equal(head.weight, probe["weight"]) and torch.equal(head.bias, probe["bias"])
    enc, _ = tv.build(fv, paths, "F1", torch.device("cpu"))
    names = [n for n, p in enc.vt.named_parameters() if p.requires_grad]
    assert names and all(n.startswith(tv.TRAINED_ENCODER) for n in names) and "proj" in names
