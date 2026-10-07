"""The fan224_v1 input contract (echo_routing/ingest/fan224.py) and the QC/export helpers of
scripts/check_fan224_input.py, on synthetic images only."""
import numpy as np
import pytest
from PIL import Image

from echo_routing.ingest import fan224
from echo_routing.ingest.frames import list_frame_paths
from echo_routing.temporal.joint_visual import crop_uint8, load_frame_u8


def _source(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))


def test_output_is_rgb_uint8_224_and_letterboxed():
    out = fan224.fan224(_source())
    assert out.dtype == np.uint8 and out.shape == (224, 224, 3)
    w, h = fan224.BBOX[2] - fan224.BBOX[0], fan224.BBOX[3] - fan224.BBOX[1]
    assert w > h   # the sector is wider than tall: full width, black bands above and below
    rows = np.where(out.reshape(224, -1).any(1))[0]
    assert rows.min() > 0 and rows.max() < 223 and abs((rows.min()) - (223 - rows.max())) <= 1


def test_pixels_outside_the_template_never_reach_the_output():
    base = np.zeros((240, 320, 3), np.uint8); base[~fan224.MASK] = 255   # bright only outside the template
    out = fan224.fan224(Image.fromarray(base))
    assert out.max() <= 160   # only bilinear bleed at the boundary, no copied interface pixels
    inside = np.zeros((240, 320, 3), np.uint8); inside[fan224.MASK] = 200
    assert fan224.fan224(Image.fromarray(inside)).max() == 200


def test_template_is_fixed_and_matches_its_parameters():
    t = fan224.TEMPLATE
    assert fan224.MASK.shape == (240, 320) and fan224.MASK.dtype == bool
    assert np.array_equal(fan224.template_mask(), fan224.MASK)
    assert not fan224.MASK[:, t["x_max"] + 1:].any()
    assert fan224.MASK[150, 157] and not fan224.MASK[5, 157] and not fan224.MASK[150, 5]
    # the same frame gives the same output; a different frame of the same size leaves the template unchanged
    a, b = _source(1), _source(2)
    assert np.array_equal(fan224.fan224(a), fan224.fan224(a))
    assert np.array_equal(fan224.template_mask(), fan224.MASK)
    assert not np.array_equal(fan224.fan224(a), fan224.fan224(b))


def test_gamma_is_applied_before_masking_and_resizing():
    im = _source(3)
    manual = np.asarray(im.point(fan224.gamma_lut(0.9))).copy(); manual[~fan224.MASK] = 0
    crop = Image.fromarray(manual).crop(fan224.BBOX)
    w, h = crop.size; s = min(224 / w, 224 / h)
    res = crop.resize((round(w * s), round(h * s)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (224, 224)); canvas.paste(res, ((224 - res.width) // 2, (224 - res.height) // 2))
    assert np.array_equal(fan224.fan224(im, 0.9), np.asarray(canvas))
    assert fan224.gamma_lut(0.9) == __import__("echo_routing.temporal.joint_visual", fromlist=["gamma_lut"]).gamma_lut(0.9)


def test_unsupported_sources_and_stored_frames_are_refused(tmp_path):
    with pytest.raises(ValueError, match="320 x 240"):
        fan224.fan224(Image.new("RGB", (224, 224)))     # a 224 x 224 image is never processed a second time
    with pytest.raises(ValueError, match="320 x 240"):
        fan224.fan224(Image.new("RGB", (640, 480)))
    good = tmp_path / "ok.png"; Image.fromarray(fan224.fan224(_source())).save(good)
    assert fan224.read_fan224_png(good).shape == (224, 224, 3)
    bad = tmp_path / "bad.png"; Image.new("RGB", (256, 256)).save(bad)
    with pytest.raises(ValueError, match="224 x 224"):
        fan224.read_fan224_png(bad)
    gray = tmp_path / "gray.png"; Image.new("L", (224, 224)).save(gray)
    with pytest.raises(ValueError, match="RGB"):
        fan224.read_fan224_png(gray)


def test_png_round_trip_is_lossless(tmp_path):
    arr = fan224.fan224(_source(4), 0.9); p = tmp_path / "f.png"; Image.fromarray(arr).save(p)
    assert np.array_equal(fan224.read_fan224_png(p), arr)


def test_legacy_path_is_unchanged(tmp_path):
    im = _source(5); p = tmp_path / "0.jpg"; im.save(p, quality=95)
    with Image.open(p) as j:
        ref = crop_uint8(j.convert("RGB"))
    assert np.array_equal(load_frame_u8(p, 1.0), ref) and ref.shape == (224, 224, 3)


def test_numeric_frame_order(tmp_path):
    for n in (0, 1, 2, 10, 11, 100):
        Image.new("RGB", (320, 240)).save(tmp_path / f"{n}.jpg")
    assert [p.stem for p in list_frame_paths(tmp_path)] == ["0", "1", "2", "10", "11", "100"]


def test_spec_digest_is_stable_and_names_the_version():
    s = fan224.spec()
    assert s["version"] == "fan224_v1" and s["gamma_order"].startswith("gamma LUT before")
    assert fan224.spec_digest() == fan224.spec_digest() and len(fan224.spec_digest()) == 64
    assert 0.8 < fan224.scale_factor() < 1.0


def test_qc_list_and_metrics(tmp_path):
    from scripts.check_fan224_input import CLASSES, pick, qc_list, recording_metrics, pilot_mask
    rows = [{"video_id": f"{c}_{i:03d}", "family5_label": c, "split": "train"} for c in CLASSES for i in range(40)]
    qc = qc_list(rows)
    assert len(qc) == 100 and [r["qc"] for r in qc[:2]] == ["Q001", "Q002"]
    assert sum(r["source"] == "pilot" for r in qc) == 30 and all(sum(r["family5_label"] == c for r in qc) == 20 for c in CLASSES)
    assert pick(rows, 20261006, 6, set()) == pick(rows, 20261006, 6, set())   # deterministic
    with pytest.raises(SystemExit):
        pick(rows[:10], 1, 14, set())
    g = np.zeros((5, 240, 320), np.uint8); g[:, 150, 157] = 200   # a constant bright dot deep inside the template
    g[:, 120, 100:110] = np.arange(5)[:, None] * 40              # a moving segment
    m = recording_metrics(g, fan224.MASK, pilot_mask())
    assert m["constant_bright_inside"] == 1 and m["active_dropped_share"] == 0.0


def test_export_records_tie_both_inputs_to_one_source_frame(tmp_path):
    from scripts.check_fan224_input import export_one
    d = tmp_path / "frames" / "vid"; d.mkdir(parents=True)
    for n in range(12):
        _source(n).save(d / f"{n}.jpg", quality=95)
    recs = export_one(("vid", d, [0, 3, 10], [0.0, 0.1, 0.333], tmp_path, "fan224_test", "digest"))
    assert len(recs) == 6 and {r["variant"] for r in recs} == {"orig", "gamma090"}
    orig = [(r["frame_idx"], r["t_sec"], r["source_file"]) for r in recs if r["variant"] == "orig"]
    g09 = [(r["frame_idx"], r["t_sec"], r["source_file"]) for r in recs if r["variant"] == "gamma090"]
    assert orig == g09 == [(0, 0.0, "0.jpg"), (3, 0.1, "3.jpg"), (10, 0.333, "10.jpg")]
    for r in recs:
        arr = fan224.read_fan224_png(tmp_path / r["png"])
        with Image.open(d / r["source_file"]) as im:
            assert np.array_equal(arr, fan224.fan224(im, r["gamma"]))
