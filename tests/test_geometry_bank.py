"""Person-balanced sampling and the final model input of the pair bank."""
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from echo_routing.geometry import bank


def registry():
    vols = [bank.VolumeInfo(f"echoslicer:A_{i}", "echoslicer:A", 20) for i in range(15)]
    vols += [bank.VolumeInfo("echoslicer:B_0", "echoslicer:B", 15), bank.VolumeInfo("echoslicer:C_0", "echoslicer:C", 15)]
    vols += [bank.VolumeInfo(f"stacom:v{i}", f"stacom:v{i}", 14) for i in (1, 2, 14)]
    return vols


def full_coverage(vols, value=0.5):
    return {v.volume_id: np.full(len(bank.PSI_GRID), value) for v in vols}


def test_people_split_and_balance():
    vols = registry(); held = {"echoslicer:C", "stacom:v14"}
    train = bank.people_of(vols, held, pretraining=True); test = bank.people_of(vols, held, pretraining=False)
    assert set(train) == {"echoslicer:A", "echoslicer:B", "stacom:v1", "stacom:v2"} and set(test) == held
    specs = bank.sample_specs(np.random.default_rng(0), train, full_coverage(vols), 4000)
    per_person = Counter(v.person for v, _ in specs)
    assert all(abs(c / 4000 - 0.25) < 0.03 for c in per_person.values())          # A's 15 volumes count as one person
    assert len({v.volume_id for v, _ in specs if v.person == "echoslicer:A"}) == 15


def test_psi_drawn_only_where_coverage_is_enough():
    vols = registry()[:1]; cov = np.zeros(len(bank.PSI_GRID)); cov[[10, 50]] = [0.3, 0.6]
    specs = bank.sample_specs(np.random.default_rng(1), {"echoslicer:A": vols}, {vols[0].volume_id: cov}, 200)
    assert {s.psi_deg for _, s in specs} == {float(bank.PSI_GRID[10]), float(bank.PSI_GRID[50])}
    assert all(s.psi_deg == round(s.psi_deg) for _, s in specs)


def test_volume_without_any_usable_psi_is_refused():
    vols = registry()[:1]
    with pytest.raises(ValueError):
        bank.sample_specs(np.random.default_rng(2), {"echoslicer:A": vols}, {vols[0].volume_id: np.zeros(len(bank.PSI_GRID))}, 5)


def test_final_input_matches_e1_geometry_on_rgb():
    rng = np.random.default_rng(3); img = rng.uniform(0, 200, size=(300, 320)).astype(np.float32)
    mask = np.zeros((300, 320), bool); mask[20:280, 30:300] = True
    x, m = bank.final_input(img, mask, mean=60.0, sd=40.0)
    assert x.shape == m.shape == (224, 224) and x.dtype == np.uint8
    # E1 converts frames to RGB before resizing; on a grey image every channel must equal the grey result
    u8 = np.round(bank.standardise(img, mask, 60.0, 40.0)).astype(np.uint8)
    rgb = np.asarray(bank.crop_like_e1(Image.fromarray(u8).convert("RGB")))
    assert np.array_equal(rgb[..., 0], x) and np.array_equal(rgb[..., 2], x)
    assert 0.3 < m.mean() < 1.0 and bank.final_features(x, m).shape == (4,)
