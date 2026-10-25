"""A33: 32-slot window plans against an independent loop, marker rule, and the 32-frame adaptation."""
import numpy as np
import pytest
import torch

import echo_routing.temporal.long_support as ls
import echo_routing.vit.target_mvit as tm
from echo_routing.evaluate.window_target import window_frames


def stream(frags):
    """frags: (source, gamma, first raw frame, step, length) -> a constructed stream dict."""
    refs, fr = [], []
    for src, g, f0, step, n in frags:
        refs += [(src, f0 + step * k, g) for k in range(n)]; fr.append([src, 0, "v", 0, n])
    return {"kind": "multi", "refs": refs, "frags": fr}


def loop_reference(a, refs, frag):
    """Independent restatement of the protocol: even slots and slot 31 repeat; odd slots insert only inside a fragment."""
    i32 = []
    for i in range(16):
        i32.append(refs[a[i]])
        if i == 15:
            i32.append(refs[a[i]]); continue
        L, R = refs[a[i]], refs[a[i + 1]]; ok = a[i] != a[i + 1] and frag[a[i]] == frag[a[i + 1]] and L[0] == R[0] and L[2] == R[2]
        mid = (L[1] + R[1]) // 2
        i32.append((L[0], mid, L[2]) if ok and L[1] < mid < R[1] else L)
    return i32


@pytest.mark.parametrize("frags", [
    [("A", 1.0, 0, 3, 40)],                                                 # one fragment, step 3
    [("A", 1.0, 0, 3, 10), ("B", 0.9, 30, 3, 12), ("A", 1.0, 90, 3, 10)],   # source and gamma boundaries, source re-used
    [("A", 1.0, 0, 3, 8), ("A", 1.0, 24, 3, 9)],                           # two contiguous fragments of one source
    [("A", 1.0, 0, 3, 8), ("A", 0.9, 24, 3, 9)],                           # same source, gamma change
    [("A", 1.0, 0, 1, 20)], [("A", 1.0, 0, 2, 20)], [("A", 1.0, 0, 5, 20)],  # no frame between / step 2 / step 5
    [("A", 1.0, 0, 3, 7)], [("A", 1.0, 0, 3, 1)],                           # N < 16: repeated positions
])
def test_plans_match_independent_loop(frags):
    q = stream(frags); frag = ls.fragment_ids(q); n = len(q["refs"])
    for c in range(n):
        a = window_frames(c, n, 16, 16); p = ls.slot_plan(a, q["refs"], frag)
        assert p["I32"] == loop_reference(a, q["refs"], frag)
        assert [p["R32"][2 * i] for i in range(16)] == [p["I32"][2 * i] for i in range(16)] and p["R32"][31] == p["I32"][31]
        assert p["R32"][0] == q["refs"][a[0]] and p["R32"][31] == q["refs"][a[15]]                      # support end points unchanged
        for i in range(15):                                                                            # inserted frames stay inside their pair
            x = p["I32"][2 * i + 1]
            if x != p["R32"][2 * i + 1]:
                assert x[0] == q["refs"][a[i]][0] and q["refs"][a[i]][1] < x[1] < q["refs"][a[i + 1]][1] and frag[a[i]] == frag[a[i + 1]]
        mk = ls.marker32(p["coord"], c); assert p["coord"][mk] == 2 * c


def test_marker_is_even_for_unrepeated_targets_and_middle_for_repeats():
    q = stream([("A", 1.0, 0, 3, 40)]); frag = ls.fragment_ids(q)
    a = window_frames(20, 40, 16, 16); p = ls.slot_plan(a, q["refs"], frag); mk = ls.marker32(p["coord"], 20)
    assert mk % 2 == 0 and p["R32"][mk] == q["refs"][20]
    q = stream([("A", 1.0, 0, 3, 5)]); frag = ls.fragment_ids(q); a = window_frames(2, 5, 16, 16); p = ls.slot_plan(a, q["refs"], frag)
    slots = np.flatnonzero(p["coord"] == 4); assert ls.marker32(p["coord"], 2) == slots[slots.size // 2]


def test_reasons_are_counted():
    q = stream([("A", 1.0, 0, 3, 10), ("B", 0.9, 30, 3, 12)]); frag = ls.fragment_ids(q); a = window_frames(9, 22, 16, 16); p = ls.slot_plan(a, q["refs"], frag)
    assert p["reasons"].count("fragment_boundary") == 1 and p["reasons"][-1] == "last_slot" and len(p["reasons"]) == 16


def test_fragment_check_refuses_mixed_gamma():
    q = stream([("A", 1.0, 0, 3, 10)]); q["refs"][3] = ("A", 9, 0.9)
    with pytest.raises(ValueError):
        ls.fragment_ids(q)


def test_resize_rows_is_torchvision_rule():
    from torchvision.models.video.mvit import _interpolate
    t = torch.randn(15, 96)
    assert torch.equal(ls.resize_rows(t, 31), _interpolate(t, 31)) and torch.equal(ls.resize_rows(t, 15), t)


def test_adapt_changes_only_rel_pos_t_and_matches_runtime_path():
    torch.manual_seed(0); m16 = tm.TargetMViT(tm.build_source_encoder(None)).eval()
    with torch.no_grad():
        m16.marker_stem.weight.normal_(0, 0.01)
    m32, ch = ls.adapt(m16)
    assert len(ch) == 16 and all(c["name"].endswith("attn.rel_pos_t") and c["from"] == [15, 96] and c["to"] == [31, 96] for c in ch)
    s16, s32 = m16.state_dict(), m32.state_dict(); assert all(torch.equal(s16[k], s32[k]) for k in s16 if not k.endswith("rel_pos_t"))
    x = torch.randn(1, 3, 32, 224, 224); mk = tm.marker_planes(torch.tensor([9]), n_slots=32)
    with torch.no_grad():
        assert torch.allclose(m32(x, mk), ls.on_the_fly(m16)(x, mk), atol=1e-5)
