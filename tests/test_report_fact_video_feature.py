"""Report side of the FACT video feature round (configs/experiments/fact_video_feature_v1.yaml): the half-bounded reader
with an I/O spy, access before labels, the committed evaluation-manifest barrier with its git-object proof, the one-time
selection marker, reuse of saved outputs of the same identity, the stored reference rows, the goal without a B3
condition, and every reading branch (table-driven). CPU only, synthetic data; no real feature, label or id.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.report_fact_video_feature as rf

POS, NEG, NONE = "consistent positive", "consistent negative", "no consistent difference detected"
A, L, O, M = "FACT - A01 ASRF", "FACT - A01 Linear + B3", "FACT - frozen Ordered + B3", "FACT - A01 MS-TCN"
GOAL_CFG = {"selection_min": 0.944409687184662, "calibration_min": 0.9821421709894332, "calibration_sensitivity": 0.9838935784747763,
            "boundary_recall_min": 0.95, "native_macro_f1_min": {"selection": 0.973, "calibration": 0.980}}


# ----------------------------------------------------------------------------------------------- the half-bounded reader
def write_dev_cache(path: Path, first: list[str], second: list[str]) -> dict:
    rng = np.random.default_rng(1); ids = first + second; n = rng.integers(7, 15, len(ids)); s_off = np.concatenate([[0], np.cumsum(n)])
    nwin = (n + 4) // 5; w_off = np.concatenate([[0], np.cumsum(nwin)])
    samples = rng.standard_normal((int(s_off[-1]), 512)).astype(np.float32); window = rng.standard_normal((int(w_off[-1]), 512)).astype(np.float32)
    k = len(first); samples[:s_off[k]] = 1e30; window[:w_off[k]] = 1e30
    with open(path, "wb") as f:
        np.savez(f, identity="dev-id", ids=np.asarray(ids), n=n, sample_offsets=s_off, window_offsets=w_off, centres=np.arange(int(w_off[-1])),
                 markers=np.zeros(int(w_off[-1]), np.int64), window=window, samples=samples)
    return {"samples": samples, "s_off": s_off, "w_off": w_off}


class Recorder:
    def __init__(self) -> None:
        self.reads: list[tuple[int, int]] = []

    def __call__(self, path, mode="rb"):
        rec = self.reads

        class F(io.FileIO):
            def read(self, size=-1):
                pos = self.tell(); out = super().read(size); rec.append((pos, len(out) if out else 0)); return out
        return F(path, "r")


def test_calibration_reads_never_touch_selection_rows(tmp_path):
    p = tmp_path / "dev.npz"; sel, cal = ["s0", "s1", "s2"], ["c0", "c1", "c2"]; d = write_dev_cache(p, sel, cal); rec = Recorder()
    got = rf.open_half(p, "calibration", cal, rf.DevAccess({"calibration"}), opener=rec)
    assert list(got["samples"]) == cal and not any((v > 1e29).any() for v in got["samples"].values())
    lay = rf.member_layout(p, "samples"); lo, hi = lay.offset, lay.offset + int(d["s_off"][3]) * 512 * 4
    assert not any(n > 0 and pos < hi and pos + n > lo for pos, n in rec.reads)
    with pytest.raises(SystemExit):
        rf.open_half(p, "selection", sel, rf.DevAccess({"calibration"}))


def test_label_metadata_of_a_closed_half_is_never_read():
    c = object.__new__(rf.Ctx)
    with pytest.raises(SystemExit, match="not open"):
        rf.Ctx.half(c, "selection", rf.DevAccess({"calibration"}))


def test_report_never_calls_an_encoder_or_an_eager_loader():
    src = inspect.getsource(rf)
    for name in ("validation(", "Inputs(", "dev_features(", "build_source(", "pv.extract(", "np.load(", "build_model(", ".embed("):
        assert name not in src, name


# ----------------------------------------------------------------------------------------------- the manifest barrier
def git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_manifest_barrier_accepts_only_the_committed_manifest(tmp_path):
    r = tmp_path / "repo"; rel = "docs/results/x/evaluation_manifest.json"; (r / "docs" / "results" / "x").mkdir(parents=True); git(r, "init", "-q")
    prov = {"report_code": "c", "config_sha256": "k", "runs": {"FACT_s0": {"identity": "i", "final_sha256": "f"}}}
    data = json.dumps({"provenance": prov, "decoding": "official"}, indent=1).encode(); (r / rel).write_bytes(data)
    git(r, "add", "."); git(r, "commit", "-q", "-m", "manifest"); commit = git(r, "rev-parse", "HEAD")
    twin = tmp_path / "twin.json"; twin.write_bytes(data); proof = tmp_path / "proof.json"; proof.write_text(json.dumps(rf.git_proof(r, commit, rel)))
    sha = hashlib.sha256(data).hexdigest()
    assert rf.manifest_barrier(r / rel, twin, proof, rel, sha, commit, prov)["decoding"] == "official"
    for args in ((r / rel, twin, proof, rel, "f" * 64, commit, prov), (r / rel, twin, proof, rel, sha, commit[:12], prov),
                 (r / rel, twin, proof, rel, sha, commit, {**prov, "report_code": "other"}), (r / rel, twin, tmp_path / "none.json", rel, sha, commit, prov),
                 (r / rel, twin, proof, rel, None, commit, prov)):
        with pytest.raises(SystemExit):
            rf.manifest_barrier(*args)
    twin.write_bytes(data + b"\n")
    with pytest.raises(SystemExit):
        rf.manifest_barrier(r / rel, twin, proof, rel, sha, commit, prov)


def test_selection_is_opened_once_and_a_repeat_needs_a_reason(tmp_path):
    ctx = SimpleNamespace(manifest_sha256="a" * 64, manifest_commit="b" * 40, repeat_reason=None, ledger=tmp_path / "ledger.jsonl"); out = tmp_path / "selection"
    assert len(rf.open_selection(ctx, out)) == 1
    with pytest.raises(SystemExit):
        rf.open_selection(ctx, out)
    ctx.repeat_reason = "interrupted after the selection half was opened"
    assert len(rf.open_selection(ctx, out)) == 2
    (out / "done.json").write_text("{}")
    with pytest.raises(SystemExit):
        rf.open_selection(ctx, out)


def test_saved_outputs_are_reused_only_under_the_same_identity(tmp_path):
    outs = {"a": {"labels": np.array([0, 0, 1]), "frame_labels": np.array([0, 1, 1]), "tokens_used": 2, "runs": 2, "tokens": 2, "separate_intervals": False},
            "b": {"labels": np.array([4]), "frame_labels": np.array([4]), "tokens_used": 1, "runs": 1, "tokens": 1, "separate_intervals": True}}
    p = tmp_path / "FACT_s0.npz"
    assert rf.load_outputs(p, "ident") is None
    rf.save_outputs(p, "ident", outs); back = rf.load_outputs(p, "ident")
    assert set(back) == set(outs) and all(np.array_equal(back[k]["labels"], outs[k]["labels"]) and back[k]["tokens_used"] == outs[k]["tokens_used"]
                                          and back[k]["separate_intervals"] == outs[k]["separate_intervals"] for k in outs)
    with pytest.raises(SystemExit):
        rf.load_outputs(p, "other identity")


# ----------------------------------------------------------------------------------------------- references and goal
def test_stored_rows_are_listed_never_dropped():
    row = {"segment_f1_50_multi": 0.9}
    summary = {"systems": {**{k: {"selection": row, "calibration": row} for k in ("E0 + B3", "E1 + B3", "B7 + B3")},
                           **{f"Original FACT s{k}": {"selection": row, "calibration": row} for k in (0, 1, 2)}}}
    rows = rf.stored_rows(summary)
    assert set(rows) == {"E0 + B3", "E1 + B3", "B7 + B3", "F02 FACT s0", "F02 FACT s1", "F02 FACT s2"}
    del summary["systems"]["Original FACT s1"]
    with pytest.raises(SystemExit):
        rf.stored_rows(summary)


def row(seg, rec=0.96, nat=0.99):
    return {"segment_f1_50_multi": seg, "boundary_recall_multi": rec, "native_macro_f1": nat}


def test_goal_has_no_b3_condition_and_keeps_the_floors():
    ok = [{"selection": row(0.95), "calibration": row(0.985)} for _ in range(3)]
    g = rf.goal_of(ok, GOAL_CFG)
    assert g["met"] and not g["constraint_failures"]
    low = [dict(r) for r in ok]; low[1] = {"selection": row(0.95, nat=0.972), "calibration": row(0.985)}
    g = rf.goal_of(low, GOAL_CFG)
    assert not g["met"] and g["constraint_failures"] == [{"seed": 1, "half": "selection", "b3_feasible": True, "boundary_recall": 0.96, "native_macro_f1": 0.972}]


# ----------------------------------------------------------------------------------------------- readings, table-driven
def ver(a=(NONE, NONE), l=(NONE, NONE), o=(NONE, NONE), m=(NONE, NONE)):
    return {c: dict(zip(("selection", "calibration"), v)) for c, v in ((A, a), (L, l), (O, o), (M, m))}


MET = {"met": True, "constraint_failures": []}
UNMET = {"met": False, "constraint_failures": []}
FAIL = {"met": False, "constraint_failures": [{"seed": 0, "half": "selection"}]}
PP = (POS, POS)
CASES = [
    ("nothing", ver(), MET, [], [], ["uncertain_or_no_qualifying_gain", "end"]),
    ("complete", ver(PP, PP, PP), MET, [], [], ["complete_system_progress", "one_half_or_local_effect", "end"]),
    ("no strong reference win", ver(PP, PP), MET, [], [], ["learned_system_gain_without_strong_reference_win", "one_half_or_local_effect",
                                                            "uncertain_or_no_qualifying_gain", "end"]),
    ("all comparisons but a floor fails", ver(PP, PP, PP), FAIL, [], [], ["learned_system_gain_without_strong_reference_win", "one_half_or_local_effect",
                                                                           "uncertain_or_no_qualifying_gain", "end"]),
    ("all comparisons but the goal", ver(PP, PP, PP), UNMET, [], [], ["learned_system_gain_without_strong_reference_win", "one_half_or_local_effect",
                                                                       "uncertain_or_no_qualifying_gain", "end"]),
    ("one half against MS-TCN", ver(m=(POS, NONE)), MET, [], [], ["one_half_or_local_effect", "uncertain_or_no_qualifying_gain", "end"]),
    ("only ASRF in both halves", ver(PP), MET, [], [], ["one_half_or_local_effect", "uncertain_or_no_qualifying_gain", "end"]),
    ("point estimates only", ver(), MET, [{"control": "A01 ASRF", "half": "selection", "what": "short"}], [],
     ["one_half_or_local_effect", "uncertain_or_no_qualifying_gain", "end"]),
    ("trade-off", ver(), MET, [], [{"seed": 0, "half": "calibration", "control": "frozen Ordered + B3"}],
     ["trade_off", "uncertain_or_no_qualifying_gain", "end"]),
    ("negative", ver((NEG, NEG), (NEG, NEG)), MET, [], [], ["uncertain_or_no_qualifying_gain", "end"]),
]


@pytest.mark.parametrize("name,v,goal,points,trade,want", CASES, ids=[c[0] for c in CASES])
def test_readings(name, v, goal, points, trade, want):
    local = rf.local_evidence(v, points)
    r = rf.readings(v, goal, local, trade)
    assert r["applied"] == want
    if "learned_system_gain_without_strong_reference_win" in want:
        failed = r["notes"]["learned_system_gain_without_strong_reference_win"]["failed"]
        assert failed and all(isinstance(x, str) for x in failed)
    if "uncertain_or_no_qualifying_gain" in want:
        assert r["notes"]["uncertain_or_no_qualifying_gain"]["primary"] == v[A]


def seg(rec=0, lost=0, new=0, rec_short=0, lost_short=0, rec_xyx=0, lost_xyx=0):
    b = lambda n, s=0, x=0: {"segments": n, "short": s, "xyx_a4c_a5c": x, "sessions": n, "few_units": True}  # noqa: E731
    return {"recovered": b(rec, rec_short, rec_xyx), "lost": b(lost, lost_short, lost_xyx), "new_unmatched_pred": b(new), "removed_unmatched_pred": b(0)}


def test_point_estimate_items_come_from_pooled_multi_stream_recoveries():
    acc = {}
    for s in (0, 1, 2):
        for h in ("selection", "calibration"):
            for c in ("A01 ASRF", "A01 Linear + B3", "frozen Ordered + B3", "A01 MS-TCN"):
                multi = seg(rec=2, lost=1, rec_short=1, lost_short=1) if (c == "A01 ASRF" and h == "selection") else seg(rec=1, lost=1)
                acc[f"s{s}|FACT vs {c}|{h}"] = {"segments": {"native": seg(), "pairs": seg(), "multi": multi}}
    items = rf.point_estimate_items(acc)
    assert items == [{"control": "A01 ASRF", "half": "selection", "population": "multi", "what": "all reference segments", "recovered": 6, "lost": 3}]
    acc["s0|FACT vs frozen Ordered + B3|calibration"]["segments"]["pairs"] = seg(rec=1, rec_short=1)       # a pair-stream recovery counts in its own population
    assert {(i["control"], i["population"], i["what"]) for i in rf.point_estimate_items(acc)} == {
        ("A01 ASRF", "multi", "all reference segments"), ("frozen Ordered + B3", "pairs", "all reference segments"),
        ("frozen Ordered + B3", "pairs", "segments of at most 16 samples")}


def test_denominators_count_reference_segments_per_population():
    seqs = [{"kind": "native", "labels": np.full(20, 2)}, {"kind": "pairs", "labels": np.array([2] * 10 + [3] * 30)},
            {"kind": "multi", "labels": np.array([2] * 30 + [3] * 8 + [2] * 30 + [0] * 5)}]
    d = rf.denominators(seqs)
    assert d["native"] == {"all reference segments": 1, "segments of at most 16 samples": 0, "A4C/A5C X-Y-X middles": 0}
    assert d["pairs"] == {"all reference segments": 2, "segments of at most 16 samples": 1, "A4C/A5C X-Y-X middles": 0}
    assert d["multi"] == {"all reference segments": 4, "segments of at most 16 samples": 2, "A4C/A5C X-Y-X middles": 1}


def test_trade_off_cells_need_a_recovery_and_a_cost_in_the_same_cell():
    acc, rows = {}, {}
    for s in (0, 1, 2):
        for h in ("selection", "calibration"):
            rows[f"FACT s{s}"] = {hh: row(0.95) for hh in ("selection", "calibration")}
            for c in ("A01 Linear + B3", "frozen Ordered + B3"):
                rows[f"{c} s{s}"] = {hh: row(0.95) for hh in ("selection", "calibration")}
                acc[f"s{s}|FACT vs {c}|{h}"] = {"segments": {"native": seg(), "pairs": seg(), "multi": seg(rec=1, new=0)}}
    assert rf.trade_cells(acc, rows, []) == []
    acc["s1|FACT vs frozen Ordered + B3|calibration"]["segments"]["pairs"] = seg(new=2)
    cells = rf.trade_cells(acc, rows, [])
    assert [(c["seed"], c["half"], c["control"]) for c in cells] == [(1, "calibration", "frozen Ordered + B3")] and cells[0]["new_unmatched_pred"] == 2
    rows["FACT s2"]["selection"] = row(0.95, nat=0.97)
    cells = rf.trade_cells(acc, rows, [])
    assert {(c["seed"], c["half"]) for c in cells} == {(1, "calibration"), (2, "selection")}
    assert len(rf.trade_cells(acc, rows, [{"seed": 0, "half": "selection"}])) == 5    # seed 1, seed 2 against both controls, seed 0's floor failure against both
