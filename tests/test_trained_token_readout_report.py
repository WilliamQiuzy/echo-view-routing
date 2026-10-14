"""Checks of the trained token readout's fixed readings and summaries (configs/experiments/trained_token_readout_v1.yaml,
Task 2), written before any new token metric exists: the same 40-setting budget for the CLS and token systems, no
feasible setting, a win over the unsupervised or shuffled controls only, a classification gain without a segment gain,
one half only, native or recall failures, the stream-sensitivity conflict, full success, and few changed cases that must
carry their source counts and a flag."""
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.raw_video_context_tables as rt
import scripts.trained_token_readout_tables as tb
from scripts.target_patch_readout_tables import UNCERTAIN

ROOT = Path(__file__).resolve().parents[1]
TT = yaml.safe_load((ROOT / "configs/experiments/trained_token_readout_v1.yaml").read_text())["trained_token_readout"]
P, N, NEG = tb.POS, tb.NONE, tb.NEG


def v(sel: str, cal: str) -> dict:
    return {"selection": sel, "calibration": cal}


def read(cls=(N, N), ordered=(N, N), cls_only=(N, N), shuffled=(N, N), mixed=(N, N), constraints=True, goal=False) -> dict:
    sysv = {"token - CLS": v(*cls), "token - Ordered": v(*ordered), "token - CLS-only token": v(*cls_only), "token - Shuffled token": v(*shuffled)}
    return tb.decide(sysv, v(*mixed), constraints, goal, TT["decision_rules"])


def test_rule_names_match_the_config():
    assert set(tb.RULE_ORDER) == {k for k in TT["decision_rules"] if k != "several"}


def test_full_success():
    d = read(cls=(P, P), ordered=(P, P), cls_only=(P, P), shuffled=(P, P), mixed=(P, P), constraints=True, goal=True)
    assert d["applied"] == ["complete_progress", "useful_readout", "end"] and not d["notes"]["only_controls"]


def test_useful_readout_without_the_goal_or_the_strong_reference():
    d = read(cls=(P, P), mixed=(P, P), constraints=True, goal=False)
    assert d["applied"] == ["useful_readout", "end"]


def test_no_feasible_setting_or_a_native_or_recall_failure_blocks_every_gain_reading():
    d = read(cls=(P, P), ordered=(P, P), cls_only=(P, P), shuffled=(P, P), mixed=(P, P), constraints=False, goal=False)
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["constraints_failed"]
    rows = [{h: {"segment_f1_50_multi": 0.99, "boundary_recall_multi": 0.99, "native_macro_f1": 0.99} for h in tb.HALVES} for _ in range(3)]
    g = rt.goal(rows, [True, False, True], TT["goal"])
    assert not g["met"] and g["constraint_failures"]


def test_beating_only_the_controls_is_no_progress():
    d = read(cls_only=(P, P), shuffled=(P, P))
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["only_controls"]


def test_a_classification_gain_without_a_segment_gain():
    d = read(mixed=(P, P), cls=(N, N))
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["classification_without_segments"]


def test_one_half_only_is_reported_as_such():
    d = read(cls=(P, N), mixed=(P, N))
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["one_half_only"] == ["selection"]


def test_a_stream_conflict_makes_that_half_uncertain():
    d = read(cls=(P, P), mixed=(UNCERTAIN, P), constraints=True)
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["uncertain_halves"] == ["selection"]


def test_negative_results_are_named():
    d = read(cls=(NEG, NEG), mixed=(NEG, N))
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["negative"] == {"segment F1 vs CLS": ["selection", "calibration"], "classification vs CLS": ["selection"]}


def test_choice_records_use_the_same_forty_setting_budget():
    grid = [{"order": k, "params": {"half_window": h, "threshold_quantile": q, "nms_radius": 5}} for k, (h, q) in
            enumerate((h, q) for h in (5, 7, 9, 11, 15) for q in (0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999))]
    tb.check_choice_record({"status": "feasible", "params": grid[3]["params"], "grid": grid}, TT["b3"]["grid"])
    with pytest.raises(SystemExit):
        tb.check_choice_record({"status": "feasible", "params": grid[3]["params"], "grid": grid[:39]}, TT["b3"]["grid"])
    with pytest.raises(SystemExit):
        tb.check_choice_record({"status": "feasible", "params": {"half_window": 13, "threshold_quantile": 0.9, "nms_radius": 5}, "grid": grid}, TT["b3"]["grid"])
    tb.check_choice_record({"status": "infeasible (diagnostic setting shown)", "params": grid[20]["params"], "grid": grid}, TT["b3"]["grid"])


def test_changed_cases_carry_sources_and_a_few_unit_flag():
    a = np.array([1, 1, 0, 1, 0, 1], bool); b = np.array([0, 1, 1, 1, 0, 0], bool)
    rec = np.array(["r1", "r1", "r2", "r3", "r4", "r5"]); ses = np.array(["s1", "s1", "s2", "s3", "s3", "s4"])
    c = tb.changes(a, b, rec, ses)
    assert c["fixed"] == {"windows": 2, "recordings": 2, "sessions": 2, "few_units": True}
    assert c["new"] == {"windows": 1, "recordings": 1, "sessions": 1, "few_units": True}


def test_anchor_offset_effects_are_split_by_exactness():
    off = np.array([0, 0, -1, -1, 0]); a = np.array([1, 0, 1, 1, 1], bool); b = np.array([1, 1, 0, 1, 1], bool)
    s = tb.offset_split(off, a, b)
    assert s["exact"] == {"windows": 3, "accuracy_token": 2 / 3, "accuracy_cls": 1.0} and s["non_exact"] == {"windows": 2, "accuracy_token": 1.0, "accuracy_cls": 0.5}


# ----------------------------------------------------------------------------------------------- selection stage guards
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import scripts.trained_token_readout as tt  # noqa: E402

GRID = [{"order": k, "params": {"half_window": h, "threshold_quantile": q, "nms_radius": 5, "threshold": 0.1 + k / 100},
         "native_macro_f1": 0.99, "boundary_recall_multi": 0.96, "segment_f1_50_multi": 0.95 if k == 3 else 0.90}
        for k, (h, q) in enumerate((h, q) for h in (5, 7, 9, 11, 15) for q in (0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999))]


def test_native_or_recall_losses_are_stated_apart():
    sysv = {c: v(N, N) for c in tb.SYS}
    d = tb.decide(sysv, v(N, N), True, False, TT["decision_rules"], {"native macro-F1": v(NEG, N), "multi boundary recall": v(N, N)})
    assert d["notes"]["native_or_recall_losses"] == {"native macro-F1": ["selection"]} and d["applied"] == ["labels_learned_no_usable_gain", "end"]


def test_per_class_f1_comes_from_the_confusion_matrix():
    conf = np.diag([4, 3, 2, 1, 5]).astype(int); conf[2, 3] = 2           # two A4C clips read as A5C
    r = tt.with_f1({"native_confusion": conf.tolist()})
    assert r["native_f1"]["A4C"] == pytest.approx(2 * 2 / (2 + 4)) and r["native_f1"]["A5C"] == pytest.approx(2 * 1 / (3 + 1))
    assert r["native_precision"]["A5C"] == pytest.approx(1 / 3) and r["native_f1"]["PLAX"] == 1.0


def test_samples_take_the_offset_of_the_window_that_serves_them():
    seqs = [{"id": "q0", "labels": np.zeros(12, int)}, {"id": "q1", "labels": np.ones(6, int)}]
    W = {"seq": np.array([0, 0, 0, 1, 1]), "centre": np.array([0, 5, 11, 0, 5])}
    served = tt.sample_windows(W, seqs)
    assert served["q0"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2] and served["q1"].tolist() == [3, 3, 3, 4, 4, 4]   # ties go to the earlier centre
    off = np.array([0, 0, -1, 0, -1])
    a = {"q0": np.zeros(12, int), "q1": np.r_[np.ones(3, int), np.zeros(3, int)]}; b = {"q0": np.zeros(12, int), "q1": np.ones(6, int)}
    s = tt.sample_offset_split(seqs, served, off, a, b)
    assert s["exact"] == {"samples": 12, "accuracy_token": 1.0, "accuracy_cls": 1.0}
    assert s["non_exact"] == {"samples": 6, "accuracy_token": 0.5, "accuracy_cls": 1.0}


def test_a_stored_row_that_is_not_reproduced_stops():
    row = {f: 0.5 for f in tt.FLAT} | {"native_confusion": [[1, 0], [0, 1]], "A4C read as A5C": 0}
    tt.check_stored(row, dict(row), "x")
    for bad in ({"segment_f1_50_multi": 0.4}, {"native_confusion": [[1, 0], [1, 0]]}, {"A4C read as A5C": 1}):
        with pytest.raises(SystemExit):
            tt.check_stored(row, row | bad, "x")


def _ctx(tmp_path, records=None):
    tmp_path.mkdir(parents=True, exist_ok=True); (tmp_path / "c.yaml").write_text("x: 1\n")
    preds = {"aligned_dense_s0": {"path": "p.npz", "sha256": "ab" * 32}}
    ctx = SimpleNamespace(results=tmp_path / "res", out=tmp_path / "out", root=tmp_path, config="c.yaml", src={"predictions": {"calibration": preds}},
                          tt={"b3": {"grid": TT["b3"]["grid"]}, "goal": TT["goal"]}, ledger=tmp_path / "ledger.jsonl")
    sysrec = records or {"token_aligned_dense_s0": {"status": "feasible", "params": GRID[3]["params"], "n_feasible": 40, "grid": GRID}}
    rec = {"half": "calibration", "code": tt.code_digest(), "config_sha256": tt.sha256_file(tmp_path / "c.yaml"),
           "predictions": {"aligned_dense_s0": "ab" * 32}, "systems": sysrec,
           "choices": {k: {x: r[x] for x in ("status", "params", "n_feasible")} for k, r in sysrec.items()}}
    for f in (ctx.results / "choices.json", ctx.out / "calibration" / "choices.json"):
        f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(rec))
    ctx.choices_sha256 = tt.sha256_file(ctx.results / "choices.json")
    return ctx, rec


def test_the_choices_are_rederived_from_their_own_grid(tmp_path):
    ctx, rec = _ctx(tmp_path / "ok")
    tt.load_choices(ctx, ctx.choices_sha256)
    bad_setting = {"token_aligned_dense_s0": {"status": "feasible", "params": GRID[4]["params"], "n_feasible": 40, "grid": GRID}}
    bad_count = {"token_aligned_dense_s0": {"status": "feasible", "params": GRID[3]["params"], "n_feasible": 39, "grid": GRID}}
    none = [r | {"native_macro_f1": 0.97} for r in GRID]
    diag = next(r for r in none if r["params"]["half_window"] == 9 and r["params"]["threshold_quantile"] == 0.95)
    infeasible_ok = {"token_aligned_dense_s0": {"status": "infeasible (diagnostic setting shown)", "params": diag["params"], "n_feasible": 0, "grid": none}}
    infeasible_bad = {"token_aligned_dense_s0": {"status": "infeasible (diagnostic setting shown)", "params": none[0]["params"], "n_feasible": 0, "grid": none}}
    ctx2, _ = _ctx(tmp_path / "inf", infeasible_ok)
    tt.load_choices(ctx2)
    for name, recs in (("setting", bad_setting), ("count", bad_count), ("diag", infeasible_bad)):
        ctx3, _ = _ctx(tmp_path / name, recs)
        with pytest.raises(SystemExit):
            tt.load_choices(ctx3)
    ctx4, rec4 = _ctx(tmp_path / "summary")
    rec4["choices"]["token_aligned_dense_s0"]["n_feasible"] = 1
    for f in (ctx4.results / "choices.json", ctx4.out / "calibration" / "choices.json"):
        f.write_text(json.dumps(rec4))
    with pytest.raises(SystemExit):
        tt.load_choices(ctx4)                                  # the short choice differs from its full record


def test_selection_runs_only_on_the_committed_choices(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tt, "evaluate_all", lambda ctx, ch, data, sha: calls.append(dict(data)) or ({"identity": {"choices_sha256": sha}, "decision": {"applied": []}}, {}, set()))
    monkeypatch.setattr(tt, "publish", lambda ctx, flat, per, out, final, extra, ids: calls.append(("publish", out.name, final)) or True)
    ctx, rec = _ctx(tmp_path / "a")
    tt.stage_dry_run(ctx)
    assert calls[-2:] == [{"calibration": "calibration", "selection": "calibration"}, ("publish", "dry_run", False)]
    tt.stage_selection(ctx)
    assert calls[-2:] == [{"calibration": "calibration", "selection": "selection"}, ("publish", "selection", True)]
    ctx.choices_sha256 = None
    with pytest.raises(SystemExit):
        tt.stage_selection(ctx)                                # no committed sha given
    ctx.choices_sha256 = "0" * 64
    with pytest.raises(SystemExit):
        tt.stage_selection(ctx)                                # not the committed record
    for k, bad in enumerate(({"code": "0" * 64}, {"config_sha256": "0" * 64}, {"predictions": {"aligned_dense_s0": "cd" * 32}})):
        ctx2, _ = _ctx(tmp_path / f"b{k}")
        for f in (ctx2.results / "choices.json", ctx2.out / "calibration" / "choices.json"):
            f.write_text(json.dumps(rec | bad))
        ctx2.choices_sha256 = tt.sha256_file(ctx2.results / "choices.json")
        with pytest.raises(SystemExit):
            tt.stage_selection(ctx2)
    ctx3, _ = _ctx(tmp_path / "c")
    (ctx3.out / "calibration" / "choices.json").write_text(json.dumps(rec | {"note": "other"}))
    with pytest.raises(SystemExit):
        tt.stage_selection(ctx3)                               # the two copies differ
    ctx4, _ = _ctx(tmp_path / "d")
    (ctx4.out / "selection").mkdir(parents=True); (ctx4.out / "selection" / "done.json").write_text(json.dumps({"choices_sha256": "0" * 64}))
    with pytest.raises(SystemExit):
        tt.stage_selection(ctx4)                               # an earlier run used other choices


def test_publish_writes_once_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "position_figure", lambda path: Path(path).write_text("figure"))
    saved = {"token_aligned_dense_s0": {"token_logits": np.zeros((2, 5), np.float32), "anchor": np.array([4, 4]), "offset": np.array([0, 0]),
                                         "probs": {"q": np.full((3, 5), 0.2)}, "decoded": {"q": np.zeros(3, int)}}}
    per = {h: {"seqs": [{"id": "q"}], "saved": saved} for h in tb.HALVES}
    ctx = SimpleNamespace(results=tmp_path / "res", out=tmp_path / "out"); ctx.results.mkdir()
    flat = {"identity": {"choices_sha256": "ab" * 32}, "systems": {}, "decision": {"applied": ["end"]}}
    out = ctx.out / "selection"
    assert tt.publish(ctx, flat, per, out, True, {"cost": {"cpu_hours_before_this_stage": 0.1}}, set())
    assert (out / "done.json").is_file() and (out / "selection" / "token_aligned_dense_s0.npz").is_file()
    stamp = {f: f.stat().st_mtime_ns for f in [*out.rglob("*"), *ctx.results.rglob("*")] if f.is_file()}
    assert not tt.publish(ctx, flat, per, out, True, {"cost": {"cpu_hours_before_this_stage": 0.2}}, set())   # reproduced: nothing rewritten
    assert stamp == {f: f.stat().st_mtime_ns for f in [*out.rglob("*"), *ctx.results.rglob("*")] if f.is_file()}
    with pytest.raises(SystemExit):
        tt.publish(ctx, flat | {"decision": {"applied": ["other"]}}, per, out, True, {}, set())
    assert stamp == {f: f.stat().st_mtime_ns for f in [*out.rglob("*"), *ctx.results.rglob("*")] if f.is_file()}
    assert json.loads((ctx.results / "summary.json").read_text())["cost"] == {"cpu_hours_before_this_stage": 0.1}
    dry = ctx.out / "dry_run"
    assert tt.publish(ctx, flat, per, dry, False, {}, set()) and (dry / "summary.json").is_file() and not (dry / "done.json").exists()


def test_the_dry_run_reads_the_calibration_half_once(monkeypatch):
    read = []
    monkeypatch.setattr(tt, "half_systems", lambda ctx, h, *a: read.append(h) or {"W": None, "meta": None, "logits": None, "offsets": None, "seqs": None})
    monkeypatch.setattr(tt, "same_target", lambda *a: ({}, {})); monkeypatch.setattr(tt, "position_summary", lambda *a: {})
    per = tt.read_halves(None, {"systems": {}}, {"calibration": "calibration", "selection": "calibration"}, {}, {}, {})
    assert read == ["calibration"] and per["selection"] is per["calibration"]
    read.clear(); tt.read_halves(None, {"systems": {}}, {"calibration": "calibration", "selection": "selection"}, {}, {}, {})
    assert read == ["calibration", "selection"]


def test_stages_need_the_records_of_this_code(tmp_path):
    env = tt.environment()
    ctx = SimpleNamespace(out=tmp_path, src={"environment": {k: env[k] for k in ("python", "numpy", "pandas", "scipy")}})
    tt.preconditions(ctx, "prepare")
    with pytest.raises(SystemExit):
        tt.preconditions(ctx, "check")                         # no preparation yet
    (tmp_path / "prepare.json").write_text(json.dumps({"code": tt.code_digest(), "environment": env}))
    tt.preconditions(ctx, "check")
    with pytest.raises(SystemExit):
        tt.preconditions(ctx, "calibrate")                     # no passed check yet
    (tmp_path / "check.json").write_text(json.dumps({"code": tt.code_digest(), "ok": True}))
    tt.preconditions(ctx, "calibrate")
    (tmp_path / "check.json").write_text(json.dumps({"code": "0" * 64, "ok": True}))
    with pytest.raises(SystemExit):
        tt.preconditions(ctx, "selection")                     # a check made by other code
    ctx.src["environment"]["numpy"] = "0.0"
    with pytest.raises(SystemExit):
        tt.preconditions(ctx, "prepare")                       # another environment


def test_a_record_is_written_once(tmp_path):
    f = tmp_path / "r.json"; f.write_text(json.dumps({"a": 1}))
    tt.refuse_other([f], {"a": 1}, "record")
    with pytest.raises(SystemExit):
        tt.refuse_other([f], {"a": 2}, "record")


def test_every_table_row_has_as_many_cells_as_its_header():
    b = {m: {"delta": 0.01, "ci95": [-0.01, 0.02]} for m in tt.METRICS}
    unit = {"segments": 4, "matched": 3, "wrong view": 1, "no segment at IoU 0.5": 0, "taken by another reference": 0, "recall": 0.75,
            "missed_recordings": 1, "missed_sessions": 1}
    boot = {"windows": 10, "units": 4, "few_units": True, "delta": 0.1, "ci95": [0.0, 0.2], "estimable": True}
    ch = {"windows": 2, "recordings": 2, "sessions": 1, "few_units": True}
    row = {"segment_f1_50_multi": 0.9, "native_macro_f1": 0.98, "native_recall": {c: 1.0 for c in tb.CLASSES}, "native_f1": {c: 1.0 for c in tb.CLASSES}}
    res = {"systems": {"token_aligned_dense_s0": {"b3": {"selection": row}, "b3_choice": {"status": "feasible", "params": {"half_window": 7}, "n_feasible": 3}}},
           "choices": {"token_aligned_dense_s0": {"status": "feasible", "params": {"half_window": 7, "threshold_quantile": 0.99, "threshold": 0.2}, "n_feasible": 3}},
           "system_comparisons": {"s0|b3|token - CLS|selection": {"bootstrap": b, "diagnostic_setting": True, "native_changes": {"all clips": {"fixed": 1, "new": 0}},
                                                                 "segment_changes": {"matched by a only": {"segments": 2}, "matched by b only": {"segments": 1}}}},
           "verdicts": {"b3": {"token - CLS": {"multi segment F1@50": {"selection": tb.NONE, "calibration": tb.POS}}}},
           "classification": {"accuracy": {"token_aligned_dense_s0": {"selection": {"native": 0.9, "all mixed": 0.8}}},
                              "comparisons": {"token_aligned_dense_s0|all mixed|selection": {"session": boot, "stream": boot, "changes": {"fixed": ch, "new": ch}},
                                              "token_aligned_dense_s0|offset split|selection": {"exact": {"windows": 5, "accuracy_token": 1.0, "accuracy_cls": 0.8},
                                                                                                "non_exact": {"windows": 1, "accuracy_token": 0.0, "accuracy_cls": 1.0}}},
                              "verdicts": {"token_aligned_dense|all mixed": {"selection": tb.NONE, "calibration": UNCERTAIN}}},
           "positions": {"selection": {"windows": 6, "exact": 5, "non_exact": 1, "non_exact_native": 1, "label_differs": 0, "offset_counts": {"-1": 1, "0": 5},
                                       "anchor_counts": {"8": 5, "6": 1}, "sample_split": {"token_aligned_dense_s0": {"b3": {
                                           "exact": {"samples": 20, "accuracy_token": 1.0, "accuracy_cls": 1.0}, "non_exact": {"samples": 4, "accuracy_token": 0.5, "accuracy_cls": 1.0}}}}}},
           "units": {"token_aligned_dense_s0|b3|selection": {g: unit for g in tb.UNIT_GROUPS}},
           "goal": {"token_aligned_dense": {"mean_segment_f1": {"selection": 0.95, "calibration": 0.98}, "margin": {"selection": 0.01, "calibration": -0.002},
                                            "calibration_sensitivity_margin": -0.004, "met": False,
                                            "constraint_failures": [{"seed": 1, "half": "selection", "b3_feasible": True, "boundary_recall": 0.94, "native_macro_f1": 0.98}]}},
           "stored_rows": {"E1 + B3": {"selection": row}}, "decision": {"applied": ["end"]}}
    md = tb.tables_md(res)
    header = None
    for line in md.splitlines():
        if not line.startswith("|"):
            header = None; continue
        n = line.count("|")
        if header is None:
            header = n
        assert n == header, line
    assert "| 0 | b3 | token - CLS | selection | diagnostic |" in md and "seed 1 selection recall 0.9400 native 0.9800" in md
