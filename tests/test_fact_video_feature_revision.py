"""A06-R1 (configs/experiments/fact_video_feature_v2.yaml): the revision keeps v1's science field for field, the v1 check
still fails under the original gate, the limited fallback gate accepts only single-segment all-null rows with a correct
official output, the inherited checks are verified field by field, the old training code is identified by its digest
and compared function by function, the official all-null fallback is the pinned decoder's own branch, the v2 namespace
reaches every output, the CLI and the queue use the launch record of their configuration, the v1 ledger is counted
once, and the monitor's token counts come from its own forwards. CPU only; no real feature, label or model.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import scripts.revalidate_fact_video_feature as rv
import scripts.train_fact_video_feature as tf
from echo_routing.temporal import fact_head as fh
from scripts._common import load_cfg

ROOT = Path(__file__).resolve().parents[1]
V1, V2 = "configs/experiments/fact_video_feature_v1.yaml", "configs/experiments/fact_video_feature_v2.yaml"
OLD_CHECK = ROOT / "docs" / "results" / "2026-10-15_fact_video_feature" / "checks.json"
REPO = ROOT / "third_party" / "fact"
needs_fact = pytest.mark.skipif(not (REPO / "models" / "blocks.py").is_file(), reason="the pinned official FACT checkout is not present")
needs_git = pytest.mark.skipif(not (ROOT / ".git").exists(), reason="the repository's git history is not present (the server has none)")
ENV = {"python": "3.12.3", "numpy": "2.5.2", "pandas": "3.0.5", "scipy": "1.18.1", "torch": "2.14.0+cu130", "torchvision": "0.29.0+cu130", "cuda": "13.0", "cudnn": 92400}
RULE = {"max_transcript": 8, "average_transcript": 2.654333333333333, "token_ratio": 1.5, "ntoken": 12}


def fv(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text())["fact_video_feature"]


def old_check() -> dict:
    return json.loads(OLD_CHECK.read_text())


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def set_path(d: dict, path: tuple, value) -> dict:
    out = copy.deepcopy(d); node = out
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return out


# ----------------------------------------------------------------------------------------------- the configuration
def test_revision_keeps_the_v1_science_field_for_field():
    a, b = fv(V1), fv(V2)
    got = rv.check_scientific_equality(a, b)
    assert set(got["equal"]) >= {"seeds", "data", "fact", "monitor", "train_fit", "evaluation", "goal", "readings", "source", "queue", "checks"}
    for path, value in ((("fact", "optimizer", "lr"), 2e-4), (("data", "total_updates"), 20000), (("fact", "init"), "another state"),
                        (("data", "features"), "other features"), (("goal", "calibration_min"), 0.95), (("seeds",), [0, 1]),
                        (("readings", "end"), "continue"), (("evaluation", "primary"), "FACT - A01 Linear + B3"), (("monitor",), "other"),
                        (("source", "a01_config", "sha256"), "0" * 64), (("checks", "smoke", "updates"), 3000),
                        (("revision", "scientific_sections"), ["seeds"])):
        with pytest.raises(SystemExit):
            rv.check_scientific_equality(a, set_path(b, path, value))


def test_revision_pins_the_v1_evidence_and_a_cumulative_budget():
    r, b = fv(V2)["revision"], fv(V2)["budget"]
    for k in ("task_sheet", "config", "preparation", "check", "readme"):
        assert sha(ROOT / r["original"][k]["path"]) == r["original"][k]["sha256"], k
    assert r["original"]["code_digest"].startswith("ae7531732dd0") and (b["gpu_hours_max"], b["cpu_hours_max"], b["gpu_reserve_hours"]) == (12.0, 4.0, 1.25)
    assert b["inherited_ledgers"][0]["path"] == "runs/fact_video_feature_v1/ledger.jsonl" and b["revalidation_gpu_hours_max"] == 0.05


def test_the_original_evidence_is_verified_by_hash_and_by_its_own_code_fields(tmp_path):
    org = fv(V2)["revision"]["original"]
    for k in ("task_sheet", "config", "preparation", "check", "readme"):
        (tmp_path / org[k]["path"]).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / org[k]["path"], tmp_path / org[k]["path"])
    got = rv.verify_original_evidence(tmp_path, org, server=False)
    assert got["check"]["sha256"] == org["check"]["sha256"]
    with pytest.raises(SystemExit):
        rv.verify_original_evidence(tmp_path, org, server=True)              # the server copies are missing here
    for k in ("preparation", "check"):
        (tmp_path / org["server_copies"][k]).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / org[k]["path"], tmp_path / org["server_copies"][k])
    assert rv.verify_original_evidence(tmp_path, org, server=True)["check"]["server_copy"] == org["server_copies"]["check"]
    (tmp_path / org["server_copies"]["check"]).write_text("{}")
    with pytest.raises(SystemExit):
        rv.verify_original_evidence(tmp_path, org, server=True)              # a server copy differs
    c = old_check(); c["passed"] = True; p = tmp_path / org["check"]["path"]; p.write_text(json.dumps(c))
    with pytest.raises(SystemExit):
        rv.verify_original_evidence(tmp_path, org, server=False)             # an edited old check fails its pin
    c = old_check(); c["code"] = "0" * 64; p.write_text(json.dumps(c))
    with pytest.raises(SystemExit):                                         # ...and a re-pinned one made by other code
        rv.verify_original_evidence(tmp_path, {**org, "check": {"path": org["check"]["path"], "sha256": sha(p)}}, server=False)


# ----------------------------------------------------------------------------------------------- the gates
def test_the_v1_check_still_fails_under_the_original_gate():
    c = old_check(); f = c["fact"]
    assert c["passed"] is False and f["ok"] is False
    req = tf.smoke_passed(f["smoke"]["outcome"], [f["isolation_initial"], f["isolation_overfit"]])
    assert req == f["smoke"]["requirements"] and [k for k, v in req.items() if not v] == ["matched_classes_right", "all"]


def test_the_revised_gate_accepts_exactly_the_four_single_segment_fallbacks():
    g = rv.revised_gate(old_check()["fact"]["smoke"]["rows"])
    assert g["passed"] and g["common"] and len(g["rows"]) == 16
    exc = [r for r in g["rows"] if r["result"] == "official fallback"]
    assert [r["index"] for r in exc] == [0, 4, 6, 10] and all(r["segments"] == 1 and r["tokens_used"] == 0 for r in exc)
    assert sum(r["result"] == "tokens right" for r in g["rows"]) == 12
    assert g["groups"]["single"]["rows"] == 9 and g["groups"]["single"]["all_null"] == 4 and g["groups"]["multi"]["rows"] == 7 and g["groups"]["multi"]["all_null"] == 0


@pytest.mark.parametrize("index,change", [
    (0, {"segments": 2}),                                                 # an all-null row with two reference segments
    (10, {"segments": 3}),                                                # an all-null stream with three
    (0, {"tokens_used": 1}),                                              # an active token that names the segment wrongly
    (3, {"matched_token_class_correct": 0.0}),                            # twelve active tokens, the matched one wrong
    (8, {"matched_token_class_correct": 0.75}),                           # a multi-segment row with a wrong matched token
    (8, {"tokens_used": 0}),                                              # a multi-segment all-null row
    (0, {"transcript_exact": False}),
    (4, {"segment_f1_50": 0.5}),
    (13, {"match_one_to_one": False}),
    (0, {"matched_token_class_correct": 1.0}),                            # an all-null row must carry its wrong-token record
])
def test_the_revised_gate_rejects_everything_else(index, change):
    rows = copy.deepcopy(old_check()["fact"]["smoke"]["rows"]); rows[index].update(change)
    assert not rv.revised_gate(rows)["passed"]


def test_the_revised_gate_needs_all_sixteen_rows():
    rows = old_check()["fact"]["smoke"]["rows"]
    assert not rv.revised_gate(rows[:15])["passed"] and not rv.revised_gate(rows + rows[1:2])["passed"]


def test_inherited_checks_are_verified_field_by_field():
    names = fv(V2)["revision"]["known_zero_gradients"]["names"]; c = old_check()
    got = rv.inherited_checks(c, names)
    assert got["passed"] and got["failed_only"] == ["matched_classes_right"] and got["zero_gradients"] == sorted(names)
    for path, value in ((("fact", "isolation_initial", "changed_predictions"), 1), (("fact", "isolation_overfit", "max_abs_logit_difference"), 1e-3),
                        (("fact", "isolation_overfit", "adapter_decode_differs"), 2), (("fact", "isolation_initial", "comparisons"), 800),
                        (("fact", "padding", "predictions_equal"), False), (("fact", "padding", "loss_diff"), 1e-7),
                        (("fact", "reproducibility", "params_max_abs"), 1e-5), (("fact", "resume", "losses_max_abs"), 2e-6),
                        (("fact", "resume", "bound"), False), (("fact", "smoke", "outcome", "transcript_exact"), 15),
                        (("fact", "optimizer_matches_config"), False), (("fact", "smoke", "updates"), 1500),
                        (("fact", "smoke", "requirements", "matched_classes_right"), True), (("passed",), True),
                        (("init", "FACT_s1", "tensors"), 325)):
        assert not rv.inherited_checks(set_path(c, path, value), names)["passed"], path
    bad = copy.deepcopy(c); bad["fact"]["smoke"]["rows"][0]["transcript_exact"] = False       # rows no longer match the outcome
    assert not rv.inherited_checks(bad, names)["passed"]
    bad = copy.deepcopy(c); bad["fact"]["zero_or_missing_gradients"] = bad["fact"]["zero_or_missing_gradients"] + ["block_list.1.frame_branch.weight"]
    assert not rv.inherited_checks(bad, names)["passed"]
    assert not rv.inherited_checks(c, names[:2])["passed"]


@needs_fact
def test_the_official_decoder_falls_back_to_frames_only_when_every_token_is_null():
    blocks = fh.official(REPO)["blocks"]; g = torch.Generator().manual_seed(0); T, K, C, w = 7, 4, 5, 0.1
    frame = torch.randn(T, 1, C, generator=g); attn = torch.rand(1, T, K, generator=g)
    null = torch.zeros(K, 1, C + 1); null[:, 0, C] = 9.0                  # every token's argmax is the null class
    assert torch.equal(blocks.Block._eval(null, attn, frame, w), torch.softmax(frame.squeeze(1), -1).argmax(1))
    act = null.clone(); act[1, 0, 2], act[3, 0, 4] = 12.0, 11.0          # tokens 1 and 3 active
    loc = torch.tensor([1, 3]); qp = torch.softmax(act.squeeze(1)[:, :-1], 1); pick = loc[attn.squeeze(0)[:, loc].argmax(-1)]
    want = ((1 - w) * qp[pick] + w * torch.softmax(frame.squeeze(1), -1)).argmax(1)
    assert torch.equal(blocks.Block._eval(act, attn, frame, w), want)


# ----------------------------------------------------------------------------------------------- the old code
@needs_git
def test_the_old_code_is_identified_by_its_digest_and_differs_only_in_allowed_functions(tmp_path):
    r = fv(V2)["revision"]
    rec = rv.export_old_code(ROOT, r["original"]["code_commit"], tmp_path / "old")
    assert rec["digest"] == r["original"]["code_digest"] and rv.old_code_digest(tmp_path / "old") == r["original"]["code_digest"]
    cmp = rv.compare_code(tmp_path / "old", r["allowed_code_changes"], r["allowed_added_functions"])
    assert cmp["ok"], cmp["disallowed"]
    me = "scripts/train_fact_video_feature.py"
    assert all(m["identical"] for name, m in cmp["modules"].items() if name != me) and len(cmp["modules"]) == len(tf.CODE_FILES)
    t = cmp["train_script"]
    assert set(t["changed"]) <= set(r["allowed_code_changes"]) and set(t["added"]) <= set(r["allowed_added_functions"]) and not t["removed"]
    for name in ("train", "run_updates", "resume_check", "padding_check", "new_model", "make_head", "strict_init", "check_batches", "stage_prepare", "<module>"):
        assert name in t["identical"], name
    (tmp_path / "old" / me).write_text((tmp_path / "old" / me).read_text() + "\n")
    with pytest.raises(SystemExit):
        rv.verify_old_code(tmp_path / "old", r["original"]["code_digest"])     # any byte changes the digest
    with pytest.raises(SystemExit, match="missing"):
        rv.verify_old_code(tmp_path / "absent", r["original"]["code_digest"])  # an export never copied to this host
    with pytest.raises(SystemExit):
        rv.export_old_code(tmp_path / "no_repo", r["original"]["code_commit"], tmp_path / "x")   # no git history here


def test_the_code_comparison_flags_every_other_change():
    old = "X = 1\n\n\ndef train(x):\n    return x + 1\n\n\ndef identity(x):\n    return x\n\n\nclass C:\n    def f(self):\n        return 1\n"
    new = old.replace("x + 1", "x + 2").replace("return x\n", "return (x, 'v2')\n")
    got = rv.compare_sources(old, new, {"identity": "plumbing"}, [])
    assert got["changed"] == ["identity", "train"] and got["disallowed"] == ["train"]
    assert rv.compare_sources(old, old, {}, [])["disallowed"] == []
    assert rv.compare_sources(old, old + "\n\ndef extra():\n    pass\n", {}, [])["disallowed"] == ["extra"]
    assert rv.compare_sources(old, old.replace("X = 1", "X = 2"), {}, [])["disallowed"] == ["<module>"]
    assert rv.compare_sources(old, old.replace("return 1", "return 2"), {}, [])["disallowed"] == ["C.f"]
    assert rv.compare_sources(old, old.replace("def identity(x):\n    return x\n", ""), {"identity": "x"}, [])["disallowed"] == ["identity"]


# ----------------------------------------------------------------------------------------------- namespace
def test_the_namespaces_are_separate_and_guarded():
    c1 = tf.Ctx(load_cfg(SimpleNamespace(config=V1, set=[])), V1); c2 = tf.Ctx(load_cfg(SimpleNamespace(config=V2, set=[])), V2)
    assert (c1.ns, c2.ns) == ("fact_video_feature_v1", "fact_video_feature_v2") and c1.revision is None and c2.revision["name"] == "A06-R1"
    assert c2.frozen == c2.root / "models_frozen" / "fact_video_feature_v2" and c1.frozen == c1.root / "models_frozen" / "fact_video_feature_v1"
    for a, b in ((c1.out, c2.out), (c1.ckpt, c2.ckpt), (c1.results, c2.results), (c1.ledger, c2.ledger), (c1.frozen, c2.frozen)):
        assert a != b and "fact_video_feature_v1" not in str(b)
    cfg = load_cfg(SimpleNamespace(config=V2, set=[]))
    for path, value in ((("fact_video_feature", "outputs", "root"), "runs/fact_video_feature_v1"), (("fact_video_feature", "outputs", "checkpoints"), "checkpoints/x"),
                        (("fact_video_feature", "outputs", "frozen"), "models_frozen/fact_video_feature_v1/FACT_s{seed}"),
                        (("fact_video_feature", "budget", "ledger"), "runs/fact_video_feature_v1/ledger.jsonl"),
                        (("fact_video_feature", "outputs", "results_dir"), "docs/results/2026-10-15_fact_video_feature"), (("experiment",), "fact_video_feature_v1"),
                        (("fact_video_feature", "outputs", "results_dir"), "docs/results/2026-10-15_fact_video_feature/revised"),
                        (("fact_video_feature", "outputs", "results_dir"), "docs/results"),
                        (("fact_video_feature", "outputs", "manifest"), "docs/results/2026-10-15_fact_video_feature/evaluation_manifest.json")):
        with pytest.raises(SystemExit):
            tf.Ctx(set_path(cfg, path, value), V2)
    p1 = tf.identity_payload(c1.fv, 0, {"head": "c"}, "e", {}, {"plan": "p"}, "code")
    p2 = tf.identity_payload(c2.fv, 0, {"head": "c"}, "e", {}, {"plan": "p"}, "code", c2.ns, tf.revision_identity(c2))
    assert p1["experiment"] == "fact_video_feature_v1" and "revision" not in p1 and tf.revision_identity(c1) is None
    assert p1 == tf.identity_payload(c1.fv, 0, {"head": "c"}, "e", {}, {"plan": "p"}, "code", c1.ns, tf.revision_identity(c1))   # v1's identity is unchanged
    assert p2["experiment"] == "fact_video_feature_v2" and p2["revision"]["task_sheet"] == c2.fv["task_sheet"]["sha256"]
    assert p2["revision"]["original"]["check"] == c2.revision["original"]["check"]["sha256"] and tf.identity_of(p1) != tf.identity_of(p2)


def test_the_freeze_and_the_report_use_the_configured_namespace(tmp_path, monkeypatch):
    import scripts.fact_video_feature_post as post
    import scripts.report_fact_video_feature as rf
    ns, key = "fact_video_feature_v2", "FACT_s0"; out = tmp_path / "out"; ck = tmp_path / "ckpt" / "final.pt"; ck.parent.mkdir(parents=True); ck.write_bytes(b"weights")
    man = out / "train" / key / "run_manifest.json"; man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"final_checkpoint": str(ck), "final_sha256": sha(ck), "completed_updates": 18750, "interrupted": False, "parameters": 1, "minutes": 1.0, "records": []}))
    ctx = SimpleNamespace(root=tmp_path, out=out, total=18750, config=V2, ns=ns); calls = []

    def fake_run(cmd, cwd, capture_output, text):
        calls.append(cmd); d = Path(cwd) / "models_frozen" / cmd[2] / cmd[3]; d.mkdir(parents=True); shutil.copyfile(ck, d / "final.pt")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(post.subprocess, "run", fake_run); monkeypatch.setattr(tf, "run_state", lambda *a, **k: "done")
    got = post.freeze(ctx, {key: "ident"}, {"inits": {key: {"path": "inits/FACT_s0.pt"}}})
    assert calls[0][2] == ns and got[key]["path"] == f"models_frozen/{ns}/{key}" and not (tmp_path / "models_frozen" / "fact_video_feature_v1").exists()
    monkeypatch.setattr(tf, "read_record", lambda c, n: {"token_rule": RULE}); monkeypatch.setattr(tf, "make_head", lambda c, r: None)
    monkeypatch.setattr(tf, "identities", lambda c, h, p: {key: "ident"})
    assert rf.verify_runs(ctx)[key]["frozen"] == f"models_frozen/{ns}/{key}"
    assert not hasattr(rf, "NS") and "tf.NS" not in inspect.getsource(post) and 'models_frozen" / ctx.ns' in inspect.getsource(post.stage_train_fit)
    assert "--config configs/experiments/fact_video_feature_v2.yaml" in rf.tables_md({}, V2)


# ----------------------------------------------------------------------------------------------- the launch records
def tmp_ctx(config: str, tmp: Path, inherited: bool = True):
    ctx = tf.Ctx(load_cfg(SimpleNamespace(config=config, set=[])), config)
    ctx.out, ctx.results, ctx.ckpt, ctx.ledger = tmp / "out", tmp / "results", tmp / "ckpt", tmp / "out" / "ledger.jsonl"
    if not inherited:
        ctx.fv = {**ctx.fv, "budget": {**ctx.fv["budget"], "inherited_ledgers": []}}
    return ctx


def write_launch(ctx, passed: bool, code: str | None = None, **extra) -> None:
    code = code or tf.code_digest(); ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "preparation.json").write_text(json.dumps({"code": tf.code_digest(), "config_sha256": ctx.config_sha(), "token_rule": RULE}))
    rec = {"passed": passed, "code": code, "config_sha256": ctx.config_sha(), **extra}
    if ctx.revision is None:
        (ctx.out / "checks.json").write_text(json.dumps(rec))
    else:
        rec = {"revision": ctx.revision["name"], "original": ctx.revision["original"], "original_gate_passed": False, **rec}
        (ctx.out / "revalidation.json").write_text(json.dumps(rec))


def test_the_launch_record_belongs_to_the_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "environment", lambda: dict(ENV))
    c1 = tmp_ctx(V1, tmp_path / "v1"); write_launch(c1, False)
    with pytest.raises(SystemExit):
        tf.preconditions(c1, "queue")                                    # v1's check failed: v1 never launches
    (c1.out / "revalidation.json").write_text(json.dumps({"passed": True, "code": tf.code_digest(), "config_sha256": c1.config_sha()}))
    with pytest.raises(SystemExit):
        tf.preconditions(c1, "run")                                      # a revalidation record never opens v1
    write_launch(c1, True); tf.preconditions(c1, "queue")                # the old rule itself is unchanged
    c2 = tmp_ctx(V2, tmp_path / "v2"); write_launch(c2, True)
    for stage in ("queue", "run", "train_fit"):
        tf.preconditions(c2, stage)
    (c2.out / "checks.json").write_text(json.dumps({"passed": True, "code": tf.code_digest(), "config_sha256": c2.config_sha()}))
    for bad in ({"passed": False}, {"code": "0" * 64}, {"original_gate_passed": True}, {"revision": "another"},
                {"original": {**c2.revision["original"], "check": {"path": "x", "sha256": "0" * 64}}}):
        write_launch(c2, True); rec = json.loads((c2.out / "revalidation.json").read_text()); (c2.out / "revalidation.json").write_text(json.dumps({**rec, **bad}))
        with pytest.raises(SystemExit):
            tf.preconditions(c2, "run")
        with pytest.raises(SystemExit):
            tf.stage_queue(c2, V2)                                       # the queue itself refuses before anything else
    with pytest.raises(SystemExit):
        tf.stage_check(c2)                                               # no repeated smoke under the revision


def test_the_cli_routes_through_the_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "environment", lambda: dict(ENV)); calls = []
    real = tf.Ctx

    class TmpCtx(real):
        def __init__(self, cfg, config):
            super().__init__(cfg, config); d = tmp_path / self.ns
            self.out, self.results, self.ckpt, self.ledger = d / "out", d / "results", d / "ckpt", d / "out" / "ledger.jsonl"
            self.fv = {**self.fv, "budget": {**self.fv["budget"], "inherited_ledgers": []}}
    monkeypatch.setattr(tf, "Ctx", TmpCtx)
    monkeypatch.setattr(tf, "run_stage", lambda ctx, name, kind, fn: calls.append((ctx.ns, name, fn.__name__)))
    monkeypatch.setattr(tf, "stage_queue", lambda ctx, config, logs="logs": calls.append((ctx.ns, "queue", config)))
    monkeypatch.setattr(tf, "train", lambda ctx, seed, device: calls.append((ctx.ns, "train", seed)))

    def cli(config, *args):
        monkeypatch.setattr(sys, "argv", ["train_fact_video_feature.py", "--config", config, *args]); tf.main()
    cli(V1, "--check"); assert calls[-1] == ("fact_video_feature_v1", "check", "stage_check")
    n = len(calls)
    with pytest.raises(SystemExit):
        cli(V2, "--check")                                               # no repeated smoke under the revision
    for config in (V1, V2):
        with pytest.raises(SystemExit):
            cli(config, "--queue")                                       # no launch record yet
        with pytest.raises(SystemExit):
            cli(config, "--seed", "0")
    assert len(calls) == n
    c1 = TmpCtx(load_cfg(SimpleNamespace(config=V1, set=[])), V1); write_launch(c1, False)
    with pytest.raises(SystemExit):
        cli(V1, "--queue")                                               # the failed v1 check keeps v1 closed
    c2 = TmpCtx(load_cfg(SimpleNamespace(config=V2, set=[])), V2); write_launch(c2, True); c2.ledger.write_text("")
    cli(V2, "--queue"); assert calls[-1] == ("fact_video_feature_v2", "queue", V2)
    assert [json.loads(x)["job"] for x in c2.ledger.read_text().splitlines()] == ["queue overhead and freeze"]


def test_the_queue_launches_under_the_launch_record_and_the_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "ROOT", tmp_path); seen = []
    c2 = tmp_ctx(V2, tmp_path / "v2", inherited=False); idents = {k: f"id{k}" for k in c2.fv["queue"]}
    write_launch(c2, True, identities=idents, hours_per_run={k: 3.4 for k in idents}, post_training_hours=0.11)
    monkeypatch.setattr(tf, "make_head", lambda c, r: None); monkeypatch.setattr(tf, "identities", lambda c, h, p: idents)

    def fake_run(cmd, cwd, stdout, stderr):
        seen.append((cmd, stdout.name)); return SimpleNamespace(returncode=1)
    monkeypatch.setattr(tf.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        tf.stage_queue(c2, V2)                                           # the fake run fails after its launch
    cmd, log = seen[0]
    assert cmd[cmd.index("--config") + 1] == V2 and cmd[cmd.index("--seed") + 1] == "0" and log == str(tmp_path / "logs" / "fact_video_feature_v2_FACT_s0.log")
    st = json.loads((c2.out / "queue_status.json").read_text())
    assert st["projection"]["spent"] == 0.0 and abs(st["projection"]["total"] - (3 * 3.4 + 0.11 + 1.25)) < 1e-9 and st["jobs"]["FACT_s0"]["state"] == "failed"
    assert [json.loads(x)["job"] for x in c2.ledger.read_text().splitlines()] == ["train FACT_s0"]
    write_launch(c2, True, identities=idents, hours_per_run={k: 3.7 for k in idents}, post_training_hours=0.11); seen.clear()
    with pytest.raises(SystemExit):
        tf.stage_queue(c2, V2)                                           # 3 x 3.7 + 0.11 + 1.25 exceeds 12: nothing starts
    assert not seen and "exceed" in json.loads((c2.out / "queue_status.json").read_text())["stopped"]["reason"]


# ----------------------------------------------------------------------------------------------- the ledgers
def test_the_cumulative_ledger_counts_v1_once(tmp_path):
    v1 = tmp_path / "v1.jsonl"; tf.ledger_append(v1, "check", "gpu", "t", 1220.0, False); tf.ledger_append(v1, "prepare", "cpu", "t", 100.0, True)
    pin = {"path": "v1.jsonl", "sha256": sha(v1), "lines": 2, "gpu_hours": 1220.0 / 3600, "cpu_hours": 100.0 / 3600}
    assert abs(tf.inherited_ledger_hours(tmp_path, [pin], "gpu") - 1220.0 / 3600) < 1e-12
    ctx = SimpleNamespace(root=tmp_path, ledger=tmp_path / "v2.jsonl", fv={"budget": {"inherited_ledgers": [pin]}})
    tf.ledger_append(ctx.ledger, "revalidation check", "gpu", "t", 90.0, True)
    assert abs(tf.spent_hours(ctx, "gpu") - (1220.0 + 90.0) / 3600) < 1e-12 and abs(tf.spent_hours(ctx, "cpu") - 100.0 / 3600) < 1e-12
    for bad in ({**pin, "lines": 3}, {**pin, "gpu_hours": 0.1}):
        with pytest.raises(SystemExit):
            tf.inherited_ledger_hours(tmp_path, [bad], "gpu")
    with pytest.raises(SystemExit):
        tf.spent_hours(SimpleNamespace(root=tmp_path, ledger=v1, fv=ctx.fv), "gpu")         # a ledger is never counted twice
    tf.ledger_append(v1, "late line", "gpu", "t", 10.0, True)
    with pytest.raises(SystemExit):
        tf.spent_hours(ctx, "gpu")                                       # the inherited boundary changed
    assert abs(tf.spent_hours(SimpleNamespace(root=tmp_path, ledger=ctx.ledger, fv={"budget": {}}), "gpu") - 90.0 / 3600) < 1e-12


# ----------------------------------------------------------------------------------------------- the monitor
def same_rng(a: dict, b: dict) -> bool:
    return a["python"] == b["python"] and all(np.array_equal(x, y) for x, y in zip(a["numpy"], b["numpy"])) and torch.equal(a["torch"], b["torch"])


def test_the_monitor_groups_token_counts_exactly(monkeypatch):
    import scripts.fact_integration as fi
    plan = iter([([5] * 12, [0], [0], [2]),                            # single segment, all null (the official fallback case)
                 ([2] + [5] * 11, [0], [0], [2]),                      # single segment, one active token, right
                 ([1, 3, 1] + [5] * 9, [0, 1, 2], [0, 1, 2], [1, 3, 0]),   # three segments, three active tokens, two right
                 ([5] * 12, [4, 7], [0, 1], [0, 4])])                  # two segments, all null

    class Model:
        block_list = [SimpleNamespace(action_clogit=None)]; mcriterion = SimpleNamespace(transcript=None)

        def eval(self):
            return self

        def _forward_one_video(self, x):
            cls, a, sg, trans = next(plan); logits = torch.full((12, 1, 6), -5.0); logits[torch.arange(12), 0, torch.tensor(cls)] = 5.0
            self.block_list[-1].action_clogit, self.cur = logits, (a, sg, trans)

    def fake_parts(head, m, lab):
        a, sg, trans = m.cur; m.mcriterion.transcript = torch.tensor(trans)
        return {"blocks": [{"frame": torch.tensor(1.0), "weights": {"frame": 1.0}}], "total": torch.tensor(1.0), "match": (torch.tensor(a), torch.tensor(sg))}
    monkeypatch.setattr(fi, "loss_parts", fake_parts)
    seqs = [SimpleNamespace(seq_id=f"s{i}", feat=np.zeros((4, 3), np.float32), labels=np.zeros(4, np.int64)) for i in range(4)]
    out = tf.term_values(SimpleNamespace(num_classes=5), Model(), seqs, "cpu")
    assert out["tokens"] == {"single": {"sequences": 2, "all_null_sequences": 1, "active_tokens": 1, "null_tokens": 23, "matched": 2, "matched_correct": 1},
                             "multi": {"sequences": 2, "all_null_sequences": 1, "active_tokens": 3, "null_tokens": 21, "matched": 5, "matched_correct": 2}}
    assert out["loss_frame"] == 1.0 and out["loss_total"] == 1.0


@needs_fact
def test_the_monitor_counts_tokens_from_its_own_forwards():
    import scripts.fact_integration as fi
    import scripts.train_fact_asrf as tfa
    from echo_routing.temporal.dev_data import Seq
    rng = np.random.default_rng(3); seqs = []
    for i, labels in enumerate((np.full(20, 2), np.array([1] * 9 + [3] * 7), np.full(15, 4), np.array([0] * 6 + [2] * 8 + [0] * 5))):
        seqs.append(Seq(f"s{i}", i % 2 == 0, rng.standard_normal((labels.size, 512)).astype(np.float32), labels.astype(np.int64), np.zeros(0, np.int64)))
    head = fh.FACTHead(REPO, 512, fh.TokenRule(**RULE), num_classes=5, base_config="gtea.yaml"); torch.manual_seed(0); model = head.build()
    chosen = [("native" if s.is_native else "stream", s) for s in seqs]
    start = tfa.rng_state(); model.eval()
    with torch.no_grad():                                               # the old monitor, verbatim
        old = {**tfa.monitor(head, model, chosen, "cpu"), **{f"loss_{k}": float(v) for k, v in fi.term_losses(head, model, seqs, "cpu").items()}}
    after_old = tfa.rng_state(); tfa.set_rng_state(start)
    new = tf.monitor(head, model, seqs, "cpu"); after_new = tfa.rng_state()
    assert same_rng(after_old, after_new) and {k: v for k, v in new.items() if k != "tokens"} == old
    tok = new["tokens"]
    assert tok["single"]["sequences"] == 2 and tok["multi"]["sequences"] == 2
    assert all(tok[g]["active_tokens"] + tok[g]["null_tokens"] == 12 * tok[g]["sequences"] for g in ("single", "multi"))
    assert tok["single"]["matched"] == 2 and tok["multi"]["matched"] == 5 and 0 <= tok["multi"]["matched_correct"] <= 5
    model.eval()
    with torch.no_grad():
        rows = fi.outcome(head, model, chosen, "cpu")["rows"]
    single = [r for r in rows if r["segments"] == 1]; multi = [r for r in rows if r["segments"] > 1]
    assert tok["single"]["active_tokens"] == sum(r["tokens_used"] for r in single) and tok["multi"]["active_tokens"] == sum(r["tokens_used"] for r in multi)
    assert tok["single"]["all_null_sequences"] == sum(r["tokens_used"] == 0 for r in single)
    assert tok["multi"]["matched_correct"] == round(sum(r["matched_token_class_correct"] * r["segments"] for r in multi))
    assert tok["single"]["matched_correct"] == round(sum(r["matched_token_class_correct"] for r in single))
