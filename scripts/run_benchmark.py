#!/usr/bin/env python
"""Run registered frozen systems on the frozen EV9V test benchmark (docs/benchmark/README.md).

  --stage dev-check                 no test read: our adapter on the development calibration half (online A08 parity
                                    against A13's verified cache; A08 + B3 and A14 P stored rows reproduced through the
                                    adapter), EchoPrime online/cache parity on validation pair streams, a hand case
  --stage predict  --run-id ID      predictions of every listed system (online or cached), with their checks and cost
  --stage evaluate --run-id ID      scores, H09 reproduction of the cached external rows, paired comparisons, JSON / CSV
  --stage report   --run-id ID      the Chinese Markdown comparison tables of a finished run
A finished run directory is never overwritten; a new model or version runs under a new run id.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.benchmark_adapters as ba  # noqa: E402
import scripts.benchmark_evaluate as be  # noqa: E402
import scripts.benchmark_manifest as bm  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

H09 = {"runs/_demo/true_order_joint/joint.json": {"STFM_numeric_s100": "B6 STFM windowed, numeric-order model", "EchoViewCLIP_stage1_windowed": "B7 EchoViewCLIP windowed",
                                                 "EchoPrime_view_per_frame": "B8 EchoPrime per frame", "EchoPrime_view_B3": "B8 + audited B3 (EchoPrime setting)"},
       "runs/_demo/true_order_joint_b5/joint.json": {"ASFormer_v2_frameorder": "B5 ASFormer v2"}}
OURS = ("A14_P_s0", "A14_P_s2", "A08_s0", "A08_s2")


def ctx_of(path: str) -> SimpleNamespace:
    cfg = load_cfg(SimpleNamespace(config=path, set=[])); fb = cfg["frozen_external_benchmark"]; o = fb["outputs"]
    return SimpleNamespace(cfg=cfg, fb=fb, config=path, out=ROOT / o["root"], results=ROOT / o["results_dir"], ledger=ROOT / fb["budget"]["ledger"])


def book(c, what: str, started: str, total: float, acc: dict, workers: int, ok: bool, note: str) -> None:
    """A19's split: GPU inference seconds as GPU; the main process outside them and the extra decode workers as CPU."""
    g = min(acc["gpu_seconds"], total)
    vsh.ledger_append(c.ledger, f"{what} (GPU inference seconds)", "gpu", started, g, ok, note)
    vsh.ledger_append(c.ledger, f"{what} (CPU: main process outside GPU + {max(0, workers - 1)} more decode workers)", "cpu", started, (total - g) + acc["decode_seconds"] * max(0, workers - 1), ok, note)


def new_acc() -> dict:
    return {"gpu_seconds": 0.0, "decode_seconds": 0.0, "windows": 0}


# ----------------------------------------------------------------------------------------------- development check
def echoprime_parity(n_streams: int, device) -> dict:
    """EchoPrime's released classifier online, by the historical runner's code path, on validation pair streams."""
    import torchvision
    from PIL import Image
    from echo_routing.config import load_paths
    from echo_routing.ingest.frames import apply_gamma, list_frame_paths
    from echo_routing.temporal.windowed import ECHOPRIME_TO_FAMILY5, ECHOPRIME_VIEWS, map_probs
    paths = load_paths(); repo = ROOT / "third_party" / "EchoPrime"; cwd = os.getcwd(); os.chdir(repo); sys.path.insert(0, str(repo))
    import utils  # noqa: E402
    os.chdir(cwd)
    vc = torchvision.models.convnext_base(); vc.classifier[-1] = torch.nn.Linear(vc.classifier[-1].in_features, 11)
    vc.load_state_dict(torch.load(ROOT / "models_frozen/echoprime_view_classifier/release_v1.0.0/view_classifier.pt", map_location="cpu")); vc.to(device).eval()
    mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1); std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1)
    plans = json.loads((paths.data_root / "window_plans" / bm.ENC / "plans.json").read_text()); root = Path(plans["frames_root"])
    sids = sorted(k for k, v in plans["streams"].items() if v["bank"] == "pairs_validation")[:n_streams]; diffs, n, sec = [], 0, 0.0
    with torch.no_grad():
        for sid in sids:
            xs = []
            for vid, fi, g in plans["streams"][sid]["refs"]:
                fr = list_frame_paths(root / vid)
                with Image.open(fr[min(fi, len(fr) - 1)]) as im:
                    im = im.convert("RGB"); im = apply_gamma(im, g) if g != 1.0 else im
                    xs.append((torch.as_tensor(utils.crop_and_scale(np.asarray(im)), dtype=torch.float).permute(2, 0, 1) - mean) / std)
            torch.cuda.synchronize(); t0 = time.time(); raw = np.concatenate([torch.softmax(vc(torch.stack(xs[i:i + 64]).to(device)).float(), 1).cpu().numpy() for i in range(0, len(xs), 64)])
            torch.cuda.synchronize(); sec += time.time() - t0; n += len(xs); fam = map_probs(raw, ECHOPRIME_VIEWS, ECHOPRIME_TO_FAMILY5)
            with np.load(paths.cache_root / "echoprime_windowed" / bm.ENC / f"{sid}.npz") as z:
                diffs.append(float(np.abs(fam - z["prob"]).max()))
    return {"streams": len(sids), "samples": n, "prob_max_abs": max(diffs), "gpu_seconds_per_frame": sec / n, "parameters": int(sum(p.numel() for p in vc.parameters())), "ok": max(diffs) <= 1e-4}


def dev_check(c, device, workers: int) -> None:
    t0, started, ok, note, acc, res = time.time(), vsh.now(), False, "", new_acc(), {"config_sha256": sha256_file(ROOT / c.config), "hand_case": be.hand_case()}
    try:
        a8 = rsh.a08_ctx(); ev, seqs = a8.half("calibration", rso.DevAccess({"calibration"})); items = [{"id": q["id"], "n": len(q["labels"]), "refs": q["refs"]} for q in seqs]
        a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())["systems"]
        a14rows = json.loads((ROOT / "docs/results/2026-10-21_local_router/screen.json").read_text())["systems"]; reg = ba.registry(c.cfg); a14 = ba.A14(); rep = {}
        for s in (0, 2):
            e = reg[f"A14_P_s{s}"]; ba.verify_weights(f"A14_P_s{s}", e)
            with np.load(ROOT / "runs/video_frozen_switch_v1/evaluation/calibration" / f"a08_s{s}.npz") as z:
                cached = {"logits": z["logits"].astype(np.float32), "emb": z["emb"].astype(np.float32), "seq": z["seq"], "centre": z["centre"]}
            sq = ba.adapter_seqs(items); W, _, _ = ba.windows(sq)
            if not (np.array_equal(cached["seq"], W["seq"]) and np.array_equal(cached["centre"], W["centre"])):
                raise SystemExit("A13's development cache does not match the adapter's windows")
            out = {"W": W, "seqs": sq, "logits": cached["logits"], "emb": cached["emb"]}
            b = ba.a08_b3_labels(out, a14.ctx.base_params(s)); r = a14.labels(out, s, e)
            rso.check_stored(rso.row_of(a8.cfg, b, ev)["row"], a8sum[f"A + B3 s{s}"]["calibration"], f"A08 + B3 s{s} calibration through the adapter")
            rso.check_stored(rso.row_of(a8.cfg, r["labels"], ev)["row"], a14rows[f"P router s{s}"]["calibration"], f"A14 P s{s} calibration through the adapter")
            sub = items[:60]; online = ba.a08_outputs(sub, "dev-check", e["weights"]["a08"], c.out / "dev_check_store", device, workers, acc)
            m = len(online["W"]["seq"]); rep[f"s{s}"] = {"rows_reproduced": ["A08 + B3 calibration", "A14 P calibration"], "a14_diag": r["diag"],
                                                        "online_vs_cache_logits_max_abs (first 60 sequences)": float(np.abs(online["logits"] - cached["logits"][:m]).max()),
                                                        "online_vs_cache_emb_max_abs": float(np.abs(online["emb"] - cached["emb"][:m]).max()), "windows": m, "state_unchanged": online["state_unchanged"]}
        res["ours"] = rep; res["echoprime_parity"] = echoprime_parity(4, device)
        res["ok"] = bool(res["hand_case"]["ok"] and res["echoprime_parity"]["ok"] and all(v["online_vs_cache_logits_max_abs (first 60 sequences)"] <= 1e-4 and v["state_unchanged"] for v in rep.values()))
        for p in (c.out / "dev_check.json", c.results / "dev_check.json"):
            p.parent.mkdir(parents=True, exist_ok=True); vsh.write_json(p, json.loads(json.dumps(res, default=str)))
        print(json.dumps(res, indent=1, default=str)[:4000], flush=True)
        if not res["ok"]:
            raise SystemExit("a development check failed")
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        book(c, "development check", started, time.time() - t0, acc, workers, ok, note)


# ----------------------------------------------------------------------------------------------- predict
def predict(c, systems: list[str], run_id: str, device, workers: int) -> None:
    t0, started, ok, note, acc = time.time(), vsh.now(), False, "", new_acc(); rd = c.out / "runs" / run_id; pdir = rd / "predictions"
    try:
        if (rd / "evaluation.json").exists():
            raise SystemExit(f"run {run_id} is finished; use a new run id")
        man = bm.load(c.cfg); items = man["items"]; bdig = bm.digest(items); reg = ba.registry(c.cfg); pdir.mkdir(parents=True, exist_ok=True); meta = {}; a14 = None; outs = {}
        for name in systems:
            e = reg[name]; ba.verify_weights(name, e); g0 = acc["gpu_seconds"]; w0 = acc["windows"]
            if e["adapter"] == "cached_probs":
                r = ba.cached_probs(items, name, e); labels, m = r["labels"], r["meta"]
            else:
                s = int(e["seed"])
                if s not in outs:
                    outs[s] = ba.a08_outputs(items, bdig, e["weights"]["a08"], c.out / "store", device, workers, acc)
                if e["adapter"] == "a08_b3":
                    a14 = a14 or ba.A14(); labels = ba.a08_b3_labels(outs[s], a14.ctx.base_params(s)); m = {"store": outs[s]["store"], "key": outs[s]["key"], "state_unchanged": outs[s]["state_unchanged"]}
                else:
                    a14 = a14 or ba.A14(); r = a14.labels(outs[s], s, e); labels = r["labels"]; m = {"store": outs[s]["store"], "key": outs[s]["key"], **r["diag"]}
                    if not all(np.array_equal(r["base"][k], v) for k, v in ba.a08_b3_labels(outs[s], a14.ctx.base_params(s)).items()):
                        raise SystemExit(f"{name}: the router's Base differs from A08 + B3")
            np.savez_compressed(pdir / f"{name}.npz", ids=np.asarray(list(labels)), offsets=np.cumsum([0, *[len(v) for v in labels.values()]]), labels=np.concatenate(list(labels.values())).astype(np.int8))
            meta[name] = {"adapter": e["adapter"], "weights": {k: w["sha256"] for k, w in e["weights"].items()}, **m, "gpu_seconds_new": acc["gpu_seconds"] - g0, "windows_new": acc["windows"] - w0}
            print(f"{name}: predicted ({e['adapter']})", flush=True)
        rec = {"run_id": run_id, "benchmark_digest": bdig, "config_sha256": sha256_file(ROOT / c.config), "registry_sha256": sha256_file(ROOT / c.fb["registry"]["path"]), "systems": meta,
               "cost": {"a08_gpu_seconds": acc["gpu_seconds"], "a08_windows": acc["windows"], "a08_seconds_per_window": acc["gpu_seconds"] / max(1, acc["windows"]), "decode_seconds": acc["decode_seconds"],
                        "a08_parameters": 34640549, "a08_encoder_forwards_per_window": 1}, "time": vsh.now()}
        vsh.write_json(rd / "predictions.json", json.loads(json.dumps(rec, default=str))); ok = True; note = f"GPU {acc['gpu_seconds']:.0f} s"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        book(c, f"predict run {run_id}", started, time.time() - t0, acc, workers, ok, note)


# ----------------------------------------------------------------------------------------------- evaluate
def load_preds(rd: Path, name: str) -> dict:
    with np.load(rd / "predictions" / f"{name}.npz") as z:
        o = z["offsets"]; return {k: z["labels"][o[i]:o[i + 1]].astype(int) for i, k in enumerate(z["ids"].tolist())}


def evaluate(c, systems: list[str], run_id: str) -> None:
    t0, started, ok, note = time.time(), vsh.now(), False, ""; rd = c.out / "runs" / run_id
    try:
        if (rd / "evaluation.json").exists():
            raise SystemExit(f"run {run_id} is finished; use a new run id")
        B = be.Benchmark(c.cfg); pred = json.loads((rd / "predictions.json").read_text())
        if pred["benchmark_digest"] != B.digest:
            raise SystemExit("the predictions belong to another benchmark manifest")
        evals = {n: B.evaluate(n, load_preds(rd, n)) for n in systems}; ext = [n for n in systems if n not in OURS]; ours = [n for n in systems if n in OURS]
        boot = {ref: be.bootstrap(evals, ref) for ref in ext + [n for n in ours if n.startswith("A08")]}; vals = next(iter(boot.values()))
        repro = {}
        for path, names in H09.items():
            j = json.loads((ROOT / path).read_text())["endpoints"]
            for n, old in names.items():
                if n in vals:
                    d = {ep: max(abs(vals[n][ep]["value"] - j[old][ep]["value"]), *(abs(a - b) for a, b in zip(vals[n][ep]["ci95"], j[old][ep]["ci95"]))) for ep in j[old]}
                    repro[n] = {"historical_row": old, "source": path, "max_abs": max(d.values()), "ok": max(d.values()) <= 1e-9}
        if not all(r["ok"] for r in repro.values()):
            raise SystemExit(f"H09 rows do not reproduce: {repro}")
        cmp_ = {f"{a} - {r}": {ep: {k: boot[r][a][ep][k] for k in ("diff", "diff_ci95")} for ep in vals[a]} for r in boot for a in ours if a != r and (r in ext or a.startswith("A14") and r == f"A08_s{a[-1]}")}
        res = {"run_id": run_id, "benchmark_digest": B.digest, "config_sha256": sha256_file(ROOT / c.config), "values": {n: {ep: {k: vals[n][ep][k] for k in ("value", "ci95")} for ep in vals[n]} for n in systems},
               "comparisons": cmp_, "h09_reproduction": repro, "predictions": pred["systems"], "cost": pred["cost"], "time": vsh.now(),
               "units": {"native": "session (file prefix)", "pairs": "stream", "multi segments": "stream", "multi boundaries": "recording"}}
        flat = json.loads(json.dumps(res, default=float)); vsh.write_json(rd / "evaluation.json", flat); c.results.mkdir(parents=True, exist_ok=True); vsh.write_json(c.results / f"evaluation_{run_id}.json", flat)
        eps = list(be.endpoints())
        with (c.results / f"values_{run_id}.csv").open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["system", *eps])
            for n in systems:
                w.writerow([n, *[f"{vals[n][ep]['value']:.6f}" for ep in eps]])
        print(json.dumps({n: {ep: round(vals[n][ep]["value"], 4) for ep in ("segment F1@50, multi", "clip macro-F1", "boundary F1, multi", "multi FP segments", "multi FN segments")} for n in systems}, indent=1), flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        vsh.ledger_append(c.ledger, f"evaluate run {run_id} (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- report (Chinese tables)
ZH = {"A14_P_s0": "A14 P 局部路由 s0", "A14_P_s2": "A14 P 局部路由 s2", "A08_s0": "A08 + B3 s0", "A08_s2": "A08 + B3 s2", "STFM_numeric_s100": "STFM（数值帧序，seed 100）",
      "ASFormer_v2_frameorder": "ASFormer v2_frameorder（B0 特征）", "EchoPrime_view_per_frame": "EchoPrime 视图分类器，逐帧", "EchoPrime_view_B3": "EchoPrime 视图分类器 + B3（验证集设置）",
      "EchoViewCLIP_stage1_windowed": "EchoViewCLIP stage 1，滑窗"}
COLS = [("segment F1@50, multi", "多流分段 F1@50", 4), ("clip macro-F1", "原生五类 macro-F1", 4), ("clip accuracy", "原生准确率", 4), ("boundary precision, multi", "多流边界 P", 3),
        ("boundary recall, multi", "多流边界 R", 3), ("boundary F1, multi", "多流边界 F1", 3), ("multi FP segments", "多流 FP", 0), ("multi FP segments <= 16", "其中短", 0),
        ("multi FN segments", "多流 FN", 0), ("multi missed true segments <= 16", "其中短", 0), ("segment F1@50, pairs", "两段流分段 F1", 4), ("boundary F1, pairs", "两段流边界 F1", 3),
        ("native cuts per clip", "原生切点/段", 3), ("same-class cut rate, multi", "同类拼接误切率", 3), ("A4C read as A5C (clips)", "A4C→A5C", 0), ("A5C read as A4C (clips)", "A5C→A4C", 0)]


def fmt(v: float, d: int) -> str:
    return f"{v:.{d}f}" if d else f"{v:.1f}".rstrip("0").rstrip(".")


def report(c, run_id: str) -> None:
    ev = json.loads((c.out / "runs" / run_id / "evaluation.json").read_text()); V, C = ev["values"], ev["comparisons"]; systems = list(V)
    rows = ["| 系统 | " + " | ".join(z for _, z, _ in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for n in systems:
        rows.append(f"| {ZH.get(n, n)} | " + " | ".join(fmt(V[n][k]["value"], d) for k, _, d in COLS) + " |")
        if n in ("A14_P_s2", "A08_s2"):
            a = n.replace("s2", "s0"); rows.append(f"| **{ZH[n][:-3]} 两 seed 均值** | " + " | ".join(fmt((V[a][k]["value"] + V[n][k]["value"]) / 2, d) for k, _, d in COLS) + " |")
    ext = [n for n in systems if n not in OURS]; ep = "segment F1@50, multi"
    diff = ["| 我方 − 外部（多流分段 F1@50，95% 配对区间，按流重采样） | " + " | ".join(ZH[e] for e in ext) + " |", "|---|" + "---|" * len(ext)]
    for a in ("A14_P", "A08"):
        for s in (0, 2):
            k = f"{a}_s{s}"; diff.append(f"| {ZH[k]} | " + " | ".join(f"{C[f'{k} - {e}'][ep]['diff']:+.4f} [{C[f'{k} - {e}'][ep]['diff_ci95'][0]:+.4f}, {C[f'{k} - {e}'][ep]['diff_ci95'][1]:+.4f}]" for e in ext) + " |")
        diff.append(f"| {ZH[a + '_s0'][:-3]} 两 seed 均值（点估计） | " + " | ".join(f"{(C[f'{a}_s0 - {e}'][ep]['diff'] + C[f'{a}_s2 - {e}'][ep]['diff']) / 2:+.4f}" for e in ext) + " |")
    rt = ["| A14 P − 同 seed A08 | " + " | ".join(z for _, z, _ in COLS[:10]) + " |", "|---|" + "---|" * 10]
    for s in (0, 2):
        k = f"A14_P_s{s} - A08_s{s}"; rt.append(f"| seed {s} | " + " | ".join(f"{C[k][e]['diff']:+.{max(d, 1)}f} [{C[k][e]['diff_ci95'][0]:+.{max(d, 1)}f}, {C[k][e]['diff_ci95'][1]:+.{max(d, 1)}f}]" for e, _, d in COLS[:10]) + " |")
    out = c.results / f"comparison_{run_id}_zh.md"; out.write_text("\n".join(["## 同场主表", "", *rows, "", "## 配对差值", "", *diff, "", "## 路由相对 A08", "", *rt, ""]))
    print(out.read_text(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("dev-check", "predict", "evaluate", "report"), required=True); ap.add_argument("--run-id")
    ap.add_argument("--systems", nargs="+"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); c = ctx_of(a.config); systems = a.systems or list(c.fb["systems"])
    if a.stage == "dev-check":
        vsh.determinism(); return dev_check(c, torch.device("cuda"), a.workers)
    if not a.run_id:
        raise SystemExit("--run-id is required")
    if a.stage == "predict":
        vsh.determinism(); return predict(c, systems, a.run_id, torch.device("cuda"), a.workers)
    if a.stage == "report":
        return report(c, a.run_id)
    evaluate(c, systems, a.run_id)


if __name__ == "__main__":
    main()
