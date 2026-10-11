#!/usr/bin/env python
"""Evaluate, freeze and report the raw-video context round (docs/superpowers/plans/2026-10-10-raw-video-recognition.md,
Tasks 3-4; configs/experiments/raw_video_context_v1.yaml). `--evaluate` runs once, after the nine trainings.

1. Runs. Every run's identity is recomputed, its 3,750 updates and final weights checked, and the nine final weights
   are frozen with scripts/freeze_model.py. Only then are development frames loaded.
2. Windows. Centres every 5 samples on every native clip and constructed stream of both halves; 16 samples by
   window_frames; the marker slot from s and c; for Shuffled models the permutation of the non-target slots from
   SHA256(seed|sequence id|centre). Each model reads its own group's input. Per-window logits, slot orders, markers
   and centres and the per-sample probabilities (nearest centre) are saved on the server only.
3. Recognition. Native windows: A4C/A5C balanced accuracy, margin AUC and all-window accuracy for the nine models and
   E1 at the same requested samples (whose frame must be the target frame), with the session-paired bootstrap of the
   within-seed comparisons. Mixed windows of constructed streams and windows clamped at sequence ends.
4. Systems. Direct argmax rows and B3 chosen once on calibration per model; the references reproduced with their
   stored settings (E0/E1/B7 + B3, historical E1 + ASRF, the paired ASRF and Original FACT, C + B3, G1 + ASRF rows);
   paired bootstrap, verdicts, goal, fixed decision, units, boundary offsets and fixed/new errors with recordings and
   sessions; costs and training curves.
Results go to the results directory through id guards after every stage. Nothing here chooses an epoch, a threshold
beyond the fixed B3 rule, a seed or a model.

  python scripts/report_raw_video_context.py --config configs/experiments/raw_video_context_v1.yaml --evaluate
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.raw_video_context_tables as rt  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_fact_asrf as ra  # noqa: E402
import scripts.train_raw_video_context as tr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import FLAT, boundary_changes, native_changes, segment_changes, summary, write_atomic  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_config  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import nearest_centre, window_frames  # noqa: E402
from echo_routing.temporal.joint_visual import load_frame_u8  # noqa: E402
from echo_routing.temporal.windowed import window_centres  # noqa: E402

NS = "raw_video_context_v1"
HALVES = rt.HALVES
SHORT = 16
DIAG = {"evaluation": {"b3_infeasible_diagnostic": {"half_window": 9, "quantile": 0.95}}}
WITHIN = {"Ordered - Shuffled": ("Ordered context", "Shuffled context"), "Ordered - Target-only": ("Ordered context", "Target-only"),
          "Shuffled - Target-only": ("Shuffled context", "Target-only")}
REFERENCES = {"Ordered - (E1 + B3)": "E1 + B3", "Ordered - (E0 + B3)": "E0 + B3", "Ordered - (B7 + B3)": "B7 + B3", "Ordered - (C + B3)": "C + B3 s{k}",
              "Ordered - Original FACT": "Original FACT s{k}", "Ordered - paired ASRF": "paired ASRF s{k}", "Ordered - historical E1 + ASRF": "E1 + ASRF s{k}"}


# ----------------------------------------------------------------------------------------------- runs
def verify_runs(ctx) -> dict:
    runs = {}
    for key in ctx.rv["queue"]:
        g, s = tr.GROUP_OF[key.rsplit("_s", 1)[0]], int(key.rsplit("_s", 1)[1]); man_path = ctx.out / "train" / key / "run_manifest.json"
        if not man_path.is_file():
            raise SystemExit(f"{key}: no finished run")
        man = json.loads(man_path.read_text())
        ident = tr.identity(ctx, g, s, ctx.entry(s)["files"]["manifest"]["sha256"], ctx.entry(s)["files"]["init"]["sha256"])
        if man["identity"] != ident or int(man["updates"]) != ctx.total or sha256_file(Path(man["final_checkpoint"])) != man["final_sha256"]:
            raise SystemExit(f"{key}: identity, update count or final weights do not verify")
        runs[key] = {"group": g, "seed": s, "manifest": man}
    return runs


def freeze(ctx, runs: dict) -> dict:
    out = {}
    for key, r in runs.items():
        man, dest = r["manifest"], ctx.root / "models_frozen" / NS / key
        try:
            if not dest.exists():
                files = [man["final_checkpoint"], str(ctx.out / "train" / key / "run_manifest.json"), str(ctx.mdir / f"init_s{r['seed']}.pt"),
                         str(ctx.mdir / "index.json"), str(ctx.root / ctx.config)]
                subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={man['identity']}",
                                "--meta", f"group={r['group']}", "--meta", f"seed={r['seed']}",
                                "--notes", "Raw-video context round (2026-10-10); final weights after 3,750 updates; EchoPrime MViTv2-S with a target marker."],
                               cwd=ctx.root, check=True, capture_output=True)
            out[key] = str(dest) if sha256_file(dest / "final.pt") == man["final_sha256"] else f"frozen copy differs from {man['final_sha256'][:12]}"
        except (subprocess.CalledProcessError, OSError) as e:
            out[key] = f"not frozen: {type(e).__name__}"
    if any(not v.startswith(str(ctx.root)) for v in out.values()):
        raise SystemExit(f"freezing failed: {out}")
    return out


def load_model(ctx, run: dict, device) -> tm.TargetMViT:
    init, _ = ctx.init(run["seed"]); model = tr.make_model(ctx, init, device)
    model.load_state_dict(torch.load(run["manifest"]["final_checkpoint"], map_location=device, weights_only=True), strict=True)
    return model.eval()


# ----------------------------------------------------------------------------------------------- frames and windows
class RefFrames(Dataset):
    def __init__(self, refs: list[tuple], fp, chunk: int = 256) -> None:
        self.refs, self.fp, self.chunk = refs, fp, chunk

    def __len__(self) -> int:
        return (len(self.refs) + self.chunk - 1) // self.chunk

    def __getitem__(self, i: int) -> torch.Tensor:
        out = []
        for v, f, g in self.refs[i * self.chunk:(i + 1) * self.chunk]:
            p = self.fp(v); out.append(load_frame_u8(p[min(int(f), len(p) - 1)], round(float(g), 6)))
        return torch.from_numpy(np.stack(out))


def decode_frames(refs: list[tuple], fp, workers: int) -> np.ndarray:
    frames = np.empty((len(refs), 224, 224, 3), np.uint8); k = 0
    for x in DataLoader(RefFrames(refs, fp), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None):
        frames[k:k + len(x)] = x.numpy(); k += len(x)
    if k != len(refs):
        raise SystemExit(f"decoded {k} of {len(refs)} frames")
    return frames


def half_windows(seqs_h: list[dict], at: dict, stride: int) -> dict:
    """Every window of a half: sequence index, centre, samples s, frame indices, marker slot and the reference labels."""
    seq, cen, ss, ref, mk, lab, lab16 = [], [], [], [], [], [], []
    for qi, q in enumerate(seqs_h):
        n = len(q["labels"]); labels = np.asarray(q["labels"], int)
        for c in window_centres(n, stride):
            s = window_frames(int(c), n, 16, 16)
            seq.append(qi); cen.append(int(c)); ss.append(s); ref.append([at[q["refs"][i]] for i in s]); mk.append(tm.marker_slot(s, int(c)))
            lab.append(int(labels[c])); lab16.append(labels[s])
    return {"seq": np.asarray(seq), "centre": np.asarray(cen), "s": np.asarray(ss), "ref": np.asarray(ref), "marker": np.asarray(mk),
            "label": np.asarray(lab), "labels16": np.asarray(lab16)}


def slot_orders(group: str, seed: int, W: dict, seqs_h: list[dict]) -> np.ndarray:
    out = np.empty(W["s"].shape, np.int64)
    for i in range(len(out)):
        s, c = W["s"][i], int(W["centre"][i])
        sh = tm.shuffle_order(tm.inference_rng(seed, seqs_h[W["seq"][i]]["id"], c), s, c) if group == "Shuffled context" else np.arange(16)
        out[i] = tm.group_order(group, s, c, sh)
    return out


@torch.no_grad()
def model_logits(model, W: dict, orders: np.ndarray, frames: np.ndarray, device, batch: int = 32) -> tuple[np.ndarray, float]:
    out = []; torch.cuda.synchronize(); t0 = time.time()
    for b in range(0, len(orders), batch):
        idx = np.take_along_axis(W["ref"][b:b + batch], orders[b:b + batch], 1)
        u8 = torch.from_numpy(frames[idx]).to(device, non_blocking=True)
        out.append(model(tm.to_rgb(u8), tm.marker_planes(torch.from_numpy(W["marker"][b:b + batch]).long().to(device))).float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(out), time.time() - t0


def per_sample(W: dict, logits: np.ndarray, seqs_h: list[dict]) -> dict:
    p = torch.softmax(torch.from_numpy(logits), -1).numpy(); out = {}
    for qi, q in enumerate(seqs_h):
        rows = np.flatnonzero(W["seq"] == qi); out[q["id"]] = p[rows][nearest_centre(len(q["labels"]), W["centre"][rows])]
    return out


def save_outputs(path: Path, ident: str, W: dict, orders: np.ndarray, logits: np.ndarray, probs: dict, seqs_h: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); ids = [q["id"] for q in seqs_h]
    np.savez(path, identity=ident, sequence_ids=np.asarray(ids), seq=W["seq"], centre=W["centre"], marker=W["marker"], order=orders.astype(np.int8),
             logits=logits.astype(np.float32), sample_offsets=np.cumsum([0, *[len(probs[i]) for i in ids]]),
             sample_probs=np.concatenate([probs[i] for i in ids]).astype(np.float32))


# ----------------------------------------------------------------------------------------------- systems
def system_rows(cfg: dict, probs: dict, inputs) -> dict:
    """Direct argmax and B3 (chosen once on calibration, then applied to selection) rows, tables and predictions."""
    out = {"argmax": {}, "b3": {}}
    ch = rcs.b3_choose(cfg, DIAG, probs["calibration"], inputs.halves["calibration"])
    for h in HALVES:
        for row, preds in (("argmax", {k: v.argmax(1) for k, v in probs[h].items()}), ("b3", decode(ch["params"], probs[h]))):
            m, t = rcs.evaluate(cfg, preds, inputs.halves[h])
            out[row][h] = {"row": {**summary(m), "native_precision": rcs.summary(m)["native_precision"]}, "tables": t, "preds": preds}
    out["b3_choice"] = {k: v for k, v in ch.items() if k != "grid"}; out["b3_grid"] = [{k: v for k, v in r.items()} for r in ch["grid"]]
    return out


def reference_c(ctx, cfg: dict, paths, inputs, seqs: dict, frames: np.ndarray, at: dict, device) -> dict:
    """The centre-supervision C models with their stored B3 settings; their stored B3 rows must be reproduced."""
    import scripts.train_center_supervision as tcs
    ref = ctx.rv["evaluation"]["reference_inputs"]["center_supervision"]
    for k in ("config", "report"):
        if sha256_file(ctx.root / ref[k]["path"]) != ref[k]["sha256"]:
            raise SystemExit(f"C reference {k} differs from its pin")
    cs = load_config(ctx.root / ref["config"]["path"])["center_supervision"]; stored = json.loads((ctx.root / ref["report"]["path"]).read_text())["models"]
    out = {}
    for key, pins in ref["frozen"].items():
        seed = int(key.split("_s")[1]); d = ctx.root / "models_frozen" / "center_supervision_v1" / key
        if sha256_file(d / "final.pt") != pins["final"] or sha256_file(d / f"init_s{seed}.pt") != pins["init"]:
            raise SystemExit(f"{key}: frozen files differ from their pins")
        model, _ = tcs.build_model(cs, paths, d / f"init_s{seed}.pt", device); st = torch.load(d / "final.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for n, p in model.encoder.trainable():
                p.copy_(st["encoder_trainable"][n].to(device))
        model.head.load_state_dict(st["head"]); model.eval()
        with torch.no_grad():
            feats = np.concatenate([model.encoder(torch.from_numpy(frames[i:i + 256]).to(device)).float().cpu().numpy() for i in range(0, len(frames), 256)])
        params = stored[key]["rows"]["b3_choice"]["params"]; name = f"C + B3 s{seed}"; out[name] = {}
        for h in HALVES:
            probs = {q["id"]: rcs.windows_of(model, feats, at, q, int(cs["window"]["stride"]), device)["prob"] for q in seqs[h]}
            preds = decode(params, probs); m, t = rcs.evaluate(cfg, preds, inputs.halves[h]); row = summary(m)
            want = stored[key]["rows"][f"b3|{h}"]; diff = {f: (row[f], want[f]) for f in FLAT if row[f] != want[f]}
            out[name][h] = {"row": row, "tables": t, "preds": preds, "stored_setting": params, "differing_fields": diff}
        del model, feats; torch.cuda.empty_cache()
    return out


def reference_fact_asrf(ctx, paths, inputs, device) -> dict:
    """Original FACT and the paired ASRF from their frozen weights with official decoding; stored rows reproduced."""
    import scripts.train_fact_asrf as tfa
    ref = ctx.rv["evaluation"]["reference_inputs"]["fact_asrf"]
    for k in ("config", "summary"):
        if sha256_file(ctx.root / ref[k]["path"]) != ref[k]["sha256"]:
            raise SystemExit(f"FACT/ASRF reference {k} differs from its pin")
    fctx = tfa.Ctx(load_config(ctx.root / ref["config"]["path"]), paths); stored = json.loads((ctx.root / ref["summary"]["path"]).read_text())["systems"]
    out = {}
    for key, sha in ref["frozen"].items():
        arm, seed = key.split("_s"); seed = int(seed); path = ctx.root / "models_frozen" / "fact_asrf_paired_v1" / key / "final.pt"
        if sha256_file(path) != sha:
            raise SystemExit(f"{key}: frozen weights differ from the pin")
        plan, _ = tfa.load_plan(fctx, seed); head = tfa.make_head(fctx, arm, plan, device)
        model = head.build(); model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); model = model.to(device).eval()
        name = f"{'Original FACT' if arm == 'FACT' else 'paired ASRF'} s{seed}"; out[name] = {}
        torch.use_deterministic_algorithms(arm == "FACT")
        for h in HALVES:
            seqs = inputs.halves[h].sequences()
            if arm == "FACT":
                preds = {k: v["labels"] for k, v in ra.fact_outputs(head, model, seqs, device).items()}
            else:
                preds = {k: np.asarray(v["labels"], int) for k, v in head.decode(model, seqs, device).items()}
            m, t = rcs.evaluate(ctx.cfg, preds, inputs.halves[h]); row = summary(m); want = stored[key][h]["row"]
            out[name][h] = {"row": row, "tables": t, "preds": preds, "differing_fields": {f: (row[f], want[f]) for f in FLAT if row[f] != want[f]}}
        torch.use_deterministic_algorithms(True, warn_only=True)
    return out


# ----------------------------------------------------------------------------------------------- recognition and windows
def native_rows(W: dict, seqs_h: list[dict]) -> np.ndarray:
    return np.flatnonzero(np.array([seqs_h[i]["kind"] == "native" for i in W["seq"]]))


def mixed_rows(W: dict, seqs_h: list[dict]) -> np.ndarray:
    stream = np.array([seqs_h[i]["kind"] != "native" for i in W["seq"]]); mixed = np.array([len(set(x.tolist())) >= 2 for x in W["labels16"]])
    return np.flatnonzero(stream & mixed)


def e1_at_centres(W: dict, seqs_h: list[dict], inputs, h: str) -> tuple[np.ndarray, int]:
    """E1's probabilities at every window's requested sample, and the number of windows whose target frame differs from
    the frame E1's cache used for that sample (must be 0)."""
    ev = inputs.halves[h]; prob = {s.seq_id: (inputs.store.get(s.seq_id, "orig").prob, inputs.store.get(s.seq_id, "orig").frame_idx) for s in ev.natives}
    prob.update({s.recipe.recipe_id: (s.prob, None) for s in (*ev.pairs, *ev.multi)})
    out = np.empty((len(W["seq"]), 5)); bad = 0
    for i, (qi, c) in enumerate(zip(W["seq"], W["centre"])):
        q = seqs_h[qi]; p, fidx = prob[q["id"]]; out[i] = p[c]
        if fidx is not None and int(fidx[c]) != int(q["refs"][c][1]):
            bad += 1
    return out, bad


def recognition_block(models: dict, W: dict, seqs_h: list[dict], e1: np.ndarray, rv: dict) -> tuple[dict, dict]:
    """Per system metrics on native windows and the within-seed and E1 comparisons (session-paired bootstrap)."""
    rows = native_rows(W, seqs_h); y = W["label"][rows]; ses = np.array([session_of(seqs_h[i]["id"]) for i in W["seq"][rows]])
    sysd = {k: {"pred": v["logits"][rows].argmax(1), "margin": v["logits"][rows][:, rt.A5C] - v["logits"][rows][:, rt.A4C]} for k, v in models.items()}
    lp = np.log(np.clip(e1[rows], 1e-12, None)); sysd["E1 (same requested samples)"] = {"pred": e1[rows].argmax(1), "margin": lp[:, rt.A5C] - lp[:, rt.A4C]}
    metrics = {k: rt.recognition(v["pred"], v["margin"], y) for k, v in sysd.items()}; comp = {}
    for seed in rv["training"]["seeds"]:
        for name, (a, b) in WITHIN.items():
            comp[f"s{seed}|{name}"] = rt.window_bootstrap(sysd[tr.run_key(a, seed)], sysd[tr.run_key(b, seed)], y, ses)
        for g in tm.GROUPS:
            comp[f"s{seed}|{g} - E1"] = rt.window_bootstrap(sysd[tr.run_key(g, seed)], sysd["E1 (same requested samples)"], y, ses)
    return metrics, comp


def window_tables(models: dict, W: dict, seqs_h: list[dict], rv: dict) -> dict:
    """Mixed windows (accuracy by target share, fixed/new with recordings and sessions) and sequence-end windows."""
    mr = mixed_rows(W, seqs_h); y = W["label"][mr]; share = (W["labels16"][mr] == y[:, None]).sum(1)
    rec = np.array([seqs_h[q]["refs"][c][0] for q, c in zip(W["seq"][mr], W["centre"][mr])]); ses = np.array([session_of(v) for v in rec])
    correct = {k: v["logits"][mr].argmax(1) == y for k, v in models.items()}; allc = {k: v["logits"].argmax(1) == W["label"] for k, v in models.items()}
    out = {"mixed": {k: rt.mixed_table(c, share) for k, c in correct.items()}, "mixed_changes": {},
           "ends": {k: rt.end_table(c, W["marker"]) for k, c in allc.items()}}
    for seed in rv["training"]["seeds"]:
        for name, (a, b) in WITHIN.items():
            out["mixed_changes"][f"s{seed}|{name}"] = rt.changes(correct[tr.run_key(a, seed)], correct[tr.run_key(b, seed)], rec, ses)
    return out


# ----------------------------------------------------------------------------------------------- comparisons
def system_comparisons(rv: dict, systems: dict, inputs, tol: float) -> dict:
    out = {}
    for seed in rv["training"]["seeds"]:
        pairs = {n: (tr.run_key(a, seed), tr.run_key(b, seed)) for n, (a, b) in WITHIN.items()}
        pairs |= {n: (tr.run_key("Ordered context", seed), r.format(k=seed)) for n, r in REFERENCES.items()}
        for name, (ka, kb) in pairs.items():
            rows = ("argmax", "b3") if name in WITHIN else ("b3",)
            for row in rows:
                for h in HALVES:
                    A, B = systems[ka][row][h], pick(systems, kb, row)[h]
                    c = {"bootstrap": paired_bootstrap(A["tables"], B["tables"]), "native_changes": native_changes(A["tables"]["native"], B["tables"]["native"])}
                    if B.get("preds") is not None:
                        c["segment_changes"] = segment_changes(inputs.halves[h], A["preds"], B["preds"])
                        c["boundary_changes"] = boundary_changes(inputs.halves[h], A["preds"], B["preds"], tol)
                    out[f"s{seed}|{row}|{name}|{h}"] = c
    return out


def pick(systems: dict, key: str, row: str) -> dict:
    """The row of a system: the round's models have argmax and b3 rows; references have one row."""
    s = systems[key]
    return s[row] if row in s else s["row"]


def verdicts(rv: dict, rec: dict, comp: dict) -> dict:
    seeds = rv["training"]["seeds"]
    out = {"recognition": {n: {m: {h: rt.consistency([rec[h][f"s{s}|{n}"][m] for s in seeds]) for h in HALVES} for m in rt.REC_METRICS}
                           for n in (*WITHIN, *(f"{g} - E1" for g in tm.GROUPS))}, "system": {}}
    for row in ("argmax", "b3"):
        names = WITHIN if row == "argmax" else (*WITHIN, *REFERENCES)
        out["system"][row] = {n: {h: rt.consistency([comp[f"s{s}|{row}|{n}|{h}"]["bootstrap"]["multi segment F1@50"] for s in seeds]) for h in HALVES} for n in names}
    return out


def units(inputs, systems: dict, tol: float) -> dict:
    """Short segments, A4C/A5C middles, boundary offsets per system row with predictions."""
    out = {}
    for key, s in systems.items():
        rows = [("row", s["row"])] if "row" in s else [(r, s[r]) for r in ("argmax", "b3") if r in s]
        for row, by in rows:
            for h, x in by.items():
                if x.get("preds") is None:
                    continue
                ev = inputs.halves[h]; offs = []
                for st in ev.multi:
                    pred = np.flatnonzero(np.diff(np.asarray(x["preds"][st.recipe.recipe_id])) != 0) + 1
                    offs += rt.match_offsets(np.asarray(st.semantic_boundaries, int), pred, tol)
                out[f"{key}|{row}|{h}"] = {**ra.mechanism(ev, x["preds"], SHORT), "boundary_offsets": rt.offset_summary(offs)}
    return out


# ----------------------------------------------------------------------------------------------- output
def strip(systems: dict) -> dict:
    out = {}
    for k, s in systems.items():
        out[k] = {r: {h: x["row"] | ({"differing_fields": x["differing_fields"]} if "differing_fields" in x else {}) for h, x in s[r].items()}
                  for r in ("argmax", "b3", "row") if r in s}
        if "b3_choice" in s:
            out[k]["b3_choice"] = s["b3_choice"]
    return out


def write(res: dict, out_dir: Path, ids: set) -> None:
    flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "summary"); tables = rt.tables_md(res)
    if any(s in tables for s in ids):
        raise SystemExit("an id appears in the report tables")
    write_atomic(out_dir / "summary.json", json.dumps(flat, indent=1)); write_atomic(out_dir / "report_tables.md", tables)


def plot_curves(runs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2)); col = {"Target-only": "#8172b2", "Shuffled context": "#dd8452", "Ordered context": "#4c72b0"}
    for key, r in runs.items():
        ep = r["manifest"]["epochs"]; x = [e["updates"] for e in ep]; kw = {"color": col[r["group"]], "ls": ("-", "--", ":")[r["seed"]], "lw": 1.2, "label": key}
        ax[0].plot(x, [e["loss"] for e in ep], **kw); ax[1].plot(x, [e["train_accuracy"] for e in ep], **kw); ax[2].plot(x, [e["lr_encoder_last"] for e in ep], **kw)
    for a, t in zip(ax, ("training loss (mean per epoch)", "training-batch accuracy (fit, not generalisation)", "encoder learning rate (last update of each epoch)")):
        a.set_title(t, fontsize=10); a.set_xlabel("updates"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=3); fig.suptitle("Raw-video context round: final weights kept, nothing chosen from these curves", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_evaluate(ctx, workers: int) -> None:
    t0 = time.time(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); tr.determinism()
    rv, cfg, paths = ctx.rv, ctx.cfg, ctx.paths; out_dir = ctx.root / rv["outputs"]["results_dir"]; out_dir.mkdir(parents=True, exist_ok=True)
    runs = verify_runs(ctx); frozen = freeze(ctx, runs); print("nine runs verified and frozen", flush=True)
    train_ids = set(tr.allowlist(rv, ctx.root)); queue = json.loads((ctx.out / "queue_status.json").read_text())
    res = {"runs": {k: {x: r["manifest"][x] for x in ("identity", "group", "seed", "updates", "final_sha256", "manifest_sha256", "init_sha256", "parameters")}
                    | {"epochs": r["manifest"]["epochs"]} for k, r in runs.items()}, "frozen": frozen, "decision": "pending"}
    ids = train_ids | {session_of(v) for v in train_ids}; write(res, out_dir, ids); plot_curves(runs, out_dir / "training_curves.png")
    pin = rv["evaluation"]["reference_inputs"]["e1_feature_manifest"]
    if sha256_file(ctx.root / pin["path"]) != pin["sha256"]:
        raise SystemExit("the E1 feature manifest differs from its pin")
    inputs, seqs = rcs.validation(cfg, {"inputs": {"protocol": rv["source"]["protocol"], "feature_manifest": pin}}, paths)   # development data from here on
    dev_ids = {q["id"] for h in HALVES for q in seqs[h]} | {r[0] for h in HALVES for q in seqs[h] for r in q["refs"]}
    ids |= dev_ids | {session_of(v) for v in dev_ids}
    refs = sorted({r for h in HALVES for q in seqs[h] for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}
    from scripts.prepare_raw_video_context import frame_dirs
    from scripts.train_joint_visual_temporal import FramePaths
    fp = FramePaths(frame_dirs(paths, rv, {r[0] for r in refs}))          # only video_id and frames_dir of these recordings
    t1 = time.time(); frames = decode_frames(refs, fp, workers); t_dec = time.time() - t1; print(f"{len(refs)} frames decoded ({t_dec:.0f} s)", flush=True)
    stride = 5; W = {h: half_windows(seqs[h], at, stride) for h in HALVES}; systems, rec_models, cost = {}, {h: {} for h in HALVES}, {}
    for key, r in runs.items():
        model = load_model(ctx, r, device); probs = {}; secs = 0.0
        for h in HALVES:
            orders = slot_orders(r["group"], r["seed"], W[h], seqs[h]); logits, sec = model_logits(model, W[h], orders, frames, device); secs += sec
            probs[h] = per_sample(W[h], logits, seqs[h]); rec_models[h][key] = {"logits": logits}
            save_outputs(ctx.out / "evaluation" / h / f"{key}.npz", r["manifest"]["identity"], W[h], orders, logits, probs[h], seqs[h])
        systems[key] = system_rows(cfg, probs, inputs); n_win = sum(len(W[h]["seq"]) for h in HALVES); n_smp = sum(len(q["labels"]) for h in HALVES for q in seqs[h])
        cost[key] = {"windows": n_win, "ms_per_window": 1000 * secs / n_win, "ms_per_sample": 1000 * secs / n_smp}
        print(f"{key}: B3 {systems[key]['b3_choice']['status']} | sel {systems[key]['b3']['selection']['row']['segment_f1_50_multi']:.4f} "
              f"cal {systems[key]['b3']['calibration']['row']['segment_f1_50_multi']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    e1 = {h: e1_at_centres(W[h], seqs[h], inputs, h) for h in HALVES}
    if any(e1[h][1] for h in HALVES):
        raise SystemExit("a window's target frame differs from the frame E1's cache used")
    rec = {h: recognition_block(rec_models[h], W[h], seqs[h], e1[h][0], rv) for h in HALVES}
    res.update({"systems": strip(systems), "recognition": {k: {h: rec[h][0][k] for h in HALVES} for k in rec["selection"][0]},
                "recognition_comparisons": {f"{k}|{h}": v for h in HALVES for k, v in rec[h][1].items()},
                "windows": {h: window_tables(rec_models[h], W[h], seqs[h], rv) for h in HALVES},
                "window_counts": {h: {"windows": int(len(W[h]["seq"])), "native": int(native_rows(W[h], seqs[h]).size), "mixed": int(mixed_rows(W[h], seqs[h]).size)} for h in HALVES},
                "target_frame_mismatches": {h: e1[h][1] for h in HALVES}, "cost": {"inference": cost, "frame_decode_seconds": t_dec}, "seconds": time.time() - t0})
    write(res, out_dir, ids); print("round systems evaluated", flush=True)
    try:
        refs_cs, e1_preds, g1 = ra.reference_systems(cfg, {"inputs": {"protocol": rv["source"]["protocol"]}}, paths, inputs, seqs)
        for name, by in refs_cs.items():
            systems[name] = {"row": {h: {"row": by[h]["metrics"], "tables": by[h]["tables"], "preds": e1_preds[h] if name == "E1 + B3" else None} for h in HALVES}}
        systems |= {k: {"row": v} for k, v in reference_c(ctx, cfg, paths, inputs, seqs, frames, at, device).items()}
        systems |= {k: {"row": v} for k, v in reference_fact_asrf(ctx, paths, inputs, device).items()}
    except (SystemExit, Exception) as e:  # noqa: BLE001 - a reproduction failure is recorded with its cause; comparisons then stay open
        res.update({"references": {"not reproduced": f"{type(e).__name__}: {e}"}, "decision": "not computed: the references did not reproduce", "seconds": time.time() - t0})
        write(res, out_dir, ids); print(json.dumps(res["references"], indent=1)); return
    tr.determinism(); del frames
    tol = float(cfg["selection"]["boundary_tolerance_s"]) * float(cfg["protocol"]["hz"])
    comp = system_comparisons(rv, systems, inputs, tol); rec_comp = {h: rec[h][1] for h in HALVES}; ver = verdicts(rv, rec_comp, comp)
    goals = {g: rt.goal([{h: systems[tr.run_key(g, s)]["b3"][h]["row"] for h in HALVES} for s in rv["training"]["seeds"]],
                        [systems[tr.run_key(g, s)]["b3_choice"]["status"] == "feasible" for s in rv["training"]["seeds"]], rv["goal"]) for g in tm.GROUPS}
    res.update({"systems": strip(systems), "references_g1": g1, "system_comparisons": {k: {x: v[x] for x in v} for k, v in comp.items()}, "verdicts": ver,
                "goal": goals, "decision": rt.decide(rv["decision_rules"], ver["recognition"], ver["system"]["b3"], goals), "units": units(inputs, systems, tol),
                "cost": {**res["cost"], "training_hours": {k: v.get("hours") for k, v in queue["jobs"].items()}, "check_hours": queue.get("check_hours"),
                         "failed_attempt_hours": queue.get("failed_attempt_hours", 0.0), "evaluation_hours": (time.time() - t0) / 3600},
                "seconds": time.time() - t0})
    write(res, out_dir, ids); print(json.dumps(res["decision"], indent=1, default=str)); print(f"wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--evaluate", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); ctx = tr.Ctx(cfg, a.config)
    if not a.evaluate:
        raise SystemExit("nothing to do: pass --evaluate after the nine runs")
    run_evaluate(ctx, a.workers)


if __name__ == "__main__":
    main()
