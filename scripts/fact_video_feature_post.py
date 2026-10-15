#!/usr/bin/env python
"""Post-training stages of the FACT video feature round (configs/experiments/fact_video_feature_v1.yaml), kept outside
the training identity so that a repair here never invalidates a finished run.

- freeze(): verify each finished run of the current identity and freeze its final weights with scripts/freeze_model.py
  into models_frozen/fact_video_feature_v1/FACT_s{k} (the queue calls it after the third run).
- --train-fit: after the freeze, the official decoding (all-zero labels) of all 1,500 training streams per seed: frame
  accuracy, segment F1@50, exact transcripts, the missed reference segments by length and class, and non-null tokens
  against reference segments. It locates under-fitting only; it never adds updates, budget or another configuration.

  python scripts/fact_video_feature_post.py --config configs/experiments/fact_video_feature_v1.yaml --train-fit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_video_feature_temporal as pv  # noqa: E402
import scripts.train_fact_asrf as tfa  # noqa: E402
import scripts.train_fact_video_feature as tf  # noqa: E402
import scripts.train_video_feature_temporal as tv  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.token_stability_attribution_tables import match_pairs  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.fact_head as fh  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

CLASSES = tf.CLASSES


def freeze(ctx, idents: dict, prep: dict) -> dict:
    out = {}
    for key, ident in idents.items():
        man_path = ctx.out / "train" / key / "run_manifest.json"; dest = ctx.root / "models_frozen" / tf.NS / key
        if tf.run_state(man_path, ident, ctx.total) != "done":
            raise SystemExit(f"{key}: not a finished run of this identity")
        man = json.loads(man_path.read_text())
        if not dest.exists():
            s = key.rsplit("_s", 1)[1]; files = [man["final_checkpoint"], str(man_path), str(ctx.root / prep["inits"][key]["path"]), str(ctx.root / ctx.config)]
            r = subprocess.run([sys.executable, "scripts/freeze_model.py", tf.NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={ident}",
                                "--meta", f"seed={s}", "--meta", f"updates={man['completed_updates']}",
                                "--notes", "FACT video feature round (2026-10-15): official GTEA FACT, final weights after 18,750 updates on frozen V01 Ordered features."],
                               cwd=ctx.root, capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"{key}: freeze_model.py failed ({(r.stderr or r.stdout)[-500:]}); a partly created frozen directory must be inspected by hand")
        if sha256_file(dest / "final.pt") != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen copy differs from the final weights")
        out[key] = {"path": str(dest.relative_to(ctx.root)), "final_sha256": man["final_sha256"], "updates": man["completed_updates"], "interrupted": man["interrupted"]}
    return out


def length_bin(n: int) -> str:
    return "<=16" if n <= 16 else "17-32" if n <= 32 else "33-64" if n <= 64 else ">64"


@torch.no_grad()
def stream_fit(head, model, streams: list, device) -> dict:
    """The official decoding of training streams (eval mode, all-zero labels): pooled segment F1@50 with the evaluator's
    matching, frame accuracy, exact transcripts, the missed reference segments by length and class, and the sign of
    non-null tokens minus reference segments."""
    model.eval(); tp = fp = fn = exact = correct = n = 0; missed = {"by_length": defaultdict(int), "by_class": defaultdict(int)}; tokens = defaultdict(int)
    for s in streams:
        pr = head.predict(model, torch.from_numpy(np.ascontiguousarray(s.feat, np.float32)).to(device)); pred = np.asarray(pr["labels"], int)
        ps, ref = labels_to_segments(pred), labels_to_segments(s.labels); pairs = match_pairs(ps, ref); hit = {j for _, j in pairs}
        tp += len(pairs); fp += len(ps) - len(pairs); fn += len(ref) - len(pairs); exact += int([g.label for g in ps] == [g.label for g in ref])
        correct += int((pred == s.labels).sum()); n += int(s.labels.size)
        tokens[{-1: "fewer", 0: "equal", 1: "more"}[int(np.sign(pr["tokens_used"] - len(ref)))]] += 1
        for k, r in enumerate(ref):
            if k not in hit:
                missed["by_length"][length_bin(r.length)] += 1; missed["by_class"][CLASSES[r.label]] += 1
    return {"streams": len(streams), "tp": tp, "fp": fp, "fn": fn, "segment_f1_50": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
            "frame_accuracy": correct / n if n else None, "exact_transcripts": exact, "missed": {k: dict(v) for k, v in missed.items()},
            "tokens_minus_segments": dict(tokens)}


def stage_train_fit(ctx) -> None:
    tfa.determinism(tf.ARM); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(pv.gpu_memory_fraction())
    prep = tf.read_record(ctx, "preparation.json"); a01 = ctx.a01(); head = tf.make_head(ctx, fh.TokenRule(**prep["token_rule"])); idents = tf.identities(ctx, head, prep)
    status = json.loads((ctx.out / "queue_status.json").read_text()); out = {}
    for s in ctx.seeds:
        key = tf.run_key(s); fz = ctx.root / "models_frozen" / tf.NS / key / "final.pt"
        if tf.run_state(ctx.out / "train" / key / "run_manifest.json", idents[key], ctx.total) != "done" or sha256_file(fz) != status["frozen"][key]["final_sha256"]:
            raise SystemExit(f"{key}: not a frozen run of this identity")
        model = head.build(); model.load_state_dict(torch.load(fz, map_location="cpu", weights_only=True), strict=True); model = model.to(device)
        data = tv.TrainSet(a01, s); out[key] = stream_fit(head, model, list(data.streams), device); del data, model; torch.cuda.empty_cache()
        print(f"{key}: training streams decoded {json.dumps({k: out[key][k] for k in ('segment_f1_50', 'frame_accuracy', 'exact_transcripts')})}", flush=True)
    rec = {"code": tf.code_digest(), "post_code_sha256": sha256_file(Path(__file__)), "config_sha256": ctx.config_sha(), "fit": out}; lib, _ = tf.library(ctx)
    assert_no_ids(rec, {q["id"] for q in lib} | {session_of(q["id"]) for q in lib if q["kind"] == "native"}, "train fit")
    for p in (ctx.out / "train_fit.json", ctx.results / "train_fit.json"):
        pv.write_atomic_json(p, rec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--train-fit", action="store_true")
    a = ap.parse_args(); a.set = []
    if not a.train_fit:
        raise SystemExit("pass --train-fit (the freeze runs at the end of the queue)")
    ctx = tf.Ctx(load_cfg(a), a.config); tf.run_stage(ctx, "train_fit", "gpu", stage_train_fit)


if __name__ == "__main__":
    main()
