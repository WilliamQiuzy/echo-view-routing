#!/usr/bin/env python
"""Diagnostic for the FACT smoke check of the paired FACT-ASRF round (configs/experiments/fact_asrf_paired_v1.yaml):
does the overfit requirement fail through an implementation error or through too few updates on the 16 sequences?

The smoke's overfit is repeated with identical settings (GTEA FACT from the FACT_s0 initial state, Adam at the constant
smoke rate with the arm's betas, eps, weight decay and clip, official augmentation on, the 16 fixed training sequences
in two alternating batches of 8) and continued to --updates. The evaluation loss at update 1,000 must reproduce the
smoke's logged value. The outcome of the overfit check is recorded every 500 updates. Nothing here changes a setting of
the formal runs; training data only.

  python scripts/fact_overfit_extension.py --config configs/experiments/fact_asrf_paired_v1.yaml --updates 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.train_fact_asrf as tr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.fact_integration import outcome, term_losses  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.temporal.dev_data import collate  # noqa: E402
from echo_routing.temporal.dev_train import seed_everything  # noqa: E402

KEYS = ("transcript_exact", "match_one_to_one", "tokens_used_equals_segments", "segment_f1_50_min", "matched_token_class_correct_min",
        "matched_token_class_correct_mean", "frame_accuracy_mean")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--updates", type=int, default=3000)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); t0 = time.time()
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); ctx = tr.Ctx(cfg, paths); tr.determinism("FACT")
    fa = ctx.fa; o, sm = fa["arms"]["FACT"]["optimizer"], fa["smoke"]["fact_overfit"]; seed = int(fa["smoke"]["seed"])
    plan, _ = tr.load_plan(ctx, seed); head = tr.make_head(ctx, "FACT", plan, device); chosen = tr.monitor_set(ctx); seqs = [s for _, s in chosen]
    smoke = json.loads((ctx.server / "smoke" / f"FACT_s{seed}.json").read_text()); logged = {r["update"]: r["eval_total"] for r in smoke["overfit"]["log"]}
    seed_everything(seed); model = tr.new_model(head, ctx.init_path("FACT", seed), device)   # as tr.fact_overfit
    opt = torch.optim.Adam(model.parameters(), lr=float(sm["lr"]), betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))
    batches, clip, log, checks, sums = [seqs[:8], seqs[8:16]], float(o["grad_clip"]), [], {}, defaultdict(float)
    for step in range(a.updates):
        model.train(); loss, _ = head.loss(model, collate(batches[step % 2], device))
        opt.zero_grad(set_to_none=True); loss.backward(); sums["grad_norm"] += float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)); opt.step()
        sums["train_loss"] += float(loss)
        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                terms = {f"eval_{k}": float(v) for k, v in term_losses(head, model, seqs, device).items()}
            log.append({"update": step + 1, **{k: v / 50 for k, v in sums.items()}, **terms}); sums.clear()
        if (step + 1) % 500 == 0:
            out = outcome(head, model, chosen, device)
            checks[step + 1] = {**{k: out[k] for k in KEYS}, "failing_rows": [{x: r[x] for x in ("kind", "segments", "tokens_used", "matched_token_class_correct")}
                                                                             for r in out["rows"] if r["matched_token_class_correct"] < 1.0]}
            print(json.dumps({"update": step + 1, "eval_total": log[-1]["eval_total"], **{k: checks[step + 1][k] for k in KEYS}}), flush=True)
    res = {"updates": a.updates, "replay_at_1000": {"eval_total": next(r["eval_total"] for r in log if r["update"] == 1000), "smoke_logged": logged[1000]},
           "outcomes": checks, "log": log, "seconds": time.time() - t0}
    res["replay_at_1000"]["abs_difference"] = abs(res["replay_at_1000"]["eval_total"] - res["replay_at_1000"]["smoke_logged"])
    assert_no_ids(res, ctx.data.ids(), "overfit_extension"); text = json.dumps(res, indent=1)
    (ctx.server / "smoke" / "FACT_overfit_extension.json").write_text(text)
    (ctx.root / fa["outputs"]["results_dir"] / "fact_overfit_extension.json").write_text(text)
    print(json.dumps({"replay_at_1000": res["replay_at_1000"]}, indent=1))


if __name__ == "__main__":
    main()
