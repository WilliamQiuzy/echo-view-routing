#!/usr/bin/env python
"""Integration checks of the official FACT on the frozen E1 features (plan docs/superpowers/plans/2026-10-08-fact-integration.md;
configs/experiments/fact_integration_v1.yaml). Training data only; the development halves and test are never read.

  --data   (CPU) input identities; the training bank rebuilt and checked against the protocol; every bank stream's
           fragments, features, frame indices, labels, joins and source records against independently loaded cache
           files; every training clip; the seed-0 epoch-0 native windows; the collated and unpadded seed-0 epoch-0
           batches against their sequences; the official transcript parse against our segments; coverage counts; the
           token rule; the fixed 16-sequence overfit set.
  --model  (GPU) parameter counts; inference label isolation (initial and overfit model); each loss term's gradient
           path at initialisation; the 16-sequence overfit check; the token flow of one stream; measured training cost
           against ASRF on the same batches.
Writes data_checks.json, model_checks.json and overfit_curves.png to the results directory (counts only); sequence ids
stay in runs/fact_integration_v1/ on the server.

  python scripts/fact_integration.py --config configs/experiments/fact_integration_v1.yaml --data
  python scripts/fact_integration.py --config configs/experiments/fact_integration_v1.yaml --model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS for the isolation check

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_to_dict  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import segment_f1  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore, Seq, TrainData, collate, native_seq, stream_seq  # noqa: E402
from echo_routing.temporal.dev_heads import ASRFHead, asrf_weights, check_pinned  # noqa: E402
from echo_routing.temporal.dev_run import check_index, digest, load_json, sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import seed_everything  # noqa: E402
from echo_routing.temporal.fact_head import FACTHead, loss_parts, official, parameter_group, token_rule  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

NS = "fact_integration_v1"
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
A4C, A5C = CLASSES.index("A4C"), CLASSES.index("A5C")
TERMS = ("frame", "seg", "token", "f2a", "a2f", "smooth")


# ----------------------------------------------------------------------------------------------- training data
class TrainOnly:
    """Protocol, E1 feature store, the 1,500-stream training bank (digest checked) and the training clips. No
    validation feature is loaded; the cache index is read only to check its rows."""

    def __init__(self, cfg: dict, fi: dict, paths) -> None:
        root, inp = paths.repo_root, fi["inputs"]
        self.proto, self.proto_sha = load_json(root / inp["protocol"]["path"])
        self.fman, self.fman_sha = load_json(root / inp["e1_feature_manifest"]["path"])
        if self.proto_sha != inp["protocol"]["sha256"] or self.fman_sha != inp["e1_feature_manifest"]["sha256"]:
            raise SystemExit("an input sha256 differs from the config")
        self.key = self.fman["encoder"]["cache_key"]; self.variants = tuple(inp["variants"])
        if self.key != inp["e1_cache_key"] or self.fman["frame_order"] != inp["frame_order"] or int(self.fman["feature_dim"]) != int(inp["feature_dim"]):
            raise SystemExit("the E1 feature manifest differs from the config (cache key, frame order or dimension)")
        self.cache_root = paths.cache_root
        for v in self.variants:
            if sha256_file(variant_dir(paths.cache_root, self.key, v) / "index.csv") != self.fman["variants"][v]["index_sha256"]:
                raise SystemExit(f"{v}: index.csv differs from the feature manifest")
        idx = read_index(variant_dir(paths.cache_root, self.key, "orig")); check_index(idx, self.proto)
        self.labels = dict(zip(idx["video_id"], idx["label_index"].astype(int)))
        self.train_clips = tuple(self.proto["train"]["clips"])
        if not set(self.train_clips) <= set(idx.loc[idx["split"] == "train", "video_id"]):
            raise SystemExit("a protocol training clip is not a training row of the cache")
        bank = multi_bank(idx, "train", cfg); self.bank_digest = digest([recipe_to_dict(r) for r in bank.recipes])
        if self.bank_digest != self.proto["train"]["multi_bank"]["digest"]:
            raise SystemExit(f"training bank digest {self.bank_digest[:16]} != protocol")
        self.recipes = bank.recipes; self.store = FeatureStore(paths.cache_root, self.key)
        self.streams = tuple(stream_seq(render(r, self.store.get)) for r in self.recipes)
        self.train = TrainData(self.streams, tuple((v, self.labels[v]) for v in self.train_clips), self.store)

    def batches(self, tr: dict, seed: int, epoch: int) -> list[list[Seq]]:
        return self.train.epoch_batches(seed, epoch, int(tr["native_per_epoch"]), int(tr["native_window"]), int(tr["streams_per_batch"]),
                                        int(tr["natives_per_batch"]))

    def ids(self) -> set[str]:
        clips = set(self.labels)
        return clips | {session_of(v) for v in clips}


class RawCache:
    """Cache files read directly with numpy, independent of FeatureStore (features as stored, fp16)."""

    def __init__(self, cache_root: Path, key: str) -> None:
        self.cache_root, self.key, self._z = cache_root, key, {}

    def get(self, vid: str, variant: str) -> dict:
        if (vid, variant) not in self._z:
            with np.load(variant_dir(self.cache_root, self.key, variant) / f"{vid}.npz") as z:
                self._z[(vid, variant)] = {k: z[k] for k in ("frame_idx", "t_sec", "feat")}
        return self._z[(vid, variant)]


def runs_of(labels: np.ndarray) -> list[int]:
    return [s.label for s in labels_to_segments(labels)]


def categories(labels: np.ndarray, joins: np.ndarray) -> dict[str, bool]:
    r = runs_of(labels)
    return {"single_view": len(r) == 1, "same_view_join": any(labels[j] == labels[j - 1] for j in joins),
            "A4C_A5C_A4C": any(r[k:k + 3] == [A4C, A5C, A4C] for k in range(len(r) - 2)),
            "A5C_A4C_A5C": any(r[k:k + 3] == [A5C, A4C, A5C] for k in range(len(r) - 2)),
            "X_Y_X": any(r[k] == r[k + 2] != r[k + 1] for k in range(len(r) - 2)), "repeated_class": len(set(r)) < len(r)}


# ----------------------------------------------------------------------------------------------- --data
def check_streams(data: TrainOnly, raw: RawCache, mods: dict) -> dict:
    """Every bank stream against the cache files, its source records and the official transcript parse."""
    train, st, dts = set(data.train_clips), Counter(), []
    for r, s in zip(data.recipes, data.streams):
        pos, joins = 0, []
        for i, fr in enumerate(r.fragments):
            if fr.video_id not in train or int(fr.label) != data.labels[fr.video_id] or fr.variant not in data.variants:
                raise SystemExit(f"{r.recipe_id}: fragment {i} has a source outside the training clips, a wrong label or variant")
            z = raw.get(fr.video_id, fr.variant); n = z["frame_idx"].size; length = fr.end - fr.start
            if not 0 <= fr.start < fr.end <= n:
                raise SystemExit(f"{r.recipe_id}: fragment {i} lies outside its clip")
            fidx, ts = z["frame_idx"][fr.start:fr.end], z["t_sec"][fr.start:fr.end]
            if np.any(np.diff(fidx) <= 0) or np.any(np.diff(ts) <= 0):
                raise SystemExit(f"{r.recipe_id}: frame indices are not increasing")
            if not np.array_equal(s.feat[pos:pos + length], z["feat"][fr.start:fr.end]) or np.any(s.labels[pos:pos + length] != fr.label):
                raise SystemExit(f"{r.recipe_id}: features or labels differ from the cache rows of fragment {i}")
            if i > 0:
                joins.append(pos)
            pos += length; st["fragments"] += 1; st[f"fragments_{fr.variant}"] += 1; dts.append(np.diff(ts))
        if pos != s.labels.size or list(map(int, s.joins)) != joins:
            raise SystemExit(f"{r.recipe_id}: length or joins differ from the recipe")
        ours = labels_to_segments(s.labels); lab = torch.from_numpy(s.labels)
        for fn in (mods["basic"].torch_class_label_to_segment_label, mods["loss"].torch_class_label_to_segment_label):
            trans, seg_label = fn(lab)
            expect = np.concatenate([np.full(g.length, k) for k, g in enumerate(ours)])
            if trans.tolist() != [g.label for g in ours] or not np.array_equal(seg_label.numpy(), expect):
                raise SystemExit(f"{r.recipe_id}: the official transcript parse differs from our segments")
        st["streams"] += 1; st["samples"] += int(s.labels.size)
    d = np.concatenate(dts)
    return {**st, "sample_spacing_s": {"min": float(d.min()), "median": float(np.median(d)), "max": float(d.max())}}


def check_natives(data: TrainOnly, raw: RawCache) -> dict:
    st = Counter()
    for v in data.train_clips:
        fidx = [raw.get(v, var)["frame_idx"] for var in data.variants]
        if any(f.size == 0 or np.any(np.diff(f) <= 0) for f in fidx) or any(not np.array_equal(fidx[0], f) for f in fidx[1:]):
            raise SystemExit(f"{v}: frame indices are empty, not increasing or differ between variants")
        st["clips"] += 1; st[f"class_{CLASSES[data.labels[v]]}"] += 1; st["samples"] += int(fidx[0].size)
    st["sessions"] = len({session_of(v) for v in data.train_clips})
    return dict(st)


def check_batches(data: TrainOnly, raw: RawCache, tr: dict) -> dict:
    """The seed-0 epoch-0 plan: native windows against the cache and every collated batch unpadded against its sequences."""
    st, train = Counter(), set(data.train_clips)
    for b in data.batches(tr, 0, 0):
        for s in b:
            if s.is_native:
                vid, start = s.seq_id.rsplit("@", 1); start = int(start)
                rows = raw.get(vid, "orig")["feat"][start:start + s.labels.size]
                if vid not in train or not np.array_equal(s.feat, rows) or np.any(s.labels != data.labels[vid]):
                    raise SystemExit(f"{s.seq_id}: a native window differs from its training clip")
                st["native_windows"] += 1
            else:
                st["streams"] += 1
        seqs, labels = FACTHead.unpad(collate(b))
        for x, lab, s in zip(seqs, labels, b):
            if not np.array_equal(x.numpy(), s.feat.astype(np.float32)) or not np.array_equal(lab.numpy(), s.labels):
                raise SystemExit(f"{s.seq_id}: the unpadded batch differs from the sequence")
        st["batches"] += 1; st["padded_positions"] += int((collate(b)["valid"] == 0).sum())
    if st["batches"] != int(tr["updates_per_epoch"]):
        raise SystemExit(f"{st['batches']} batches, expected {tr['updates_per_epoch']}")
    return dict(st)


def overfit_set(data: TrainOnly, cats: dict[str, dict]) -> list[tuple[str, Seq]]:
    """The fixed rule of the config: five single-view native windows, then bank streams in bank order without repeats."""
    out = []
    for c in range(len(CLASSES)):
        vid = next(v for v in data.train_clips if data.labels[v] == c); vf = data.store.get(vid, "orig")
        out.append(("single_view_native", native_seq(vid, c, vf, 0, min(256, vf.n), f"{vid}@0")))
    longest = max(len(runs_of(s.labels)) for s in data.streams)
    rules = [("same_view_join", 3, lambda s: cats[s.seq_id]["same_view_join"]), ("A4C_A5C_A4C", 3, lambda s: cats[s.seq_id]["A4C_A5C_A4C"]),
             ("repeated_class", 3, lambda s: cats[s.seq_id]["repeated_class"]), ("longest_transcript", 2, lambda s: len(runs_of(s.labels)) == longest)]
    used = set()
    for kind, k, ok in rules:
        picked = [s for s in data.streams if s.seq_id not in used and ok(s)][:k]
        if len(picked) != k:
            raise SystemExit(f"only {len(picked)} bank streams of kind {kind}")
        used |= {s.seq_id for s in picked}; out += [(kind, s) for s in picked]
    return out


def run_data(cfg: dict, fi: dict, paths, out_dir: Path) -> None:
    t0 = time.time(); repo = paths.repo_root / fi["fact"]["repo"]; check_pinned(repo, fi["fact"]["commit"])
    data = TrainOnly(cfg, fi, paths); raw = RawCache(paths.cache_root, data.key); mods = official(repo)
    cats = {s.seq_id: categories(s.labels, s.joins) for s in data.streams}
    rule = token_rule([s.labels for s in data.streams], int(fi["training_data"]["native_per_epoch"]), float(fi["fact"]["token_ratio"]))
    lens = Counter(len(runs_of(s.labels)) for s in data.streams); seg_len = [g.length for s in data.streams for g in labels_to_segments(s.labels)]
    chosen = overfit_set(data, cats)
    res = {"inputs_sha256": {"protocol": data.proto_sha, "e1_feature_manifest": data.fman_sha}, "cache_key": data.key,
           "bank_digest": data.bank_digest, "streams": check_streams(data, raw, mods), "natives": check_natives(data, raw),
           "seed0_epoch0_plan": check_batches(data, raw, fi["training_data"]),
           "coverage": {"bank_streams": {k: sum(c[k] for c in cats.values()) for k in next(iter(cats.values()))},
                        "native_clips_single_view": len(data.train_clips), "transcript_lengths": dict(sorted(lens.items())),
                        "segment_length_samples": {q: float(np.percentile(seg_len, p)) for q, p in (("min", 0), ("q25", 25), ("median", 50), ("q75", 75), ("max", 100))}},
           "token_rule": rule.__dict__, "overfit_set": {"kinds": dict(Counter(k for k, _ in chosen)), "transcript_lengths": [len(runs_of(s.labels)) for _, s in chosen],
                                                       "samples": [int(s.labels.size) for _, s in chosen]},
           "seconds": time.time() - t0}
    assert_no_ids(res, data.ids(), "data_checks")
    server = paths.repo_root / fi["outputs"]["server_dir"]; server.mkdir(parents=True, exist_ok=True)
    (server / "data_details.json").write_text(json.dumps({"overfit_set": [{"kind": k, "seq_id": s.seq_id} for k, s in chosen], "token_rule": rule.__dict__}, indent=1))
    write_json(out_dir / "data_checks.json", res, data)
    print(json.dumps({k: res[k] for k in ("coverage", "token_rule", "overfit_set")}, indent=1))


# ----------------------------------------------------------------------------------------------- --model
def as_input(s: Seq, device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(s.feat.astype(np.float32)).to(device), torch.from_numpy(s.labels).to(device)


@torch.no_grad()
def isolation(head: FACTHead, model, seqs: list[Seq], device, seed: int) -> dict:
    """Predictions under five label sequences passed to the official forward, and the adapter's dummy-label decode."""
    model.eval(); rng = np.random.default_rng(seed); st = Counter(); worst = 0.0
    for s in seqs:
        x, ref = as_input(s, device)
        variants = {"reference": ref, "zeros": torch.zeros_like(ref), "random": torch.from_numpy(rng.integers(0, len(CLASSES), ref.numel())).to(device),
                    "time_reversed": ref.flip(0), "single_sample": ref[:1]}
        outs = {}
        for k, lab in variants.items():
            pred = np.asarray(model([x], [lab])[0]["pred"]); last = model.block_list[-1]
            outs[k] = (pred, last.frame_clogit.clone(), last.action_clogit.clone())
        base = outs["reference"]
        for k, (pred, fl, al) in outs.items():
            if k != "reference":
                st["comparisons"] += 1; st["changed_predictions"] += int(not np.array_equal(pred, base[0]))
                worst = max(worst, float((fl - base[1]).abs().max()), float((al - base[2]).abs().max()))
        st["adapter_decode_differs"] += int(not np.array_equal(head.predict(model, x)["labels"], base[0])); st["sequences"] += 1
    return {**st, "max_abs_logit_difference": worst}


def term_losses(head: FACTHead, model, seqs: list[Seq], device) -> dict[str, torch.Tensor]:
    """Each term's weighted contribution to the official total, averaged over blocks and videos as the official total
    is. The contributions of a video must add up to its official total."""
    acc: dict[str, list] = defaultdict(list)
    for s in seqs:
        x, lab = as_input(s, device); model._forward_one_video(x.unsqueeze(1)); p = loss_parts(head, model, lab); n = len(p["blocks"])
        contrib = {t: sum(b[t] * b["weights"][t] for b in p["blocks"] if t in b) / n for t in TERMS if any(t in b for b in p["blocks"])}
        if not torch.allclose(sum(contrib.values()), p["total"], rtol=1e-5, atol=1e-6):
            raise AssertionError(f"{s.seq_id}: the term contributions do not add up to the official total")
        for t, v in contrib.items():
            acc[t].append(v)
        acc["total"].append(p["total"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def gradient_paths(head: FACTHead, model, seqs: list[Seq], device, seed: int) -> dict:
    groups = sorted({parameter_group(n, model) for n, _ in model.named_parameters()}); out = {}
    for term in (*TERMS, "official_total"):
        seed_everything(seed); model.train(); model.zero_grad(set_to_none=True)
        if term == "official_total":
            xs, labs = zip(*(as_input(s, device) for s in seqs)); loss, _ = model(list(xs), list(labs), compute_loss=True)
        else:
            loss = term_losses(head, model, seqs, device)[term]
        loss.backward(); has = Counter(); zero = []
        for n, p in model.named_parameters():
            g = 0.0 if p.grad is None else float(p.grad.abs().max())
            has[parameter_group(n, model)] += int(g > 0 and np.isfinite(g))
            if term == "official_total" and not (g > 0 and np.isfinite(g)):
                zero.append(n)
        out[term] = {"loss": float(loss), "groups_with_gradient": [g for g in groups if has[g]], "groups_without_gradient": [g for g in groups if not has[g]]}
        if term == "official_total":
            out[term]["zero_gradient_parameters"] = zero
    model.zero_grad(set_to_none=True)
    return out


def zero_gradient_parameters(model, seqs: list[Seq], device, seed: int) -> list[str]:
    seed_everything(seed); model.train(); model.zero_grad(set_to_none=True)
    xs, labs = zip(*(as_input(s, device) for s in seqs)); loss, _ = model(list(xs), list(labs), compute_loss=True); loss.backward()
    zero = [n for n, p in model.named_parameters() if p.grad is None or not float(p.grad.abs().max()) > 0]
    model.zero_grad(set_to_none=True)
    return zero


@torch.no_grad()
def outcome(head: FACTHead, model, chosen: list[tuple[str, Seq]], device) -> dict:
    """Eval-mode predictions with dummy labels on the overfit set, and the official matching on its eval forward."""
    model.eval(); rows = []; null = head.num_classes
    for kind, s in chosen:
        x, lab = as_input(s, device); pr = head.predict(model, x); pred = pr["labels"]
        f1 = segment_f1(labels_to_segments(pred), labels_to_segments(s.labels), 0.5)["f1"]
        model._forward_one_video(x.unsqueeze(1)); p = loss_parts(head, model, lab); a, sg = (t.tolist() for t in p["match"])
        trans = model.mcriterion.transcript.tolist(); cls = model.block_list[-1].action_clogit[:, 0].argmax(-1).tolist()
        unmatched = [t for t in range(len(cls)) if t not in set(a)]
        rows.append({"kind": kind, "frame_accuracy": float((pred == s.labels).mean()), "segment_f1_50": f1,
                     "transcript_exact": runs_of(pred) == trans, "segments": len(trans), "tokens_used": pr["tokens_used"],
                     "match_one_to_one": len(set(a)) == len(a) == len(trans) and sorted(sg) == list(range(len(trans))),
                     "matched_token_class_correct": sum(int(cls[t] == trans[k]) for t, k in zip(a, sg)) / len(a),
                     "unmatched_tokens_null": sum(int(cls[t] == null) for t in unmatched) / max(len(unmatched), 1)})
    keys = ("frame_accuracy", "segment_f1_50", "matched_token_class_correct", "unmatched_tokens_null")
    return {"sequences": len(rows), **{f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys},
            **{f"{k}_min": float(np.min([r[k] for r in rows])) for k in keys},
            "transcript_exact": sum(r["transcript_exact"] for r in rows), "match_one_to_one": sum(r["match_one_to_one"] for r in rows),
            "tokens_used_equals_segments": sum(r["tokens_used"] == r["segments"] for r in rows), "rows": rows}


def overfit(head: FACTHead, chosen: list[tuple[str, Seq]], device, oc: dict) -> tuple[torch.nn.Module, dict]:
    seed_everything(int(oc["seed"])); model = head.build().to(device); seqs = [s for _, s in chosen]
    opt = torch.optim.AdamW(model.parameters(), lr=float(oc["optimizer"]["lr"]), weight_decay=float(oc["optimizer"]["weight_decay"]))
    batches, clip, every = [seqs[:8], seqs[8:16]], float(oc["optimizer"]["grad_clip"]), int(oc["log_every"])
    log, sums = [], defaultdict(float)
    for step in range(int(oc["updates"])):
        model.train(); loss, _ = head.loss(model, collate(batches[step % 2], device))
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)); opt.step()
        sums["train_loss"] += float(loss); sums["grad_norm"] += gn
        if (step + 1) % every == 0:
            model.eval()
            with torch.no_grad():
                terms = {f"eval_{k}": float(v) for k, v in term_losses(head, model, seqs, device).items()}
            log.append({"update": step + 1, **{k: v / every for k, v in sums.items()}, **terms}); sums.clear()
            print(f"overfit {step + 1}: train {log[-1]['train_loss']:.4f} eval total {terms['eval_total']:.4f}", flush=True)
    return model, {"log": log, "outcome": outcome(head, model, chosen, device)}


@torch.no_grad()
def token_flow(head: FACTHead, model, s: Seq, device) -> dict:
    """Tensor shapes through the blocks of one stream, and the final frame-to-token assignment against the reference runs."""
    model.eval(); x, _ = as_input(s, device); outs = model._forward_one_video(x.unsqueeze(1)); rows = []
    for i, (b, (ff, af)) in enumerate(zip(model.block_list, outs)):
        r = {"block": i, "type": type(b).__name__, "frame_features": list(ff.shape), "token_features": list(af.shape),
             "frame_class_logits": list(b.frame_clogit.shape), "token_class_logits": list(b.action_clogit.shape)}
        for name, attr in (("frame_to_token_attention", "a2f_attn"), ("token_to_frame_attention", "f2a_attn")):
            if hasattr(b, attr):
                r[name] = list(getattr(b, attr).shape)
        if hasattr(b, "tdu"):
            r["predicted_segments"] = int(b.tdu.num_seg)
        rows.append(r)
    last = model.block_list[-1]; cls = last.action_clogit[:, 0].argmax(-1); active = torch.where(cls != head.num_classes)[0]
    runs = []
    if active.numel():
        assign = active[last.a2f_attn[0][:, active].argmax(-1)].tolist()   # the official _eval's frame-to-token choice
        for g in labels_to_segments(np.asarray(assign)):
            runs.append({"token": g.label, "start": g.start, "end": g.end, "token_class": CLASSES[int(cls[g.label])]})
    return {"input": [int(x.shape[0]), 1, int(x.shape[1])], "tokens": int(head.cfg.FACT.ntoken), "blocks": rows, "token_runs": runs,
            "reference_runs": [{"class": CLASSES[g.label], "start": g.start, "end": g.end} for g in labels_to_segments(s.labels)]}


def timed(head, model, batches: list, device, warm: int, opt_cfg: dict) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"]))
    for i, b in enumerate(batches):
        if i == warm:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device); t0 = time.time()
        model.train(); loss, _ = head.loss(model, collate(b, device)); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip"])); opt.step()
    torch.cuda.synchronize(); spu = (time.time() - t0) / (len(batches) - warm)
    return {"seconds_per_update": spu, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20}


def cost(heads: dict, data: TrainOnly, fi: dict, device) -> dict:
    cc, tr, oc = fi["checks"]["cost"], fi["training_data"], fi["checks"]["overfit"]["optimizer"]
    warm, n = int(cc["warmup_updates"]), int(cc["timed_updates"]); batches = data.batches(tr, 0, 0)[:warm + n]
    updates = 50 * int(tr["updates_per_epoch"]); out = {}
    for name, head in heads.items():
        seed_everything(0); model = head.build().to(device); r = timed(head, model, batches, device, warm, oc)
        torch.cuda.synchronize(); t0 = time.time(); head.decode(model, list(data.streams[:200]), device); torch.cuda.synchronize()
        r["decode_seconds_per_stream"] = (time.time() - t0) / 200
        r["parameters"] = int(sum(p.numel() for p in model.parameters())); r["hours_per_18750_updates"] = r["seconds_per_update"] * updates / 3600
        out[name] = r; del model; torch.cuda.empty_cache()
        print(f"cost {name}: {r['seconds_per_update']:.3f} s/update, {r['peak_gpu_memory_mb']:.0f} MB, {r['parameters']} parameters", flush=True)
    return out


def plot_overfit(log: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["update"] for r in log]; fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(x, [r["train_loss"] for r in log], label="training loss (train mode)"); ax[0].plot(x, [r["eval_total"] for r in log], label="official total (eval mode)")
    for t in TERMS:
        if f"eval_{t}" in log[0]:
            ax[1].plot(x, [r[f"eval_{t}"] for r in log], label=t)
    for a in ax:
        a.set_yscale("log"); a.set_xlabel("update (batch of 8)"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    ax[0].set_title("FACT on the 16 fixed training sequences", fontsize=10); ax[1].set_title("each term's contribution to the official total (eval mode)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130)


def run_model(cfg: dict, fi: dict, paths, out_dir: Path) -> None:
    t0 = time.time(); root = paths.repo_root
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    data = TrainOnly(cfg, fi, paths); details = json.loads((root / fi["outputs"]["server_dir"] / "data_details.json").read_text())
    rule = token_rule([s.labels for s in data.streams], int(fi["training_data"]["native_per_epoch"]), float(fi["fact"]["token_ratio"]))
    if rule.__dict__ != details["token_rule"]:
        raise SystemExit("the token rule differs from the data step")
    fa = fi["fact"]; head = FACTHead(root / fa["repo"], int(fi["inputs"]["feature_dim"]), rule, base_config=fa["base_config"])
    if head.commit != fa["commit"]:
        raise SystemExit(f"FACT commit {head.commit} != {fa['commit']}")
    by_id = {s.seq_id: s for s in data.streams}; chosen = []
    for row in details["overfit_set"]:
        if row["kind"] == "single_view_native":
            vid = row["seq_id"].rsplit("@", 1)[0]; vf = data.store.get(vid, "orig")
            chosen.append((row["kind"], native_seq(vid, data.labels[vid], vf, 0, min(256, vf.n), row["seq_id"])))
        else:
            chosen.append((row["kind"], by_id[row["seq_id"]]))
    iso_seqs = [s for _, s in chosen] + list(data.streams[:200]); oc = fi["checks"]["overfit"]
    seed_everything(int(oc["seed"])); init = head.build().to(device)
    res = {"fact_commit": head.commit, "config": {"base_config": fa["base_config"], "ntoken": int(head.cfg.FACT.ntoken), "null_weight": float(head.cfg.Loss.nullw),
                                                  "average_transcript_len": float(head.cfg.average_transcript_len), "block": head.cfg.FACT.block,
                                                  "trans": bool(head.cfg.FACT.trans), "match": head.cfg.Loss.match, "Bi": dict(head.cfg.Bi), "Loss": dict(head.cfg.Loss),
                                                  "TM": dict(head.cfg.TM), "FACT": dict(head.cfg.FACT)},
           "parameters": {"total": int(sum(p.numel() for p in init.parameters()))}}
    pg = Counter()
    for n, p in init.named_parameters():
        pg[parameter_group(n, init)] += p.numel()
    res["parameters"]["by_group"] = dict(sorted(pg.items()))
    res["isolation_initial"] = isolation(head, init, iso_seqs, device, seed=1)
    res["gradient_paths_initial"] = gradient_paths(head, init, [s for _, s in chosen[:8]], device, int(oc["seed"]))
    res["token_flow_initial"] = token_flow(head, init, next(s for k, s in chosen if k == "A4C_A5C_A4C"), device); del init
    model, ov = overfit(head, chosen, device, oc); res["overfit"] = {"log": ov["log"], "outcome": {k: v for k, v in ov["outcome"].items() if k != "rows"},
                                                                     "rows": [{k: v for k, v in r.items()} for r in ov["outcome"]["rows"]]}
    res["isolation_overfit"] = isolation(head, model, iso_seqs, device, seed=2)
    res["zero_gradient_after_overfit"] = zero_gradient_parameters(model, [s for _, s in chosen[:8]], device, int(oc["seed"]))
    res["token_flow_overfit"] = token_flow(head, model, next(s for k, s in chosen if k == "A4C_A5C_A4C"), device); del model; torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)   # the cost runs under the planned training settings
    first = [s for b in data.batches(fi["training_data"], 0, 0) for s in b]; cw, pw = asrf_weights(first, num_classes=len(CLASSES))
    heads = {"fact_" + Path(fa["base_config"]).stem: head,
             "fact_" + Path(fa["alternative_base_config"]).stem: FACTHead(root / fa["repo"], int(fi["inputs"]["feature_dim"]), rule, base_config=fa["alternative_base_config"]),
             "asrf": ASRFHead(root / "third_party" / "asrf", int(fi["inputs"]["feature_dim"]), cw, pw, device=device)}
    res["cost"] = cost(heads, data, fi, device); res["seconds"] = time.time() - t0
    assert_no_ids(res, data.ids(), "model_checks"); out_dir.mkdir(parents=True, exist_ok=True); plot_overfit(ov["log"], out_dir / "overfit_curves.png")
    write_json(out_dir / "model_checks.json", res, data)
    print(json.dumps({k: res[k] for k in ("isolation_initial", "isolation_overfit")} | {"outcome": res["overfit"]["outcome"]}, indent=1, default=str))


def write_json(path: Path, res: dict, data: TrainOnly) -> None:
    text = json.dumps(res, indent=1, default=str)
    if any(sess in text for sess in {session_of(v) for v in data.labels}):
        raise SystemExit(f"{path.name}: a session id appears in the output")
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp"); tmp.write_text(text); tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); step = ap.add_mutually_exclusive_group(required=True)
    step.add_argument("--data", action="store_true"); step.add_argument("--model", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); fi = cfg["fact_integration"]; paths = load_paths()
    out_dir = paths.repo_root / fi["outputs"]["results_dir"]
    (run_data if a.data else run_model)(cfg, fi, paths, out_dir)


if __name__ == "__main__":
    main()
