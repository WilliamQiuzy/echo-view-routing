#!/usr/bin/env python
"""A30 diagnostic (configs/experiments/post_a29_supervision_fit_diagnostic_v1.yaml), training data only.

  --stage audit    reference-level identity audit of every A15 large-fit request window and A27 T/P interval (CPU)
  --stage select   the fixed fit groups and bystanders per seed (CPU), manifests frozen with sha256
  --stage cpucheck a CPU fake case of the pooled objective and the fit loop
  --stage cache    prefix cache and pixel digests of the selected windows only (GPU), checked against A15's logits
  --stage fit      F and T, 1,000 full-group updates each, with fixed readouts (GPU)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # this process only; A27's CPU-only trainer would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.train_interval_recognition as t27  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_native_window_recognition as t29  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.interval_clips as ic  # noqa: E402
import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

A27_CONFIG = "configs/experiments/video_interval_recognition_v1.yaml"
A08 = {0: "9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc", 2: "f95fed7ba78cc0b5e2474c71a156356507b4588a4b324957fd5b53fe66ee8472"}
A27_INTERVALS = {0: "6409f64a4776e55312181676de6cddfa5f83eed35cce3bae41e5de62197819c9", 2: "5d31723194a780380e8db1c9b46d9e5781634bc988df53e6b82e030d6150b718"}
SEEDS, N_PICK, UPDATES, READ_AT = (0, 2), 16, 1000, (0, 100, 300, 1000)
LR_HEAD, LR_REP = 1e-3, 1e-4
now, write_json, ledger_append = vsh.now, vsh.write_json, vsh.ledger_append


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.d = cfg, config, cfg["post_a29_supervision_fit_diagnostic"]; o = self.d["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.d["budget"]["ledger"]
        self.a27 = t27.Ctx(load_cfg(SimpleNamespace(config=A27_CONFIG, set=[])), A27_CONFIG); self.t28 = SimpleNamespace(pin={"A08_weights": A08})

    def a08(self, s: int, device):
        return t28.Ctx.a08(self.t28, s, device)


def sha1(x) -> str:
    return hashlib.sha1(repr(x).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- identities
def windows_of(P: dict, fit: list[int]) -> dict:
    """Per global window row: input identity (ordered refs + marker), centre reference, target-sample label."""
    wid, cref, tlab = {}, {}, {}
    for i in fit:
        q = P["seqs"][i]; refs = [(str(r[0]), int(r[1]), round(float(r[2]), 6)) for r in q["refs"]]; at = {r: k for k, r in enumerate(sorted(set(refs)))}
        W = rr.half_windows([{"labels": q["labels"], "refs": refs}], at, 5); rows = np.flatnonzero(P["W"]["seq"] == i)
        if not (np.array_equal(P["W"]["centre"][rows], W["centre"]) and np.array_equal(P["W"]["marker"][rows], W["marker"])):
            raise SystemExit("rebuilt request windows differ from A15's")
        lab = np.asarray(q["labels"])
        for k, g in enumerate(rows):
            wid[int(g)] = sha1((tuple(refs[int(x)] for x in W["s"][k]), int(W["marker"][k]))); c = int(W["centre"][k]); cref[int(g)] = refs[c]; tlab[int(g)] = int(lab[c])
    return {"wid": wid, "cref": cref, "tlab": tlab}


def runs_of(ctx: Ctx, s: int) -> tuple[dict, list, list]:
    """A27's T and P lists rebuilt with (sequence, start, end, label[, base class]) and checked against A27's file."""
    D = t27.pool_data(ctx.a27, s); P = ctx.a27.pool(); fit = bc.members(P, "large"); T, Pl = [], []
    for i in fit:
        T += [(i, a, b, c, c) for a, b, c in ir.runs(D["labels"][i])]; Pl += [(i, a, b, ir.majority(D["labels"][i][a:b])[0], c) for a, b, c in ir.runs(D["base"][i])]
    f = ROOT / f"runs/video_interval_recognition_v1/data/intervals_s{s}.npz"
    if sha256_file(f) != A27_INTERVALS[s]:
        raise SystemExit("A27's interval file differs from its pin")
    with np.load(f) as z:
        if not (np.array_equal(z["T_labels"], [x[3] for x in T]) and np.array_equal(z["P_labels"], [x[3] for x in Pl]) and np.array_equal(z["T_lengths"], [x[2] - x[1] for x in T])
                and np.array_equal(z["P_lengths"], [x[2] - x[1] for x in Pl])):
            raise SystemExit("rebuilt lists differ from A27's")
    return D, T, Pl


def interval_key(r: tuple, nearest: list, wid: dict) -> str:
    i, a, b = r[0], r[1], r[2]; L = b - a; m = defaultdict(Fraction)
    for row, cnt in Counter(int(x) for x in nearest[i][a:b]).items():
        m[wid[row]] += Fraction(cnt, L)
    return sha1(tuple(sorted(m.items())))


def describe(r: tuple, q: dict, nearest: list, tlab: dict) -> dict:
    i, a, b, y, c = r; lab = np.asarray(q["labels"]); recs = {x[0] for x in q["refs"][a:b]}; frags = q.get("frags") or [] if q["kind"] != "native" else []
    pos, inside = 0, None
    for f in frags:
        n = int(f[4]) - int(f[3])
        if pos <= a and b <= pos + n:
            inside = True
        pos += n
    rows = Counter(int(x) for x in nearest[i][a:b]); off = sum(c_ for row, c_ in rows.items() if tlab[row] != y) / (b - a)
    return {"kind": q["kind"], "one_fragment": True if q["kind"] == "native" else bool(inside), "pure": bool(np.all(lab[a:b] == y)), "multi_recording": len(recs) > 1, "short": (b - a) <= 16,
            "off_label_target_weight": off, "recordings": sorted(recs), "sessions": sorted({session_of(v) for v in recs})}


def audit(ctx: Ctx) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "seeds": {}}
    try:
        P = ctx.a27.pool(); fit = bc.members(P, "large"); seqs = P["seqs"]; Wd = windows_of(P, fit); wid, cref, tlab = Wd["wid"], Wd["cref"], Wd["tlab"]
        by_c = defaultdict(set)
        for g, c in cref.items():
            by_c[c].add(wid[g])
        rec["windows"] = {"fit_windows": len(wid), "distinct_window_identities": len(set(wid.values())), "rows_sharing_an_identity": int(sum(v for v in Counter(wid.values()).values() if v > 1)),
                          "centre_refs": len(by_c), "centre_refs_with_several_identities": int(sum(1 for v in by_c.values() if len(v) > 1))}
        priv = {}
        for s in SEEDS:
            D, T, Pl = runs_of(ctx, s); near = D["nearest"]; items = [("T", k, r) for k, r in enumerate(T)] + [("P", k, r) for k, r in enumerate(Pl)]
            keys = [interval_key(r, near, wid) for _, _, r in items]; groups = defaultdict(list)
            for (lst, k, r), key in zip(items, keys):
                groups[key].append((lst, k, r))
            conflicts = {key: g for key, g in groups.items() if len({x[2][3] for x in g}) > 1}
            win_labels = defaultdict(set)
            for lst, k, r in items:
                for row in set(int(x) for x in near[r[0]][r[1]:r[2]]):
                    win_labels[wid[row]].add(r[3])
            tense = {w for w, ls in win_labels.items() if len(ls) > 1}; tense_runs = sum(1 for lst, k, r in items if any(wid[int(x)] in tense for x in set(near[r[0]][r[1]:r[2]])))
            conf_P = {k for g in conflicts.values() for lst, k, r in g if lst == "P"}; wrongP = [k for k, r in enumerate(Pl) if r[3] != r[4]]
            lb = sum(len(g) - max(Counter(x[2][3] for x in g).values()) for g in conflicts.values())
            ce_lb = sum(len(g) * -sum((c / len(g)) * math.log(c / len(g)) for c in Counter(x[2][3] for x in g).values()) for g in conflicts.values())
            crec = [{"size": len(g), "labels": dict(Counter(int(x[2][3]) for x in g)), "lists": dict(Counter(x[0] for x in g)), "kinds": dict(Counter(seqs[x[2][0]]["kind"] for x in g)),
                     "P_base_wrong": int(sum(1 for x in g if x[0] == "P" and x[2][3] != x[2][4])), "P_base_correct": int(sum(1 for x in g if x[0] == "P" and x[2][3] == x[2][4])),
                     "recordings": len({v for x in g for v in describe(x[2], seqs[x[2][0]], near, tlab)["recordings"]})} for g in conflicts.values()]
            desc = [describe(Pl[k], seqs[Pl[k][0]], near, tlab) for k in wrongP]
            cnt = lambda f: int(sum(1 for d in desc if f(d)))  # noqa: E731
            conc = {"base_wrong_P": len(wrongP), "native": cnt(lambda d: d["kind"] == "native"), "constructed_within_one_fragment": cnt(lambda d: d["kind"] != "native" and d["one_fragment"]),
                    "constructed_across_a_join": cnt(lambda d: d["kind"] != "native" and not d["one_fragment"]), "pure": cnt(lambda d: d["pure"]), "mixed": cnt(lambda d: not d["pure"]),
                    "single_recording": cnt(lambda d: not d["multi_recording"]), "multi_recording": cnt(lambda d: d["multi_recording"]), "short_le16": cnt(lambda d: d["short"]),
                    "with_off_label_target_windows": cnt(lambda d: d["off_label_target_weight"] > 0), "mean_off_label_target_weight": float(np.mean([d["off_label_target_weight"] for d in desc])),
                    "pure_with_off_label_target_windows": cnt(lambda d: d["pure"] and d["off_label_target_weight"] > 0),
                    "recordings": len({v for d in desc for v in d["recordings"]}), "sessions": len({x for d in desc for x in d["sessions"]}),
                    "true_class": dict(Counter(int(Pl[k][3]) for k in wrongP)), "base_class": dict(Counter(int(Pl[k][4]) for k in wrongP)),
                    "pairs_true_to_base": dict(Counter(f"{Pl[k][3]}->{Pl[k][4]}" for k in wrongP))}
            nat_rec = {seqs[i]["refs"][0][0]: i for i in fit if seqs[i]["kind"] == "native"}; nat_right = {v: bool(D["base"][i][0] == np.asarray(seqs[i]["labels"])[0] and len(set(D["base"][i].tolist())) == 1) for v, i in nat_rec.items()}
            cons_single = [d for d in desc if d["kind"] != "native" and not d["multi_recording"]]; with_nat = [d for d in cons_single if d["recordings"][0] in nat_right]
            conc["constructed_single_recording"] = len(cons_single); conc["of_which_recording_has_a_pool_native"] = len(with_nat)
            conc["of_which_that_native_is_Base_correct_whole"] = int(sum(1 for d in with_nat if nat_right[d["recordings"][0]]))
            rec["seeds"][str(s)] = {"intervals": {"T": len(T), "P": len(Pl)}, "distinct_interval_identities": len(groups),
                                    "direct_conflicts": {"groups": len(conflicts), "intervals": int(sum(len(g) for g in conflicts.values())), "P_base_wrong_in_conflicts": int(len(conf_P & set(wrongP))),
                                                         "unavoidable_error_lower_bound": int(lb), "ce_lower_bound_sum": float(ce_lb), "ce_lower_bound_mean_over_all_intervals": float(ce_lb / len(items)), "groups_detail": crec[:50]},
                                    "shared_window_tension": {"window_identities_with_several_labels": len(tense), "of_identities": len(win_labels), "intervals_touching_them": int(tense_runs), "of_intervals": len(items)},
                                    "base_wrong_concentration": conc}
            priv[str(s)] = {"conflict_P": sorted(conf_P)}
            print(json.dumps({k: v for k, v in rec["seeds"][str(s)].items() if k != "base_wrong_concentration"})[:1500], flush=True); print(json.dumps(conc), flush=True)
        (ctx.out / "audit_private.json").write_text(json.dumps(priv)); rec["private_sha256"] = sha256_file(ctx.out / "audit_private.json"); rec["finished"] = now()
        for p in (ctx.out / "audit.json", ctx.results / "audit.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "audit (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- selection
def length_bin(L: int) -> int:
    return 0 if L <= 16 else (1 if L <= 40 else 2)


def pick(cands: list[int], Pl: list, seqs: list, info: dict, n: int, used_s: set, used_r: set) -> list[int]:
    """Greedy over a fixed order: distinct sessions first, then distinct recordings, then anything not yet taken."""
    out = []
    for rule in ("session", "recording", "any"):
        for k in cands:
            if len(out) >= n:
                break
            if k in out:
                continue
            d = info[k]
            if rule == "session" and set(d["sessions"]) & used_s or rule == "recording" and set(d["recordings"]) & used_r:
                continue
            out.append(k); used_s |= set(d["sessions"]); used_r |= set(d["recordings"])
    return out


def match(wrong: list[int], cands: list[int], Pl: list, info: dict, used_s: set, used_r: set, taken: set) -> list[int]:
    out = []
    for w in wrong:
        y, b = Pl[w][3], length_bin(Pl[w][2] - Pl[w][1]); chosen = None
        for rule in (lambda k: Pl[k][3] == y and length_bin(Pl[k][2] - Pl[k][1]) == b and not set(info[k]["sessions"]) & used_s, lambda k: Pl[k][3] == y and not set(info[k]["sessions"]) & used_s,
                     lambda k: Pl[k][3] == y, lambda k: True):
            chosen = next((k for k in cands if k not in taken and rule(k)), None)
            if chosen is not None:
                break
        if chosen is None:
            break
        out.append(chosen); taken.add(chosen); used_s |= set(info[chosen]["sessions"]); used_r |= set(info[chosen]["recordings"])
    return out


def select(ctx: Ctx) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "seeds": {}}
    try:
        P = ctx.a27.pool(); fit = bc.members(P, "large"); seqs = P["seqs"]; Wd = windows_of(P, fit); priv = json.loads((ctx.out / "audit_private.json").read_text()); man = {}
        for s in SEEDS:
            D, T, Pl = runs_of(ctx, s); near = D["nearest"]; conf = set(priv[str(s)]["conflict_P"]); rng = np.random.default_rng([20261024, 30, s])
            info = {k: describe(r, seqs[r[0]], near, Wd["tlab"]) for k, r in enumerate(Pl)}
            wc = [k for k, r in enumerate(Pl) if r[3] != r[4] and k not in conf]; cc = [k for k, r in enumerate(Pl) if r[3] == r[4] and k not in conf]
            pw, pc = [int(x) for x in rng.permutation(wc)], [int(x) for x in rng.permutation(cc)]
            us, ur = set(), set(); fw = pick(pw, Pl, seqs, info, N_PICK, us, ur); taken = set(fw); fc = match(fw, pc, Pl, info, us, ur, taken)
            fitset = fw + fc; fit_w = {Wd["wid"][int(x)] for k in fitset for x in near[Pl[k][0]][Pl[k][1]:Pl[k][2]]}
            clean = lambda k: k not in taken and not ({Wd["wid"][int(x)] for x in near[Pl[k][0]][Pl[k][1]:Pl[k][2]]} & fit_w)  # noqa: E731
            us2, ur2 = set(), set(); bw = pick([k for k in pw if clean(k)], Pl, seqs, info, N_PICK, us2, ur2); taken |= set(bw); bcn = match(bw, [k for k in pc if clean(k)], Pl, info, us2, ur2, taken)
            def summ(ks):
                ds = [info[k] for k in ks]
                return {"n": len(ks), "classes": dict(Counter(int(Pl[k][3]) for k in ks)), "length_bins": dict(Counter(length_bin(Pl[k][2] - Pl[k][1]) for k in ks)), "pure": int(sum(d["pure"] for d in ds)),
                        "multi_recording": int(sum(d["multi_recording"] for d in ds)), "native": int(sum(d["kind"] == "native" for d in ds)), "recordings": len({v for d in ds for v in d["recordings"]}),
                        "sessions": len({x for d in ds for x in d["sessions"]})}
            sets = {"fit_wrong": fw, "fit_correct": fc, "bystander_wrong": bw, "bystander_correct": bcn}
            fr, fs = {v for k in fitset for v in info[k]["recordings"]}, {x for k in fitset for x in info[k]["sessions"]}; br, bs = {v for k in bw + bcn for v in info[k]["recordings"]}, {x for k in bw + bcn for x in info[k]["sessions"]}
            rec["seeds"][str(s)] = {"candidates": {"wrong": len(wc), "correct": len(cc), "wrong_excluded_by_conflicts": int(sum(1 for k, r in enumerate(Pl) if r[3] != r[4] and k in conf))},
                                    **{k: summ(v) for k, v in sets.items()}, "bystander_overlap": {"recordings_shared_with_fit": len(fr & br), "sessions_shared_with_fit": len(fs & bs), "window_identities_shared": 0}}
            man[str(s)] = {k: [[int(Pl[x][0]), int(Pl[x][1]), int(Pl[x][2]), int(Pl[x][3]), int(Pl[x][4]), int(x)] for x in v] for k, v in sets.items()}
            print(s, json.dumps(rec["seeds"][str(s)]), flush=True)
        (ctx.out / "subsets_private.json").write_text(json.dumps(man)); rec["subsets_sha256"] = sha256_file(ctx.out / "subsets_private.json"); rec["finished"] = now()
        for p in (ctx.out / "selection.json", ctx.results / "selection.json"):
            write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "select (CPU)", "cpu", started, time.time() - t0, ok, note)


def subsets(ctx: Ctx) -> dict:
    sel = json.loads((ctx.out / "selection.json").read_text())
    if sha256_file(ctx.out / "subsets_private.json") != sel["subsets_sha256"]:
        raise SystemExit("subset manifest differs from its record")
    return json.loads((ctx.out / "subsets_private.json").read_text())


def packed(groups: dict, nearest: list) -> dict:
    """Per group name: interval packing (rows, weights, offsets, labels, base classes) in A27's multiplicity convention."""
    out = {}
    for name, rs in groups.items():
        items = []
        for i, a, b, y, c, _k in rs:
            r, w = ir.interval_rows(nearest[i], a, b); items.append({"rows": r, "weights": w, "label": y, "length": b - a, "purity": 1.0})
        pk = ir.pack(items) if items else None
        if pk is not None:
            pk["base"] = np.asarray([x[4] for x in rs], np.int64)
        out[name] = pk
    return out


# ----------------------------------------------------------------------------------------------- CPU fake case
def cpucheck(ctx: Ctx) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        torch.manual_seed(0); emb = torch.randn(40, 512); head = torch.nn.Linear(512, 5); w = torch.full((40,), 0.25); ix = torch.arange(40) // 4; y = torch.randint(0, 5, (10,))
        z = torch.randn(9, 5, dtype=torch.float64); ww = torch.tensor([.5, .3, .2, 1., .25, .25, .25, .25, 1.], dtype=torch.float64); ii = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 3]); yy = torch.tensor([1, 4, 0, 2])
        p = torch.softmax(z, -1).numpy(); ref = [-np.log(sum(ww[k].item() * p[k, yy[j]] for k in range(9) if ii[k] == j)) for j in range(4)]
        opt = torch.optim.AdamW(head.parameters(), lr=LR_HEAD, weight_decay=0.0); first = None
        for u in range(300):
            loss = t29.pooled_ce(head(emb), w, ix, y).mean(); first = first if first is not None else float(loss); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        rec = {"pooled_ce_against_numpy_max_abs": float(np.abs(t29.pooled_ce(z, ww, ii, yy).numpy() - np.asarray(ref)).max()), "fake_fit_loss_first": first, "fake_fit_loss_after_300": float(loss),
               "ok": bool(float(loss) < 0.05 and float(np.abs(t29.pooled_ce(z, ww, ii, yy).numpy() - np.asarray(ref)).max()) < 1e-10), "time": now()}
        write_json(ctx.out / "cpucheck.json", rec); print(json.dumps(rec), flush=True)
        if not rec["ok"]:
            raise SystemExit("the CPU fake case failed")
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "cpu fake case (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- small prefix cache (GPU)
def cache(ctx: Ctx, device) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"seeds": {}}
    try:
        man = subsets(ctx); P = ctx.a27.pool(); fp = t28.frame_paths(P["seqs"])
        for s in SEEDS:
            D = t27.pool_data(ctx.a27, s); near = D["nearest"]; rows = sorted({int(x) for rs in man[str(s)].values() for i, a, b, *_ in rs for x in near[i][a:b]})
            m = ic.branch(ctx.a08(s, device), True); bys = defaultdict(list)
            for g in rows:
                bys[int(P["W"]["seq"][g])].append(g)
            toks, lg, digs = {}, {}, {}
            for qi, gs in bys.items():
                W, refs, first = t29.pool_windows(P, [qi]); frames = rr.decode_frames(refs, fp, 0); ks = [g - first for g in gs]
                with torch.no_grad():
                    x, thw = ic.prefix(m, tm.to_rgb(torch.from_numpy(frames[W["ref"][ks]]).to(device)), tm.marker_planes(torch.from_numpy(W["marker"][ks]).long().to(device)))
                    z = m.classifier(ic.suffix(m, x, thw))
                for j, g in enumerate(gs):
                    toks[g] = x[j].float().cpu().numpy(); lg[g] = z[j].float().cpu().numpy(); digs[g] = tm.window_digest([tm.frame_hash(frames[r]) for r in W["ref"][ks[j]]]).hex()
            err = max(float(np.abs(lg[g] - D["logits"][g]).max()) for g in rows); mism = int(sum(int(lg[g].argmax() != D["logits"][g].argmax()) for g in rows))
            Wd = windows_of(P, sorted({int(P["W"]["seq"][g]) for g in rows})); dup = defaultdict(set)
            for g in rows:
                dup[digs[g]].add(Wd["wid"][g])
            np.savez(ctx.out / f"prefix_small_s{s}.npz", rows=np.asarray(rows), tokens=np.stack([toks[g] for g in rows]), logits=np.stack([lg[g] for g in rows]))
            rec["seeds"][str(s)] = {"windows": len(rows), "logits_max_abs_against_A15": err, "argmax_mismatches": mism, "pixel_digests": len(set(digs.values())),
                                    "different_identities_with_identical_pixels": int(sum(1 for v in dup.values() if len(v) > 1)), "sha256": sha256_file(ctx.out / f"prefix_small_s{s}.npz")}
            print(s, json.dumps(rec["seeds"][str(s)]), flush=True)
            if err > 1e-4 or mism:
                raise SystemExit("the small prefix cache does not reproduce A15's logits")
        write_json(ctx.out / "cache.json", rec); write_json(ctx.results / "cache.json", rec); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "small prefix cache (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- fit (GPU)
def group_read(z_rows: torch.Tensor, pk: dict) -> dict:
    """CE, accuracy, true-class margin (mean / worst), fixed / broken against the Base class, per-run learned flags."""
    idx = np.arange(len(pk["labels"])); pp = t29.pooled_probs(z_rows, pk, idx); ce, margin, pred = t29.scores(pp, pk["labels"]); y = pk["labels"]; b = pk["base"]
    wrong = b != y; out = {}
    for name, mask in (("all", np.ones(len(y), bool)), ("base_wrong", wrong), ("base_correct", ~wrong)):
        if mask.any():
            out[name] = {"n": int(mask.sum()), "ce_mean": float(ce[mask].mean()), "ce_max": float(ce[mask].max()), "accuracy": float((pred[mask] == y[mask]).mean()),
                         "margin_mean": float(margin[mask].mean()), "margin_worst": float(margin[mask].min())}
    out["fixed"] = int((wrong & (pred == y)).sum()); out["broken"] = int((~wrong & (pred != y)).sum()); out["learned"] = [bool(p == t) for p, t in zip(pred, y)]
    return out


def fit(ctx: Ctx, device) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"seeds": {}}
    try:
        man = subsets(ctx); cj = json.loads((ctx.out / "cache.json").read_text())
        for s in SEEDS:
            if sha256_file(ctx.out / f"prefix_small_s{s}.npz") != cj["seeds"][str(s)]["sha256"]:
                raise SystemExit("small prefix cache differs from its record")
            D = t27.pool_data(ctx.a27, s); near = D["nearest"]; emb = D["emb"]
            with np.load(ctx.out / f"prefix_small_s{s}.npz") as z:
                crow, ctok = z["rows"], torch.from_numpy(z["tokens"])
            pos = {int(g): k for k, g in enumerate(crow)}; G = packed({"fit": man[str(s)]["fit_wrong"] + man[str(s)]["fit_correct"], "by": man[str(s)]["bystander_wrong"] + man[str(s)]["bystander_correct"]}, near)
            init = ctx.a27.init(s); out = {}
            for arm in ("F", "T"):
                a = time.time(); log = []
                if arm == "F":
                    model = ir.copied_head(*init).to(device); params = list(model.parameters()); opt = torch.optim.AdamW(params, lr=LR_HEAD, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0); rep0 = None
                    def logits(rows):
                        return model(torch.from_numpy(emb[rows]).to(device))
                else:
                    model = ic.branch(ctx.a08(s, device), True); pd0 = t28.prefix_digest(model); tr = ic.trainable(model); params = [p for _, p in tr]
                    rep = [p for n, p in tr if not n.startswith("classifier.")]; hd = [p for n, p in tr if n.startswith("classifier.")]
                    opt = torch.optim.AdamW([{"params": rep, "lr": LR_REP}, {"params": hd, "lr": LR_HEAD}], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0); rep0 = [p.detach().clone() for p in rep]
                    def logits(rows):
                        return t29.t_forward(model, ctok[[pos[int(g)] for g in rows]], device)
                if sorted(id(p) for g in opt.param_groups for p in g["params"]) != sorted(id(p) for p in params):
                    raise SystemExit(f"{arm}: optimizer binding")
                head0 = [p.detach().clone() for p in (model.parameters() if arm == "F" else hd)]; reads = {}
                rows_fit = np.unique(G["fit"]["rows"]); rows_by = np.unique(G["by"]["rows"]) if G["by"] is not None else None
                fw = torch.from_numpy(G["fit"]["weights"]).float().to(device); fix = torch.from_numpy(np.repeat(np.arange(len(G["fit"]["labels"])), np.diff(G["fit"]["offsets"]))).to(device)
                fy = torch.from_numpy(G["fit"]["labels"]).to(device); inv = torch.from_numpy(np.searchsorted(rows_fit, G["fit"]["rows"])).to(device)
                def dense(rows):
                    full = torch.zeros(int(max(rows.max(), rows_by.max() if rows_by is not None else 0)) + 1, 5)
                    with torch.no_grad():
                        full[rows] = logits(rows).float().cpu()
                    return full
                for u in range(UPDATES + 1):
                    if u in READ_AT:
                        moved_h = float(torch.sqrt(sum(((p.detach() - q) ** 2).sum() for p, q in zip(model.parameters() if arm == "F" else hd, head0))))
                        moved_r = float(torch.sqrt(sum(((p.detach() - q) ** 2).sum() for p, q in zip(rep, rep0)))) if rep0 is not None else 0.0
                        reads[u] = {"fit": group_read(dense(rows_fit), G["fit"]), "bystanders": group_read(dense(rows_by), G["by"]) if rows_by is not None else None,
                                    "head_moved_l2": moved_h, "representation_moved_l2": moved_r, "grad_norm_last": log[-1]["grad_norm"] if log else None}
                    if u == UPDATES:
                        break
                    loss = t29.pooled_ce(logits(rows_fit)[inv], fw, fix, fy).mean(); opt.zero_grad(set_to_none=True); loss.backward()
                    gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); nz = all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params) if u == 0 else None; opt.step()
                    log.append({"u": u, "loss": float(loss.detach()), "grad_norm": gn, **({"all_grads_nonzero": nz} if nz is not None else {})})
                    if (u + 1) % 100 == 0:
                        print(f"  [{arm}_s{s}] update {u + 1}/{UPDATES}: loss {log[-1]['loss']:.4f} grad {gn:.3f}", flush=True)
                fin = reads[UPDATES]["fit"]; f_ok = bool(fin["all"]["accuracy"] == 1.0 and fin["all"]["ce_mean"] <= 0.05)
                info = {"reads": reads, "updates": len(log), "first_grads_nonzero": log[0].get("all_grads_nonzero"), "loss_first": log[0]["loss"], "loss_last": log[-1]["loss"], "fitted": f_ok, "seconds": time.time() - a,
                        "trainable": int(sum(p.numel() for p in params))}
                if arm == "T":
                    info["frozen_prefix_unchanged"] = t28.prefix_digest(model) == pd0
                wd = ctx.out / "fit" / f"{arm}_s{s}"; wd.mkdir(parents=True, exist_ok=True)
                torch.save({n: p.detach().cpu() for n, p in (model.named_parameters() if arm == "F" else ic.trainable(model))}, wd / "final.pt"); (wd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
                info["final_sha256"] = sha256_file(wd / "final.pt"); out[arm] = info
                print(f"[{arm}_s{s}] fitted {f_ok}: acc {fin['all']['accuracy']:.3f} CE {fin['all']['ce_mean']:.4f} max {fin['all']['ce_max']:.3f}; wrong learned {fin['fixed']}/{fin.get('base_wrong', {}).get('n')} broken {fin['broken']}", flush=True)
            rec["seeds"][str(s)] = out
        rec["a08_files_unchanged"] = all(sha256_file(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt") == A08[s] for s in SEEDS); rec["finished"] = now()
        flat = json.loads(json.dumps(rec, default=str))
        for p in (ctx.out / "fit.json", ctx.results / "fit.json"):
            write_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "small-group fits F/T s0/s2 (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("audit", "select", "cpucheck", "cache", "fit"), required=True)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("audit", "select", "cpucheck"):
        return {"audit": audit, "select": select, "cpucheck": cpucheck}[a.stage](ctx)
    vsh.determinism(); device = torch.device("cuda")
    (cache if a.stage == "cache" else fit)(ctx, device)


if __name__ == "__main__":
    main()
