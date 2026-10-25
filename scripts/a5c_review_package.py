#!/usr/bin/env python
"""A32 private review package (server only): per set S1-S4 up to 6 unique recordings, chosen by a fixed permutation.

Each case page shows the source recording (frame player over links to the original frames), the 16 slots of the target
run's central A08 window as A08 loads them (marker slot outlined), and, in closed layers, the stream timeline (reference,
A08 + B3, A31 F_E / T_E, both external rules) and the A08 / external A4C and A5C probability traces. Expert fields are
blank. Neutral case ids only; the key that maps them back stays in the package directory on the server.
Built by scripts/diagnose_a5c_evidence.py --stage package.
"""
from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import scripts.diagnose_a5c_evidence as d32
import scripts.report_raw_video_context as rr
import scripts.train_session_correction as t31
from echo_routing.config import load_paths
from echo_routing.evaluate.failure_units import session_of
from echo_routing.ingest.frames import list_frame_paths

SETS = ("S1", "S2", "S3", "S4")
PER_SET, SLOT = 6, 112
CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
COLOURS = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b", "#9c755f"]
TRACE = {"A08 + B3": "#222222", "EchoViewCLIP": "#1f77b4", "EchoPrime": "#d62728"}
EXPERT_FIELDS = ["case", "view_per_expert", "LVOT_visible", "image_quality", "label_agrees_with_view", "comments"]


def choose(cands: dict) -> dict:
    """Fixed permutation per set (one generator, sets in order); distinct sessions first, then the remaining recordings."""
    rng, out = np.random.default_rng([20261024, 32, 9]), {}
    for name in SETS:
        vids = sorted(cands[name]); perm = rng.permutation(len(vids)); chosen, used = [], set()
        for i in perm:
            if len(chosen) < PER_SET and session_of(vids[i]) not in used:
                chosen.append(vids[i]); used.add(session_of(vids[i]))
        for i in perm:
            if len(chosen) < PER_SET and vids[i] not in chosen:
                chosen.append(vids[i])
        out[name] = chosen
    return out


def bar(labels: np.ndarray, y0: int, h: int, scale: float) -> str:
    return "".join(f'<rect x="{a * scale:.1f}" y="{y0}" width="{(b - a) * scale:.1f}" height="{h}" fill="{COLOURS[c]}"><title>{CLASSES[c]} [{a}, {b})</title></rect>' for a, b, c in d32.ir.runs(np.asarray(labels, int)))


def timeline(rows: dict, src: list[str], a: int, b: int, n: int) -> str:
    width = 900; sc = width / n; h = 18; out = [f'<svg width="{width + 160}" height="{(len(rows) + 1) * (h + 6) + 30}" font-size="11">']
    for j, (name, lab) in enumerate(rows.items()):
        y = j * (h + 6) + 4; out.append(f'<text x="0" y="{y + 13}">{html.escape(name)}</text><g transform="translate(150,0)">{bar(lab, y, h, sc)}</g>')
    y = len(rows) * (h + 6) + 4; letters = {v: ("R" if k == 0 else f"o{k}") for k, v in enumerate(dict.fromkeys(src))}; s0 = 0
    out.append(f'<text x="0" y="{y + 13}">source</text>')
    for k in range(1, n + 1):
        if k == n or src[k] != src[s0]:
            out.append(f'<g transform="translate(150,0)"><rect x="{s0 * sc:.1f}" y="{y}" width="{(k - s0) * sc:.1f}" height="{h}" fill="{"#555" if letters[src[s0]] == "R" else "#bbb"}" stroke="#fff"/>'
                       f'<text x="{s0 * sc + 2:.1f}" y="{y + 13}" fill="#fff">{letters[src[s0]]}</text></g>'); s0 = k
    out.append(f'<g transform="translate(150,0)"><rect x="{a * sc:.1f}" y="0" width="{max(1.5, (b - a) * sc):.1f}" height="{y + h + 4}" fill="none" stroke="#000" stroke-width="2"/></g>')
    legend = "".join(f'<rect x="{150 + 90 * c}" y="{y + h + 10}" width="10" height="10" fill="{COLOURS[c]}"/><text x="{164 + 90 * c}" y="{y + h + 19}">{CLASSES[c]}</text>' for c in range(5))
    return "".join(out) + legend + "</svg>"


def traces(probs: dict, a: int, b: int, n: int) -> str:
    width, hgt = 900, 90; sc = width / max(1, n - 1); out = []
    for c in (2, 3):
        g = [f'<svg width="{width + 160}" height="{hgt + 24}" font-size="11"><text x="0" y="14">P({CLASSES[c]})</text><g transform="translate(150,4)">',
             f'<rect x="{a * sc:.1f}" y="0" width="{max(1.5, (b - a) * sc):.1f}" height="{hgt}" fill="#eee"/><rect x="0" y="0" width="{width}" height="{hgt}" fill="none" stroke="#999"/>']
        for name, p in probs.items():
            pts = " ".join(f"{i * sc:.1f},{hgt * (1 - float(p[i, c])):.1f}" for i in range(n)); g.append(f'<polyline points="{pts}" fill="none" stroke="{TRACE[name]}" stroke-width="1.3"><title>{name}</title></polyline>')
        out.append("".join(g) + "</g></svg>")
    key = " ".join(f'<span style="color:{TRACE[k]}">&#9632; {k}</span>' for k in probs)
    return f"<p>{key} (shaded: target run)</p>" + "".join(out)


def slots_png(refs: list, fp, sample_idx: np.ndarray, marker: int, a: int, b: int, path: Path) -> list[str]:
    clip = t31.t28.load_clip(refs, fp); im = Image.new("RGB", (8 * SLOT + 18, 2 * SLOT + 12), "white"); dr = ImageDraw.Draw(im); cap = []
    for j in range(16):
        x, y = 2 + (j % 8) * (SLOT + 2), 2 + (j // 8) * (SLOT + 8); im.paste(Image.fromarray(clip[j]).resize((SLOT, SLOT), Image.BILINEAR), (x, y))
        inside = a <= int(sample_idx[j]) < b
        if j == marker:
            dr.rectangle([x - 2, y - 2, x + SLOT + 1, y + SLOT + 1], outline="#ffd400", width=3)
        cap.append(f"slot {j + 1}: sample {int(sample_idx[j])}, {'inside' if inside else 'outside'} the run{', marker (window centre)' if j == marker else ''}")
    im.save(path); return cap


def player(frames: list[Path], case_dir: Path) -> str:
    fd = case_dir / "frames"; fd.mkdir(exist_ok=True); names = []
    for k, p in enumerate(frames):
        link = fd / f"{k:04d}{p.suffix.lower()}"
        if not link.exists():
            link.symlink_to(p.resolve())
        names.append(f"frames/{link.name}")
    return (f'<img id="v" src="{names[0]}" width="320"><br><button onclick="t=!t">play / pause</button> <span id="c"></span>'
            f'<script>const F={json.dumps(names)};let i=0,t=true;setInterval(()=>{{if(t){{i=(i+1)%F.length;document.getElementById("v").src=F[i];document.getElementById("c").textContent=(i+1)+"/"+F.length+" (30 fps source frames)";}}}},33);</script>')


def run_samples(pk: dict, seq_i: int, pred: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -1, int)
    for k in np.flatnonzero(pk["seq"] == seq_i):
        out[pk["a"][k]:pk["b"][k]] = pred[k]
    if (out < 0).any():
        raise SystemExit("the P runs do not tile the stream")
    return out


def build(ctx) -> str:
    out = ctx.out / "review_package"; out.mkdir(exist_ok=True); sp = np.load(ctx.out / "sets_private.npy", allow_pickle=True).item(); ro = np.load(ctx.out / "readout_private.npy", allow_pickle=True).item()
    Ds, cands, seeds_of = {}, {n: {} for n in SETS}, {n: {} for n in SETS}
    for s in d32.SEEDS:
        for h in d32.FOLDS:
            D = d32.fold_data(ctx, s, h); Ds[(s, h)] = D
            if D["key"] != sp[s][h]["key"]:
                raise SystemExit("runs differ from the sets stage")
            for name in SETS:
                for k in np.flatnonzero(sp[s][h]["masks"][name]):
                    v = max(D["src"][k], key=D["src"][k].get); cands[name].setdefault(v, (s, h, int(k))); seeds_of[name].setdefault(v, set()).add(s)
    chosen = choose(cands); root = load_paths().ev9v_images; key, rows_csv, index, ext = {}, [], [], {}; probs08 = {}
    for name in SETS:
        index.append(f"<h2>{name}</h2><ul>")
        for j, v in enumerate(chosen[name], 1):
            cid = f"{name}-{j:02d}"; s, h, k = cands[name][v]; D = Ds[(s, h)]; pk = D["pk"]; i = int(pk["seq"][k]); q = D["seqs"][i]; a, b = int(pk["a"][k]), int(pk["b"][k]); n = len(q["labels"])
            if h not in ext:
                ext[h] = d32.external_probs(ctx, t31.load_pool(ctx.a31, h), h)
            if (s, h) not in probs08:
                W = t31.pool_W(D["seqs"]); O = t31.outputs(ctx.a31, s, h); probs08[(s, h)] = (W, rr.per_sample(W, O["logits"], D["seqs"]))
            W, P08 = probs08[(s, h)]; cdir = out / cid; cdir.mkdir(exist_ok=True)
            o = pk["offsets"]; rws = pk["rows"][o[k]:o[k + 1]]; wt = pk["weights"][o[k]:o[k + 1]]; off = int(t31.outputs(ctx.a31, s, h)["offset"]); jw = int(rws[np.argmax(wt)] - off)
            if W["seq"][jw] != i:
                raise SystemExit("the central window belongs to another stream")
            fp = t31.t28.frame_paths([q]); cap = slots_png([q["refs"][x] for x in W["s"][jw]], fp, W["s"][jw], int(W["marker"][jw]), a, b, cdir / "slots.png")
            pr = ro[(s, h)]
            tl = {"reference": q["labels"], "A08 + B3": D["base_preds"][q["id"]], **{f"A31 {x}": run_samples(pk, i, D["arms"][x], n) for x in ("F_E", "T_E")},
                  **{f"{m} rule": run_samples(pk, i, pr[m], n) for m in d32.MODELS}}
            outs = "".join(f"<li>A31 {x}: {CLASSES[int(D['arms'][x][k])]}</li>" for x in d32.ARMS) + "".join(f"<li>{m} rule: {CLASSES[int(pr[m][k])]}</li>" for m in d32.MODELS)
            src = [r[0] for r in q["refs"]]; gam = sorted({float(r[2]) for r in q["refs"][a:b]}); nsrc = len(set(src[a:b]))
            pg = (f'<!doctype html><meta charset="utf-8"><title>{cid}</title><body style="font-family:sans-serif;max-width:1100px">'
                  f"<h1>Case {cid}</h1><p>Set {name}. Stream type: {'native recording' if q['kind'] == 'native' else 'constructed stream'}; target run samples [{a}, {b}) of {n} (10 Hz); "
                  f"{nsrc} source recording(s) inside the run; gamma {gam}; appears in this set under seed(s) {sorted(seeds_of[name][v])}.</p>"
                  f"<h3>Source recording R</h3>{player(list_frame_paths(root / v), cdir)}"
                  f'<h3>The 16 slots of the target run\'s central A08 window (as A08 loads them)</h3><img src="slots.png"><details><summary>slot positions</summary><ul>{"".join(f"<li>{c}</li>" for c in cap)}</ul></details>'
                  f"<h3>Expert fields (blank, not filled by any model)</h3><table border=1 cellpadding=4>{''.join(f'<tr><td>{x}</td><td style=min-width:300px></td></tr>' for x in EXPERT_FIELDS[1:])}</table>"
                  f"<details><summary>Layer 1: reference label and model outputs for the target run</summary><ul><li>reference (clip code): {CLASSES[int(pk['labels'][k])]}</li><li>A08 + B3: {CLASSES[int(pk['base'][k])]}</li>"
                  f"{outs}</ul></details>"
                  f"<details><summary>Layer 2: stream timeline</summary>{timeline(tl, src, a, b, n)}<p>R = this case's source recording; o1, o2 ... = other sources.</p></details>"
                  f"<details><summary>Layer 3: probability traces</summary>{traces({'A08 + B3': P08[q['id']], 'EchoViewCLIP': ext[h]['EchoViewCLIP'][q['id']], 'EchoPrime': ext[h]['EchoPrime'][q['id']]}, a, b, n)}</details></body>")
            (cdir / "index.html").write_text(pg); key[cid] = {"set": name, "seed": s, "heldout_fold": h, "stream": q["id"], "run": [a, b], "recording": v, "session": session_of(v)}
            rows_csv.append({"case": cid, **{x: "" for x in EXPERT_FIELDS[1:]}}); index.append(f'<li><a href="{cid}/index.html">{cid}</a></li>')
        index.append("</ul>")
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>A32 review package</title><body style="font-family:sans-serif"><h1>A32 A4C/A5C review package (private, server only)</h1>'
                                    "<p>No expert judgement has been recorded. Fields are blank; labels and model outputs sit in closed layers.</p>" + "".join(index) + "</body>")
    with open(out / "review_template.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=EXPERT_FIELDS); wr.writeheader(); wr.writerows(rows_csv)
    (out / "private_key.json").write_text(json.dumps(key, indent=1))
    rec = {"location": "server only: runs/a5c_source_external_evidence_v1/review_package (not copied, not committed)", "per_set": {n: {"candidate_recordings": len(cands[n]), "chosen": len(chosen[n]),
           "distinct_sessions": len({session_of(v) for v in chosen[n]}), "native_cases": int(sum(Ds[cands[n][v][:2]]["seqs"][int(Ds[cands[n][v][:2]]["pk"]["seq"][cands[n][v][2]])]["kind"] == "native" for v in chosen[n]))} for n in SETS},
           "cases": len(key), "expert_review": "none recorded (fields blank)", "files": int(sum(1 for _ in out.rglob("*"))), "kinds": dict(Counter(x["set"] for x in key.values()))}
    d32.publish(ctx, "package.json", rec); print(json.dumps(rec, indent=1)); return f"{len(key)} cases"
