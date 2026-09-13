#!/usr/bin/env python
"""Build demo/demo_page.html: a self-contained meeting page (ladder table, four-cell chart, risk-coverage
curves, sample cines) from the merged ladder metrics under runs/. Videos are referenced as samples/<file>."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from echo_routing.evaluate.report import collect_metrics  # noqa: E402

METHODS = {
    "b_file": ("B-file", "Per-file mean probability; whole-file accept or defer", "Operational comparator: what an archive curator can already do with known file boundaries."),
    "b0": ("B0", "ResNet-18 frame classifier, per-frame argmax", "The encoder alone. Establishes classification quality and raw flicker with no temporal processing."),
    "b1": ("B1", "Probability moving average + hysteresis", "Cheapest temporal baseline. A change is accepted only after it persists for 0.5 s."),
    "b2": ("B2", "Sticky HMM, Viterbi over posteriors", "Tests whether simple state persistence is enough."),
    "b3": ("B3", "JS divergence of left/right windows + persistent label change", "Parameter-free semantic-change detector; the essential comparator for any learned boundary head."),
    "b4": ("B4", "MS-TCN on frozen features", "Standard learned temporal-segmentation competitor (compact re-implementation)."),
    "b6": ("B6", "STFM official code (EV9V authors) via adapter", "Recent echo-specific video classifier, run unmodified at a pinned commit on its own nine-code task."),
}
SERIES = ["b0", "b1", "b2", "b3", "b4"]
CELLS = [("same_none", "same view · no edit", "false_split_rate", "false split"),
         ("same_edit", "same view · gamma edit", "false_split_rate", "false split"),
         ("diff_none", "different view · no edit", "missed_rate", "missed change"),
         ("diff_edit", "different view · gamma edit", "missed_rate", "missed change")]
SAMPLES = [("PLHLA", "PLAX", "PLHLA_2022-07-18_14-35-49_320x240_0.mp4"), ("PMASA", "PSAX", "PMASA_2022-07-27_14-17-08_001_320x240_3.mp4"),
           ("A4C", "A4C", "A4C_2022-06-17_09-22-19_320x240_1.mp4"), ("A5C", "A5C", "A5C_2022-05-24_14-30-39_320x240_12.mp4"),
           ("SC4C", "SC4C", "SC4C_2022-07-19_15-21-13_320x240_12.mp4"), ("PMPALA", "excluded from family5", "PMPALA_2022-07-27_14-17-08_001_320x240_12.mp4")]


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def esc(s):
    return html.escape(str(s))


def bar_chart(ms: dict[str, dict]) -> str:
    W, H = 640, 300; L, R, T, B = 44, 12, 16, 58
    pw, ph = W - L - R, H - T - B
    ymax = 0.15
    vals = {c[0]: [g(ms.get(m, {}), "constructed", "cells", c[0], c[2]) for m in SERIES] for c in CELLS}
    top = max([v for vs in vals.values() for v in vs if v is not None] + [0.01])
    ymax = 0.15 if top <= 0.15 else (0.25 if top <= 0.25 else 0.5 if top <= 0.5 else 1.0)
    ticks = [ymax * i / 3 for i in range(4)]
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-labelledby="fc-title"><title id="fc-title">Four-cell error rates per method</title>']
    for t in ticks:
        y = T + ph - t / ymax * ph
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{L-6}" y="{y+4:.1f}" class="tick" text-anchor="end">{t:.2f}</text>')
    gw = pw / 4; bw = min(18, (gw - 24) / len(SERIES) - 2)
    for gi, (cell, label, key, kind) in enumerate(CELLS):
        x0 = L + gi * gw + (gw - len(SERIES) * (bw + 2)) / 2
        for si, m in enumerate(SERIES):
            v = vals[cell][si]
            if v is None:
                continue
            h = max(0.0, v / ymax * ph); x = x0 + si * (bw + 2); y = T + ph - h
            r = 4 if h >= 4 else 0
            out.append(f'<rect class="bar s{si+1}" x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="{r}" '
                       f'data-tip="{METHODS[m][0]} · {esc(label)}: {kind} rate {v:.3f}"/>')
            out.append(f'<rect class="hit" x="{x-1:.1f}" y="{T}" width="{bw+2:.1f}" height="{ph}" data-tip="{METHODS[m][0]} · {esc(label)}: {kind} rate {v:.3f}"/>')
        xc = L + gi * gw + gw / 2
        l1, l2 = label.split(" · ")
        out.append(f'<text x="{xc:.1f}" y="{T+ph+18}" class="axis" text-anchor="middle">{esc(l1)}</text>'
                   f'<text x="{xc:.1f}" y="{T+ph+33}" class="axis" text-anchor="middle">{esc(l2)}</text>'
                   f'<text x="{xc:.1f}" y="{T+ph+50}" class="axis muted" text-anchor="middle">{kind}</text>')
    out.append(f'<line x1="{L}" x2="{W-R}" y1="{T+ph}" y2="{T+ph}" class="axisline"/></svg>')
    return "".join(out)


def curve_chart(ms: dict[str, dict]) -> str:
    W, H = 640, 300; L, R, T, B = 48, 12, 16, 44
    pw, ph = W - L - R, H - T - B
    curves = {m: [p for p in g(ms.get(m, {}), "constructed", "risk_coverage_curve", default=[]) if p.get("risk") is not None] for m in SERIES}
    top = max([p["risk"] for c in curves.values() for p in c] + [0.05])
    ymax = 0.05 if top <= 0.05 else (0.1 if top <= 0.1 else 0.25 if top <= 0.25 else 1.0)
    yt = [ymax * i / 5 for i in range(6)]
    def X(c): return L + c * pw
    def Y(r): return T + ph - min(r, ymax) / ymax * ph
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-labelledby="rc-title"><title id="rc-title">Risk versus coverage per method</title>']
    for t in yt:
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L-6}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">{t*100:.0f}%</text>')
    for c in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<text x="{X(c):.1f}" y="{T+ph+18}" class="tick" text-anchor="middle">{c:.2f}</text>')
    out.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(0.05):.1f}" y2="{Y(0.05):.1f}" class="target"/><text x="{W-R}" y="{Y(0.05)-5:.1f}" class="axis muted" text-anchor="end">5% contamination target</text>')
    out.append(f'<text x="{L + pw/2:.1f}" y="{H-6}" class="axis" text-anchor="middle">coverage (accepted duration ÷ all duration)</text>')
    for si, m in enumerate(SERIES):
        pts = sorted(curves[m], key=lambda p: p["coverage"])
        if not pts:
            continue
        d = " ".join(f'{"M" if i == 0 else "L"}{X(p["coverage"]):.1f},{Y(p["risk"]):.1f}' for i, p in enumerate(pts))
        out.append(f'<path class="line s{si+1}" d="{d}"/>')
        for p in pts:
            out.append(f'<circle class="hit" cx="{X(p["coverage"]):.1f}" cy="{Y(p["risk"]):.1f}" r="7" data-tip="{METHODS[m][0]} · τ {p["tau"]:.3f}: coverage {p["coverage"]:.3f}, risk {p["risk"]:.3f}"/>')
        r = g(ms[m], "constructed", "routing")
        if r and r.get("risk") is not None:
            out.append(f'<circle class="op s{si+1}" cx="{X(r["coverage"]):.1f}" cy="{Y(r["risk"]):.1f}" r="6" data-tip="{METHODS[m][0]} frozen 5% policy · τ {r["tau"]:.3f}: coverage {r["coverage"]:.3f}, achieved risk {r["risk"]:.3f}"/>')
    out.append(f'<line x1="{L}" x2="{W-R}" y1="{T+ph}" y2="{T+ph}" class="axisline"/></svg>')
    return "".join(out)


ROW_ORDER = ["b_file", "b0", "b1", "b2", "b3", "b4"]


def streams_section(streams_json: Path) -> str:
    if not streams_json.is_file():
        return '<p class="muted">No rendered streams found (run scripts/make_demo_streams.py on the server and pull runs/_demo).</p>'
    d = json.loads(streams_json.read_text()); classes = d["classes"]
    keys = '<div class="keys">' + "".join(f'<span><i class="sw" style="background:var(--s{i+1})"></i>{c}</span>' for i, c in enumerate(classes)) + \
           '<span><i class="sw hatch"></i>deferred (score below frozen τ)</span><span><i class="ln dash"></i>file join, never shown to the router</span><span><i class="ln"></i>true view change</span></div>'
    panels = []
    for st in d["streams"]:
        dur = st["duration_s"]
        def seg_html(ivs, truth=False):
            out = []
            for iv in ivs:
                view = iv.get("label", iv.get("view")); acc = True if truth else iv.get("accepted", True)
                l = iv["start_s"] / dur * 100; w = max(0.0, (iv["end_s"] - iv["start_s"]) / dur * 100)
                out.append(f'<span class="seg v{view}{"" if acc else " defer"}" style="left:{l:.2f}%;width:{w:.2f}%" title="{classes[view]} {iv["start_s"]:.1f}–{iv["end_s"]:.1f} s{"" if acc else " · deferred"}"></span>')
            return "".join(out)
        rows = [f'<span class="lab truth">truth</span><span class="badge off" data-row="truth">—</span><div class="strip">{seg_html(st["truth"], truth=True)}</div>']
        for m in ROW_ORDER:
            if m not in st["methods"]: continue
            rows.append(f'<span class="lab">{METHODS[m][0]}</span><span class="badge off" data-row="{m}">—</span><div class="strip">{seg_html(st["methods"][m]["intervals"])}</div>')
        marks = "".join(f'<i class="mark" style="left:{j/dur*100:.2f}%"></i>' for j in st["joins_s"] if j not in st["semantic_s"]) + \
                "".join(f'<i class="mark sem" style="left:{b/dur*100:.2f}%"></i>' for b in st["semantic_s"])
        frags = "".join(f'<span>{fr["start_s"]:.1f}–{fr["end_s"]:.1f} s · {fr["family"]} ({fr["raw_label"]}){" · gamma " + str(VARIANT_GAMMA_TXT.get(fr["variant"], fr["variant"])) if fr["variant"] != "orig" else ""} · {fr["video_id"]}</span>' for fr in st["fragments"])
        data = {"dur": dur, "rows": {"truth": [{"s": t["start_s"], "e": t["end_s"], "v": t["label"], "a": True} for t in st["truth"]],
                                     **{m: [{"s": i["start_s"], "e": i["end_s"], "v": i["view"], "a": i["accepted"]} for i in st["methods"][m]["intervals"]] for m in ROW_ORDER if m in st["methods"]}}}
        panels.append(f"""<div class="stream" data-dur="{dur:.3f}">
  <div><h3>{esc(st["title"])}</h3><video src="streams/{st["file"]}" controls loop muted playsinline preload="metadata"></video><div class="frags">{frags}</div></div>
  <div><div class="rows">{"".join(rows)}</div><div class="stripcol-marks" hidden></div></div>
  <script type="application/json" class="rows-data">{json.dumps(data)}</script>
</div>""".replace('<div class="rows">', f'<div class="rows"><div class="stripcol" style="grid-column:3;grid-row:1/-1;position:absolute;left:0;right:0;top:0;bottom:0;pointer-events:none"></div>'))
        # marks + playhead overlay spans the strip column: place an absolutely positioned layer inside .rows aligned to the strips
        panels[-1] = panels[-1].replace('<div class="stripcol" style="grid-column:3;grid-row:1/-1;position:absolute;left:0;right:0;top:0;bottom:0;pointer-events:none"></div>',
                                        f'<div class="overlay" style="position:absolute;top:0;bottom:0;left:0;width:0;pointer-events:none">{marks}<i class="playhead" style="left:0"></i></div>')
    intro = ('<p>Press play on any stream. Each row is one router reading the <em>same</em> frames: the badge shows the view it assigns at the playhead, the strip shows its whole timeline. '
             'Hatched = deferred. Click a strip to seek. Frozen thresholds (validation, 5% target): ' +
             " · ".join(f'{METHODS[m][0]} τ {d["taus"][m]:.3f}' for m in ROW_ORDER if d["taus"].get(m) is not None) + '.</p>')
    return intro + keys + '<div class="streams">' + "".join(panels) + '</div>'


VARIANT_GAMMA_TXT = {"gamma090": "0.90", "gamma110": "1.10"}


def legend() -> str:
    return '<div class="legend">' + "".join(f'<span><i class="sw" style="background:var(--m{i+1})"></i>{METHODS[m][0]} {esc(METHODS[m][1].split(",")[0])}</span>' for i, m in enumerate(SERIES)) + '<span><i class="sw op-key"></i>frozen 5% operating point</span></div>'


def ladder_table(ms: dict[str, dict]) -> str:
    head = ["method", "cine macro-F1", "balanced acc", "frame acc", "fragments / min", "boundary F1 @0.25 s", "@0.5 s", "@1.0 s",
            "false split same/none", "same/edit", "missed diff/none", "diff/edit", "τ @5%", "coverage @5%", "achieved risk"]
    rows = []
    for m in SERIES:
        if m not in ms: continue
        d = ms[m]; c = g(d, "constructed", "cells", default={}); b = g(d, "constructed", "boundary", default={})
        rows.append([f'<b>{METHODS[m][0]}</b> <span class="muted">{esc(METHODS[m][1])}</span>',
                     f(g(d, "native", "test", "cine", "macro_f1")), f(g(d, "native", "test", "cine", "balanced_accuracy")), f(g(d, "native", "test", "frame", "accuracy")),
                     f(g(d, "native", "test", "stability", "fragments_per_minute_mean"), 1),
                     f(g(b, "0.25", "f1")), f(g(b, "0.5", "f1")), f(g(b, "1.0", "f1")),
                     f(g(c, "same_none", "false_split_rate")), f(g(c, "same_edit", "false_split_rate")), f(g(c, "diff_none", "missed_rate")), f(g(c, "diff_edit", "missed_rate")),
                     f(g(d, "policy", "by_target", "0.05", "tau")), f(g(d, "constructed", "routing", "coverage")), f(g(d, "constructed", "routing", "risk"))])
    t = '<table><thead><tr>' + "".join(f'<th>{esc(h)}</th>' for h in head) + '</tr></thead><tbody>'
    t += "".join('<tr>' + "".join(f'<td>{v}</td>' for v in r) + '</tr>' for r in rows) + '</tbody></table>'
    return t


def bfile_table(bf: dict | None) -> str:
    if not bf: return '<p class="muted">B-file metrics not found.</p>'
    rows = []
    for t, v in g(bf, "policy", "by_target", default={}).items():
        rows.append([f"{float(t)*100:.0f}%", f(v.get("tau")), esc(v.get("status")), f(g(v, "validation", "coverage")), f(g(v, "validation", "risk")), f(g(v, "test", "coverage")), f(g(v, "test", "risk"))])
    head = ["target", "τ", "status", "val coverage", "val risk", "test coverage", "test risk"]
    t = '<table><thead><tr>' + "".join(f'<th>{h}</th>' for h in head) + '</tr></thead><tbody>'
    t += "".join('<tr>' + "".join(f'<td>{v}</td>' for v in r) + '</tr>' for r in rows) + '</tbody></table>'
    acc = g(bf, "native", "test", "cine", "accuracy"); maj = bf.get("clip_majority_test_accuracy")
    return f'<p>Test cine accuracy <b class="num">{f(acc)}</b> (mean probability) · <b class="num">{f(maj)}</b> (frame majority vote) · macro-F1 <b class="num">{f(g(bf,"native","test","cine","macro_f1"))}</b></p>' + t


CSS = r"""
:root{color-scheme:light;--bg:#f6f8f7;--surface:#ffffff;--ink:#17222a;--ink-2:#4e5d66;--muted:#7a8891;--line:#d7dedb;--line-2:#eaeeec;--accent:#0f7c78;--accent-ink:#0b5f5c;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--m1:#a9dbd6;--m2:#6fc2bb;--m3:#3aa39c;--m4:#1a7f79;--m5:#0c5652;--chip:#e6f2f1;--ok:#1a7f37;--warn:#b26a00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0f1517;--surface:#161e21;--ink:#e6ecea;--ink-2:#b4bfba;--muted:#8b979a;--line:#2a353a;--line-2:#1f292d;--accent:#3fb8b2;--accent-ink:#7fd6d1;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--m1:#c4ece8;--m2:#93d6d0;--m3:#5fbab3;--m4:#379a93;--m5:#1f7871;--chip:#173436;--ok:#4fbf6a;--warn:#e0a33a}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0f1517;--surface:#161e21;--ink:#e6ecea;--ink-2:#b4bfba;--muted:#8b979a;--line:#2a353a;--line-2:#1f292d;--accent:#3fb8b2;--accent-ink:#7fd6d1;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--m1:#c4ece8;--m2:#93d6d0;--m3:#5fbab3;--m4:#379a93;--m5:#1f7871;--chip:#173436;--ok:#4fbf6a;--warn:#e0a33a}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.55;margin:0;padding-block:32px 64px;padding-inline:clamp(16px,4vw,40px)}
.wrap{max-width:1120px;margin:0 auto;display:grid;gap:44px}
h1,h2,h3{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:clamp(28px,4vw,40px);font-weight:600;letter-spacing:-.01em}
h2{font-size:22px;font-weight:600}
p{margin:0;max-width:68ch}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent-ink);margin-bottom:8px}
header{display:grid;gap:12px;padding-bottom:24px;border-bottom:1px solid var(--line)}
header .lede{color:var(--ink-2);max-width:72ch}
.facts{display:flex;flex-wrap:wrap;gap:8px 20px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;color:var(--ink-2)}
.facts b{color:var(--ink);font-weight:500}
section{display:grid;gap:16px}
.methods{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.method{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 16px;display:grid;grid-template-columns:auto 1fr;gap:4px 12px;align-content:start}
.method .id{grid-row:span 3;font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500;font-size:13px;color:var(--accent-ink);background:var(--chip);border-radius:4px;padding:2px 8px;height:fit-content}
.method .name{font-weight:600}
.method .why{color:var(--ink-2);font-size:14px}
.method .status{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted)}
.method .status.ok{color:var(--ok)} .method .status.warn{color:var(--warn)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line-2);white-space:nowrap}
th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);background:var(--surface);position:sticky;top:0}
td:first-child,th:first-child{text-align:left}
tbody tr:last-child td{border-bottom:0}
td .muted{font-size:12.5px}
.muted{color:var(--muted)} .num{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}
figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 14px 10px;display:grid;gap:8px}
figcaption{font-weight:600;font-size:14px}
figcaption small{display:block;font-weight:400;color:var(--muted);font-size:12.5px;margin-top:2px}
.chart{width:100%;height:auto;display:block;font-family:"IBM Plex Mono",ui-monospace,monospace}
.chart .grid{stroke:var(--line-2);stroke-width:1} .chart .axisline{stroke:var(--line);stroke-width:1}
.chart .tick,.chart .axis{fill:var(--ink-2);font-size:11px} .chart .axis.muted{fill:var(--muted);font-size:10.5px}
.chart .target{stroke:var(--ink-2);stroke-width:1;stroke-dasharray:4 4}
.chart .bar{stroke:var(--surface);stroke-width:1} .chart .line{fill:none;stroke-width:2;stroke-linejoin:round}
.chart .op{stroke:var(--surface);stroke-width:2} .chart .hit{fill:transparent;cursor:crosshair}
.s1{fill:var(--m1);stroke:var(--m1)} .s2{fill:var(--m2);stroke:var(--m2)} .s3{fill:var(--m3);stroke:var(--m3)} .s4{fill:var(--m4);stroke:var(--m4)} .s5{fill:var(--m5);stroke:var(--m5)}
.chart .line.s1,.chart .line.s2,.chart .line.s3,.chart .line.s4,.chart .line.s5{fill:none}
.v0{--c:var(--s1)} .v1{--c:var(--s2)} .v2{--c:var(--s3)} .v3{--c:var(--s4)} .v4{--c:var(--s5)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.sw.op-key{border-radius:50%;background:transparent;border:2px solid var(--ink-2);width:10px;height:10px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font:12.5px/1.4 "IBM Plex Mono",ui-monospace,monospace;padding:6px 9px;border-radius:4px;max-width:320px;opacity:0;transform:translate(-50%,-120%);transition:opacity .08s}
#tip.on{opacity:1}
.streams{display:grid;gap:16px}
.samples{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.sample{background:#000;border-radius:6px;overflow:hidden;border:1px solid var(--line);display:grid}
.sample video{width:100%;aspect-ratio:4/3;display:block;background:#000}
.sample .cap{display:flex;justify-content:space-between;gap:8px;padding:8px 10px;background:var(--surface);font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--ink-2)}
.sample .cap b{color:var(--ink);font-weight:500}
.stream{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px;display:grid;grid-template-columns:minmax(220px,380px) 1fr;gap:14px 20px;align-items:start}
@media (max-width:760px){.stream{grid-template-columns:1fr}}
.stream h3{font-size:16px;margin:0 0 6px}
.stream video{width:100%;aspect-ratio:4/3;background:#000;border-radius:4px;display:block}
.stream .frags{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted);margin-top:8px;display:grid;gap:2px;overflow-wrap:anywhere}
.rows{display:grid;grid-template-columns:auto auto 1fr;gap:6px 10px;align-items:center;position:relative}
.rows .lab{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--ink-2);white-space:nowrap}
.rows .lab.truth{color:var(--ink);font-weight:500}
.badge{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;min-width:78px;text-align:center;padding:2px 7px;border-radius:4px;background:var(--c,var(--line-2));color:#fff;white-space:nowrap}
.badge.off{background:transparent;color:var(--muted);border:1px dashed var(--line)}
.badge.defer{background:transparent;color:var(--c,var(--muted));border:1.5px dashed var(--c,var(--line))}
.strip{position:relative;height:22px;background:var(--line-2);border-radius:3px;overflow:hidden}
.seg{position:absolute;top:0;bottom:0;background:var(--c);opacity:.95}
.seg.defer{background:repeating-linear-gradient(45deg,var(--c) 0 3px,transparent 3px 7px);opacity:.7}
.seg+.seg{border-left:2px solid var(--surface)}
.stripcol{position:relative}
.mark{position:absolute;top:0;bottom:0;width:0;border-left:1.5px dashed var(--muted);pointer-events:none}
.mark.sem{border-left:2px solid var(--ink)}
.playhead{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--accent);pointer-events:none;box-shadow:0 0 0 1px var(--surface)}
.strips{display:grid;gap:6px;cursor:pointer}
.keys{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--ink-2);align-items:center}
.keys .sw{border-radius:2px} .keys .hatch{background:repeating-linear-gradient(45deg,var(--ink-2) 0 3px,transparent 3px 6px)}
.keys .ln{display:inline-block;width:18px;border-top:2px solid var(--ink);vertical-align:middle} .keys .ln.dash{border-top:1.5px dashed var(--muted)}
.reads{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.read{border-left:3px solid var(--accent);padding:2px 0 2px 14px;display:grid;gap:4px}
.read b{font-weight:600}
.read p{color:var(--ink-2);font-size:14px}
ul{margin:0;padding-left:20px;display:grid;gap:6px;color:var(--ink-2);max-width:78ch}
footer{border-top:1px solid var(--line);padding-top:16px;font-size:13px;color:var(--muted);font-family:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-reduced-motion:reduce){#tip{transition:none}}
"""

JS = r"""
const tip=document.getElementById('tip');
document.querySelectorAll('.stream').forEach(panel=>{
  const v=panel.querySelector('video'), data=JSON.parse(panel.querySelector('.rows-data').textContent), dur=data.dur;
  const rows=panel.querySelector('.rows'), overlay=panel.querySelector('.overlay'), head=panel.querySelector('.playhead');
  const strips=[...panel.querySelectorAll('.strip')];
  function place(){const r=rows.getBoundingClientRect(), s0=strips[0].getBoundingClientRect(); overlay.style.left=(s0.left-r.left)+'px'; overlay.style.width=s0.width+'px';}
  const badges={}; panel.querySelectorAll('.badge').forEach(b=>badges[b.dataset.row]=b);
  const names=["PLAX","PSAX","A4C","A5C","SC4C"];
  function update(){const t=v.currentTime||0; head.style.left=(t/dur*100)+'%';
    for(const [row,ivs] of Object.entries(data.rows)){const b=badges[row]; if(!b) continue; const iv=ivs.find(i=>t>=i.s&&t<i.e)||ivs[ivs.length-1];
      if(!iv){b.textContent='—'; b.className='badge off'; continue;}
      b.textContent=names[iv.v]+(iv.a?'':' · defer'); b.className='badge v'+iv.v+(iv.a?'':' defer');}}
  let raf; function loop(){update(); if(!v.paused&&!v.ended) raf=requestAnimationFrame(loop);}
  v.addEventListener('play',loop); v.addEventListener('timeupdate',update); v.addEventListener('seeked',update);
  strips.forEach(st=>st.addEventListener('click',e=>{const r=st.getBoundingClientRect(); v.currentTime=Math.max(0,Math.min(dur,(e.clientX-r.left)/r.width*dur)); update();}));
  window.addEventListener('resize',place); place(); update();
});
document.querySelectorAll('[data-tip]').forEach(el=>{
  el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.left=e.clientX+'px';tip.style.top=(e.clientY-8)+'px';tip.classList.add('on');});
  el.addEventListener('mouseleave',()=>tip.classList.remove('on'));
  el.addEventListener('focus',()=>{const r=el.getBoundingClientRect();tip.textContent=el.dataset.tip;tip.style.left=(r.left+r.width/2)+'px';tip.style.top=(r.top-8)+'px';tip.classList.add('on');});
  el.addEventListener('blur',()=>tip.classList.remove('on'));
  el.setAttribute('tabindex','0');
});
"""


def build(runs_root: Path, out: Path) -> Path:
    runs_root = Path(runs_root)
    metrics, h = collect_metrics(runs_root)
    if not metrics:
        raise SystemExit("no ladder metrics under runs/")
    ms = {m["method_id"]: m for m in metrics}
    b0 = ms.get("b0", {})
    n_native = g(b0, "native", "test", "cine", "n"); n_streams = g(b0, "constructed", "test_n")
    status = {"b_file": ("run", "ok"), "b0": ("run", "ok"), "b1": ("run", "ok"), "b2": ("run", "ok"), "b3": ("run", "ok"),
              "b4": ("run · training bank not class-balanced", "warn"), "b6": ("run · adaptation, 9-class video task", "warn")}
    methods_html = "".join(
        f'<div class="method"><span class="id">{k if k != "b_file" else "B-file"}</span><span class="name">{esc(v[1])}</span>'
        f'<span class="why">{esc(v[2])}</span><span class="status {status[k][1]}">{esc(status[k][0])}</span></div>'
        for k, v in METHODS.items())
    samples_html = "".join(
        f'<div class="sample"><video src="samples/{fn}" controls loop muted playsinline preload="metadata"></video>'
        f'<div class="cap"><b>{code}</b><span>{esc(fam)}</span></div></div>' for code, fam, fn in SAMPLES)
    page = f"""<title>EV9V Routing Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow">Capstone · selective temporal routing of echocardiography video · public-data study</div>
  <h1>EV9V routing demo</h1>
  <p class="lede">Watch six routers read the same echocardiography frames, then compare them on the full test split. Can a router tell a real change of echocardiographic view from a same-view recording join or a brightness edit? Seven baselines from the proposal (§7.1), run on the public EV9V dataset with the five-family task (PLAX, PSAX, A4C, A5C, SC4C). Every threshold was chosen on the validation split and frozen before the test split was touched.</p>
  <div class="facts"><span>encoder <b>ResNet-18 · {h}</b></span><span>test <b>{n_native} native cines · {n_streams} constructed streams</b></span><span>routing rate <b>10 Hz</b></span><span>policy <b>5% contamination target, validation-selected</b></span><span>date <b>2026-09-13</b></span></div>
</header>

<section>
  <div><div class="eyebrow">1 · watch the routers work</div><h2>One video, every baseline, side by side</h2></div>
  {streams_section(runs_root / "_demo" / "streams" / "streams.json")}
</section>

<section>
  <div><div class="eyebrow">2 · what we compare</div><h2>The seven baselines</h2></div>
  <div class="methods">{methods_html}</div>
  <p class="muted">The proposed model (P0–P2: small TCN + semantic boundary head) is not started; the ladder exists so that it has something honest to beat.</p>
</section>

<section>
  <div><div class="eyebrow">3 · results</div><h2>Test-split ladder, frozen validation-selected 5% policy</h2></div>
  <div class="tablewrap">{ladder_table(ms)}</div>
  <p class="muted">Native columns: untouched single-view test cines (cine label = majority of per-sample labels; fragments per minute is a stability proxy, not annotated truth). Constructed columns: two-fragment streams in a 2×2 design; boundary F1 uses one-to-one matching within ±0.25 / 0.5 / 1.0 s. Coverage and achieved risk are measured on test streams at the τ chosen on validation.</p>
  <div>
    <h3 style="font-size:16px;margin-bottom:6px">B-file comparator (native files, known file boundaries)</h3>
    {bfile_table(ms.get("b_file"))}
  </div>
</section>

<section>
  <div><div class="eyebrow">4 · the two figures</div><h2>Four-cell errors and the risk–coverage trade-off</h2></div>
  {legend()}
  <div class="charts">
    <figure>{bar_chart(ms)}<figcaption>Four-cell design (test streams, ±0.5 s)<small>Left pair: how often a same-view join is wrongly split. Right pair: how often a genuine view change is missed. Both must be low at once.</small></figcaption></figure>
    <figure>{curve_chart(ms)}<figcaption>Risk versus coverage on test streams<small>Accepting more duration (→) admits more wrongly routed duration (↑). Rings mark the frozen 5% policy; the 5% target is not binding here, the 1% target is.</small></figcaption></figure>
  </div>
</section>

<section>
  <div><div class="eyebrow">5 · how to read it</div><h2>Three sentences for the meeting</h2></div>
  <div class="reads">
    <div class="read"><b>File-level routing is near ceiling on curated single-view files.</b><p>B-file reaches 97.5% cine accuracy. Temporal routing must earn its keep on mixed inputs, exactly what the proposal's week-2 gate asks.</p></div>
    <div class="read"><b>The simplest smoothing already solves two-fragment streams.</b><p>B1 lifts boundary F1 from 0.50 to 0.92 with almost no false splits, and B1 ≈ B2 ≈ B3. Any learned method must beat these on harder settings, not here.</p></div>
    <div class="read"><b>The 5% target is not the binding one.</b><p>Near-full acceptance still yields ~1% contamination. The 1% target separates methods (coverage drops to ~0.83) and should be reported alongside.</p></div>
  </div>
</section>

<section>
  <div><div class="eyebrow">6 · what the input looks like</div><h2>One test-split cine per raw view code</h2></div>
  <div class="samples">{samples_html}</div>
  <p class="muted">EV9V, CC-BY-4.0, 320×240 at 30 fps. The router never sees file names or displayed text. PMPALA is kept in the manifest but excluded from the five-family task until its conflicting code expansion is resolved.</p>
</section>

<section>
  <div><div class="eyebrow">7 · caveats</div><h2>What these numbers cannot say</h2></div>
  <ul>
    <li>Constructed streams are feature-space concatenations of saved cines, not probe sweeps; nothing here transfers to natural bedside transitions.</li>
    <li>Streams have two fragments; the 4–8 fragment, 10–45 s banks from the proposal are the next step.</li>
    <li>The encoder overfits after its first epoch (best validation checkpoint = epoch 0); a lower learning-rate run is pending.</li>
    <li>B4's training bank was not class-balanced, so minority classes are under-predicted; its row is a pipeline result, not a fair comparison yet.</li>
    <li>B6 (STFM) ran its own nine-code, video-level task with a 15-epoch cap: test accuracy 0.939, macro-F1 0.904. It is an adaptation, not an exact reproduction, and is not comparable to the five-family table.</li>
    <li>No patient identifiers are available; uncertainty can only be grouped by cine.</li>
  </ul>
</section>

<footer>echo-view-routing · runs merged by scripts/report.py · thresholds, seeds and recipes recorded in docs/EXPERIMENT_LEDGER.md</footer>
</div>
<div id="tip" role="status" aria-live="polite"></div>
<script>{JS}</script>
"""
    out.write_text(page)
    return out


if __name__ == "__main__":
    print(build(REPO / "runs", REPO / "demo" / "demo_page.html"))
