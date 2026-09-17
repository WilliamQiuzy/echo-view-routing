#!/usr/bin/env python
"""Demo v2: one video per stream with the five real baselines side by side, plus a compact results table. Minimal text.
Reads runs/_demo/streams/streams.json (decoded by scripts/make_demo_streams.py) and the newest five-baseline ladder metrics."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from echo_routing.evaluate.report import collect_metrics  # noqa: E402

GH = "https://github.com/WilliamQiuzy/echo-view-routing/blob/main/"
BASELINES = {  # registry id -> (display name, one-line description, links)
    "b6": ("STFM", "EV9V authors' CNN-LSTM view classifier, sliding window", [("paper", "https://arxiv.org/abs/2606.17437"), ("official code", "https://github.com/bgx666/stfm"),
           ("our driver", GH + "scripts/run_stfm.py"), ("our sliding window", GH + "scripts/run_windowed_stfm.py"), ("frozen checkpoint", GH + "docs/models_frozen/stfm_official/seed100/MANIFEST.json")]),
    "b7": ("EchoViewCLIP", "CLIP-based view recognition (MICCAI 2025), retrained on EV9V, sliding window", [("paper", "https://link.springer.com/chapter/10.1007/978-3-032-05169-1_18"), ("official code", "https://github.com/xmed-lab/EchoViewCLIP"),
           ("our driver", GH + "scripts/run_echoviewclip.py"), ("our config", GH + "third_party/adapters/echoviewclip/ev9v_stage1.yaml"), ("our sliding window", GH + "scripts/run_windowed_echoviewclip.py"), ("frozen checkpoint", GH + "docs/models_frozen/echoviewclip_stage1/v1/MANIFEST.json")]),
    "b8": ("EchoPrime", "released 11-view frame classifier, per frame", [("paper", "https://arxiv.org/abs/2410.09704"), ("official code + weights", "https://github.com/echonet/EchoPrime"),
           ("our evaluation", GH + "scripts/eval_echoprime_views.py"), ("our per-frame runner", GH + "scripts/run_windowed_echoprime.py"), ("frozen weights", GH + "docs/models_frozen/echoprime_view_classifier/release_v1.0.0/MANIFEST.json")]),
    "b4": ("MS-TCN", "temporal segmentation on frozen frame features", [("paper", "https://arxiv.org/abs/1903.01945"), ("official code", "https://github.com/yabufarha/ms-tcn"),
           ("our driver", GH + "scripts/run_mstcn_official.py"), ("our patch (py3)", GH + "third_party/patches/ms-tcn-py3.patch"), ("frozen checkpoint", GH + "docs/models_frozen/mstcn_official/v1/MANIFEST.json")]),
    "b5": ("ASFormer", "transformer temporal segmentation on frozen frame features", [("paper", "https://arxiv.org/abs/2110.08568"), ("official code", "https://github.com/ChinaYi/ASFormer"),
           ("our driver", GH + "scripts/run_asformer_official.py"), ("our patch (py3)", GH + "third_party/patches/asformer-py3.patch"), ("frozen checkpoint", GH + "docs/models_frozen/asformer_official/v1/MANIFEST.json")]),
}
ROWS = ["b6", "b7", "b8", "b4", "b5"]
CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]


def esc(s):
    return html.escape(str(s))


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


CSS = r"""
:root{color-scheme:light;--bg:#f6f8f7;--surface:#fff;--ink:#17222a;--ink-2:#4e5d66;--muted:#7a8891;--line:#d7dedb;--line-2:#eaeeec;--accent:#0f7c78;--accent-ink:#0b5f5c;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0f1517;--surface:#161e21;--ink:#e6ecea;--ink-2:#b4bfba;--muted:#8b979a;--line:#2a353a;--line-2:#1f292d;--accent:#3fb8b2;--accent-ink:#7fd6d1;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0f1517;--surface:#161e21;--ink:#e6ecea;--ink-2:#b4bfba;--muted:#8b979a;--line:#2a353a;--line-2:#1f292d;--accent:#3fb8b2;--accent-ink:#7fd6d1;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.5;margin:0;padding-block:28px 56px;padding-inline:clamp(16px,4vw,40px)}
.wrap{max-width:1120px;margin:0 auto;display:grid;gap:28px}
section{display:grid;gap:12px;scroll-margin-top:16px}
h2{scroll-margin-top:16px}
h3{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;font-size:15px;font-weight:600;margin:6px 0 0;color:var(--ink-2)}
.side{position:fixed;left:12px;top:16px;width:170px;display:grid;gap:4px;font-size:12.5px;padding:10px;background:var(--surface);border:1px solid var(--line);border-radius:6px;z-index:5}
.side a{color:var(--ink-2);text-decoration:none;padding:2px 4px;border-radius:3px}.side a:hover{background:var(--line-2);color:var(--ink)}
.side input{width:100%;font:inherit;padding:5px 7px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--ink);margin-bottom:4px}
.navsep{height:1px;background:var(--line);margin:4px 0}
@media (max-width:1500px){.side{position:sticky;top:0;width:auto;max-width:1120px;margin:0 auto 16px;display:flex;flex-wrap:wrap;gap:4px 10px;align-items:center}.side input{width:180px;margin:0}.navsep{display:none}}
.stream.hide{display:none}
h1{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;font-size:clamp(26px,4vw,36px);font-weight:600;margin:0;letter-spacing:-.01em;text-wrap:balance}
h2{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;font-size:20px;font-weight:600;margin:0}
p{margin:0;color:var(--ink-2);max-width:72ch}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
.keys{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--ink-2);align-items:center}
.sw{width:12px;height:12px;border-radius:2px;display:inline-block;margin-right:6px;vertical-align:-1px}
.hatch{background:repeating-linear-gradient(45deg,var(--ink-2) 0 3px,transparent 3px 6px)}
.ln{display:inline-block;width:18px;border-top:2px solid var(--ink);vertical-align:middle;margin-right:6px}.ln.dash{border-top:1.5px dashed var(--muted)}
.streams{display:grid;gap:14px}
.stream{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px;display:grid;grid-template-columns:minmax(200px,340px) 1fr;gap:12px 18px;align-items:start}
@media (max-width:720px){.stream{grid-template-columns:1fr}}
.stream h3{font-size:15px;margin:0 0 6px;font-weight:600}
.stream video{width:100%;aspect-ratio:4/3;background:#000;border-radius:4px;display:block}
.frags{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;color:var(--muted);margin-top:6px;display:grid;gap:1px;overflow-wrap:anywhere}
.rows{display:grid;grid-template-columns:auto auto 1fr;gap:6px 10px;align-items:center;position:relative}
.lab{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--ink-2);white-space:nowrap}.lab.truth{color:var(--ink);font-weight:500}
.badge{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;min-width:78px;text-align:center;padding:2px 7px;border-radius:4px;background:var(--c,var(--line-2));color:#fff;white-space:nowrap}
.badge.off{background:transparent;color:var(--muted);border:1px dashed var(--line)}.badge.defer{background:transparent;color:var(--c,var(--muted));border:1.5px dashed var(--c,var(--line))}
.strip{position:relative;height:22px;background:var(--line-2);border-radius:3px;overflow:hidden;cursor:pointer}
.seg{position:absolute;top:0;bottom:0;background:var(--c);opacity:.95}.seg.defer{background:repeating-linear-gradient(45deg,var(--c) 0 3px,transparent 3px 7px);opacity:.7}.seg+.seg{border-left:2px solid var(--surface)}
.v0{--c:var(--s1)}.v1{--c:var(--s2)}.v2{--c:var(--s3)}.v3{--c:var(--s4)}.v4{--c:var(--s5)}
.overlay{position:absolute;top:0;bottom:0;pointer-events:none}.mark{position:absolute;top:0;bottom:0;width:0;border-left:1.5px dashed var(--muted)}.mark.sem{border-left:2px solid var(--ink)}
.playhead{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--accent);box-shadow:0 0 0 1px var(--surface)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--line-2);white-space:nowrap}
th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
td:first-child,th:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}
td a,.links a{color:var(--accent-ink)}
.links{font-size:13px;color:var(--ink-2);display:grid;gap:4px}
footer{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}
"""

JS = r"""
const filt=document.getElementById('filter');
if(filt){filt.addEventListener('input',()=>{const q=filt.value.trim().toLowerCase(); document.querySelectorAll('.stream').forEach(p=>{p.classList.toggle('hide', q && !p.querySelector('h3').textContent.toLowerCase().includes(q));});});}
document.querySelectorAll('.stream').forEach(panel=>{
  const v=panel.querySelector('video'), data=JSON.parse(panel.querySelector('.rows-data').textContent), dur=data.dur;
  const rows=panel.querySelector('.rows'), overlay=panel.querySelector('.overlay'), head=panel.querySelector('.playhead'), strips=[...panel.querySelectorAll('.strip')];
  function place(){const r=rows.getBoundingClientRect(), s0=strips[0].getBoundingClientRect(); overlay.style.left=(s0.left-r.left)+'px'; overlay.style.width=s0.width+'px';}
  const badges={}; panel.querySelectorAll('.badge').forEach(b=>badges[b.dataset.row]=b); const names=["PLAX","PSAX","A4C","A5C","SC4C"];
  function update(){const t=v.currentTime||0; head.style.left=(t/dur*100)+'%';
    for(const [row,ivs] of Object.entries(data.rows)){const b=badges[row]; if(!b) continue; const iv=ivs.find(i=>t>=i.s&&t<i.e)||ivs[ivs.length-1];
      if(!iv){b.textContent='—'; b.className='badge off'; continue;} b.textContent=names[iv.v]+(iv.a?'':' · defer'); b.className='badge v'+iv.v+(iv.a?'':' defer');}}
  let raf; function loop(){update(); if(!v.paused&&!v.ended) raf=requestAnimationFrame(loop);}
  v.addEventListener('play',loop); v.addEventListener('timeupdate',update); v.addEventListener('seeked',update);
  strips.forEach(st=>st.addEventListener('click',e=>{const r=st.getBoundingClientRect(); v.currentTime=Math.max(0,Math.min(dur,(e.clientX-r.left)/r.width*dur)); update();}));
  window.addEventListener('resize',place); place(); update();
});
"""


KINDS = [("native", "Single view"), ("same_none", "Same view, two recordings joined"), ("same_edit", "Same view, second half brightness-edited"),
         ("diff_none", "View change"), ("diff_edit", "View change, second half brightness-edited"), ("multi", "Multi-view streams")]


def streams_html(d: dict) -> str:
    groups = {k: [] for k, _ in KINDS}
    for st in d["streams"]:
        groups.setdefault(st["kind"], []).append(st)
    out = []
    for kind, title in KINDS:
        if groups.get(kind):
            out.append(f'<section id="{kind}"><h2>{esc(title)}</h2>{_panels(groups[kind])}</section>')
    return "".join(out)


def _panels(streams: list) -> str:
    panels = []
    for st in streams:
        dur = st["duration_s"]
        def segs(ivs, truth=False):
            out = []
            for iv in ivs:
                view = iv.get("label", iv.get("view")); acc = True if truth else iv.get("accepted", True)
                l = iv["start_s"] / dur * 100; w = max(0.0, (iv["end_s"] - iv["start_s"]) / dur * 100)
                out.append(f'<span class="seg v{view}{"" if acc else " defer"}" style="left:{l:.2f}%;width:{w:.2f}%"></span>')
            return "".join(out)
        rows = [f'<span class="lab truth">truth</span><span class="badge off" data-row="truth">—</span><div class="strip">{segs(st["truth"], True)}</div>']
        data_rows = {"truth": [{"s": t["start_s"], "e": t["end_s"], "v": t["label"], "a": True} for t in st["truth"]]}
        for m in ROWS:
            if m not in st["methods"]:
                continue
            rows.append(f'<span class="lab">{BASELINES[m][0]}</span><span class="badge off" data-row="{m}">—</span><div class="strip">{segs(st["methods"][m]["intervals"])}</div>')
            data_rows[m] = [{"s": i["start_s"], "e": i["end_s"], "v": i["view"], "a": i["accepted"]} for i in st["methods"][m]["intervals"]]
        marks = "".join(f'<i class="mark" style="left:{j/dur*100:.2f}%"></i>' for j in st["joins_s"] if j not in st["semantic_s"]) + \
                "".join(f'<i class="mark sem" style="left:{b/dur*100:.2f}%"></i>' for b in st["semantic_s"])
        frags = "".join(f'<span>{fr["start_s"]:.1f}–{fr["end_s"]:.1f} s · {fr["family"]}{" · gamma 0.9" if fr["variant"] != "orig" else ""}</span>' for fr in st["fragments"])
        panels.append(f'''<div class="stream" data-dur="{dur:.3f}">
  <div><h3>{esc(st["title"])}</h3><video src="streams/{st["file"]}" controls loop muted playsinline preload="metadata"></video><div class="frags">{frags}</div></div>
  <div class="rows"><div class="overlay" style="left:0;width:0">{marks}<i class="playhead" style="left:0"></i></div>{"".join(rows)}</div>
  <script type="application/json" class="rows-data">{json.dumps({"dur": dur, "rows": data_rows})}</script>
</div>''')
    return '<div class="streams">' + "".join(panels) + "</div>"


def _table(head, body):
    return '<div class="tablewrap"><table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>" + \
        "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in body) + "</tbody></table></div>"


def table_html(metrics: list[dict]) -> str:
    by = {m["method_id"]: m for m in metrics}
    rows = [m for m in ROWS if m in by]
    if not rows:
        return '<p class="mono">no five-baseline metrics pulled yet</p>'
    # 1) recognition + stability on the 800 untouched test clips
    h1 = ["model", "clip acc", "clip macro-F1", "balanced acc", "frame acc", "frame macro-F1", "fragments / min", "share of clips fragmented"]
    b1 = [[f'<b>{BASELINES[m][0]}</b>', f(g(by[m], "native", "test", "cine", "accuracy")), f(g(by[m], "native", "test", "cine", "macro_f1")),
           f(g(by[m], "native", "test", "cine", "balanced_accuracy")), f(g(by[m], "native", "test", "frame", "accuracy")), f(g(by[m], "native", "test", "frame", "macro_f1")),
           f(g(by[m], "native", "test", "stability", "fragments_per_minute_mean"), 1), f(g(by[m], "native", "test", "stability", "share_fragmented"))] for m in rows]
    # 2) segmentation on the 240 two-fragment test streams (2x2 design) and the 120 multi-fragment streams
    h2 = ["model", "boundary F1 @0.25 s", "@0.5 s", "@1.0 s", "false split same/none", "same/edit", "missed diff/none", "diff/edit", "multi-fragment boundary F1 @0.5 s"]
    b2 = []
    for m in rows:
        c = g(by[m], "constructed", "cells", default={}); b = g(by[m], "constructed", "boundary", default={})
        b2.append([f'<b>{BASELINES[m][0]}</b>', f(g(b, "0.25", "f1")), f(g(b, "0.5", "f1")), f(g(b, "1.0", "f1")), f(g(c, "same_none", "false_split_rate")), f(g(c, "same_edit", "false_split_rate")),
                   f(g(c, "diff_none", "missed_rate")), f(g(c, "diff_edit", "missed_rate")), f(g(by[m], "constructed_multi", "boundary", "0.5", "f1"))])
    # 3) selective routing at validation-selected thresholds
    h3 = ["model", "τ @5%", "coverage @5%", "achieved risk", "τ @1%", "val coverage @1%", "multi-fragment coverage", "multi-fragment risk"]
    b3 = []
    for m in rows:
        pol = g(by[m], "policy", "by_target", default={})
        b3.append([f'<b>{BASELINES[m][0]}</b>', f(g(pol, "0.05", "tau")), f(g(by[m], "constructed", "routing", "coverage")), f(g(by[m], "constructed", "routing", "risk")),
                   f(g(pol, "0.01", "tau")), f(g(pol, "0.01", "coverage")), f(g(by[m], "constructed_multi", "routing", "coverage")), f(g(by[m], "constructed_multi", "routing", "risk"))])
    return ('<h2 id="recognition">Recognition on 800 untouched test clips</h2>' + _table(h1, b1) + '<h2 id="segmentation">Segmentation on constructed test streams (±tolerance, one-to-one matching)</h2>' + _table(h2, b2) +
            '<h2 id="routing">Selective routing (thresholds chosen on validation, frozen)</h2>' + _table(h3, b3))


# Numbers recorded in docs/REPRODUCTION.md and docs/EXPERIMENT_LEDGER.md (2026-09-13/14). Δ = ours − authors, percentage points.
def _d(a, b):
    return "—" if a is None or b is None else f"{a - b:+.1f}"


def repro_html() -> str:
    out = ['<h2 id="reproduction">Reproduction: our trained models vs the authors\' reported numbers</h2>']
    # STFM: same code, same recipe, same seeds as the paper (EV9V nine-code video task)
    stfm = [("seed 100", 93.92, 90.14), ("seed 200", 93.58, 89.57), ("seed 300", 93.81, 89.92)]
    rows = [[f"ours, {n}", f"{a:.2f}", f"{f1:.2f}", "", ""] for n, a, f1 in stfm]
    rows.append(["<b>ours, mean ± std (seeds 100/200/300)</b>", "<b>93.77 ± 0.17</b>", "<b>89.88 ± 0.29</b>", "", ""])
    rows.append(["authors, Table 5 ResNet-18, seeds 100/200/300", "94.07 ± 0.66", "90.30 ± 1.03", "", ""])
    rows.append(["<b>Δ ours − authors</b>", f"<b>{_d(93.77, 94.07)}</b>", f"<b>{_d(89.88, 90.30)}</b>", "within one std of the paper", ""])
    rows.append(["ours, extra seed 666 (code default)", "93.36", "89.49", "", ""])
    out.append('<h3>STFM · retrained with the authors\' code and recipe on EV9V (nine-code, video-level test set)</h3>' + _table(["run", "test accuracy", "test macro-F1", "verdict", ""], rows))
    # MS-TCN: no echo result to compare, so the environment is checked on the authors' GTEA benchmark
    gtea = [("split 1", 83.0, 80.3, 63.0, 75.4, 76.1), ("split 2", 84.1, 81.2, 65.9, 80.3, 75.5), ("split 3", 89.1, 85.3, 77.7, 84.2, 78.4), ("split 4", 87.0, 86.2, 74.3, 84.2, 77.6)]
    rows = [[f"ours, {n}", f"{a:.1f}", f"{b:.1f}", f"{c:.1f}", f"{d:.1f}", f"{e:.1f}"] for n, a, b, c, d, e in gtea]
    rows.append(["<b>ours, 4-split average</b>", "<b>85.8</b>", "<b>83.2</b>", "<b>70.3</b>", "<b>81.0</b>", "<b>76.9</b>"])
    rows.append(["authors, CVPR 2019 (4-split average)", "85.8", "83.4", "69.8", "79.0", "76.3"])
    rows.append(["<b>Δ ours − authors</b>", f"<b>{_d(85.8, 85.8)}</b>", f"<b>{_d(83.2, 83.4)}</b>", f"<b>{_d(70.3, 69.8)}</b>", f"<b>{_d(81.0, 79.0)}</b>", f"<b>{_d(76.9, 76.3)}</b>"])
    out.append('<h3>MS-TCN · authors\' code and constants, trained by us on their GTEA benchmark (environment check; the EV9V model above is ours)</h3>' + _table(["run", "F1@10", "F1@25", "F1@50", "edit", "accuracy"], rows))
    # ASFormer: released models through our environment
    rows = [["ours (authors\' released GTEA models, our environment + 2-line patch)", "90.1", "88.8", "79.2", "84.6", "79.7"],
            ["authors, BMVC 2021 Table 7", "90.1", "88.8", "79.2", "84.6", "79.7"],
            ["<b>Δ</b>", "<b>+0.0</b>", "<b>+0.0</b>", "<b>+0.0</b>", "<b>+0.0</b>", "<b>+0.0</b>"]]
    out.append('<h3>ASFormer · authors\' released GTEA models re-evaluated in our environment (the EV9V model above is ours, trained with their code)</h3>' + _table(["run", "F1@10", "F1@25", "F1@50", "edit", "accuracy"], rows))
    # EchoViewCLIP: retrained on EV9V; paper numbers are on a private 38-view dataset
    rows = [["ours, EV9V test (nine-code, video-level)", "94.14", "90.7", "best validation accuracy 95.41 (epoch 21 of 30)"],
            ["authors, MICCAI 2025 Table 1 (private 38-view dataset)", "96.8", "95.7", "different data and label set"],
            ["<b>Δ (not like-for-like)</b>", f"<b>{_d(94.14, 96.8)}</b>", f"<b>{_d(90.7, 95.7)}</b>", "no released weights exist to check against; on EV9V it sits 0.4 pp above our STFM reproduction (93.77)"]]
    out.append('<h3>EchoViewCLIP · retrained on EV9V with the authors\' code and recipe</h3>' + _table(["run", "accuracy", "macro-F1", "note"], rows))
    # EchoPrime: no training
    rows = [["ours, EV9V test, five-family mapping (released weights, no training)", "95.9", "90.6", "0% of predictions fell outside the five families"],
            ["authors, Nature 2026 (internal 58-view test set)", "AUC 0.997", "—", "no comparable accuracy reported"]]
    out.append('<h3>EchoPrime · released weights only</h3>' + _table(["run", "accuracy", "macro-F1", "note"], rows))
    return "".join(out)


def build(out: Path) -> Path:
    streams = json.loads((REPO / "runs" / "_demo" / "streams" / "streams.json").read_text())
    metrics, h = collect_metrics(REPO / "runs")
    metrics = [m for m in metrics if m["method_id"] in ROWS]
    keys = '<div class="keys">' + "".join(f'<span><i class="sw" style="background:var(--s{i+1})"></i>{c}</span>' for i, c in enumerate(CLASSES)) + \
           '<span><i class="sw hatch"></i>deferred</span><span><i class="ln dash"></i>file join</span><span><i class="ln"></i>true view change</span></div>'
    links = '<div class="links">' + "".join(f'<span><b>{BASELINES[m][0]}</b> — {esc(BASELINES[m][1])} · ' + " · ".join(f'<a href="{u}" target="_blank" rel="noopener">{esc(l)}</a>' for l, u in BASELINES[m][2]) + "</span>" for m in ROWS) + "</div>"
    page = f"""<title>EV9V Routing Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<nav class="side" aria-label="sections">
  <input id="filter" type="search" placeholder="filter streams…" aria-label="filter streams">
  <a href="#native">Single view</a><a href="#same_none">Same view</a><a href="#same_edit">Same view, edited</a><a href="#diff_none">View change</a><a href="#diff_edit">View change, edited</a><a href="#multi">Multi-view</a>
  <span class="navsep"></span><a href="#recognition">Recognition</a><a href="#segmentation">Segmentation</a><a href="#routing">Routing</a><a href="#reproduction">Reproduction</a><a href="#models">Models</a>
</nav>
<div class="wrap">
<header id="top"><h1>Five baselines on EV9V</h1><p>Press play. Each row is one published model reading the same frames. Hatched = deferred. Click a strip to seek.</p>{keys}</header>
{streams_html(streams)}
<section style="display:grid;gap:12px">{table_html(metrics)}</section>
<section>{repro_html()}</section>
<section id="models"><h2>Models</h2>{links}</section>
<footer>encoder {h} · EV9V (CC-BY-4.0) · <a href="https://github.com/WilliamQiuzy/echo-view-routing">github.com/WilliamQiuzy/echo-view-routing</a></footer>
</div>
<script>{JS}</script>
"""
    out.write_text(page)
    return out


if __name__ == "__main__":
    print(build(REPO / "demo" / "demo_v2.html"))
