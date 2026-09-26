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
    "b6": ("STFM", "The EV9V authors' own view classifier. A CNN reads each frame and an LSTM combines the frames. "
                   "We run it on a sliding 1.6 s window.", [("paper", "https://arxiv.org/abs/2606.17437"), ("official code", "https://github.com/bgx666/stfm"),
           ("our driver", GH + "scripts/run_stfm.py"), ("our sliding window", GH + "scripts/run_windowed_stfm.py"), ("frozen checkpoint", GH + "docs/models_frozen/stfm_official/seed100/MANIFEST.json")]),
    "b7": ("EchoViewCLIP", "A CLIP-based view classifier (MICCAI 2025). We retrained it on EV9V and run it on a sliding 1.6 s window.", [("paper", "https://link.springer.com/chapter/10.1007/978-3-032-05169-1_18"), ("official code", "https://github.com/xmed-lab/EchoViewCLIP"),
           ("our driver", GH + "scripts/run_echoviewclip.py"), ("our config", GH + "third_party/adapters/echoviewclip/ev9v_stage1.yaml"), ("our sliding window", GH + "scripts/run_windowed_echoviewclip.py"), ("frozen checkpoint", GH + "docs/models_frozen/echoviewclip_stage1/v1/MANIFEST.json")]),
    "b8": ("EchoPrime", "A released view classifier for 11 views. It labels each frame on its own, with the released weights and no training.", [("paper", "https://arxiv.org/abs/2410.09704"), ("official code + weights", "https://github.com/echonet/EchoPrime"),
           ("our evaluation", GH + "scripts/eval_echoprime_views.py"), ("our per-frame runner", GH + "scripts/run_windowed_echoprime.py"), ("frozen weights", GH + "docs/models_frozen/echoprime_view_classifier/release_v1.0.0/MANIFEST.json")]),
    "b4": ("MS-TCN", "A temporal segmentation model. It reads frozen ResNet-18 features of every frame in the stream.", [("paper", "https://arxiv.org/abs/1903.01945"), ("official code", "https://github.com/yabufarha/ms-tcn"),
           ("our driver", GH + "scripts/run_mstcn_official.py"), ("our patch (py3)", GH + "third_party/patches/ms-tcn-py3.patch"), ("frozen checkpoint", GH + "docs/models_frozen/mstcn_official/v1/MANIFEST.json")]),
    "b5": ("ASFormer", "A Transformer for temporal segmentation. It reads the same frozen frame features as MS-TCN.", [("paper", "https://arxiv.org/abs/2110.08568"), ("official code", "https://github.com/ChinaYi/ASFormer"),
           ("our driver", GH + "scripts/run_asformer_official.py"), ("our patch (py3)", GH + "third_party/patches/asformer-py3.patch"), ("frozen checkpoint", GH + "docs/models_frozen/asformer_official/v1/MANIFEST.json")]),
}
ROWS = ["b6", "b7", "b8", "b4", "b5"]
FIGURES = {  # authors' own architecture figures, reproduced from the papers / official repo with attribution
    "b6": ("figures/stfm.png", "Figure 3 of Gou et al., arXiv:2606.17437"),
    "b7": ("figures/echoviewclip.jpg", "Figure 1 in the official repository (Song et al., MICCAI 2025)"),
    "b8": ("figures/echoprime.png", "Figure 1B–C of Vukadinovic et al., Nature 650 (2026) / arXiv:2410.09704. Our baseline uses only the View Classifier (first block of C), applied per frame; the ViT-family Echo-MViT starts from the video side of the Contrastive Encoder"),
    "b4": (["figures/mstcn.png", "figures/mstcn_layer.png"], "Figures 1 and 2 of Abu Farha & Gall, CVPR 2019. The left figure shows the whole model. Input x is one feature vector per frame (here ResNet-18 features). Each stage is a stack of dilated temporal convolutions over all frames and refines the class probabilities of the stage below; every stage has its own loss. The right figure shows one grey node, a dilated residual layer. The dilation doubles from layer to layer. The official code uses 4 stages of 10 layers"),
    "b5": ("figures/asformer.png", "Figure 1 of Yi et al., BMVC 2021"),
}
CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
ROUTING_SENTENCE = ("<b>Routing</b> means sending each part of an echo recording to the analysis made for its view. "
                    "Parts the model is unsure about are held back for a person to check.")


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
td:first-child,th:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}thead th{vertical-align:bottom;line-height:1.4}
td a,.links a{color:var(--accent-ink)}
.model{display:grid;gap:8px;padding:12px 0;border-top:1px solid var(--line-2)}
.mhead{font-size:13.5px;color:var(--ink-2);line-height:1.7}.mhead b{color:var(--ink);font-size:15px}
figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:10px;display:grid;gap:6px}
figure img{max-width:100%;max-height:560px;width:auto;height:auto;display:block;margin:0 auto}
.figrow{display:flex;flex-wrap:wrap;gap:12px 36px;align-items:center;justify-content:center}.figrow img{margin:0}.figrow img.aux{max-height:300px}
figcaption{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted)}
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


def _panels(streams: list, rows_ids: list[str] | None = None, names: dict[str, str] | None = None,
            notes: dict[str, str] | None = None) -> str:
    rows_ids = rows_ids or ROWS; names = names or {m: BASELINES[m][0] for m in ROWS}; notes = notes or {}
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
        for m in rows_ids:
            if m not in st["methods"]:
                continue
            rows.append(f'<span class="lab">{names[m]}</span><span class="badge off" data-row="{m}">—</span><div class="strip">{segs(st["methods"][m]["intervals"])}</div>')
            data_rows[m] = [{"s": i["start_s"], "e": i["end_s"], "v": i["view"], "a": i["accepted"]} for i in st["methods"][m]["intervals"]]
        marks = "".join(f'<i class="mark" style="left:{j/dur*100:.2f}%"></i>' for j in st["joins_s"] if j not in st["semantic_s"]) + \
                "".join(f'<i class="mark sem" style="left:{b/dur*100:.2f}%"></i>' for b in st["semantic_s"])
        frags = "".join(f'<span>{fr["start_s"]:.1f}–{fr["end_s"]:.1f} s · {fr["family"]}{" · brightness edit" if fr["variant"] != "orig" else ""}</span>' for fr in st["fragments"])
        title = st["title"].replace("gamma-edited", "brightness-edited")
        panels.append(f'''<div class="stream" data-dur="{dur:.3f}">
  <div><h3>{esc(title)}</h3>{f'<p class="note">{esc(notes[st["id"]])}</p>' if st["id"] in notes else ""}<video src="streams/{st["file"]}" controls loop muted playsinline preload="metadata"></video><div class="frags">{frags}</div></div>
  <div class="rows"><div class="overlay" style="left:0;width:0">{marks}<i class="playhead" style="left:0"></i></div>{"".join(rows)}</div>
  <script type="application/json" class="rows-data">{json.dumps({"dur": dur, "rows": data_rows})}</script>
</div>''')
    return '<div class="streams">' + "".join(panels) + "</div>"


def _th(h: str) -> str:
    """Header cell; a '\n' in the label is the only place it breaks."""
    return "<th>" + "<br>".join(esc(part) for part in str(h).split("\n")) + "</th>"


def _table(head, body):
    return '<div class="tablewrap"><table><thead><tr>' + "".join(_th(h) for h in head) + "</tr></thead><tbody>" + \
        "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in body) + "</tbody></table></div>"


def table_html(metrics: list[dict]) -> str:
    by = {m["method_id"]: m for m in metrics}
    rows = [m for m in ROWS if m in by]
    if not rows:
        return '<p class="mono">no five-baseline metrics pulled yet</p>'
    # 1) recognition + stability on the 800 untouched test clips
    h1 = ["model", "clip\naccuracy", "clip\nmacro-F1", "balanced\naccuracy", "frame\naccuracy", "frame\nmacro-F1", "label flips\nper min",
          "clips with\na flip"]
    b1 = [[f'<b>{BASELINES[m][0]}</b>', f(g(by[m], "native", "test", "cine", "accuracy")), f(g(by[m], "native", "test", "cine", "macro_f1")),
           f(g(by[m], "native", "test", "cine", "balanced_accuracy")), f(g(by[m], "native", "test", "frame", "accuracy")), f(g(by[m], "native", "test", "frame", "macro_f1")),
           f(g(by[m], "native", "test", "stability", "fragments_per_minute_mean"), 1), f(g(by[m], "native", "test", "stability", "share_fragmented"))] for m in rows]
    # 2) segmentation on the 240 two-fragment test streams (2x2 design) and the 120 multi-fragment streams
    h2 = ["model", "change F1\n±0.25 s", "change F1\n±0.5 s", "change F1\n±1 s", "false cuts\nsame view", "false cuts\nsame view, edited",
          "missed\nview change", "missed\nview change, edited", "change F1 ±0.5 s\n4–8 clips"]
    b2 = []
    for m in rows:
        c = g(by[m], "constructed", "cells", default={}); b = g(by[m], "constructed", "boundary", default={})
        b2.append([f'<b>{BASELINES[m][0]}</b>', f(g(b, "0.25", "f1")), f(g(b, "0.5", "f1")), f(g(b, "1.0", "f1")), f(g(c, "same_none", "false_split_rate")), f(g(c, "same_edit", "false_split_rate")),
                   f(g(c, "diff_none", "missed_rate")), f(g(c, "diff_edit", "missed_rate")), f(g(by[m], "constructed_multi", "boundary", "0.5", "f1"))])
    # 3) selective routing at validation-selected thresholds
    h3 = ["model", "threshold\n5% target", "coverage\n5% target", "actual error\n5% target", "threshold\n1% target",
          "coverage on\nvalidation, 1%", "coverage\n4–8 clips", "actual error\n4–8 clips"]
    b3 = []
    for m in rows:
        pol = g(by[m], "policy", "by_target", default={})
        b3.append([f'<b>{BASELINES[m][0]}</b>', f(g(pol, "0.05", "tau")), f(g(by[m], "constructed", "routing", "coverage")), f(g(by[m], "constructed", "routing", "risk")),
                   f(g(pol, "0.01", "tau")), f(g(pol, "0.01", "coverage")), f(g(by[m], "constructed_multi", "routing", "coverage")), f(g(by[m], "constructed_multi", "routing", "risk"))])
    intro1 = ("<p>Each of the 800 test clips shows one view. A clip score gives each clip the label most of its frames get. "
              "A frame score judges every frame on its own. Macro-F1 and balanced accuracy weight every view equally. "
              "Label flips count how often the label changes inside one clip, per minute, so 0 is best. "
              "The last column is the share of clips whose label changes at least once. "
              'All metrics are explained on the <a href="vit.html#benchmarks">Vision-Transformer page</a>.</p>')
    intro2 = ("<p>EV9V has no continuous recordings with several views, so we join test clips into streams. "
              "A predicted view change is correct if it lies within the tolerance of a real change. Each real change can be matched only once. "
              "A false cut is a change predicted within 0.5 s of a join between two clips of the same view. "
              "A missed change is a real view change with no predicted change within 0.5 s. "
              "Edited means the second clip had its brightness changed. "
              "The last column uses the 120 streams of 4 to 8 clips, and the others use the 240 two-clip streams.</p>")
    intro3 = ("<p>Each predicted segment gets a confidence. It is routed if the confidence clears a threshold, and held back for a person otherwise. "
              "The threshold is chosen on validation for an error target of 5% or 1% and then kept fixed. "
              "Coverage is the share of stream time routed automatically. Actual error is the share of routed time with the wrong view. "
              "The last two columns apply the 5% threshold to the streams of 4 to 8 clips.</p>")
    return ('<h2 id="recognition">Recognition on the 800 test clips</h2>' + intro1 + _table(h1, b1)
            + '<h2 id="segmentation">Segmentation on joined test streams</h2>' + intro2 + _table(h2, b2)
            + '<h2 id="routing">Routing</h2>' + intro3 + _table(h3, b3))


# Numbers recorded in docs/REPRODUCTION.md and docs/EXPERIMENT_LEDGER.md (2026-09-13/14). Δ = ours − authors, percentage points.
def _d(a, b):
    return "—" if a is None or b is None else f"{a - b:+.1f}"


def repro_html() -> str:
    out = ['<h2 id="reproduction">Reproduction check against the authors\' reported numbers</h2>'
           "<p>We ran each baseline with its authors' code and compared our numbers with theirs. "
           "Δ is ours minus theirs, in percentage points.</p>"]
    # STFM: same code, same recipe, same seeds as the paper (EV9V nine-code video task)
    stfm = [("seed 100", 93.92, 90.14), ("seed 200", 93.58, 89.57), ("seed 300", 93.81, 89.92)]
    rows = [[f"ours, {n}", f"{a:.2f}", f"{f1:.2f}", "", ""] for n, a, f1 in stfm]
    rows.append(["<b>ours, mean ± std (seeds 100/200/300)</b>", "<b>93.77 ± 0.17</b>", "<b>89.88 ± 0.29</b>", "", ""])
    rows.append(["authors, Table 5 ResNet-18, seeds 100/200/300", "94.07 ± 0.66", "90.30 ± 1.03", "", ""])
    rows.append(["<b>Δ ours − authors</b>", f"<b>{_d(93.77, 94.07)}</b>", f"<b>{_d(89.88, 90.30)}</b>", "within one std of the paper", ""])
    rows.append(["ours, extra seed 666 (code default)", "93.36", "89.49", "", ""])
    out.append('<h3>STFM, retrained on EV9V with the authors\' code and recipe</h3>'
               "<p>This is the nine-code test, with one label per clip.</p>" + _table(["run", "test\naccuracy", "test\nmacro-F1", "verdict", ""], rows))
    # MS-TCN: no echo result to compare, so the environment is checked on the authors' GTEA benchmark
    gtea = [("split 1", 83.0, 80.3, 63.0, 75.4, 76.1), ("split 2", 84.1, 81.2, 65.9, 80.3, 75.5), ("split 3", 89.1, 85.3, 77.7, 84.2, 78.4), ("split 4", 87.0, 86.2, 74.3, 84.2, 77.6)]
    rows = [[f"ours, {n}", f"{a:.1f}", f"{b:.1f}", f"{c:.1f}", f"{d:.1f}", f"{e:.1f}"] for n, a, b, c, d, e in gtea]
    rows.append(["<b>ours, 4-split average</b>", "<b>85.8</b>", "<b>83.2</b>", "<b>70.3</b>", "<b>81.0</b>", "<b>76.9</b>"])
    rows.append(["authors, CVPR 2019 (4-split average)", "85.8", "83.4", "69.8", "79.0", "76.3"])
    rows.append(["<b>Δ ours − authors</b>", f"<b>{_d(85.8, 85.8)}</b>", f"<b>{_d(83.2, 83.4)}</b>", f"<b>{_d(70.3, 69.8)}</b>", f"<b>{_d(81.0, 79.0)}</b>", f"<b>{_d(76.9, 76.3)}</b>"])
    gtea_head = ["run", "segment F1\n10% overlap", "segment F1\n25% overlap", "segment F1\n50% overlap", "edit\nscore", "frame\naccuracy"]
    out.append("<h3>MS-TCN, trained by us on the authors' GTEA benchmark</h3>"
               "<p>There is no published echo result to compare with, so this checks our environment on the authors' own benchmark. "
               "GTEA is a video dataset of kitchen actions. The EV9V model above is trained by us with the same code. "
               "Segment F1 counts a predicted segment as correct when it overlaps a true segment by the given share. "
               "The edit score compares the predicted order of segments with the true order, so extra segments lower it.</p>"
               + _table(gtea_head, rows))
    # ASFormer: released models through our environment
    rows = [["ours (authors\' released GTEA models, our environment + 2-line patch)", "90.1", "88.8", "79.2", "84.6", "79.7"],
            ["authors, BMVC 2021 Table 7", "90.1", "88.8", "79.2", "84.6", "79.7"],
            ["<b>Δ</b>", "<b>+0.0</b>", "<b>+0.0</b>", "<b>+0.0</b>", "<b>+0.0</b>", "<b>+0.0</b>"]]
    out.append("<h3>ASFormer, the authors' released GTEA models run in our environment</h3>"
               "<p>The EV9V model above is trained by us with their code.</p>" + _table(gtea_head, rows))
    # EchoViewCLIP: retrained on EV9V; paper numbers are on a private 38-view dataset
    rows = [["ours, EV9V test (nine-code, video-level)", "94.14", "90.7", "best validation accuracy 95.41 (epoch 21 of 30)"],
            ["authors, MICCAI 2025 Table 1 (private 38-view dataset)", "96.8", "95.7", "different data and label set"],
            ["<b>Δ (not like-for-like)</b>", f"<b>{_d(94.14, 96.8)}</b>", f"<b>{_d(90.7, 95.7)}</b>",
             "No released weights exist to check against. On EV9V it is 0.4 points above our STFM reproduction (93.77)."]]
    out.append("<h3>EchoViewCLIP, retrained on EV9V with the authors' code and recipe</h3>" + _table(["run", "accuracy", "macro-F1", "note"], rows))
    # EchoPrime: no training
    rows = [["ours, EV9V test, five-family mapping (released weights, no training)", "95.9", "90.6", "0% of predictions fell outside the five families"],
            ["authors, Nature 2026 (internal 58-view test set)", "AUC 0.997", "—", "no comparable accuracy reported"]]
    out.append("<h3>EchoPrime, released weights only</h3>" + _table(["run", "accuracy", "macro-F1", "note"], rows))
    return "".join(out)


def _imgs(m: str) -> str:
    srcs = FIGURES[m][0] if isinstance(FIGURES[m][0], list) else [FIGURES[m][0]]
    return "".join(f'<img src="{src}" alt="{BASELINES[m][0]} architecture, part {i + 1}" loading="lazy"{' class="aux"' if i else ''}>'
                   for i, src in enumerate(srcs))


def build(out: Path) -> Path:
    streams = json.loads((REPO / "runs" / "_demo" / "streams" / "streams.json").read_text())
    metrics, h = collect_metrics(REPO / "runs")
    metrics = [m for m in metrics if m["method_id"] in ROWS]
    keys = '<div class="keys">' + "".join(f'<span><i class="sw" style="background:var(--s{i+1})"></i>{c}</span>' for i, c in enumerate(CLASSES)) + \
           '<span><i class="sw hatch"></i>deferred</span><span><i class="ln dash"></i>clip join</span><span><i class="ln"></i>true view change</span></div>'
    links = "".join(
        f'<div class="model" id="model-{m}"><div class="mhead"><b>{BASELINES[m][0]}</b><br>{esc(BASELINES[m][1])}<br>'
        + " · ".join(f'<a href="{u}" target="_blank" rel="noopener">{esc(l)}</a>' for l, u in BASELINES[m][2]) + '</div>'
        + f'<figure><div class="figrow">{_imgs(m)}</div><figcaption>Architecture figure by the authors, taken from {esc(FIGURES[m][1])}</figcaption></figure></div>'
        for m in ROWS)
    page = f"""<title>EV9V Routing Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<nav class="side" aria-label="sections">
  <input id="filter" type="search" placeholder="filter streams…" aria-label="filter streams">
  <a href="vit.html">Vision Transformers →</a><span class="navsep"></span>
  <a href="#native">Single view</a><a href="#same_none">Same view</a><a href="#same_edit">Same view, edited</a><a href="#diff_none">View change</a><a href="#diff_edit">View change, edited</a><a href="#multi">Multi-view</a>
  <span class="navsep"></span><a href="#recognition">Recognition</a><a href="#segmentation">Segmentation</a><a href="#routing">Routing</a><a href="#reproduction">Reproduction</a><a href="#models">Models</a>
</nav>
<div class="wrap">
<header id="top"><h1>Five baselines on EV9V</h1><p>Press play. Each row is one published model reading the same frames. Our Vision-Transformer designs are on a <a href="vit.html">separate page</a>.</p><p>{ROUTING_SENTENCE} In the rows below, held-back parts are hatched.</p>{keys}</header>
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
