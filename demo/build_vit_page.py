#!/usr/bin/env python
"""Build demo/vit.html: the Vision-Transformer family page (designs, benchmarks, per-view results, streams, conclusions).

Inputs (all already produced): the final ladder metrics (runs/<LADDER>/metrics), docs/results/2026-09-25_vit_family/
{nine_code,breakdowns}.json and runs/_demo/streams/streams.json (ViT rows added by scripts/make_demo_streams.py).
Shares CSS, JS and the stream-panel component with demo/build_demo_v2.py so both pages look the same.
Page text is written as short full sentences (no "term: fragment" lists); table headers break only at "\n".
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_demo_v2 as base  # noqa: E402
import vit_diagrams as dg  # noqa: E402

REPO = HERE.parent
LADDER = "20260925-130154-baselines-1b027c50-s0"
RES = REPO / "docs" / "results" / "2026-09-25_vit_family"
GH = "https://github.com/WilliamQiuzy/echo-view-routing/blob/main/"

NAMES = {
    "b0": "ResNet-18 (B0)", "b8": "EchoPrime", "vit_frame": "ViT-S · D1",
    "b6": "STFM", "b7": "EchoViewCLIP", "vivit_fe": "Factorised ViT · D2", "vivit_fe_centre": "Factorised ViT + centre · D2",
    "mvit_k400": "MViT (Kinetics) · D3", "mvit_echoprime": "Echo-MViT · D3", "mvit_echoprime_centre": "Echo-MViT + centre · D3",
    "b4": "MS-TCN", "b5": "ASFormer", "mstcn_vitfeat": "MS-TCN on ViT-S", "vit_router": "ViT-Router · D4",
}
OURS = {"vit_frame", "vivit_fe", "vivit_fe_centre", "mvit_k400", "mvit_echoprime", "mvit_echoprime_centre", "mstcn_vitfeat", "vit_router"}
GROUPS = [("Frame models, which see one frame at a time", ["b0", "b8", "vit_frame"]),
          ("Clip models, which see a 1.6 s window slid along the stream",
           ["b6", "b7", "vivit_fe", "vivit_fe_centre", "mvit_k400", "mvit_echoprime", "mvit_echoprime_centre"]),
          ("Stream models, which see the whole stream at once", ["b4", "b5", "mstcn_vitfeat", "vit_router"])]
STREAM_ROWS = ["b0", "b8", "b6", "b7", "b4", "b5", "mvit_echoprime", "mvit_echoprime_centre", "vit_router"]
STREAM_NAMES = {**{m: base.BASELINES[m][0] for m in ("b8", "b6", "b7", "b4", "b5")}, "b0": "ResNet-18",
                "mvit_echoprime": "Echo-MViT", "mvit_echoprime_centre": "Echo-MViT + centre", "vit_router": "ViT-Router"}
SELECTED = {
    "native_PASA": "One PSAX clip. Frame models flicker. Every temporal model is clean.",
    "native_PMASA": "PMASA is often confused with PMPALA. Both Echo-MViT rows defer the clip instead of guessing.",
    "same_none_A5C": "Two A5C clips joined. ViT-Router calls most of it A4C, its weak view.",
    "same_edit_A4C": "One view, second half darker. Only ViT-Router has no false A5C segment.",
    "diff_PLAX_A4C": "Change at 3.0 s. The ViT rows switch on time. EchoViewCLIP is 0.5 s late.",
    "diff_A4C_PSAX": "Plain Echo-MViT switches 0.5 s late. With centre supervision it is on time. ViT-Router is exact.",
    "diff_A4C_A5C": "The hardest pair on the full test set. Here ViT-Router, STFM, MS-TCN and ASFormer are exact.",
    "diff_A5C_SC4C": "Echo-MViT + centre reads the A5C half as A4C. ViT-Router, MS-TCN and ASFormer are exact.",
    "diff_A5C_A4C": "Most models read the A5C half as A4C. The Echo-MViT rows recover part of it.",
    "multi_01": "Five views. ViT-Router and Echo-MViT + centre read the A5C part as A4C.",
}

EXTRA_CSS = r"""
:root{--warn:#b4532a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--warn:#e0875e}}
:root[data-theme="dark"]{--warn:#e0875e}
.lead{font-size:16px;color:var(--ink);max-width:72ch}
ul.short{margin:0;padding-left:18px;color:var(--ink-2);display:grid;gap:4px;max-width:80ch}
.cards{display:grid;gap:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 16px;display:grid;gap:8px}
.card h3{margin:0;font-size:16px;color:var(--ink)}
.card .why{font-size:14px;color:var(--ink-2);max-width:90ch}
.tag{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;color:var(--accent-ink);border:1px solid var(--accent);border-radius:3px;padding:0 5px;margin-left:6px;vertical-align:1px}
svg.diagram{width:100%;height:auto;max-width:900px;display:block;overflow:visible}
.wrap>*,section>*,.cards>*,.card>*,.stream>*{min-width:0}
.diagwrap{overflow-x:auto;-webkit-overflow-scrolling:touch}.diagwrap svg.diagram{min-width:680px}
td.ours,tr.ours td:first-child{font-weight:600;color:var(--ink)}
tr.group td{background:var(--line-2);color:var(--muted);font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.04em;text-transform:uppercase;text-align:left;padding:5px 12px}
td.heat{background:color-mix(in srgb,var(--warn) var(--p),transparent)}
.note{font-size:13px;color:var(--ink-2);margin:0 0 6px;max-width:none}
details.more>summary{cursor:pointer;color:var(--accent-ink);font-size:14px;margin:6px 0}
table.glance td,table.glance th{white-space:normal;text-align:left;vertical-align:top}
.facts{display:grid;grid-template-columns:118px 1fr;gap:6px 14px;margin:2px 0 0;max-width:100ch}
.facts.wide{grid-template-columns:168px 1fr}
.facts dt{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;padding-top:3px}
.facts dd{margin:0;color:var(--ink-2);font-size:14px}
@media (max-width:640px){.facts,.facts.wide{grid-template-columns:1fr}.facts dd{margin-bottom:4px}}
th{white-space:normal;vertical-align:bottom;min-width:64px}
table.metrics th,table.metrics td{text-align:center}
table.metrics th:first-child,table.metrics td:first-child{text-align:left}
table.metrics thead th{white-space:nowrap;line-height:1.4}
.verdict{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;white-space:nowrap}
.yes{color:var(--accent-ink)}.no{color:var(--warn)}.mixed{color:var(--muted)}
td.l,th.l{text-align:left}
th.gstart,td.gstart{border-left:1px solid var(--line)}
tr.super th{color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-weight:600;text-transform:none;letter-spacing:0;font-size:13px;border-bottom:1px solid var(--line);padding-bottom:6px}
tr.super th:first-child{border-bottom:0}
tr.super th .sub{display:block;font-weight:400;font-size:12px;color:var(--muted)}
ol.concl{margin:0;padding-left:20px;display:grid;gap:6px;max-width:85ch}
ol.concl li{color:var(--ink)}ol.concl li span{color:var(--ink-2)}
"""


def load_metrics() -> dict[str, dict]:
    out = {}
    for p in sorted((REPO / "runs" / LADDER / "metrics").glob("*.json")):
        m = json.loads(p.read_text()); out[m["method_id"]] = m
    return out


def ms(vals: list[float], nd: int = 3) -> str:
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{statistics.mean(vals):.{nd}f} ± {statistics.stdev(vals):.{nd}f}"


def diag(svg: str) -> str:
    return f'<div class="diagwrap">{svg}</div>'


def name_cell(m: str) -> str:
    return f'<span class="{"ours" if m in OURS else ""}">{base.esc(NAMES[m])}</span>'


def _th(text: str, attr: str = "") -> str:
    """Header cell; a '\n' in `text` is the only place the label may break."""
    return f"<th{attr}>" + "<br>".join(base.esc(part) for part in text.split("\n")) + "</th>"


def simple_table(head: list[str], rows: list[list[str]]) -> str:
    return ('<div class="tablewrap"><table class="metrics"><thead><tr>' + "".join(_th(h) for h in head) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows) + "</tbody></table></div>")


def grouped_table(head: list[str], row_fn, methods: dict, super_head: list[tuple[str, str, int]] | None = None) -> str:
    """Rows grouped by GROUPS. `super_head` holds (title, subtitle, span) per column group, the first one over the model column."""
    starts, col = set(), 1
    for _, _, span in (super_head or [])[1:]:
        starts.add(col); col += span

    def cls(i: int) -> str:
        return ' class="gstart"' if i in starts else ""
    body = []
    for title, ms_ in GROUPS:
        body.append(f'<tr class="group"><td colspan="{len(head)}">{base.esc(title)}</td></tr>')
        for m in ms_:
            if m in methods:
                cells = [name_cell(m)] + row_fn(m)
                tds = []
                for i, c in enumerate(cells):
                    text, attr = (c if isinstance(c, tuple) else (c, ""))
                    if i in starts:
                        attr = attr.replace('class="', 'class="gstart ') if 'class="' in attr else attr + ' class="gstart"'
                    tds.append(f"<td{attr}>{text}</td>")
                body.append(f'<tr class="{"ours" if m in OURS else ""}">' + "".join(tds) + "</tr>")
    top = ""
    if super_head:
        cells, start = [], 0
        for t, sub, span in super_head:
            label = f'{base.esc(t)}<span class="sub">{base.esc(sub)}</span>' if t else ""
            cells.append(f'<th colspan="{span}"{cls(start)}>{label}</th>'); start += span
        top = '<tr class="super">' + "".join(cells) + "</tr>"
    return ('<div class="tablewrap"><table class="metrics"><thead>' + top + "<tr>" + "".join(_th(h, cls(i)) for i, h in enumerate(head))
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def heat(v: float | None, worst: float = 0.5) -> tuple[str, str]:
    """(text, td attributes): shade the error 1 - v, full shade at `worst`."""
    if v is None:
        return ("—", "")
    p = max(0.0, min(1.0, (1 - v) / (1 - worst))) * 45
    return (f"{v:.2f}", f' class="heat" style="--p:{p:.0f}%"')


# ---------------------------------------------------------------------------------------------------------------- sections

def labels_html() -> str:
    # EV9V official split, clips per code (data/manifests/ev9v_native.csv); names as on the dataset card
    codes = [("PLHLA", "parasternal long axis", "PLAX", (1120, 185, 275)), ("PASA", "parasternal short axis", "PSAX", (212, 16, 41)),
             ("PMASA", "short axis, apical level *", "PSAX", (317, 50, 72)), ("PMVLSA", "short axis, mitral valve level *", "PSAX", (262, 32, 50)),
             ("PPMLSA", "short axis, papillary muscle level", "PSAX", (183, 41, 56)), ("PMPALA", "pulmonary artery long axis *", "left out", (319, 46, 88)),
             ("A4C", "apical 4-chamber", "A4C", (858, 148, 231)), ("A5C", "apical 5-chamber", "A5C", (217, 29, 40)),
             ("SC4C", "subcostal 4-chamber", "SC4C", (195, 20, 35))]
    rows = [(f'<span class="mono">{c}</span>', base.esc(n), base.esc(fam), *[f"{x:,}" for x in cnt]) for c, n, fam, cnt in codes]
    rows += [("<b>nine codes</b>", "", "", "3,683", "567", "888"), ("<b>five families</b>", "", "PMPALA left out", "3,364", "521", "800")]
    head = [("code", "l"), ("view (dataset card)", "l"), ("family", "l"), ("train", ""), ("val", ""), ("test", "")]
    table = ('<div class="tablewrap"><table><thead><tr>' + "".join(f'<th class="{c}">{h}</th>' for h, c in head) + "</tr></thead><tbody>"
             + "".join("<tr>" + "".join(f'<td class="{head[i][1]}">{v}</td>' for i, v in enumerate(r)) + "</tr>" for r in rows)
             + "</tbody></table></div>")
    points = [
        "<b>Nine codes.</b> EV9V labels every clip with one of nine view codes (table below).",
        "<b>Five view families.</b> Routing only needs the family of a view, so we group the nine codes into five families. "
        "They are PLAX, PSAX, A4C, A5C and SC4C.",
        "The four short-axis codes become one family, PSAX. They differ only in the level of the cut.",
        "PMPALA is left out, because two sources define it differently. One calls it a pulmonary-artery view, "
        "the other a parasternal long-axis view.",
        "<b>Nine-code models</b> (STFM, EchoViewCLIP, D2, D3) give a probability for each code. "
        "For the five-family results we add up the probabilities of the codes in each family and drop PMPALA.",
        "<b>EchoPrime</b> was trained by its authors on its own 11 views. Five of them match our families. "
        "They are A4C, A5C, parasternal long (PLAX), parasternal short (PSAX) and subcostal (SC4C). "
        "We keep those five and drop the other six (A2C, A3C, suprasternal and three Doppler views).",
        "<b>Five-family models</b> (ResNet-18, MS-TCN, ASFormer, D1, D4) are trained on the five families directly.",
    ]
    note = ("<p>* The dataset card and the STFM README name these codes differently. "
            "The short-axis names disagree on the level, and PMPALA is a different view altogether.</p>")
    return ('<section id="labels"><h2>Views and labels</h2><ul class="short">' + "".join(f"<li>{p}</li>" for p in points) + "</ul>"
            f"{table}{note}</section>")


def _facts(facts: list[tuple[str, str]], cls: str = "") -> str:
    return f'<dl class="facts {cls}">' + "".join(f"<dt>{base.esc(k)}</dt><dd>{base.esc(v)}</dd>" for k, v in facts) + "</dl>"


def _card(title: str, tag: str, svg: str, facts: list[tuple[str, str]]) -> str:
    return f'<div class="card"><h3>{base.esc(title)}<span class="tag">{base.esc(tag)}</span></h3>{svg}{_facts(facts)}</div>'


def designs_html() -> str:
    glance = [["D1", "Plain ViT", "one frame", "image patches", "ImageNet-21k", "ResNet-18 (same recipe)"],
              ["D2", "Factorised video ViT", "1.6 s clip, sliding window", "frame embeddings from a shared ViT-S", "ImageNet-21k", "STFM, EchoViewCLIP"],
              ["D3", "Multiscale video ViT (MViTv2-S)", "1.6 s clip, sliding window", "space-time cubes of pixels", "Kinetics-400 or EchoPrime", "STFM, EchoViewCLIP, EchoPrime"],
              ["D4", "ViT-Router", "whole stream", "frame embeddings from frozen D1", "D1", "MS-TCN, ASFormer (same training streams)"]]
    table = ('<div class="tablewrap"><table class="glance"><thead><tr>' + "".join(f"<th>{h}</th>" for h in
             ["", "design", "sees", "Transformer input", "initial weights", "compared with"]) + "</tr></thead><tbody>"
             + "".join("<tr>" + "".join(f"<td>{base.esc(c)}</td>" for c in r) + "</tr>" for r in glance) + "</tbody></table></div>")
    cards = [
        _card("D1 · Plain ViT", "plain ViT", diag(dg.plain_vit()), [
            ("Input", "The model sees one frame, resized to 224 × 224 with black bars above and below."),
            ("How it works", "The frame is cut into 14 × 14 patches of 16 × 16 pixels. Each patch becomes a token. "
                             "Twelve Transformer blocks let every patch look at every other patch. A class token collects the result. "
                             "A linear layer turns it into scores for the five families."),
            ("Training", "It is trained on the five view families. Weights start from ImageNet-21k. Data, sampling, augmentation, loss and "
                         "stopping rule are the same as for the ResNet-18 encoder. Only the network and its optimiser settings change."),
            ("Why", "It tests whether a Transformer is a better frame encoder than a CNN. "
                    "A view is defined by where the chambers and valves sit. Attention can relate distant parts of the image from the first layer."),
            ("Result", "It matches ResNet-18 on frame accuracy (0.953 vs 0.951) but has lower clip macro-F1 (0.935 vs 0.957). It is weaker on A5C. "
                       "D4 uses its features."),
        ]),
        _card("D2 · Factorised video ViT", "borrows STFM + EchoViewCLIP", diag(dg.factorised_vit()), [
            ("Input", "The model sees 16 frames covering 1.6 s, taken at 10 frames per second."),
            ("How it works", "A shared ViT-S, as in D1, turns each frame into one token. The 16 tokens get a time position. "
                             "A small temporal Transformer (2 layers) reads them and outputs one view for the clip. "
                             "A second head labels each frame on its own. It is used only in training."),
            ("On a stream", "A 1.6 s window slides every 0.5 s. Each moment takes the view of the window centred nearest to it."),
            ("Training", "It is trained on the nine codes. Weights start from ImageNet-21k. "
                         "Each epoch takes one random 1.6 s clip per video, balanced over views."),
            ("Why", "Some views differ by motion, not by one frame. A5C shows the outflow tract only while it opens. "
                    "Splitting space and time keeps it cheap, because each frame is encoded once and reused by every window. "
                    "The per-frame head (from STFM) keeps each frame token useful on its own. "
                    "EchoViewCLIP averages frame features. D2 lets them attend to each other."),
            ("Result", "Its nine-code accuracy is 0.928 ± 0.002 (3 runs). Its clip labels are good, but in streams it switched late or missed changes "
                       "(change F1 0.58 to 0.78). The fix below brings it to 0.975."),
        ]),
        _card("D3 · Multiscale video ViT", "borrows EchoPrime", diag(dg.mvit()), [
            ("Input", "It sees the same 16 frames as D2."),
            ("How it works", "The clip is cut into small space-time cubes (3 frames × 7 × 7 pixels). Attention runs over space and time together. "
                             "After each of four stages, the tokens are pooled. There are fewer tokens, but each has more channels. "
                             "A class token gives the view."),
            ("Two starts", "The weights come from Kinetics-400 (human action videos) or from EchoPrime's video encoder (about 12 million echo videos with their reports)."),
            ("Training", "It is trained on the nine codes, with the same clips and windows as D2. The whole network is fine-tuned."),
            ("Why", "Joint space-time attention sees where a structure is and how it moves in one step. "
                    "Comparing the two starts measures what echo pre-training is worth, the idea behind EchoPrime."),
            ("Result", "This is our best nine-code ViT. It reaches 0.935 ± 0.006 from Kinetics and 0.927 ± 0.001 from EchoPrime (2 runs each). "
                       "EchoPrime weights learn much faster but end lower. It has the same late-switch problem as D2 (change F1 0.73 to 0.77). "
                       "With the fix, Echo-MViT reaches 0.934 accuracy and change F1 0.938 (3 runs)."),
        ]),
        _card("D4 · ViT-Router", "borrows ASFormer + MS-TCN", diag(dg.router()), [
            ("Input", "It sees the whole stream at 10 frames per second."),
            ("How it works", "D1 turns each frame into a feature vector, and D1 stays frozen. Stage 1 has six attention blocks. "
                             "Each block looks only at nearby frames. The window doubles from block to block, from 0.5 s to about 13 s. "
                             "A small convolution in each block tells the model which frames are close. "
                             "Two more stages look only at the class probabilities of the stage before and clean them up. "
                             "The output is a view for every moment, plus a confidence to accept or defer each segment."),
            ("Training", "The same 1,500 training streams as MS-TCN and ASFormer (4 to 8 clips each, from the training split). "
                         "Each stage has a classification loss plus a smoothing loss that punishes jumpy labels. "
                         "It trains for 50 epochs and keeps the last one, as MS-TCN does."),
            ("Why", "Finding view changes is a segmentation problem over the whole stream. "
                    "Local attention suits labels that stay the same for a while (the idea from ASFormer). "
                    "Cleaning up probabilities with a smoothing loss removes short false cuts (the idea from MS-TCN). "
                    "A control runs the official MS-TCN on the same D1 features."),
            ("Result", "It reaches change F1 0.988 and the lowest routing error (0.003). Its recognition is limited by D1's features. "
                       "A5C recall is only 0.57."),
        ]),
    ]
    fix = ('<div class="card"><h3>Fixing D2 and D3 by asking each window about its centre<span class="tag">ours</span></h3>'
           + _facts([("Problem", "On a stream, the 1.6 s window slides along. Near a view change, one window holds two views. "
                                 "D2 and D3 then switch too late, or miss a short clip entirely (change F1 0.58 to 0.78)."),
                     ("Why", "They were trained only on single-view clips, so they never saw a window with two views. "
                             "On such a window a few frames of one view take over. The chart mixes PLAX frames into an A4C window. "
                             "With just 2 of 16 frames PLAX, our model's probability for A4C drops from 0.97 to 0.48 (solid line). "
                             "Averaging over frames, as EchoViewCLIP does, falls smoothly (dashed line).")])
           + diag(dg.window_problem())
           + _facts([("Fix", "In training, half of the clips are two clips joined at a random frame, as in the picture below. "
                             "The target is the view of the centre frame. On a stream each moment takes the answer of the window "
                             "centred on it, so this is exactly the question a window must answer.")])
           + diag(dg.centre_supervision())
           + _facts([("Tried, failed", "We first used the share of frames as the target (here 7/16 A and 9/16 B). "
                                       "The error flipped, and changes came too early instead of too late."),
                     ("Result", "After the fix, change F1 is 0.94 to 0.975. Echo-MViT keeps its accuracy (0.934 on nine codes).")])
           + "</div>")
    return ('<section id="designs"><h2>1 · Designs</h2><p class="lead">We built four Vision-Transformer designs. '
            "They differ in how much time they see. D1 sees one frame, D2 and D3 see a 1.6 s clip, and D4 sees the whole stream. "
            f'Each is trained the same way as the baseline it is compared with.</p>{table}'
            f'<div class="cards">{"".join(cards)}{fix}</div></section>')


def benchmarks_html() -> str:
    items = [
        "<b>Data.</b> We use EV9V (public, CC-BY-4.0) with its official split. Frames are sampled at 10 per second. "
        "Labels are as described above.",
        "<b>Nine-view test.</b> Each of the 888 test clips shows one view. The model names one of the nine views for each clip.",
        '<b>Recognition.</b> This test uses the <a href="#labels">five view families</a>. It has 800 test clips, each showing one view '
        "(the 88 PMPALA clips are left out). Each clip gets one label, the label most of its frames get. "
        "We also score every frame on its own, and count how often the label flips inside a clip.",
        "<b>Segmentation.</b> EV9V has no continuous recordings with several views, so we build test streams by joining test clips. "
        "There are 240 streams of two clips (same or different view, with or without a brightness change) and 120 streams of 4 to 8 clips. "
        "A view change counts as found if the model puts a change within 0.5 s of it.",
        "<b>Routing.</b> The model cuts a stream into segments and gives each one a confidence. "
        "A segment is routed if its confidence clears a threshold. Otherwise it is held back for a person. "
        "We set the threshold on validation so that at most 5% of the routed time has the wrong view, then keep it fixed for the test.",
        "<b>Same conditions.</b> Every row uses the same streams and the same evaluation code. "
        "The five baselines reproduce the numbers on the baselines page exactly. Checkpoints and thresholds are chosen on validation only.",
    ]
    metrics = [
        ("Test split", "Clips kept aside until the end and used once. Models learn from the train split. "
                       "Every choice, such as the checkpoint or the threshold, is made on the validation split."),
        ("Accuracy", "The share of test clips given the right view. The common views dominate it."),
        ("Recall", "For one view, the share of its clips that the model finds. "
                   "On the nine-view test MViT finds 35 of the 40 A5C clips, so its A5C recall is 0.875."),
        ("Precision", "For one label, the share of the clips given that label that are right. "
                      "The same MViT calls 47 clips A5C and 35 of them are A5C, so its precision is 0.745."),
        ("F1", "One number that is high only when precision and recall are both high. It is their harmonic mean."),
        ("Macro-F1", "The F1 of each view, averaged with equal weight. A rare view counts as much as a common one."),
        ("Balanced accuracy", "The recall of each view, averaged with equal weight."),
        ("Per clip, per frame", "A per-clip score gives each clip one label. A per-frame score judges every frame on its own."),
        ("Mean ± std", "The same model trained several times from different random starts. "
                       "The ± part shows how much the result moves by chance alone."),
        ("Change F1", "A predicted view change counts as correct if a real change lies within 0.5 s of it. "
                      "The score drops with missed changes and with false cuts."),
        ("Coverage", "The share of stream time the model routes by itself. The rest is held back for a person. Higher is better."),
        ("Actual error", "Of the routed time, the share that went to the wrong view. The target was at most 5%. Lower is better."),
    ]
    return ('<section id="benchmarks"><h2>2 · Benchmarks</h2><ul class="short">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
            f'<h3>Metrics explained</h3>{_facts(metrics, "wide")}</section>')


def results_html(met: dict, nine: dict) -> str:
    # nine-view table: one label per clip (clip models average their sliding windows over the clip)
    rows = [["STFM", ms(nine["baselines"]["b6"]["acc"]), ms(nine["baselines"]["b6"]["macro_f1"]), "3"],
            ["EchoViewCLIP", ms(nine["baselines"]["b7"]["acc"]), ms(nine["baselines"]["b7"]["macro_f1"]), "1"]]
    for m in ("vivit_fe", "vivit_fe_centre", "mvit_k400", "mvit_echoprime", "mvit_echoprime_centre"):
        runs = nine["vit"][m]
        rows.append([f"<b>{base.esc(NAMES[m])}</b>", ms([r["test_acc"] for r in runs]), ms([r["test_macro_f1"] for r in runs]), str(len(runs))])
    t9 = simple_table(["model", "accuracy", "macro-F1", "runs"], rows)
    nine_text = ("<p>Each of the 888 test clips shows one view. The model looks at the whole clip and gives it one of the nine view labels. "
                 "Accuracy is the share of clips labelled correctly. Macro-F1 gives the nine views equal weight, "
                 "so rare views count as much as common ones.</p>"
                 "<p>This is one label per clip, not one per frame. Our clip models slide their 1.6 s window along the clip, "
                 "average the window outputs and pick the most likely view. STFM and EchoViewCLIP are scored with their authors' own test code. "
                 "Accuracy frame by frame, on the five families, is in the next table.</p>")

    def g(m, *k):
        return base.g(met[m], *k)

    def row(m):
        c = g(m, "constructed", "cells") or {}
        missed = [c.get(k, {}).get("missed_rate") for k in ("diff_none", "diff_edit")]
        fs = [c.get(k, {}).get("false_split_rate") for k in ("same_none", "same_edit")]
        return [base.f(g(m, "native", "test", "cine", "macro_f1")), base.f(g(m, "native", "test", "cine", "balanced_accuracy")),
                base.f(g(m, "native", "test", "frame", "accuracy")), base.f(g(m, "native", "test", "stability", "fragments_per_minute_mean"), 1),
                base.f(g(m, "constructed", "boundary", "0.5", "f1")), base.f(g(m, "constructed_multi", "boundary", "0.5", "f1")),
                f"{100 * sum(missed) / 2:.0f}%", f"{100 * sum(fs) / 2:.1f}%",
                base.f(g(m, "constructed", "routing", "coverage")), base.f(g(m, "constructed", "routing", "risk"))]
    head = ["model", "clip\nmacro-F1", "balanced\naccuracy", "frame\naccuracy", "label flips\nper min", "change F1\n2 clips",
            "change F1\n4–8 clips", "missed\nchanges", "false\ncuts", "coverage", "actual\nerror"]
    sup = [("", "", 1), ("Recognition", "is the view right?", 4), ("Segmentation", "is the change found?", 4),
           ("Routing", "error target 5%", 2)]
    main = grouped_table(head, row, met, sup)
    guide = ('<ul class="short"><li>Rows are grouped by how much of the recording a model sees at once.</li>'
             "<li><b>Recognition</b> uses the 800 single-view test clips. Clip macro-F1 and balanced accuracy give each clip one label, "
             "the label most of its frames get, and weight every view equally. Frame accuracy scores every frame on its own. "
             "The label-flip rate counts how often the label changes inside one clip, per minute. 0 is best.</li>"
             "<li><b>Segmentation</b> uses the joined test streams. Change F1 is shown for the two-clip and for the 4–8-clip streams. "
             "Missed changes is the share of real view changes the model does not find. "
             "False cuts is the share of same-view joins it cuts anyway.</li>"
             "<li><b>Routing</b> uses the two-clip streams, with the threshold fixed on validation. "
             "Coverage is the share of stream time routed automatically. Actual error is the share of routed time with the wrong view.</li></ul>")
    return (f'<section id="results"><h2>3 · Results</h2><h3>Which of the nine views does each clip show?</h3>{nine_text}{t9}'
            f'<h3>Recognition, segmentation and routing on the five view families</h3>{guide}{main}'
            "<p>Our rows show the first training run (seed 0). Means over several runs are in the nine-view table above.</p></section>")


def per_view_html(bd: dict) -> str:
    fams = bd["families"]; M = bd["methods"]
    n = {f: M["b4"]["family"][f]["n"] for f in fams}

    def fam_row(m):
        return [heat(M[m]["family"][f]["recall"]) for f in fams]
    t1 = grouped_table(["model"] + [f"{f}\n{n[f]} clips" for f in fams], fam_row, M)
    nine = [m for m in ("b6", "b7", "vivit_fe", "vivit_fe_centre", "mvit_k400", "mvit_echoprime", "mvit_echoprime_centre") if "code9" in M[m]]
    codes = bd["codes"]; nc = {c: M[nine[0]]["code9"][c]["n"] for c in codes}
    body = ["<tr class=\"" + ("ours" if m in OURS else "") + f'"><td>{name_cell(m)}</td>' + "".join(
        "<td{1}>{0}</td>".format(*heat(M[m]["code9"][c]["recall"])) for c in codes) + "</tr>" for m in nine]
    t2 = ('<div class="tablewrap"><table class="metrics"><thead><tr>' + _th("model") + "".join(_th(f"{c}\n{nc[c]} clips") for c in codes)
          + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")
    return ('<section id="per-view"><h2>4 · Recognition per view</h2>'
            "<p>Each cell is the recall of one view on the 800 test clips, that is, the share of its clips that got the right label. "
            "Darker cells mean more errors.</p>"
            f'{t1}<ul class="short"><li>PLAX and PSAX are solved by every model.</li>'
            "<li>Models differ most on A5C, from 0.57 (ViT-Router) to 0.95 (Echo-MViT). A5C has only 40 test clips.</li>"
            "<li>A4C and A5C tend to trade off. A model that finds more A5C clips often mislabels more A4C clips.</li></ul>"
            f"<h3>Recall per code, for the models trained on the nine codes</h3>{t2}"
            '<ul class="short"><li>The hardest codes are the PSAX levels (PASA, PMASA, PPMLSA) and A5C.</li>'
            "<li>The dataset card and the STFM README give different names for several PSAX codes, so part of this error may be label ambiguity.</li></ul></section>")


def per_change_html(bd: dict) -> str:
    M = bd["methods"]; pairs = list(M["b4"]["pairs"])
    npair = {p: M["b4"]["pairs"][p]["n"] for p in pairs}
    t1 = grouped_table(["model"] + [f"{p}\n{npair[p]} changes" for p in pairs], lambda m: [heat(M[m]["pairs"][p]["recall"]) for p in pairs], M)
    fams = bd["families"]

    def spur(m):
        return [(f"{M[m]['spurious_per_min'][f]:.1f}", f' class="heat" style="--p:{min(45, M[m]["spurious_per_min"][f] * 4):.0f}%"') for f in fams]
    t2 = grouped_table(["model"] + [f"inside\n{f}" for f in fams], spur, M)
    return ('<section id="per-change"><h2>5 · Segmentation per view change</h2>'
            "<p>Each cell is the share of view changes found within 0.5 s, for one pair of views. "
            "It uses the 120 streams of 4 to 8 clips (379 changes), because only they contain every pair.</p>"
            f"{t1}<h3>False cuts per minute inside one view</h3>"
            "<p>A false cut is a predicted change more than 0.5 s away from any real change. Darker cells mean more false cuts.</p>"
            f'{t2}<ul class="short"><li>A4C ↔ A5C is the hardest change. The temporal models find only 52% to 73% of these changes.</li>'
            "<li>Frame models score high there only because they cut everywhere. They make up to 86 false cuts per minute inside A5C.</li>"
            "<li>Most remaining false cuts of the good models fall inside A4C and A5C.</li></ul></section>")


def streams_html(streams: dict) -> str:
    by = {s["id"]: s for s in streams["streams"]}
    sel = [by[k] for k in SELECTED if k in by]
    rest = [s for s in streams["streams"] if s["id"] not in SELECTED]
    keys = ('<div class="keys">' + "".join(f'<span><i class="sw" style="background:var(--s{i + 1})"></i>{c}</span>' for i, c in enumerate(base.CLASSES))
            + '<span><i class="sw hatch"></i>deferred</span><span><i class="ln dash"></i>clip join</span><span><i class="ln"></i>true view change</span></div>')
    return (f'<section id="streams"><h2>6 · Streams</h2><p>Test streams, played frame by frame. Rows show each model\'s routed segments. '
            f'These are examples. The numbers above come from the full test set.</p>{keys}'
            f'{base._panels(sel, STREAM_ROWS, STREAM_NAMES, SELECTED)}'
            f'<details class="more"><summary>Show the other {len(rest)} streams</summary>{base._panels(rest, STREAM_ROWS, STREAM_NAMES)}</details></section>')


def conclusions_html() -> str:
    changes = [
        ("ViT-S instead of ResNet-18 as the frame encoder", "D1",
         "Clip macro-F1 fell from 0.957 to 0.935. MS-TCN on its features fell from 0.956 to 0.927.", "no"),
        ("Frame embeddings as the Transformer input", "D2, D4", "Each frame is encoded once and cached. This lets D4 read a whole stream.", "yes"),
        ("Temporal attention over a 1.6 s clip", "D2, D3",
         "Label flips fell from 25 to 1–2 per minute, compared with D1. Nine-code accuracy is 0.928 to 0.935, close to STFM (0.938).", "yes"),
        ("A sliding window read out by a class token", "D2, D3", "Changes came late or were missed. Change F1 was 0.58 to 0.78.", "no"),
        ("Mixed clips, with the share of frames as the target", "D2", "The bias flipped, and change F1 was 0.576.", "no"),
        ("Mixed clips, with the centre view as the target", "D2, D3", "Change F1 rose to 0.94 to 0.975. Echo-MViT kept its accuracy.", "yes"),
        ("Echo-specific pre-training (EchoPrime)", "D3", "Training was faster, but the final accuracy was lower (0.927 vs 0.935).", "no"),
        ("Local attention over the whole stream", "D4", "Change F1 reached 0.988, with the lowest routing error (0.003).", "yes"),
    ]
    cls = {"yes": ("yes", "helped"), "no": ("no", "did not help"), "mixed": ("mixed", "mixed")}
    t = ('<div class="tablewrap"><table class="glance"><thead><tr><th>change</th><th>where</th><th>evidence</th><th>verdict</th></tr></thead><tbody>'
         + "".join(f'<tr><td>{base.esc(a)}</td><td>{b}</td><td>{base.esc(c)}</td><td class="verdict {cls[v][0]}">{cls[v][1]}</td></tr>' for a, b, c, v in changes)
         + "</tbody></table></div>")
    concl = [
        ("A plain ViT is not a better frame encoder than ResNet-18 at this data size.", "With about 3,400 training clips, the built-in locality of a CNN still pays off."),
        ("The best video ViTs match STFM on nine codes, but do not beat EchoViewCLIP.", "EchoViewCLIP is itself a ViT (CLIP ViT-B/16). Its edge is web-scale image–text pre-training."),
        ("For routing, how a window is read out matters more than the backbone.", "Train a window the way it is used, by asking for the view at its centre."),
        ("As a temporal model, a Transformer works well.", "ViT-Router segments as well as MS-TCN and ASFormer and routes with the lowest error."),
        ("The remaining errors sit in A5C and the PSAX levels.", "A5C differs from A4C only while the outflow tract is open. The PSAX level names are not agreed between sources."),
    ]
    ol = '<ol class="concl">' + "".join(f"<li>{base.esc(a)} <span>{base.esc(b)}</span></li>" for a, b in concl) + "</ol>"
    inference = ("At EV9V's size, recognition is limited by the frame representation and by label ambiguity, not by temporal modelling. "
                 "Transformers help as temporal models when their training target matches how they are used. "
                 "A natural next step is a CLIP-initialised frame encoder under ViT-Router.")
    return (f'<section id="conclusions"><h2>7 · Conclusions</h2><h3>What we changed</h3>{t}<h3>What it means</h3>{ol}'
            f'<h3>Inference</h3><p class="lead">{base.esc(inference)}</p></section>')


def build(out: Path) -> Path:
    met = load_metrics()
    nine = json.loads((RES / "nine_code.json").read_text()); bd = json.loads((RES / "breakdowns.json").read_text())
    streams = json.loads((REPO / "runs" / "_demo" / "streams" / "streams.json").read_text())
    nav = ('<nav class="side" aria-label="sections"><a href="index.html">← Five baselines</a><span class="navsep"></span>'
           + "".join(f'<a href="#{i}">{t}</a>' for i, t in [("labels", "Labels"), ("designs", "Designs"), ("benchmarks", "Benchmarks"), ("results", "Results"),
                                                           ("per-view", "Per view"), ("per-change", "Per change"), ("streams", "Streams"),
                                                           ("conclusions", "Conclusions")]) + "</nav>")
    page = f"""<title>Vision Transformers on EV9V</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{base.CSS}{EXTRA_CSS}</style>
{nav}
<div class="wrap">
<header id="top"><h1>Vision Transformers on EV9V</h1>
<p class="lead">We built four Transformer designs for echo view routing. We tested them against five published baselines and a plain ResNet-18, on the same data and with the same code.</p>
<p>{base.ROUTING_SENTENCE}</p>
<p><b>In short</b>, our Transformers do not recognise views better than CNNs. They do well at finding where the view changes, once they are trained the way they are used.</p></header>
{labels_html()}
{designs_html()}
{benchmarks_html()}
{results_html(met, nine)}
{per_view_html(bd)}
{per_change_html(bd)}
{streams_html(streams)}
{conclusions_html()}
<footer>ladder run {LADDER} · EV9V (CC-BY-4.0) · <a href="https://github.com/WilliamQiuzy/echo-view-routing">github.com/WilliamQiuzy/echo-view-routing</a></footer>
</div>
<script>{base.JS}</script>
"""
    out.write_text(page)
    return out


if __name__ == "__main__":
    print(build(HERE / "vit.html"))
