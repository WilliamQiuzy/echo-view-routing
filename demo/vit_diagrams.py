"""Inline SVG pipeline diagrams for the ViT-family page: boxes, arrows, one muted accent, theme-aware colours.

Colours come from the page's CSS variables through `style` attributes (presentation attributes cannot use var()),
so the diagrams follow light/dark mode. View colours (--s1..--s5) are deliberately not used here.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

FONT = 'font-family:"IBM Plex Sans",system-ui,sans-serif'


@dataclass(frozen=True)
class Box:
    key: str
    x: float
    y: float
    w: float
    h: float
    title: str
    sub: str = ""
    kind: str = "plain"      # plain | accent | io | ghost


@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    side: str = "h"          # h: right -> left, v: bottom -> top, vu: top -> bottom of b from below
    dashed: bool = False
    label: str = ""


STYLE = {
    "plain": "fill:var(--surface);stroke:var(--line);stroke-width:1.2",
    "accent": "fill:color-mix(in srgb,var(--accent) 12%,var(--surface));stroke:var(--accent);stroke-width:1.6",
    "io": "fill:var(--line-2);stroke:var(--line);stroke-width:1.2",
    "ghost": "fill:none;stroke:var(--muted);stroke-width:1.1;stroke-dasharray:4 3",
}


def _text(x: float, y: float, s: str, size: float, color: str, weight: int = 400, anchor: str = "middle") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="{FONT};font-size:{size}px;font-weight:{weight};'
            f'fill:var({color})">{html.escape(s)}</text>')


def _box(b: Box) -> str:
    out = [f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="6" style="{STYLE[b.kind]}"/>']
    titles = b.title.split("\n"); subs = b.sub.split("\n") if b.sub else []
    total = 16 * len(titles) + 14 * len(subs)
    y = b.y + (b.h - total) / 2 + 12
    for t in titles:
        out.append(_text(b.x + b.w / 2, y, t, 13, "--ink" if b.kind != "ghost" else "--ink-2", 600)); y += 16
    for t in subs:
        out.append(_text(b.x + b.w / 2, y + 1, t, 11.5, "--ink-2" if b.kind == "accent" else "--muted")); y += 14
    return "".join(out)


def _edge(e: Edge, boxes: dict[str, Box], mid: str) -> str:
    a, b = boxes[e.a], boxes[e.b]
    if e.side == "h":
        x1, y1, x2, y2 = a.x + a.w, a.y + a.h / 2, b.x - 3, b.y + b.h / 2
        d = f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" if abs(y1 - y2) < 1 else \
            f"M{x1:.1f},{y1:.1f} C{(x1 + x2) / 2:.1f},{y1:.1f} {(x1 + x2) / 2:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
    else:
        x1, y1 = a.x + a.w / 2, a.y + a.h
        x2, y2 = b.x + b.w / 2, b.y - 3
        if e.side == "vu":
            x1, y1, x2, y2 = a.x + a.w / 2, a.y - 1, b.x + b.w / 2, b.y + b.h + 3
        d = f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" if abs(x1 - x2) < 1 else \
            f"M{x1:.1f},{y1:.1f} C{x1:.1f},{(y1 + y2) / 2:.1f} {x2:.1f},{(y1 + y2) / 2:.1f} {x2:.1f},{y2:.1f}"
    dash = ";stroke-dasharray:4 3" if e.dashed else ""
    out = f'<path d="{d}" style="fill:none;stroke:var(--ink-2);stroke-width:1.4{dash}" marker-end="url(#{mid})"/>'
    if e.label:
        lx, ly = ((x1 + x2) / 2, (y1 + y2) / 2 - 6) if e.side == "h" else ((x1 + x2) / 2 + 8, (y1 + y2) / 2)
        out += _text(lx, ly, e.label, 10.5, "--muted", anchor="middle" if e.side == "h" else "start")
    return out


def svg(uid: str, width: int, height: int, boxes: list[Box], edges: list[Edge], extra: str = "", label: str = "") -> str:
    by = {b.key: b for b in boxes}
    mid = f"arrow-{uid}"
    marker = (f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
              f'<path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink-2)"/></marker></defs>')
    body = "".join(_edge(e, by, mid) for e in edges) + "".join(_box(b) for b in boxes) + extra
    return (f'<svg class="diagram" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
            f'aria-label="{html.escape(label)}" xmlns="http://www.w3.org/2000/svg">{marker}{body}</svg>')


def _row(keys_titles: list[tuple[str, str, str, str]], y: float, h: float, x0: float = 8, gap: float = 24, width: float = 884) -> list[Box]:
    n = len(keys_titles); w = (width - gap * (n - 1)) / n
    return [Box(k, x0 + i * (w + gap), y, w, h, t, s, kind) for i, (k, t, s, kind) in enumerate(keys_titles)]


def plain_vit() -> str:
    boxes = _row([("in", "One frame", "224 × 224\nletterboxed", "io"), ("patch", "Patches", "14 × 14 tokens\nof 16 × 16 px", "plain"),
                  ("vit", "ViT-S/16", "12 blocks\nImageNet-21k", "accent"), ("cls", "CLS token", "384-d", "plain"),
                  ("head", "Linear head", "5 families", "plain"), ("out", "View", "for this frame", "io")], 14, 70)
    edges = [Edge(a.key, b.key) for a, b in zip(boxes, boxes[1:])]
    return svg("d1", 900, 100, boxes, edges, label="Plain ViT, one frame to a view")


def factorised_vit() -> str:
    boxes = _row([("in", "16 frames", "1.6 s at 10 Hz", "io"), ("vit", "ViT-S/16", "shared, one frame\nat a time", "plain"),
                  ("tok", "16 frame tokens", "+ time position", "plain"), ("temp", "Temporal\nTransformer", "2 layers, clip token", "accent"),
                  ("out", "View", "for the clip", "io")], 14, 70)
    tok = boxes[2]
    aux = Box("aux", tok.x, 124, tok.w, 46, "Frame head", "training only", "ghost")
    edges = [Edge(a.key, b.key) for a, b in zip(boxes, boxes[1:])] + [Edge("tok", "aux", "v", dashed=True)]
    return svg("d2", 900, 182, boxes + [aux], edges, label="Factorised video ViT")


def mvit() -> str:
    boxes = _row([("in", "16 frames", "1.6 s at 10 Hz", "io"), ("cube", "Space-time cubes", "3 × 7 × 7 patches", "plain"),
                  ("stages", "4 stages", "fewer tokens,\nwider channels", "accent"), ("cls", "CLS token", "768-d", "plain"),
                  ("out", "View", "for the clip", "io")], 92, 70)
    st = boxes[2]
    init = Box("init", st.x - 60, 6, st.w + 120, 60, "Initial weights", "Kinetics-400 (actions) or\nEchoPrime (12 M echo videos)", "ghost")
    edges = [Edge(a.key, b.key) for a, b in zip(boxes, boxes[1:])] + [Edge("init", "stages", "v", dashed=True)]
    return svg("d3", 900, 176, boxes + [init], edges, label="Multiscale video ViT (MViTv2-S)")


def router() -> str:
    boxes = _row([("in", "Whole stream", "10 samples / s", "io"), ("vit", "ViT-S/16", "per frame, frozen\n(design 1)", "plain"),
                  ("seq", "Feature sequence", "384-d per sample", "plain"), ("s1", "Local attention", "6 blocks, window\n0.5 s → 13 s", "accent"),
                  ("ref", "Refine × 2", "on class\nprobabilities", "accent"), ("out", "View per sample", "+ accept / defer", "io")], 14, 76)
    edges = [Edge(a.key, b.key) for a, b in zip(boxes, boxes[1:])]
    return svg("d4", 900, 104, boxes, edges, label="ViT-Router")


def centre_supervision() -> str:
    """A mixed training clip and the two targets we tried."""
    cells, x0, y0, cw, ch, k = [], 118, 34, 30, 30, 7
    for i in range(16):
        style = STYLE["plain"] if i < k else STYLE["accent"]
        cells.append(f'<rect x="{x0 + i * cw}" y="{y0}" width="{cw - 3}" height="{ch}" rx="3" style="{style}"/>')
    cx = x0 + 8 * cw + (cw - 3) / 2
    parts = cells + [
        _text(x0 + k * cw / 2, y0 - 10, "clip A · view A", 11, "--ink-2"),
        _text(x0 + k * cw + (16 - k) * cw / 2, y0 - 10, "clip B · view B", 11, "--ink-2"),
        _text(12, y0 + 20, "Training clip", 12, "--ink", 600, "start"),
        f'<path d="M{cx:.1f},{y0 - 2} L{cx:.1f},{y0 + ch + 12}" style="stroke:var(--ink);stroke-width:1.6"/>',
        _text(cx, y0 + ch + 25, "centre frame", 11, "--ink", 600),
        _text(118, 118, "✓  Target is the view at the centre frame (B). This worked.", 12, "--ink", 500, "start"),
        _text(118, 142, "✗  Target is the share of frames (7/16 A, 9/16 B). This did not.", 12, "--ink-2", 400, "start"),
    ]
    return (f'<svg class="diagram" viewBox="0 0 760 156" style="max-width:760px" role="img" aria-label="Centre-supervised mixed clips" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>')


def window_problem() -> str:
    """Why class-token read-out breaks sliding-window routing: probe result as a small line chart."""
    xs = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    cls = [0.97, 0.48, 0.23, 0.09, 0.06, 0.05, 0.01, 0.00, 0.00]
    mean = [0.68, 0.59, 0.50, 0.40, 0.32, 0.24, 0.15, 0.07, 0.00]
    L, T, W, H = 60, 16, 300, 150

    def pt(i, v):
        return L + xs[i] / 16 * W, T + (1 - v) * H
    def line(vals, style):
        return '<path d="' + " ".join(("M" if i == 0 else "L") + f"{pt(i, v)[0]:.1f},{pt(i, v)[1]:.1f}" for i, v in enumerate(vals)) + f'" style="fill:none;{style}"/>'
    grid = "".join(f'<path d="M{L},{T + (1 - g) * H:.1f} L{L + W},{T + (1 - g) * H:.1f}" style="stroke:var(--line-2);stroke-width:1"/>'
                   + _text(L - 8, T + (1 - g) * H + 4, f"{g:.1f}", 10.5, "--muted", anchor="end") for g in (0, 0.5, 1.0))
    ticks = "".join(_text(L + x / 16 * W, T + H + 16, str(x), 10.5, "--muted") for x in (0, 4, 8, 12, 16))
    parts = [grid, ticks,
             line(cls, "stroke:var(--accent);stroke-width:2.2"), line(mean, "stroke:var(--ink-2);stroke-width:1.6;stroke-dasharray:5 4"),
             _text(L + W / 2, T + H + 34, "PLAX frames mixed into a 16-frame A4C window", 11, "--ink-2"),
             _text(18, T + H / 2, "P(A4C)", 11, "--ink-2", anchor="middle").replace("<text", f'<text transform="rotate(-90 18 {T + H / 2})"'),
             _text(L + W + 20, T + 30, "our clip ViTs, before the fix", 11.5, "--ink", 600, "start"),
             _text(L + W + 20, T + 46, "(class-token read-out)", 11, "--muted", 400, "start"),
             _text(L + W + 20, T + 96, "averaging over frames", 11.5, "--ink", 600, "start"),
             _text(L + W + 20, T + 112, "(linear pooling, like EchoViewCLIP)", 11, "--muted", 400, "start"),
             f'<path d="M{L + W + 6},{T + 26} l10,0" style="stroke:var(--accent);stroke-width:2.2"/>',
             f'<path d="M{L + W + 6},{T + 92} l10,0" style="stroke:var(--ink-2);stroke-width:1.6;stroke-dasharray:5 4"/>']
    return (f'<svg class="diagram" viewBox="0 0 640 214" style="max-width:640px" role="img" aria-label="Probe of class-token read-out on mixed windows" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>')
