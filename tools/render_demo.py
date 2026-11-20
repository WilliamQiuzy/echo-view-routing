#!/usr/bin/env python3
"""Renders a stream with the system's answer under it: the frames the model reads, and a timeline of the views.

  python tools/render_demo.py clip1.mp4 clip2.mp4 clip3.mp4 --out demo.webp [--reference A4C,A5C,PLAX] [--seconds 3] [--uncertain]

The inputs are joined in the given order, as `echoview segment` joins them. The picture shows the 224 x 224 grey
frames at 10 frames per second, enlarged. Under it, one lane holds the view the system gives every 0.1 s (hatched
where it declines), a second lane the reference views if `--reference` names one per input. `--seconds` takes only
the first seconds of every input. `.webp` gives the smallest file; `.gif` and `.webp` are written with Pillow, `.mp4`
with OpenCV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEWS = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
COLOURS = ((42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164))
PAPER, INK, QUIET, STRIPE = (252, 252, 251), (11, 11, 11), (119, 117, 111), (176, 174, 166)
LANE, THIN, PAD, AXIS = 30, 12, 12, 16


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow before 10.1 has one size only
        return ImageFont.load_default()


def lane(draw: ImageDraw.ImageDraw, labels: np.ndarray, left: int, top: int, width: int, height: int, names: bool) -> None:
    """One coloured block per run of equal labels; -1 (uncertain) is hatched."""
    n = len(labels)
    edges = [0, *(np.flatnonzero(np.diff(labels)) + 1).tolist(), n]
    for a, b in zip(edges[:-1], edges[1:]):
        x0, x1 = left + round(a / n * width), left + round(b / n * width) - 1
        view = int(labels[a])
        if view < 0:
            draw.rectangle([x0, top, x1, top + height - 1], fill=(236, 235, 230))
            for x in range(x0 - height, x1, 6):
                draw.line([(max(x0, x), top + height - 1 - max(0, x0 - x)), (min(x1, x + height), top + max(0, x + height - x1))], fill=STRIPE, width=1)
            continue
        draw.rectangle([x0, top, x1, top + height - 1], fill=COLOURS[view])
        if names and x1 - x0 > 44:
            draw.text((x0 + 6, top + height // 2), VIEWS[view], fill=(255, 255, 255) if view != 3 else INK, font=font(14), anchor="lm")


def render(frames: np.ndarray, labels: np.ndarray, out: str | Path, reference: np.ndarray | None = None, scale: int = 2, fps: int = 10) -> Path:
    """Writes the animation. `frames` is (n, h, w, 3) uint8 at 10 frames per second, `labels` the view index of every
    frame (-1 for uncertain), `reference` the reference view of every frame or None."""
    frames, labels = np.asarray(frames), np.asarray(labels, int)
    n, h, w = frames.shape[:3]
    if labels.shape != (n,) or (reference is not None and len(reference) != n):
        raise ValueError("one label per frame is needed")
    width, video_h = w * scale, h * scale
    lanes_top = video_h + PAD
    height = lanes_top + LANE + (4 + THIN if reference is not None else 0) + AXIS + PAD + 22
    inner = width - 2 * PAD
    base = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(base)
    lane(draw, labels, PAD, lanes_top, inner, LANE, names=True)
    bottom = lanes_top + LANE
    if reference is not None:
        lane(draw, np.asarray(reference, int), PAD, bottom + 4, inner, THIN, names=False)
        bottom += 4 + THIN
    seconds = n / 10.0
    for second in range(0, int(seconds) + 1, max(1, int(seconds // 6) or 1)):
        draw.text((PAD + round(second / seconds * inner), bottom + 3), f"{second} s", fill=QUIET, font=font(11), anchor="ma")
    x = PAD
    for view, colour in zip((*VIEWS, "uncertain"), (*COLOURS, None)):  # the legend
        if colour is None and not (labels < 0).any():
            continue
        if colour is None:
            lane(draw, np.array([-1]), x, height - PAD - 10, 10, 10, names=False)
        else:
            draw.rectangle([x, height - PAD - 10, x + 9, height - PAD - 1], fill=colour)
        draw.text((x + 14, height - PAD - 5), view, fill=INK, font=font(11), anchor="lm")
        x += 14 + int(draw.textlength(view, font=font(11))) + 14
    pictures = []
    for i in range(n):
        picture = base.copy()
        picture.paste(Image.fromarray(frames[i]).resize((width, video_h), Image.NEAREST if scale <= 2 else Image.BILINEAR), (0, 0))
        pen = ImageDraw.Draw(picture)
        px = PAD + round((i + 0.5) / n * inner)
        pen.line([(px, lanes_top - 4), (px, bottom + 1)], fill=INK, width=2)
        name = VIEWS[labels[i]] if labels[i] >= 0 else "uncertain"
        pen.rectangle([PAD, PAD, PAD + 20 + int(pen.textlength(name, font=font(22))), PAD + 34], fill=(0, 0, 0))
        pen.text((PAD + 10, PAD + 17), name, fill=(255, 255, 255), font=font(22), anchor="lm")
        pictures.append(picture)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp4":
        import cv2

        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for picture in pictures:
            writer.write(np.asarray(picture)[:, :, ::-1])
        writer.release()
    else:
        if out.suffix.lower() == ".gif":  # a small palette keeps the file small; the grey picture loses little by it
            pictures = [p.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for p in pictures]
        more = {"quality": 60, "method": 4} if out.suffix.lower() == ".webp" else {"optimize": True}
        pictures[0].save(out, save_all=True, append_images=pictures[1:], duration=round(1000 / fps), loop=0, **more)
    return out


def main(argv: list[str] | None = None) -> int:
    import echoview
    from echoview import reference, uncertain

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default="", help="the reference view of every input, comma-separated, for a second lane")
    parser.add_argument("--uncertain", action="store_true", help="let the system decline, with the released rule")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=0.0, help="use only the first seconds of every input (0: all of it)")
    args = parser.parse_args(argv)
    recordings = [echoview.read_recording(path) for path in args.inputs]
    keep = round(args.seconds * 10) if args.seconds > 0 else None
    pieces = [r.frames[:keep] for r in recordings]
    frames = np.concatenate(pieces)
    rule = None
    if args.uncertain:
        if getattr(reference, "UNCERTAIN", None) is None:
            raise SystemExit("this version has no released rule for the uncertain output")
        rule = uncertain.Rule(**reference.UNCERTAIN)
    result = echoview.load(args.weights, device=args.device).segment(frames, rule)
    truth = None
    if args.reference:
        names = args.reference.split(",")
        if len(names) != len(recordings) or any(name not in VIEWS for name in names):
            raise SystemExit(f"--reference needs one of {', '.join(VIEWS)} for each of the {len(recordings)} inputs")
        truth = np.concatenate([np.full(len(piece), VIEWS.index(name)) for piece, name in zip(pieces, names)])
    out = render(frames, result.labels, args.out, truth, args.scale)
    for start, end, view, probability in result.segments:
        print(f"  {start:6.1f} - {end:6.1f} s  {view:9s} {probability:.2f}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
