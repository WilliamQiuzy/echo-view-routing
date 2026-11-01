"""Offline OCR-proposed nuisance regions; no anatomical or view targets."""
import csv
import io
import math
import numpy as np
from PIL import Image


def boxes_from_tsv(text):
    boxes = []
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        try:
            confidence = float(row["conf"])
            token = row.get("text", "").strip()
            if confidence < 60 or sum(c.isalnum() for c in token) < 2:
                continue
            box = {k: int(row[k]) for k in ("left", "top", "width", "height")}
            if box["width"] <= 0 or box["height"] <= 0:
                continue
            boxes.append({**box, "text": token, "confidence": confidence})
        except (ValueError, KeyError, TypeError):
            continue
    return boxes


def native_to_input_mask(shape, boxes):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for b in boxes:
        x0, y0 = max(0, b["left"] - 2), max(0, b["top"] - 2)
        x1, y1 = min(w, b["left"] + b["width"] + 2), min(h, b["top"] + b["height"] + 2)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    s = 256 / min(w, h)
    rw, rh = max(224, round(w * s)), max(224, round(h * s))
    left, top = (rw - 224) // 2, (rh - 224) // 2
    im = Image.fromarray(mask).resize((rw, rh), Image.Resampling.NEAREST)
    return np.asarray(im.crop((left, top, left + 224, top + 224))) > 0


def scaled_to_input_mask(boxes):
    mask = np.zeros((224, 224), bool)
    for b in boxes:
        x0, y0 = max(0, math.floor(b["left"] / 3) - 1), max(0, math.floor(b["top"] / 3) - 1)
        x1, y1 = min(224, math.ceil((b["left"] + b["width"]) / 3) + 1), min(224, math.ceil((b["top"] + b["height"]) / 3) + 1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask
