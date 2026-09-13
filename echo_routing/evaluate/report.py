"""Render the baseline ladder as Markdown/CSV from metrics dicts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd


def _g(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def _fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def ladder_rows(metrics: Sequence[dict]) -> list[dict]:
    rows = []
    for m in metrics:
        if m["method_id"] == "b_file":
            continue
        cells = _g(m, "constructed", "cells", default={})
        rows.append({
            "method": m["method_id"],
            "native_cine_macroF1": _g(m, "native", "test", "cine", "macro_f1"),
            "native_cine_balAcc": _g(m, "native", "test", "cine", "balanced_accuracy"),
            "native_frame_acc": _g(m, "native", "test", "frame", "accuracy"),
            "frag_per_min": _g(m, "native", "test", "stability", "fragments_per_minute_mean"),
            "share_fragmented": _g(m, "native", "test", "stability", "share_fragmented"),
            "boundaryF1@0.5s": _g(m, "constructed", "boundary", "0.5", "f1"),
            "falseSplit_same_none": _g(cells, "same_none", "false_split_rate"),
            "falseSplit_same_edit": _g(cells, "same_edit", "false_split_rate"),
            "missed_diff_none": _g(cells, "diff_none", "missed_rate"),
            "missed_diff_edit": _g(cells, "diff_edit", "missed_rate"),
            "tau@5%": _g(m, "policy", "by_target", "0.05", "tau"),
            "coverage@5%": _g(m, "constructed", "routing", "coverage"),
            "risk@5%": _g(m, "constructed", "routing", "risk"),
        })
    return rows


def bfile_rows(metrics: Sequence[dict]) -> list[dict]:
    out = []
    for m in metrics:
        if m["method_id"] != "b_file":
            continue
        for t, v in _g(m, "policy", "by_target", default={}).items():
            out.append({"target_risk": t, "tau": v.get("tau"), "status": v.get("status"),
                        "val_coverage": _g(v, "validation", "coverage"), "val_risk": _g(v, "validation", "risk"),
                        "test_coverage": _g(v, "test", "coverage"), "test_risk": _g(v, "test", "risk"),
                        "test_cine_macroF1": _g(m, "native", "test", "cine", "macro_f1")})
    return out


def to_markdown(rows: list[dict], title: str) -> str:
    if not rows:
        return f"### {title}\n\n_(no rows)_\n"
    cols = list(rows[0].keys())
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_ladder(metrics: Sequence[dict], out_dir: Path, header: str = "") -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows, brows = ladder_rows(metrics), bfile_rows(metrics)
    md = (header + "\n\n" if header else "") + to_markdown(rows, "Temporal routing ladder (test, frozen validation-selected policy)") + "\n" + \
        to_markdown(brows, "B-file operational comparator (native files; known file boundaries)")
    (out_dir / "ladder.md").write_text(md)
    pd.DataFrame(rows).to_csv(out_dir / "ladder.csv", index=False)
    pd.DataFrame(brows).to_csv(out_dir / "b_file.csv", index=False)
    return out_dir / "ladder.md"


def load_metrics_dir(metrics_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(metrics_dir).glob("*.json"))]
