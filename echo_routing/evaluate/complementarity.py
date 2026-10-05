"""Complementarity of two frozen recognition systems and one fixed equal-weight fusion (EV9V, 2026-10-05 plan, tasks
1-2; configs/experiments/ev9v_complementarity_v1.yaml). Pure functions; scripts/ev9v_complementarity.py does the I/O.

- Alignment. Both teachers must describe the same samples: per 10 Hz sample the same clip, actual frame index and
  contrast variant (E0's cached sample against B7's window-plan reference), and the same number of samples.
- Classes. B7's nine codes are summed into the five families with the frozen map (PMPALA dropped, as in
  scripts/run_windowed_echoviewclip.py). The five mapped columns are then divided by their sum.
- Fusion. q = w_E0 * p_E0 + w_B7 * p_B7 per sample, with fixed weights.
- Selection. On the calibration half, the settings that meet both constraints compete on segment F1@50; an equal score
  goes to the earlier setting. With no such setting the system is infeasible, and there is no fallback.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from echo_routing.compose.recipe_builder import Recipe
from echo_routing.features.cache import VideoFeatures

CELLS = ("both right", "only E0 right", "only B7 right", "both wrong")
SampleRef = tuple[str, int, str]          # (clip, actual frame index, contrast variant)


class AlignmentError(ValueError):
    """The two teachers' outputs do not describe the same samples."""


class InputError(ValueError):
    """An input breaks the round's fixed rules (hash, test split, forbidden source, malformed probabilities)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files_digest(root: Path, rel_paths: Iterable[str]) -> str:
    """sha256 over the sorted lines '<relative path>\\t<sha256 of the file>'."""
    lines = "".join(f"{p}\t{sha256_file(Path(root) / p)}\n" for p in sorted(set(rel_paths)))
    return hashlib.sha256(lines.encode()).hexdigest()


def check_sources(paths: Iterable[str], forbidden: Sequence[str]) -> None:
    """No input path may name a forbidden source (the test split, EchoNet-Pediatric products)."""
    for p in paths:
        hits = [f for f in forbidden if f in str(p).lower()]
        if hits:
            raise InputError(f"input {p} matches forbidden {hits}")


def check_no_test(clip_ids: Iterable[str], test_ids: set[str]) -> None:
    bad = sorted(set(clip_ids) & set(test_ids))
    if bad:
        raise InputError(f"{len(bad)} test clips among the inputs, e.g. {bad[:3]}")


def recipe_refs(recipe: Recipe, loader: Callable[[str, str], VideoFeatures]) -> list[SampleRef]:
    """E0's sample references of a rendered stream, in stream order."""
    out: list[SampleRef] = []
    for fr in recipe.fragments:
        vf = loader(fr.video_id, fr.variant)
        out += [(fr.video_id, int(vf.frame_idx[s]), fr.variant) for s in range(fr.start, fr.end)]
    return out


def native_refs(vf: VideoFeatures, variant: str = "orig") -> list[SampleRef]:
    return [(vf.video_id, int(f), variant) for f in vf.frame_idx]


def check_alignment(e0_refs: Sequence[SampleRef], b7_refs: Sequence[Sequence], gamma_of: Mapping[str, float], seq_id: str = "") -> None:
    """B7 plan references are (clip, frame index, gamma); the variant's gamma must match."""
    if len(e0_refs) != len(b7_refs):
        raise AlignmentError(f"{seq_id}: {len(e0_refs)} E0 samples against {len(b7_refs)} B7 samples")
    for i, ((v, f, var), (bv, bf, g)) in enumerate(zip(e0_refs, b7_refs)):
        if v != bv or int(f) != int(bf) or abs(float(gamma_of[var]) - float(g)) > 1e-9:
            raise AlignmentError(f"{seq_id} sample {i}: E0 ({v}, frame {f}, {var}) against B7 ({bv}, frame {bf}, gamma {g})")


def mapping_matrix(raw_order: Sequence[str], raw_to_family: Mapping[str, str | None], family: Sequence[str]) -> np.ndarray:
    m = np.zeros((len(raw_order), len(family)))
    for j, name in enumerate(raw_order):
        if raw_to_family[name] is not None:
            m[j, list(family).index(raw_to_family[name])] = 1.0
    return m


def map_b7(raw: np.ndarray, matrix: np.ndarray, stored: np.ndarray | None = None, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Five-family rows of B7's nine-code softmax and the dropped mass per sample. `stored`, the cached five-family
    rows, must equal the frozen map of `raw`."""
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or raw.shape[1] != matrix.shape[0] or not np.isfinite(raw).all() or (raw < 0).any():
        raise InputError(f"raw B7 output must be finite, non-negative and (n, {matrix.shape[0]}), got {raw.shape}")
    mapped = raw @ matrix
    if stored is not None and (np.shape(stored) != mapped.shape or np.abs(mapped - np.asarray(stored, float)).max() > tol):
        raise InputError("cached five-family rows differ from the frozen map of the raw output")
    return mapped, raw.sum(1) - mapped.sum(1)


def normalise_rows(p: np.ndarray, min_mass: float) -> np.ndarray:
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or (p < 0).any():
        raise InputError("probabilities must be finite and non-negative")
    mass = p.sum(1)
    if (mass < min_mass).any():
        raise InputError(f"{int((mass < min_mass).sum())} samples keep less than {min_mass} of their mass after the map")
    return p / mass[:, None]


def check_distribution(p: np.ndarray, tol: float) -> None:
    p = np.asarray(p, float)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any() or np.abs(p.sum(1) - 1.0).max() > tol:
        raise InputError("rows must be finite, non-negative and sum to 1")


def fuse(p_e0: np.ndarray, p_b7: np.ndarray, w_e0: float, w_b7: float) -> np.ndarray:
    if np.shape(p_e0) != np.shape(p_b7):
        raise AlignmentError(f"cannot fuse shapes {np.shape(p_e0)} and {np.shape(p_b7)}")
    if min(w_e0, w_b7) < 0 or abs(w_e0 + w_b7 - 1.0) > 1e-12:
        raise ValueError("fusion weights must be non-negative and sum to 1")
    return w_e0 * np.asarray(p_e0, float) + w_b7 * np.asarray(p_b7, float)


def select_setting(records: Sequence[Mapping], native_min: float, recall_min: float) -> dict:
    """Task-2 calibration rule: feasible = native macro-F1 >= native_min and multi boundary recall >= recall_min;
    the highest segment F1@50 wins and an equal score goes to the lowest `order`. No feasible setting: infeasible."""
    feasible = [r for r in records if r["native_macro_f1"] >= native_min and (r["boundary_recall_multi"] or 0.0) >= recall_min]
    if not feasible:
        return {"status": "infeasible", "chosen": None, "n_feasible": 0, "n_settings": len(records)}
    best = max(r["segment_f1_50_multi"] for r in feasible)
    chosen = min((r for r in feasible if r["segment_f1_50_multi"] == best), key=lambda r: r["order"])
    return {"status": "feasible", "chosen": chosen, "n_feasible": len(feasible), "n_settings": len(records)}


def cell(e0_right: bool, b7_right: bool) -> str:
    return CELLS[0] if e0_right and b7_right else CELLS[1] if e0_right else CELLS[2] if b7_right else CELLS[3]


def interval_class(prob: np.ndarray, start: int, end: int) -> int:
    """The class of a teacher's mean probability over [start, end) (diagnostic; the interval comes from the labels)."""
    return int(np.asarray(prob, float)[start:end].mean(0).argmax())


def opinion_kind(class_e0: int, class_b7: int) -> str:
    return "teachers disagree on class" if class_e0 != class_b7 else "same class opinion"


def changes(before: Mapping[str, bool], after: Mapping[str, bool]) -> tuple[list[str], list[str]]:
    """Items fixed (wrong before, right after) and newly wrong (right before, wrong after)."""
    if set(before) != set(after):
        raise ValueError("the two systems cover different items")
    return sorted(k for k in before if not before[k] and after[k]), sorted(k for k in before if before[k] and not after[k])


def concentration(fixed_sources: Sequence[Sequence[str]], new_sources: Sequence[Sequence[str]], top: Sequence[int] = (1, 3, 5)) -> dict:
    """How far fixed and newly wrong items depend on a few source recordings. An item names all its source
    recordings and counts once for each, so per-recording counts can add up to more than the items."""
    def per_recording(items: Sequence[Sequence[str]]) -> Counter:
        c: Counter = Counter()
        for s in items:
            c.update(set(s))
        return c
    fc, nc = per_recording(fixed_sources), per_recording(new_sources)
    ranked = [r for r, _ in sorted(fc.items(), key=lambda kv: (-kv[1], kv[0]))]
    out = {"fixed": len(fixed_sources), "new_errors": len(new_sources), "net": len(fixed_sources) - len(new_sources),
           "recordings_in_fixed": len(fc), "recordings_in_new_errors": len(nc)}
    for k in top:
        tops = set(ranked[:k])
        f_top = sum(1 for s in fixed_sources if tops & set(s)); n_top = sum(1 for s in new_sources if tops & set(s))
        out[f"top{k}"] = {"recordings": ranked[:k], "fixed_involving": f_top, "net_without": (len(fixed_sources) - f_top) - (len(new_sources) - n_top)}
    out["by_recording"] = sorted(([r, fc[r], nc[r], fc[r] - nc[r]] for r in set(fc) | set(nc)), key=lambda x: (-x[3], -x[1], x[0]))
    return out
