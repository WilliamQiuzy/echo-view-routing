"""A51 source contracts. Pure metadata operations; never infer anatomical truth."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib


SOURCE_VIEWS = ("A4C", "A5C")
LABELS = {*SOURCE_VIEWS, "NO_VIEW", "IVC", "PLAX", "PSAX", "S4C"}
REQUIRED_CHECKS = (
    "license", "source_identity", "patient_owner", "source_partition", "released_test_exclusion",
    "frame_binding", "all_window_labels", "duplicate_isolation", "complete_hashes",
)
MINIMUMS = {
    "fit_patients": 24, "diagnostic_patients": 6,
    "fit_videos_A4C": 16, "fit_videos_A5C": 16,
    "diagnostic_videos_A4C": 4, "diagnostic_videos_A5C": 4,
    "fit_disagreement_patients": 8, "diagnostic_disagreement_patients": 2,
}


def join_annotations(meta: list[dict], annotations: list[dict]) -> list[dict]:
    owners = {}
    for r in meta:
        name = r["video_filename"]
        if name in owners or not r["patient_id"]:
            raise ValueError("duplicate or missing patient owner")
        owners[name] = (r["patient_id"], r["view"])
    out = []
    by_video = defaultdict(list)
    for r in annotations:
        name = r["Video_FileName"]
        if not name.endswith(".avi") or "/" in name or "\\" in name:
            raise ValueError("unexpected exact filename")
        stem = name[:-4]
        if stem not in owners:
            raise ValueError("annotation has no patient owner")
        patient, directory = owners[stem]
        if directory != r["View_Folder"]:
            raise ValueError("metadata folder disagreement")
        a, b = int(r["Start_Frame"]), int(r["End_Frame"])
        if a < 0 or b <= a:
            raise ValueError("bad interval bounds")
        if r["Track"] != "view" or r["Label"] not in LABELS:
            raise ValueError("unknown annotation contract")
        item = {"video": name, "directory_label": directory, "patient": patient,
                "start": a, "end": b, "true_label": r["Label"]}
        by_video[(directory, name)].append(item)
        out.append(item)
    for rows in by_video.values():
        rows = sorted(rows, key=lambda r: (r["start"], r["end"]))
        for a, b in zip(rows, rows[1:]):
            # Shared endpoints can arise from inclusive/exclusive export. The
            # conservative interior removes them; a longer conflict is invalid.
            if b["start"] < a["end"] and a["true_label"] != b["true_label"]:
                raise ValueError("contradictory label overlap")
    return out


def common_interior(start: int, end: int) -> tuple[int, int]:
    if start < 0 or end <= start:
        raise ValueError("bad interval bounds")
    return start + 1, end - 1


def interval_requests(rec: dict, max_requests: int = 8) -> list[dict]:
    if rec["true_label"] not in SOURCE_VIEWS or rec["directory_label"] not in SOURCE_VIEWS:
        return []
    a, b = common_interior(rec["start"], rec["end"])
    centres = list(range(a + 24, b - 21, 15))
    if len(centres) > max_requests:
        inds = [i * (len(centres) - 1) // (max_requests - 1) for i in range(max_requests)]
        centres = [centres[i] for i in inds]
    return [{**rec, "centre": c, "marker": 8, "frames": [c + 3 * (i - 8) for i in range(16)]}
            for c in centres]


def marker_requests(rec: dict, n_frames: int | None = None, max_requests: int = 8) -> list[dict]:
    """Local human marker labels with genuine, potentially mixed native context."""
    if rec["true_label"] not in SOURCE_VIEWS or rec["directory_label"] not in SOURCE_VIEWS:
        return []
    a, b = common_interior(rec["start"], rec["end"])
    centres = list(range(a, b, 15))
    if len(centres) > max_requests:
        centres = [centres[i * (len(centres) - 1) // (max_requests - 1)] for i in range(max_requests)]
    out = []
    for c in centres:
        frames = [c + 3 * (i - 8) for i in range(16)]
        if n_frames is not None:
            if not 0 <= c < n_frames:
                raise ValueError("labelled marker is outside released video")
            clipped = [max(0, min(n_frames - 1, x)) for x in frames]
        else:
            clipped = frames.copy()
        out.append({**rec, "centre": c, "marker": 8, "frames": clipped,
                    "raw_frames": frames, "clipped_slots": sum(x != y for x, y in zip(frames, clipped))})
    return out


def validate_video_axis(fps: float, decoded_frames: int, intervals: list[tuple[int, int]]) -> bool:
    if abs(float(fps) - 30.) > .001:
        raise ValueError("released fps does not support the fixed annotation contract")
    if decoded_frames < 2 or any(a < 0 or b > decoded_frames + 1 for a, b in intervals):
        raise ValueError("annotation exceeds the actual decoded ordinal inventory")
    return True


def check_duplicate_isolation(requests: list[dict]) -> bool:
    owners = {}
    for r in requests:
        key = r["pixel_sha256"]
        owner = r["patient"], r["role"]
        if key in owners and owners[key] != owner:
            raise ValueError("duplicate pixel request crosses patient or fold")
        owners[key] = owner
    return True


def robust_marker_label(records: list[dict], ordinal: int, native_pts_index: int, expected: str) -> bool:
    """Require the frozen label under all tested export/index conventions."""
    for position in (ordinal, native_pts_index):
        for origin in (0, 1):
            for inclusive in (False, True):
                supported = {r["true_label"] for r in records
                             if r["start"] - origin <= position < r["end"] - origin + int(inclusive)}
                if supported != {expected}:
                    return False
    return True


def _top_mass(counts: Counter, fraction: float) -> set:
    if not counts:
        return set()
    ordered = sorted(counts, key=lambda p: (-counts[p], p))
    target = sum(counts.values()) * fraction
    total, cutoff = 0, None
    for p in ordered:
        if total < target:
            total += counts[p]
            cutoff = counts[p]
        else:
            break
    return {p for p in ordered if counts[p] >= cutoff} if cutoff is not None else set()


def reserve_publisher_candidates(records: list[dict]) -> dict:
    """Metadata quarantine, explicitly NOT proof of the unpublished PNG split.

    Publisher code ranks *surviving* PNG windows; arbitrary missing folders can
    change that ranking. Never treat a larger metadata reserve as a guarantee.
    """
    all_tests, all_reserved, convention_counts = set(), set(), []
    for origin in (0, 1):
        for inclusive in (False, True):
            counts = Counter()
            for r in records:
                a = max(0, r["start"] - origin)
                end = r["end"] - origin + int(inclusive)
                counts[r["patient"]] += len(range(a, end - 5 + 1, 2))
            all_tests.update(_top_mass(counts, .20))
            all_reserved.update(_top_mass(counts, .40))
            convention_counts.append(sum(counts.values()))
    return {"publisher_test_union": sorted(all_tests), "reserved": sorted(all_reserved),
            "convention_windows": convention_counts,
            "exact_publisher_split_proved": False,
            "limitation": "Publisher uses external PNG availability; metadata quarantine is not an original-test exclusion proof."}


def patient_split(patients: list[str]) -> dict:
    ordered = sorted(set(patients), key=lambda p: hashlib.sha256(("A51-source-v1|" + p).encode()).hexdigest())
    return {"fit": [p for i, p in enumerate(ordered) if i % 5],
            "diagnostic": [p for i, p in enumerate(ordered) if i % 5 == 0]}


def source_qualified(counts: dict, checks: dict) -> dict:
    failed = [k for k in REQUIRED_CHECKS if checks.get(k) is not True]
    failed += [k for k, v in MINIMUMS.items() if counts.get(k, 0) < v]
    return {"passed": not failed, "failed": failed, "minimums": dict(MINIMUMS)}


def source_signal_gate(l: dict, w: dict, geometry: float, constant: float) -> dict:
    checks = {
        "L_balanced_accuracy": l["balanced_accuracy"] >= .80,
        "L_recall_A4C": l["a4c_recall"] >= .70,
        "L_recall_A5C": l["a5c_recall"] >= .70,
        "L_increment": l["balanced_accuracy"] - w["balanced_accuracy"] >= .05 - 1e-12,
        "A4C_protection": l["a4c_fpr"] <= w["a4c_fpr"],
        "image_over_geometry": l["balanced_accuracy"] - geometry >= .10 - 1e-12,
        "image_over_constant": l["balanced_accuracy"] - constant >= .10 - 1e-12,
        "discordant_increment": l["discordant_accuracy"] > w["discordant_accuracy"],
    }
    return {"passed": all(checks.values()), "checks": checks}
