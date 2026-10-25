"""a5c_expert_pilot_v1 records: field schema, validation, append-only versioned store, JSON / CSV round trip and the
private training overlay. Standard library only (the server and the local UI test use it without the project stack).

Frame indices are on the recording's original decoded frame axis (0-based, inclusive intervals); times come from that
axis's own PTS. "Unanswered" (None) is never the same as "not visible" or "cannot judge".
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

SCHEMA = "a5c_expert_pilot_v1/record"
OVERLAY_SCHEMA = "a5c_expert_pilot_v1/overlay"
VIEWS = ("A4C", "A5C", "transitional_or_off_axis", "other", "cannot_judge")
CONFIDENCE = ("high", "medium", "low")
STRUCT_STATUS = ("intervals", "not_visible_anywhere", "cannot_judge")
STRUCT_STATE = ("visible", "uncertain")
QUALITY = ("good", "adequate", "poor", "cannot_judge")
LABEL_FIT = ("yes", "partly", "no", "cannot_judge")
STRUCTURES = ("lvot", "aortic_valve")
BLIND_FIELDS = ("overall_view", "overall_confidence", "view_intervals", "lvot", "aortic_valve", "image_quality", "quality_reason", "comments")
UNBLINDED_FIELDS = ("label_fit", "review_comment")
# overlay codes (int per original frame)
UNKNOWN, DISAGREE = -1, -2
VIEW_CODE = {v: i for i, v in enumerate(VIEWS)}                    # 0 A4C, 1 A5C, 2 transitional, 3 other, 4 cannot judge
STRUCT_CODE = {"not_visible": 0, "visible": 1, "uncertain": 2, "cannot_judge": 3}


class RecordError(ValueError):
    pass


def empty_fields() -> dict:
    return {"overall_view": None, "overall_confidence": None, "view_intervals": [], "lvot": {"status": None, "intervals": []},
            "aortic_valve": {"status": None, "intervals": []}, "image_quality": None, "quality_reason": "", "comments": "", "label_fit": None, "review_comment": ""}


def _intervals(items, n: int, states: tuple, what: str) -> list:
    out, last = [], -1
    for it in sorted(items, key=lambda x: (int(x["start"]), int(x["end"]))):
        s, e, st = it.get("start"), it.get("end"), it.get("state")
        if not (isinstance(s, int) and isinstance(e, int)) or isinstance(s, bool) or isinstance(e, bool):
            raise RecordError(f"{what}: start and end must be integer frame indices")
        if not 0 <= s <= e <= n - 1:
            raise RecordError(f"{what}: [{s}, {e}] is outside 0..{n - 1} or reversed")
        if s <= last:
            raise RecordError(f"{what}: [{s}, {e}] overlaps the previous interval")
        if st not in states:
            raise RecordError(f"{what}: state {st!r} is not one of {states}")
        out.append({"start": s, "end": e, "state": st}); last = e
    return out


def validate_fields(f: dict, n: int, phase: str) -> dict:
    """Return a normalised copy; refuse unknown keys, codes, out-of-range / reversed / overlapping intervals."""
    allowed = set(BLIND_FIELDS) | (set(UNBLINDED_FIELDS) if phase == "unblinded" else set())
    extra = set(f) - allowed
    if extra:
        raise RecordError(f"fields not allowed in the {phase} phase: {sorted(extra)}")
    g = empty_fields(); g.update({k: f[k] for k in f})
    if g["overall_view"] not in (None, *VIEWS):
        raise RecordError("overall_view code")
    if g["overall_confidence"] not in (None, *CONFIDENCE):
        raise RecordError("overall_confidence code")
    g["view_intervals"] = _intervals(g["view_intervals"] or [], n, VIEWS, "view_intervals")
    for k in STRUCTURES:
        s = g[k] or {"status": None, "intervals": []}; st = s.get("status")
        if st not in (None, *STRUCT_STATUS):
            raise RecordError(f"{k}: status code")
        iv = _intervals(s.get("intervals") or [], n, STRUCT_STATE, k)
        if st == "intervals" and not iv:
            raise RecordError(f"{k}: status 'intervals' needs at least one interval")
        if st != "intervals" and iv:
            raise RecordError(f"{k}: intervals given with status {st!r}")
        g[k] = {"status": st, "intervals": iv}
    if g["image_quality"] not in (None, *QUALITY):
        raise RecordError("image_quality code")
    if g["label_fit"] not in (None, *LABEL_FIT):
        raise RecordError("label_fit code")
    for k in ("quality_reason", "comments", "review_comment"):
        if not isinstance(g[k], str) or len(g[k]) > 4000:
            raise RecordError(f"{k} must be text up to 4000 characters")
    if phase == "blind":
        g["label_fit"], g["review_comment"] = None, ""
    return g


def new_record(pid: str, reader: str, axis: dict, synthetic: bool) -> dict:
    return {"schema": SCHEMA, "recording": pid, "reader": reader, "synthetic": bool(synthetic), "frame_axis": {k: axis[k] for k in ("sha256", "n_frames", "time_base", "source")},
            "state": "not_started", "unblinded": False, "versions": []}


def state_of(rec: dict) -> str:
    v = rec["versions"]
    if not v:
        return "not_started"
    if not rec["unblinded"]:
        return "blind_submitted" if any(x["phase"] == "blind" and x["submitted"] for x in v) else "blind_draft"
    return "unblinded_submitted" if v[-1]["phase"] == "unblinded" and v[-1]["submitted"] else "unblinded_draft"


def add_version(rec: dict, fields: dict, phase: str, submit: bool, n: int, now: float | None = None) -> dict:
    """Append a version (never edit one). A submitted blind version freezes the blind phase."""
    st = state_of(rec)
    if phase == "blind" and st not in ("not_started", "blind_draft"):
        raise RecordError("the blind phase is submitted; it cannot be changed")
    if phase == "unblinded" and not rec["unblinded"]:
        raise RecordError("unblinding has not been requested")
    if phase == "unblinded" and st == "unblinded_submitted":
        raise RecordError("the unblinded review is submitted")
    g = validate_fields(fields, n, phase); t = time.time() if now is None else now
    out = json.loads(json.dumps(rec)); out["versions"].append({"version": len(rec["versions"]) + 1, "phase": phase, "submitted": bool(submit), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)), "fields": g})
    out["state"] = state_of(out); return out


def unblind(rec: dict, now: float | None = None) -> dict:
    if state_of(rec) != "blind_submitted":
        raise RecordError("unblinding needs a submitted blind version")
    out = json.loads(json.dumps(rec)); out["unblinded"] = True
    out["unblinded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if now is None else now)); out["state"] = state_of(out); return out


def blind_version(rec: dict) -> dict | None:
    sub = [v for v in rec["versions"] if v["phase"] == "blind" and v["submitted"]]
    return sub[-1] if sub else None


def current(rec: dict) -> dict | None:
    return rec["versions"][-1] if rec["versions"] else None


class Store:
    """One directory per reader; one JSON file per recording; atomic writes. The store never reads another reader's directory."""

    def __init__(self, root: Path, synthetic: bool) -> None:
        self.root, self.synthetic = Path(root), bool(synthetic)

    def path(self, reader: str, pid: str) -> Path:
        if not (reader.isalnum() and pid.isalnum()):
            raise RecordError("reader and recording ids must be alphanumeric")
        return self.root / reader / f"{pid}.json"

    def load(self, reader: str, pid: str, axis: dict) -> dict:
        p = self.path(reader, pid)
        if not p.is_file():
            return new_record(pid, reader, axis, self.synthetic)
        rec = json.loads(p.read_text())
        if rec["frame_axis"]["sha256"] != axis["sha256"] or rec["synthetic"] != self.synthetic:
            raise RecordError("the stored record belongs to another frame axis or store")
        return rec

    def save(self, rec: dict) -> None:
        p = self.path(rec["reader"], rec["recording"]); p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
        os.replace(tmp, p)

    def all(self, reader: str) -> list:
        d = self.root / reader
        return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


# ----------------------------------------------------------------------------------------------- CSV round trip
CSV_COLUMNS = ("recording", "reader", "synthetic", "frame_axis_sha256", "n_frames", "time_base", "axis_source", "unblinded", "unblinded_at_utc", "version", "phase", "submitted",
               "timestamp_utc", "row", "field", "start", "end", "state", "value")


def to_csv(records: list) -> str:
    """One row per scalar field and per interval of every version; the JSON file stays the primary record."""
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n"); w.writeheader()
    for r in records:
        base = {"recording": r["recording"], "reader": r["reader"], "synthetic": int(r["synthetic"]), "frame_axis_sha256": r["frame_axis"]["sha256"], "n_frames": r["frame_axis"]["n_frames"],
                "time_base": r["frame_axis"]["time_base"], "axis_source": r["frame_axis"]["source"], "unblinded": int(r["unblinded"]), "unblinded_at_utc": r.get("unblinded_at_utc", "")}
        if not r["versions"]:
            w.writerow({**base, "row": "record"})
        for v in r["versions"]:
            vb = {**base, "version": v["version"], "phase": v["phase"], "submitted": int(v["submitted"]), "timestamp_utc": v["timestamp_utc"]}
            f = v["fields"]
            for k in ("overall_view", "overall_confidence", "image_quality", "quality_reason", "comments", "label_fit", "review_comment"):
                w.writerow({**vb, "row": "scalar", "field": k, "value": json.dumps(f[k])})
            for k in STRUCTURES:
                w.writerow({**vb, "row": "status", "field": k, "value": json.dumps(f[k]["status"])})
                for it in f[k]["intervals"]:
                    w.writerow({**vb, "row": "interval", "field": k, "start": it["start"], "end": it["end"], "state": it["state"]})
            for it in f["view_intervals"]:
                w.writerow({**vb, "row": "interval", "field": "view_intervals", "start": it["start"], "end": it["end"], "state": it["state"]})
    return buf.getvalue()


def from_csv(text: str) -> list:
    recs: dict = {}
    for row in csv.DictReader(io.StringIO(text)):
        key = (row["recording"], row["reader"])
        r = recs.setdefault(key, {"schema": SCHEMA, "recording": row["recording"], "reader": row["reader"], "synthetic": bool(int(row["synthetic"])),
                                  "frame_axis": {"sha256": row["frame_axis_sha256"], "n_frames": int(row["n_frames"]), "time_base": row["time_base"], "source": row["axis_source"]},
                                  "unblinded": bool(int(row["unblinded"])), "versions": {}, **({"unblinded_at_utc": row["unblinded_at_utc"]} if row["unblinded_at_utc"] else {})})
        if row["row"] == "record":
            continue
        v = r["versions"].setdefault(int(row["version"]), {"version": int(row["version"]), "phase": row["phase"], "submitted": bool(int(row["submitted"])), "timestamp_utc": row["timestamp_utc"], "fields": empty_fields()})
        f = v["fields"]
        if row["row"] == "scalar":
            f[row["field"]] = json.loads(row["value"])
        elif row["row"] == "status":
            f[row["field"]]["status"] = json.loads(row["value"])
        else:
            it = {"start": int(row["start"]), "end": int(row["end"]), "state": row["state"]}
            (f["view_intervals"] if row["field"] == "view_intervals" else f[row["field"]]["intervals"]).append(it)
    out = []
    for r in recs.values():
        r["versions"] = [r["versions"][k] for k in sorted(r["versions"])]; r["state"] = state_of(r); out.append(r)
    return out


def canonical(records: list) -> str:
    return json.dumps(sorted(records, key=lambda r: (r["recording"], r["reader"])), sort_keys=True)


# ----------------------------------------------------------------------------------------------- overlay (training interface)
def frame_codes(fields: dict, n: int) -> dict:
    """Per original frame: view code from view intervals (unannotated frames stay UNKNOWN; the overall view is kept as a
    separate recording-level field, never spread over frames); structure states from their own fields only."""
    view = [UNKNOWN] * n
    for it in fields["view_intervals"]:
        for k in range(it["start"], it["end"] + 1):
            view[k] = VIEW_CODE[it["state"]]
    out = {"view": view}
    for s in STRUCTURES:
        st = fields[s]["status"]; arr = [UNKNOWN] * n
        if st == "not_visible_anywhere":
            arr = [STRUCT_CODE["not_visible"]] * n
        elif st == "cannot_judge":
            arr = [STRUCT_CODE["cannot_judge"]] * n
        elif st == "intervals":
            arr = [UNKNOWN] * n                                        # frames outside a visible interval: not asserted
            for it in fields[s]["intervals"]:
                for k in range(it["start"], it["end"] + 1):
                    arr[k] = STRUCT_CODE[it["state"]]
        out[s] = arr
    return out


def merge_readers(per_reader: list) -> list:
    if not per_reader:
        return []
    n = len(per_reader[0]); out = []
    for k in range(n):
        vals = {a[k] for a in per_reader if a[k] != UNKNOWN}
        out.append(UNKNOWN if not vals else (vals.pop() if len(vals) == 1 else DISAGREE))
    return out


def build_overlay(records: list, axes: dict, links: dict) -> dict:
    """records: final records of real readers; axes: pid -> frame axis; links: pid -> {"fold", "sample_frames": [frame per
    10 Hz sample of the recording's own cached axis]}. Public labels are not an input and are never written."""
    by = {}
    for r in records:
        if r["synthetic"] != records[0]["synthetic"]:
            raise RecordError("real and synthetic records cannot be mixed in one overlay")
        if r["frame_axis"]["sha256"] != axes[r["recording"]]["sha256"]:
            raise RecordError("a record was made on another frame axis")
        cur = current(r)
        if cur is None or not (cur["submitted"] or blind_version(r)):
            continue
        use = cur if cur["submitted"] else blind_version(r)
        by.setdefault(r["recording"], []).append((r["reader"], use))
    out = {"schema": OVERLAY_SCHEMA, "synthetic": bool(records and records[0]["synthetic"]), "codes": {"unknown": UNKNOWN, "disagree": DISAGREE, "view": VIEW_CODE, "structure": STRUCT_CODE},
           "recordings": {}}
    for pid, lst in sorted(by.items()):
        n = axes[pid]["n_frames"]; codes = [frame_codes(v["fields"], n) for _, v in lst]
        fr = {k: merge_readers([c[k] for c in codes]) for k in ("view", *STRUCTURES)}
        overall = sorted({v["fields"]["overall_view"] for _, v in lst if v["fields"]["overall_view"]})
        samp = links[pid]["sample_frames"]
        out["recordings"][pid] = {"frame_axis_sha256": axes[pid]["sha256"], "n_frames": n, "fold": links[pid]["fold"], "readers": [r for r, _ in lst], "phases": [v["phase"] for _, v in lst],
                                  "overall_view": overall[0] if len(overall) == 1 else ("disagree" if overall else None), "frames": fr,
                                  "samples_10hz": {k: [fr[k][f] if 0 <= f < n else UNKNOWN for f in samp] for k in fr}}
    out["sha256"] = hashlib.sha256(json.dumps(out["recordings"], sort_keys=True).encode()).hexdigest(); return out


def check_overlay(ov: dict, axes: dict) -> dict:
    """Invariants a training loader must rely on; returns coverage counts (frames asserted vs unknown)."""
    if ov["schema"] != OVERLAY_SCHEMA:
        raise RecordError("overlay schema")
    cov = {}
    for pid, r in ov["recordings"].items():
        if r["frame_axis_sha256"] != axes[pid]["sha256"] or len(r["frames"]["view"]) != axes[pid]["n_frames"]:
            raise RecordError(f"{pid}: overlay does not match the frame axis")
        allowed_view = set(VIEW_CODE.values()) | {UNKNOWN, DISAGREE}; allowed_s = set(STRUCT_CODE.values()) | {UNKNOWN, DISAGREE}
        if not set(r["frames"]["view"]) <= allowed_view or any(not set(r["frames"][s]) <= allowed_s for s in STRUCTURES):
            raise RecordError(f"{pid}: unknown code")
        cov[pid] = {k: {"asserted": sum(c >= 0 for c in r["frames"][k]), "unknown": sum(c == UNKNOWN for c in r["frames"][k]), "disagree": sum(c == DISAGREE for c in r["frames"][k])} for k in r["frames"]}
    return cov


# ----------------------------------------------------------------------------------------------- double reading (format only)
DISAGREE_COLUMNS = ("recording", "reader_a", "reader_b", "field", "a", "b", "equal", "frames_compared", "frames_both_asserted", "frames_disagree")


def disagreement_rows(records: list, axes: dict) -> list:
    """Pairwise comparison of submitted blind versions per recording (fields and per-frame codes). A format for later
    review of real double reading; it computes no agreement statistic and needs no particular number of readers."""
    by = {}
    for r in records:
        b = blind_version(r)
        if b is not None:
            by.setdefault(r["recording"], []).append((r["reader"], b["fields"]))
    rows = []
    for pid, lst in sorted(by.items()):
        lst = sorted(lst); n = axes[pid]["n_frames"]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                (ra, fa), (rb, fb) = lst[i], lst[j]; ca, cb = frame_codes(fa, n), frame_codes(fb, n)
                for k in ("overall_view", "overall_confidence", "image_quality"):
                    rows.append({"recording": pid, "reader_a": ra, "reader_b": rb, "field": k, "a": json.dumps(fa[k]), "b": json.dumps(fb[k]), "equal": int(fa[k] == fb[k])})
                for k in STRUCTURES:
                    rows.append({"recording": pid, "reader_a": ra, "reader_b": rb, "field": f"{k}.status", "a": json.dumps(fa[k]["status"]), "b": json.dumps(fb[k]["status"]), "equal": int(fa[k]["status"] == fb[k]["status"])})
                for k in ("view", *STRUCTURES):
                    both = [x >= 0 and y >= 0 for x, y in zip(ca[k], cb[k])]; dis = sum(b_ and x != y for b_, x, y in zip(both, ca[k], cb[k]))
                    rows.append({"recording": pid, "reader_a": ra, "reader_b": rb, "field": f"{k}.frames", "frames_compared": n, "frames_both_asserted": sum(both), "frames_disagree": dis, "equal": int(dis == 0)})
    return rows


def rows_csv(rows: list, columns: tuple) -> str:
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="raise"); w.writeheader(); w.writerows(rows); return buf.getvalue()
