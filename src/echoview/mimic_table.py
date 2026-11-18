"""The MIMIC-IV-Echo view table with this model's view, probability and version in the last three columns.

The command reads each row's `dicom_file` under `--root`, copies the first seven columns of the existing view table and
writes this model's answer in the last three. It can be stopped and started again: rows already written are skipped.
Files it cannot read go to a second file ending in `.errors.csv` and are tried again on the next run.

MIMIC-IV-Echo is restricted-access data. This command needs your own credentialed copy; nothing of it is part of
this repository. The labels of the existing table are another classifier's output, so a comparison with them is
agreement between two models, not accuracy.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import frames as fr

V4_COLUMNS = ("video_name", "subject_id", "study_id", "dicom_file", "frames", "manufacturer", "manufacturer_model",
              "predicted_view", "view_probability", "view_classifier_version")
COPIED = V4_COLUMNS[:7]
BLOCK = 64  # rows read ahead by the reader threads


def rows_to_do(labels: Path, out: Path, limit: int) -> tuple[list[dict], bool]:
    """The table rows not yet written to `out`, and whether `out` is new."""
    with labels.open(newline="") as handle:
        table = list(csv.DictReader(handle))
    if not table or any(column not in table[0] for column in COPIED):
        raise SystemExit(f"the label table needs the columns {COPIED}")
    new_file = not out.exists() or out.stat().st_size == 0
    done: set[str] = set()
    if not new_file:
        with out.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != V4_COLUMNS:
                raise SystemExit(f"{out} exists and is not a table written by this command; name another --out or remove it")
            done = {row["video_name"] for row in reader}
    todo = [row for row in table if row["video_name"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"table rows {len(table)}, already written {len(done)}, to do now {len(todo)}", file=sys.stderr)
    return todo, new_file


def run(args: argparse.Namespace) -> int:
    from .cli import build

    if args.limit < 0 or args.workers < 1:
        raise SystemExit("--limit must not be negative and --workers must be at least 1")
    root, out = Path(args.root), Path(args.out)
    if out.resolve() == Path(args.labels).resolve():
        raise SystemExit("--out must not be the label table itself")
    todo, new_file = rows_to_do(Path(args.labels), out, args.limit)
    if not todo:
        return 0
    system = build(args)
    errors = out.with_name(out.stem + ".errors.csv")
    new_errors = not errors.exists() or errors.stat().st_size == 0

    def load(row: dict) -> tuple:
        try:
            return row, fr.read_recording(root / row["dicom_file"], args.fps), ""
        except fr.ReadError as error:
            return row, None, str(error)

    began, written, failed, fallback = time.time(), 0, 0, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", newline="") as handle, errors.open("a", newline="") as error_handle, ThreadPoolExecutor(args.workers) as pool:
        writer, error_writer = csv.writer(handle), csv.writer(error_handle)
        if new_file:
            writer.writerow(V4_COLUMNS)
        if new_errors:
            error_writer.writerow(["video_name", "dicom_file", "reason"])
        for start in range(0, len(todo), BLOCK):
            for row, recording, why in pool.map(load, todo[start:start + BLOCK]):
                if recording is None:
                    error_writer.writerow([row["video_name"], row["dicom_file"], why])
                    failed += 1
                    continue
                fallback += recording.reader == fr.PYDICOM_READER
                result = system.classify(recording.frames)
                writer.writerow([*[row[c] for c in COPIED], result.view, repr(result.probability), system.version])
                written += 1
            handle.flush()
            error_handle.flush()
            rate = (written + failed) / max(time.time() - began, 1e-9)
            print(f"{written + failed} of {len(todo)} | written {written} | unreadable {failed} | {rate:.1f} recordings/s", file=sys.stderr)
    if fallback:
        print(f"NOTE {fallback} recordings were decoded by pydicom, not by the reader behind the MIMIC numbers", file=sys.stderr)
    if failed:
        print(f"{failed} rows could not be read; they are listed in {errors} and are tried again on the next run", file=sys.stderr)
    return 1 if failed and not written else 0
