#!/usr/bin/env python3
"""Export training logs for the public repository. It only reads the run folders.

Per-epoch logs keep their numbers, booleans and nulls, plus the short `head` label. All other text and every list are dropped.
Per-update logs become about fifty windows per file, holding only the means of their top-level numbers. Their lists, hashes
and dictionaries never pass, because dictionary keys can carry names. A field that names the test split is dropped in both.
Runs on withheld data, engineering checks and smoke runs are left out.

  python scripts/export_training_logs.py runs docs/training_logs
"""
import argparse
import collections
import csv
import json
import math
import os
import re
from pathlib import Path

EPOCH_LOG = 'train_log.jsonl'
MANIFEST = 'run_manifest.json'
UPDATE_LOG = re.compile(r'^(updates|physical_updates|.+\.train)(\.private)?\.jsonl$')
SKIP_FOLDERS = {'cache', 'code', 'pylib', 'helena_copy', 'forward', 'evaluation', '__pycache__', '.git'}
SKIP_PREFIXES = ('engineering', 'smoke')
KEEP_TEXT = {'head'}
DROP_NUMBERS = {'pid'}
POSITION_FIELDS = {'u', 'step', 'update', 'updates', 'epoch'}
TEST_FIELD = re.compile(r'(^|_)test(_|$)', re.I)
WITHHELD_RUNS = r'pediatric|unity|adapted_head'
POINTS = 50
INDEX_FIELDS = ['run', 'log', 'kind', 'source_records', 'records', 'last_epoch', 'last_updates']


def clean_record(record, prefix=''):
    """A per-epoch record's public fields, and the full names of the fields it lost."""
    kept, dropped = {}, []
    for key, value in record.items():
        name = f'{prefix}{key}'
        if TEST_FIELD.search(str(key)):
            dropped = [*dropped, name]
        elif isinstance(value, dict):
            inner, lost = clean_record(value, f'{name}.')
            kept, dropped = {**kept, key: inner}, [*dropped, *lost]
        elif isinstance(value, float):
            kept = {**kept, key: value if math.isfinite(value) else None}
        elif value is None or isinstance(value, (bool, int)) or (isinstance(value, str) and key in KEEP_TEXT):
            kept = {**kept, key: value}
        else:
            dropped = [*dropped, name]
    return kept, dropped


def scalar_fields(record):
    """The top-level finite numbers of a per-update record, the names it lost, and how many values were not finite."""
    public = {k: float(v) for k, v in record.items() if isinstance(v, (bool, int, float)) and not TEST_FIELD.search(k) and k not in DROP_NUMBERS}
    kept = {k: v for k, v in public.items() if math.isfinite(v)}
    return kept, [k for k, v in record.items() if v is not None and k not in public], len(public) - len(kept)


def window_value(name, values):
    return int(values[-1]) if name in POSITION_FIELDS else float(f'{sum(values) / len(values):.6g}')


def window_row(start, rows, non_finite):
    names = sorted({name for row in rows for name in row})
    values = {name: window_value(name, [row[name] for row in rows if name in row]) for name in names}
    return {'first_record': start, 'records': len(rows), **values, **({'non_finite': non_finite} if non_finite else {})}


def window_records(records, points=POINTS):
    """Means over equal runs of records, at most `points` of them, and the names of the fields that were lost."""
    scalars = [scalar_fields(record) for record in records]
    size = max(1, math.ceil(len(scalars) / points))
    windows = [scalars[start:start + size] for start in range(0, len(scalars), size)]
    rows = [window_row(i * size, [row for row, _, _ in window], sum(count for _, _, count in window)) for i, window in enumerate(windows)]
    return rows, [name for _, lost, _ in scalars for name in lost]


def is_withheld(runs_root, run, pattern):
    """By the run folder's name, or by a config file at its top, because some runs on withheld data have plain names."""
    if pattern.search(run):
        return True
    return any(pattern.search(path.read_text(errors='replace')) for path in sorted((runs_root / run).glob('*.yaml')))


def read_lines(path, relative):
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f'{relative}: line {number} is not JSON') from error
    return records


def manifest_epochs(path):
    manifest = json.loads(path.read_text())
    epochs = manifest.get('epochs') if isinstance(manifest, dict) else None
    return epochs if isinstance(epochs, list) and epochs else None


def folder_logs(folder, files):
    """(source, kind, published name) of each training log in one folder. A manifest counts only when nothing else is there."""
    found = [(folder / name, 'windows', f'{UPDATE_LOG.match(name).group(1)}.windows.jsonl') for name in sorted(files) if UPDATE_LOG.match(name)]
    if EPOCH_LOG in files:
        return [(folder / EPOCH_LOG, 'per_epoch', EPOCH_LOG), *found]
    if not found and MANIFEST in files and manifest_epochs(folder / MANIFEST):
        return [(folder / MANIFEST, 'manifest', EPOCH_LOG)]
    return found


def all_logs(root):
    entries = []
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = sorted(s for s in subfolders if s not in SKIP_FOLDERS and not s.startswith(SKIP_PREFIXES))
        entries.extend(folder_logs(Path(folder), set(files)))
    return sorted(entries)


def published(path, kind, relative):
    """The records to publish, the number of source records, and the names of the fields that were lost."""
    if kind == 'windows':
        source = read_lines(path, relative)
        rows, lost = window_records(source)
        return rows, len(source), lost
    source = manifest_epochs(path) if kind == 'manifest' else read_lines(path, relative)
    cleaned = [clean_record(record) for record in source]
    return [kept for kept, _ in cleaned], len(source), [name for _, lost in cleaned for name in lost]


def index_row(run, log, kind, source_records, records):
    last = records[-1] if records else {}
    position = next((last[name] for name in ('updates', 'u', 'step') if name in last), None)
    return {'run': run, 'log': log, 'kind': 'windows' if kind == 'windows' else 'per_epoch', 'source_records': source_records, 'records': len(records), 'last_epoch': last.get('epoch'), 'last_updates': position}


def write_log(target, records):
    if target.exists():
        raise ValueError(f'two logs would share one published name: {target.name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(record, allow_nan=False) + '\n' for record in records))


def export(runs_root, out_dir, withheld=WITHHELD_RUNS, select=None):
    runs_root, out_dir = Path(runs_root), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError('the output folder is not empty')
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern, chosen = re.compile(withheld, re.I), re.compile(select) if select else None
    rows, dropped, withheld_logs = [], collections.Counter(), 0
    for path, kind, name in all_logs(runs_root):
        relative = path.relative_to(runs_root)
        run = relative.parts[0]
        if chosen and not chosen.search(run):
            continue
        if is_withheld(runs_root, run, pattern):
            withheld_logs += 1
            continue
        records, source_records, lost = published(path, kind, relative)
        dropped.update(lost)
        target = relative.with_name(name)
        write_log(out_dir / target, records)
        rows.append(index_row(run, target.as_posix(), kind, source_records, records))
    with (out_dir / 'index.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {'logs': len(rows), 'source_records': sum(row['source_records'] for row in rows), 'records': sum(row['records'] for row in rows), 'runs': len({row['run'] for row in rows}),
               'withheld_logs': withheld_logs, 'dropped_fields': dict(sorted(dropped.items()))}
    (out_dir / 'export_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs')
    parser.add_argument('out')
    parser.add_argument('--withheld', default=WITHHELD_RUNS, help='regular expression on the run folder name and its config files')
    parser.add_argument('--select', help='regular expression on the run folder name: only these runs')
    args = parser.parse_args()
    print(json.dumps(export(args.runs, args.out, args.withheld, args.select), indent=1))


if __name__ == '__main__':
    main()
