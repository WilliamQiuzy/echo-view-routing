# Training logs

These are the training curves of the project's trained runs. They let you see how a run trained without logging in to the server.

Start with [index.csv](index.csv). It has one row per log. To find the report for a named run, search `docs/` for the name in the `run` column. Names that start with a date, such as `20261002-053948-md_M4_lr0.0003-040e7258-s0`, come from the early model development rounds.

| Column | Meaning |
|---|---|
| `run` | The run's folder name on the server. |
| `log` | The file, relative to this folder. |
| `kind` | `per_epoch` or `windows`, as explained below. |
| `source_records` | How many records the original log had. |
| `records` | How many records this file has. |
| `last_epoch`, `last_updates` | Where the log ends. A blank cell means the log did not record it. |

## The two kinds of file

**`train_log.jsonl`** has one line per epoch, copied from the run. Only numbers, true or false values and blanks are kept. A value that was NaN or infinite appears as a blank.

**`*.windows.jsonl`** summarises a log that recorded every update. Each file has about 50 lines, and each line covers an equal run of updates.

- `first_record` and `records` say which original records the line covers.
- Most fields are means over those records, to six significant digits.
- `u`, `step`, `update`, `updates` and `epoch` give the last value in the line, not a mean.
- `non_finite` appears when some values were NaN or infinite. Those values are not in the means.

Field names differ between rounds, because rounds train with different losses. The round's report explains its own fields.

## Reading a log

```python
import json
from pathlib import Path

path = Path('docs/training_logs/video_within_view_v1/train/all_s0/train_log.jsonl')
rows = [json.loads(line) for line in path.read_text().splitlines()]
print([(row['updates'], round(row['loss'], 3)) for row in rows])
```

## What is not here

- **Weights, checkpoints, raw media and per-recording outputs.** They stay on the server.
- **Text, lists, hashes and dictionaries from the per-update logs.** Dictionary keys and lists can carry recording names, so only top-level numbers pass.
- **Runs on Pediatric data.** Their 15 logs are left out because the Pediatric reports are not public either.
- **Engineering checks and smoke runs.** They are short tests of the code, not training.
- **Checksum records such as `committed.jsonl`.** They are not training logs.
- **Rounds without training.** Diagnostic rounds have no training log.

These files come from development data and repeatedly used evaluation sets. A good curve here is not evidence on new patients.

## How this was built

[`scripts/export_training_logs.py`](../../scripts/export_training_logs.py) wrote these files from the server's `runs/` folder. [export_summary.json](export_summary.json) gives the totals and the names of the dropped fields. Before publishing, every file was scanned for recording and session identifiers, patient names, local paths, email addresses and secrets, and none was found.
