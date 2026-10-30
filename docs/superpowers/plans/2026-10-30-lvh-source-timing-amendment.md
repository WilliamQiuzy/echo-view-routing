# A50 LVH source-timing qualification, before model inference

The first queue completed all800 EV9V and1277 Dynamic recordings, then stopped
on the first LVH video before any LVH model forward. All343 LVH video frame
counts match the source CSV, but its column named `FPS` differs from the actual
container rate. The original contract already specifies decoded source timing.

Before resuming, decode every LVH video with PyAV and require every PTS interval
to equal the container CFR within1e-6s, with the exact source frame count. Do not
infer what the CSV field means. Preserve it as `reported_csv_fps`; use the
independently qualified playback rate for the unchanged10Hz input grid.

Preserve the original protocol/manifest and every completed prediction byte for
byte. A dated v2 source manifest/protocol is used only for LVH and CAMUS; the
per-dataset binding file keeps EV9V/Dynamic bound to v1. No weight, crop, marker,
window, B3 number, split membership, target, metric or stopping rule changes.
Repeat the unchanged development model-parity check under the v2 binding, then
run all343 LVH and50 CAMUS cases. Keep the failed job and actual resource cost.
No target outcomes or trial timing alternatives are used to choose this repair.
