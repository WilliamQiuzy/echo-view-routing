# Why A5C clips are misread, and what helps (2026-09-28)

> **Pending recomputation (2026-09-30).** These results were computed on caches that read EV9V frames in name order (0.jpg, 1.jpg, 10.jpg, 100.jpg, …, 2.jpg), which shuffles frames in time within each clip. Every number that depends on the order of frames inside a clip is unreliable, and frame- and clip-level scores also shift because different frames were sampled. Do not cite these numbers until they are recomputed from `cache_v2`. See `docs/results/2026-09-30_frame_order_fix/`. The frame grids show true-order frames, but the probability printed beside each frame came from a different frame. The Doppler-text analysis paired text flags with probabilities from different frames and must be rerun. The visual observation of a late A4C→A5C tilt stands.

Scripts: `scripts/a5c_analysis.py` (evidence, nearest training clips, sessions, frame grids), `scripts/a5c_aggregation.py`
(aggregation rules), `scripts/a5c_text_cue.py` (Doppler-text cue). Frame grids and the review page stay local in
`runs/_demo/a5c_analysis/` and `runs/_demo/a5c_audit/`; git ignores both. The visual notes below describe the frames and
are not a clinical reading.

## Findings

1. **A5C evidence is intermittent in 5 of the 6 test A5C clips that at least 4 of 6 main models misread.**
   - In every one of the six, the ResNet-18 frame classifier (B0) gives P(A5C) of 0.70–0.999 on some frames. It calls
     only a median 30% of their frames A5C, against 95% for A5C clips that all six models read correctly.
   - In the frames, the outflow tract appears and disappears within the clip. Some clips alternate between frames.
     Others reach the A5C plane only late: `2022-07-10_10-58-17_320x240_7` tilts into A5C at the end, where the LVOT
     Doppler set-up text appears, and the next clip of that session is a clear A4C.
   - Temporal models follow the A4C majority. By mean B0 feature, the 10 nearest training clips of these six are on
     average 8.2 A4C and 0.3 A5C; for clips read correctly they are 9.3 A5C.
2. **One misread clip is off-axis.** `2022-07-26_16-43-25_320x240_6` is so oblique that B0 calls it PSAX.
3. **The opposite error exists.** A4C clip `2022-05-05_10-38-37_320x240_1` is called A5C by every model. It shows a
   round central structure like the aortic root in clear A5C clips, so it is a candidate label error.
4. **There is no session leakage.** None of the 125 test sessions (video-id prefix) appears in training.
5. **Doppler text is not the main cause.**
   - "PW SV4.0mm" appears 5× more often in A5C clips (train: 12.4% of A5C against 2.3% of A4C clips).
   - Inside A4C clips, frames with the text have a lower B0 P(A5C) than frames without it (0.068 against 0.128). B0
     therefore does not read the text as A5C.
   - Five of the six misread A5C clips have no such text.
6. **Aggregation rules do not fix it.** B0 statistics, threshold chosen on validation, applied to test:
   - Mean: A5C recall 0.875; 17 of 231 A4C called A5C; 1 of 6 hard clips recovered.
   - Share of frames above 0.5: 0.90; 17; 2 of 6.
   - Maximum: 0.95; 48; 4 of 6.

   A4C clips also contain A5C-like frames. A4C and A5C are one probe position with a continuous tilt, so a post-hoc
   rule mainly trades A4C errors for A5C.

## What would help

- **Labels.** Clip-level labels hide changes inside a clip.
  - A reader should adjudicate the misread clips and the A4C clip above, so that results can also be reported on an
    adjudicated test subset.
  - Treat A4C/A5C transitions explicitly, as a transitional class or by deferral. Routing policy 2 already defers
    most of these errors.
- **Model.** The deciding structure (the LVOT) is small.
  - Test sector-cropped, higher-resolution input using the teammates' sector cropping, which is also the proposal's
    sector-only control.
  - Use frame-level A5C evidence, not only clip labels, as a training signal.
- **Data.** Training has 217 A5C clips against 858 A4C. External A5C data would help most where the classes overlap.
