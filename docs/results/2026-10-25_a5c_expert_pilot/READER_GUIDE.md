# Reader guide: A4C / A5C pilot (24 recordings)

**Who this is for:** an echocardiographer or sonographer reviewing apical recordings. **Language:** short notes are fine
in any language.

**What you are asked.** For each recording, give your own judgement of the imaging plane, and say where the left
ventricular outflow tract (LVOT) and the aortic valve are visible. You do not need any dataset label or model output for
this. They stay hidden until you submit your first-pass record for that recording.

**What we are not asking.**
- We are not asking whether a model is right.
- We are not asking you to force every frame into A4C or A5C. "Transitional or off-axis", "other" and "cannot judge" are
  valid answers.

## Opening the tool

You receive a personal link. Open it after the tunnel is running, as set up by the project team:
`http://127.0.0.1:8765/?t=<your token>`. Do not share it, because it opens only your own records. Nothing is stored on
your computer.

## For each recording (P01 … P24, in order)

1. **Play the whole recording.**
   - Use "frame ‹ / ›" or the arrow keys to step one original frame. The screen shows the frame number and its time in
     seconds. These recordings are 30 frames per second.
   - Zoom with the slider.
2. **Whole-recording view.** Choose A4C, A5C, transitional or off-axis, other, or cannot judge, and give your
   confidence.
3. **View intervals (optional).** Use these only if the plane changes during the recording, for example A4C in one part
   and A5C in another.
   - Mark a start frame and an end frame, pick the view, then press "add interval".
   - Frames you do not cover stay unannotated, which is fine.
4. **LVOT and aortic valve.** Record each structure separately:
   - **Intervals** where the structure is *visible*, or *uncertain* if you think it may be visible;
   - **not visible anywhere**;
   - **cannot judge**, for example because of image quality.
   - Seeing or not seeing a structure does not decide the view for you. Record the view in step 2 or 3.
5. **Image quality.** Good, adequate, poor or cannot judge, with a short reason.
6. **Submit the blind record.** After submitting, you cannot change it.
7. **Optional review.** Press "unblind" to see the dataset's clip-level code and the model outputs for that recording.
   Then say whether the code fits the recording (yes, partly, no or cannot judge) and add a comment. If you change your
   mind about the view, record it here. It is saved as a new version, and your blind answer is kept.

"Save draft" keeps your work without submitting it. You can stop at any time and continue later.

## Notes

- **Definitions.** These follow the ASE 2019 adult TTE guideline (Mitchell et al., J Am Soc Echocardiogr 2019;32:1–64).
  It obtains the apical five-chamber view from the A4C view by anterior tilt until the LVOT, aortic valve and proximal
  ascending aorta come into view. Use your own clinical judgement where the images are ambiguous.
- **Second reader.** If you are the second reader, do not discuss cases with the first reader until both of you have
  submitted.
