# Frozen (verified) models

Weights live on the server under `/home/william/echo-view-routing/models_frozen/<name>/<version>/` and are read-only
(`chmod 444`, directory `555`). They are never edited, retrained into, or deleted; a change means a new version directory.
Each directory has a `MANIFEST.json` (sha256 of every file, source code commit, run id, recipe, metrics); the manifests
are mirrored here so the repo records exactly which bytes each reported number came from.

| Name / version | What | Verification |
|---|---|---|
| `mstcn_official/v1` | official MS-TCN epoch-50 model trained on the EV9V class-balanced multi-fragment bank (+ the bank recipe and the py3 patch) | GTEA 4-split average matches the paper |
| `stfm_official/seed666`, `seed100`, `seed200` | official STFM `model.ckpt` (authors' recipe) with training logs | each within one std of the paper's Table 5 ResNet-18 row; seed 300 to be added |
| `echoprime_view_classifier/release_v1.0.0` | authors' released ConvNeXt-B 11-view classifier | evaluated on EV9V only |
| `frame_encoder_resnet18/v1` | our ResNet-18 family-5 encoder (`ckpt_hash 79f4a41af6e3`), the feature source for all temporal models | pipeline component |

Pending freezes: `stfm_official/seed300`, `asformer_official/v1`, `echoviewclip_stage1/v1` (when their runs finish and are checked).
