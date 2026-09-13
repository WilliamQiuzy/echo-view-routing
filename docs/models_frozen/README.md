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

`models_frozen/` holds exactly the five baselines and nothing else: `stfm_official`, `mstcn_official`, `asformer_official`, `echoprime_view_classifier`, `echoviewclip_stage1`. Pending: `stfm_official/seed300`, `asformer_official/v1`, `echoviewclip_stage1/v1` (added when their runs finish and are checked).

Our own pipeline components (e.g. the ResNet-18 frame encoder `ckpt_hash 79f4a41af6e3`) are frozen the same way under `models_internal/` (manifests in `docs/models_internal/`).
