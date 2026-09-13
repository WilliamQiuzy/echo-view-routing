# Frozen (verified) models

Weights live on the server under `/home/william/echo-view-routing/models_frozen/<name>/<version>/` and are read-only
(`chmod 444`, directory `555`). They are never edited, retrained into, or deleted; a change means a new version directory.
Each directory has a `MANIFEST.json` (sha256 of every file, source code commit, run id, recipe, metrics); the manifests
are mirrored here so the repo records exactly which bytes each reported number came from.

| Name / version | What | Verification |
|---|---|---|
| `mstcn_official/v1` | official MS-TCN epoch-50 model trained on the EV9V class-balanced multi-fragment bank (+ the bank recipe and the py3 patch) | GTEA 4-split average matches the paper |
| `stfm_official/seed100`, `seed200`, `seed300` (+ `seed666`) | official STFM `model.ckpt` (authors' recipe) with training logs | paper seeds: acc 93.77 ± 0.17, F1 89.88 ± 0.29 vs paper 94.07 ± 0.66 / 90.30 ± 1.03 — reproduced |
| `echoprime_view_classifier/release_v1.0.0` | authors' released ConvNeXt-B 11-view classifier | evaluated on EV9V only |
| `echoviewclip_stage1/v1` | official EchoViewCLIP stage-1 (ViT-B/16) trained on EV9V with the authors' recipe, plus test predictions | EV9V test Acc 94.14 / macro-F1 0.907; paper numbers are on private data |

`models_frozen/` holds exactly the five baselines and nothing else: `stfm_official`, `mstcn_official`, `asformer_official`, `echoprime_view_classifier`, `echoviewclip_stage1`. Pending: `asformer_official/v1` (added when their runs finish and are checked).

Our own pipeline components (e.g. the ResNet-18 frame encoder `ckpt_hash 79f4a41af6e3`) are frozen the same way under `models_internal/` (manifests in `docs/models_internal/`).

## Backup

Every frozen directory is also packaged as `<name>_<version>.tar.gz` and attached to the GitHub release
[`baselines-v1`](https://github.com/WilliamQiuzy/echo-view-routing/releases/tag/baselines-v1), so the reproduced
weights survive the loss of the server. To restore on a fresh machine:

```bash
gh release download baselines-v1 --repo WilliamQiuzy/echo-view-routing --dir /tmp/baselines
for f in /tmp/baselines/*.tar.gz; do tar -xzf "$f" -C /home/william/echo-view-routing/; done   # recreates models_frozen/ and models_internal/
sha256sum -c <(python -c "import json,glob; [print(f['sha256'], m.rsplit('/',1)[0]+'/'+f['file']) for m in glob.glob('models_frozen/*/*/MANIFEST.json')+glob.glob('models_internal/*/*/MANIFEST.json') for f in json.load(open(m))['files']]")
```
