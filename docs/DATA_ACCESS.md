# Data access checklist

Server layout: `/home/william/echo-view-routing/data/<dataset>/raw/` holds the untouched download, extracted
content sits next to it. The Mac keeps only a handful of example files under `demo/samples/<dataset>/`.
Agreements and accounts are signed by the project owner, never by an automated agent.

## Downloaded automatically (no agreement required)

| Dataset | Source | Licence | Server path | Status |
|---|---|---|---|---|
| EV9V | huggingface.co/datasets/bgx666/EV9V | CC-BY-4.0 | `data/ev9v/` | done (5,138 cines, Videos + Images) |
| CAMUS | humanheart-project.creatis.insa-lyon.fr, public Girder collection `CAMUS_public` (3.8 GB) | CAMUS terms (research) | `data/camus/` | downloading (`logs/download_camus.log`) |
| Unity Imaging | data.unityimaging.net `files/2020-12-05/{png-cache.zip,labels.zip}` (1.27 GB) | CC BY-NC-ND 4.0 — evaluation only, no derived data, no redistribution | `data/unity/` | downloading (`logs/download_unity.log`) |
| EchoXFlow | huggingface.co/datasets/Ahus-AIM/EchoXFlow (37,125 recordings, 666 exams) | CC BY-NC-SA 4.0 | `data/echoxflow/` | size/gating being checked; download a sample first, full only if needed |

## Needs your registration (I will download as soon as you forward the link or credentials)

| Dataset | What to do | What I need from you |
|---|---|---|
| EchoNet-Dynamic | access granted; hosted on Redivis (`aimi.echonet_dynamic:66s1:v1_0.echonet:fjdn`) | your Redivis API token (redivis.com → workspace → Settings → API tokens, read scope) placed in `secrets/redivis_token` on the server; then `scripts/download_echonet_redivis.py` |
| EchoNet-LVH | access granted; Redivis table `aimi.echonet_lvh:cchq:v1_0.echonet_lvh:m0ea` | same token |
| EchoNet-Pediatric | access granted; Azure Blob SAS URL (valid to 2026-10-13), stored in `secrets/echonet_pediatric_sas.url` on the server | downloading with azcopy into `data/echonet/pediatric/` |
| MIMIC-IV-Echo v1.0.1 + ECHOVIEW | https://physionet.org: (1) create an account, (2) complete CITI "Data or Specimens Only Research" training, (3) apply for credentialing (reference: your advisor), (4) sign the DUA for `mimic-iv-echo` and `echoview` | your PhysioNet username; the download uses `wget --user <you> --ask-password` on the server, so you type the password in `remote/shell.sh` yourself. The DICOM set is several hundred GB; we start with the ECHOVIEW CSV plus a stratified subset of studies |
| TMED-2 | the site now says: contact Mike Hughes, mhughes@cs.tufts.edu (Tufts) for access | the link or archive they send |
| TTE47 | request page is offline; contact THRIVE Centre, info@thrive-centre.com, referencing the EchoFine paper (Medical Image Analysis 2026) | the link or archive they send |

## Not downloaded on purpose

- Nothing is downloaded to the Mac except examples (`scripts/make_demo_bundle.py` / manual `scp`).
- Unity data are never used to build constructed streams (NoDerivatives clause).
