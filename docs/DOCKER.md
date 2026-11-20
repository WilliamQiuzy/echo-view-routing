# Docker

The image holds the package and its pinned requirements (`requirements.txt`, Python 3.12). It holds no weights and no data. The weights are downloaded from the Hugging Face Hub the first time a command needs them.

## Get the image

```bash
docker pull ghcr.io/williamqiuzy/echo-view-routing:latest        # NVIDIA GPU on x86-64, CPU on ARM (Apple silicon)
docker pull ghcr.io/williamqiuzy/echo-view-routing:latest-cpu    # CPU only, much smaller
```

Each release also has its own tags, for example `1.2.0` and `1.2.0-cpu`. Or build the image from a clone:

```bash
docker build -t echo-view-routing .
docker build -t echo-view-routing:cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
```

## Run it

The examples call the image `IMAGE`:

```bash
IMAGE=ghcr.io/williamqiuzy/echo-view-routing:latest
```

**Self-check.** It downloads the weights and five sample clips into the volume `echoview-cache`, runs the model and compares with stored results. It ends with `SELF-CHECK PASSED`.

```bash
docker run --rm -v echoview-cache:/cache $IMAGE echoview selfcheck
```

**On a GPU.** Add `--gpus all`. The host needs an NVIDIA driver and the NVIDIA Container Toolkit.

```bash
docker run --rm --gpus all -v echoview-cache:/cache $IMAGE echoview selfcheck
```

**Your recordings.** Mount them read-only and write the results to the current folder. `--user` makes the result files yours.

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v echoview-cache:/cache -v /path/to/recordings:/data:ro -v "$PWD":/work \
    $IMAGE echoview classify /data --out views.csv
```

**Evaluation and training** work the same way, with the dataset mounted.

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v echoview-cache:/cache -v /data/EV9V:/data/EV9V:ro -v "$PWD":/work \
    $IMAGE echoview evaluate --images /data/EV9V/Images --out report.json
```

Training stages 1 and 2 decode their frames in worker processes, which exchange data through shared memory. Docker gives a container 64 MB of it; add `--shm-size 8g` to those runs.

## What is where in the image

| Path | Content |
|---|---|
| `/work` | the working directory; mount your output folder here |
| `/cache` | downloads from the Hugging Face Hub (`HF_HOME=/cache/huggingface`); mount a volume to keep them |
| `/opt/echoview/configs` | the training recipes, for example `--config /opt/echoview/configs/stage3.json` |

- **Weights from a folder.** To run without a network, mount a folder with the weights and name it: `-v /path/to/weights:/weights:ro -e ECHOVIEW_WEIGHTS=/weights`.
- **The user.** The image runs as the unprivileged user `echoview` (uid 1000). Any other uid works too, because `/cache` and `/work` are writable for everyone.

## How the image is tested

`docker build --target test .` installs the developer requirements on top of the pinned ones and runs the linter and the whole test suite inside the image's environment. The build fails when a test fails.
