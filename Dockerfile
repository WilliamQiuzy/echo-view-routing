# syntax=docker/dockerfile:1
#
# echo-view-routing in one image: the commands `echoview`, `echoview-cache` and `echoview-train`.
# The weights are not inside. They are downloaded on first use into /cache; mount a volume there to keep them.
#
#   docker build -t echo-view-routing .
#   docker build -t echo-view-routing:cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
#   docker build --target test .          # runs the linter and the tests inside the image's environment
#
# See docs/DOCKER.md for how to run it.

FROM python:3.12-slim AS base
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /opt/echoview

# torch.compile, which training stage 3 uses, builds small pieces of C at run time
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# The pinned requirements come first: this layer changes least. With an empty TORCH_INDEX, PyTorch is PyPI's build
# with its NVIDIA libraries on x86-64, and the CPU build elsewhere (an ARM image is mostly run on a laptop, where
# those 5 GB of libraries would never be used). A TORCH_INDEX of your own is used on every architecture.
ARG TORCH_INDEX=""
ARG TARGETARCH
COPY requirements.txt ./
RUN index="$TORCH_INDEX"; \
    if [ -z "$index" ] && [ "$TARGETARCH" != "amd64" ]; then index=https://download.pytorch.org/whl/cpu; fi; \
    if [ -n "$index" ]; then pip install --index-url "$index" $(grep -o -E '^(torch|torchvision)==[^ ]+' requirements.txt); fi \
 && pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .


FROM base AS test
COPY requirements-dev.txt CITATION.cff CHANGELOG.md Dockerfile .dockerignore ./
COPY configs ./configs
COPY tools ./tools
COPY tests ./tests
RUN pip install -r requirements-dev.txt \
 && ruff check src tests tools \
 && pytest -p no:cacheprovider


FROM base AS runtime
LABEL org.opencontainers.image.title="echo-view-routing" \
      org.opencontainers.image.description="Echocardiography view recognition and view segmentation for video" \
      org.opencontainers.image.source="https://github.com/WilliamQiuzy/echo-view-routing" \
      org.opencontainers.image.licenses="MIT"
COPY configs ./configs
# Any user may run the image (`--user $(id -u):$(id -g)` keeps the files it writes yours), so the two folders it
# writes to are open: /cache for downloads, /work as the working directory.
RUN useradd --create-home --uid 1000 echoview \
 && mkdir -p /cache /work \
 && chmod 1777 /cache /work
ENV HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache
USER echoview
WORKDIR /work
CMD ["echoview", "--help"]
