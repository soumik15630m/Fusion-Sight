# GPU inference image for the detector + fusion backend.
#
# Build context is the REPO ROOT (see docker-compose.yml), not detector/, because
# detector/app/__init__.py adds the repo root to sys.path so the sibling top-level
# `fusion/` package imports -- both directories must sit under one root in the
# image, exactly like the source tree.
#
# CUDA 12.6 base matches the pinned torch==2.13.0+cu126 / torchvision==0.28.0+cu126
# in detector/requirements.txt (those wheels come from PyTorch's cu126 index, which
# the requirements file's --extra-index-url already points at). The NVIDIA driver
# comes from the host via the container runtime; only the CUDA userspace libs are
# in the image.
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.12 (via deadsnakes) -- the pinned requirements were frozen on 3.12
# (numpy 2.5.x / contourpy 1.3.3 refuse to install on anything older). A venv
# gives a clean pip that owns its own site-packages (no PEP-668 friction). Plus
# the two system libs OpenCV needs at runtime (libGL, glib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

# venv on PATH: `python` / `pip` / `uvicorn` all resolve to the 3.12 venv.
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /srv

# Dependencies first, as their own layer, so app/weights edits don't re-run the
# multi-GB torch install. The --extra-index-url for the cu126 wheels lives inside
# requirements.txt itself.
COPY detector/requirements.txt detector/requirements.txt
RUN pip install -r detector/requirements.txt

# Detector only -- fusion is a separate service/image now, reached over the
# network via app/fusion_client.py (FUSION_WS_URL).
COPY detector ./detector

# Run from detector/ with `app.main:app`, as detector/README.md documents.
WORKDIR /srv/detector

# "0" = first GPU (app/config.py default). Override to "cpu" to run without a GPU.
# FUSION_WS_URL points at the fusion service (compose service name, or a mesh
# hostname/IP in the distributed deployment).
ENV BATTLESIGHT_DEVICE=0 \
    FUSION_WS_URL=ws://fusion:8100/ws/ingest

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
