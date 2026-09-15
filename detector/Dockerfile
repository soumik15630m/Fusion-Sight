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

# python + the two system libs OpenCV needs at runtime (libGL, glib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies first, as their own layer, so app/weights edits don't re-run the
# multi-GB torch install. The --extra-index-url for the cu126 wheels lives inside
# requirements.txt itself.
COPY detector/requirements.txt detector/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r detector/requirements.txt

# fusion/ and detector/ under one root -> /srv/fusion, /srv/detector.
COPY fusion ./fusion
COPY detector ./detector

# Run from detector/ with `app.main:app`, exactly as detector/README.md documents;
# app/__init__.py then makes /srv (repo root) importable for `fusion`.
WORKDIR /srv/detector

# "0" = first GPU (app/config.py default). Override to "cpu" to run without a GPU.
ENV BATTLESIGHT_DEVICE=0 \
    FUSION_ENABLED=1

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
