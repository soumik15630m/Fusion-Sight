FROM python:3.12-slim

WORKDIR /srv

ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=15

# opencv-python (non-headless -- required, see requirements-docker.txt's note)
# needs these X11/GL shared libraries at import time; python:3.12-slim has
# none of them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# requirements-docker.txt is requirements.txt minus the pinned torch==...+cu126
# / torchvision==...+cu126 GPU wheels and the --extra-index-url pointing at
# them -- this image is CPU-only (BATTLESIGHT_DEVICE=cpu below), so it pulls
# plain CPU torch/torchvision separately below instead of the much larger
# CUDA-bundled wheels. Not a replacement for requirements.txt; that stays the
# real pinned stack for the GPU dev machine.
#
# CPU torch/torchvision MUST install first: ultralytics (in
# requirements-docker.txt) depends on `torch` with no pin of its own, so if
# it installs first, pip resolves that unpinned dependency from the default
# PyPI index -- which pulled torch==2.14.0 plus ~1GB of nvidia_cudnn_cu13/
# cuda_toolkit wheels the first time this was tried, exactly the multi-GB
# CUDA download this split was meant to avoid. Installing the pinned CPU
# build first means that requirement is already satisfied by the time
# ultralytics is installed, so pip never reaches for another one.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0

COPY detector/requirements-docker.txt detector/requirements-docker.txt
RUN pip install --no-cache-dir -r detector/requirements-docker.txt

COPY detector detector
COPY fusion fusion

WORKDIR /srv/detector
ENV BATTLESIGHT_DEVICE=cpu
# uvicorn adds its own cwd (/srv/detector) to sys.path, which resolves
# detector's own `app` package -- but fusion/ is a sibling one level up, so it
# needs /srv on the path too for `from fusion import ...` (app/main.py and the
# routers) to resolve.
ENV PYTHONPATH=/srv
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
