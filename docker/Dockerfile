FROM docker.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/dam-venv/bin:$PATH

WORKDIR /workspace/project

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxft2 \
    libxss1 \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/dam-venv

COPY . /workspace/project

RUN pip install --upgrade pip setuptools wheel && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
      torch torchvision torchaudio && \
    pip install -v .

CMD ["bash", "-lc", "trap : TERM INT; sleep infinity & wait"]