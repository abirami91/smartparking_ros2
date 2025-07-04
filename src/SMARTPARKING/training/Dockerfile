# Dockerfile.slim
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 1. System deps (no-op for NumPy via apt if you like)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. Install NumPy first
RUN pip install --no-cache-dir numpy

# 3. Then install PyTorch, TorchVision & ULTRALYTICS
RUN pip install --no-cache-dir \
      torch==2.1.1 torchvision==0.16.1 \
      -f https://download.pytorch.org/whl/torch_stable.html \
      ultralytics==8.1.0

# 4. Finally the rest of your deps
RUN pip install --no-cache-dir \
      opencv-python \
      pyyaml \
      shapely \
      scipy \
      tqdm

# 5. Copy your config & scripts
COPY pklot50.yaml .
COPY scripts/     ./scripts/
RUN chmod +x scripts/*.py

CMD ["bash"]
