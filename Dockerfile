FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/opt/sam3d \
    MOMENTUM_ENABLED=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/facebookresearch/sam-3d-body.git /opt/sam3d \
    && git -C /opt/sam3d checkout b5c765a0d89d789985e186d396315e7590887b94

WORKDIR /app

COPY requirements-space.txt ./
COPY LICENSE NOTICE.md ./
COPY motion2mixamorig ./motion2mixamorig
COPY private_assets ./private_assets

RUN python -m pip install -r requirements-space.txt \
    && python -c "from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body" \
    && python -c "import torch; assert 'dinov3_vith16plus' in torch.hub.list('facebookresearch/dinov3', trust_repo=True)"

COPY app.py ./
COPY scripts/test_sam3d.py ./scripts/test_sam3d.py
COPY scripts/test_mannequin.py ./scripts/test_mannequin.py

RUN python -m unittest scripts.test_sam3d scripts.test_mannequin

EXPOSE 7860

CMD ["python", "app.py"]
