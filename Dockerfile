FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY motion2mixamorig ./motion2mixamorig

RUN python -m pip install --upgrade pip setuptools wheel "numpy>=1.26,<2" \
    && python -m pip install --no-build-isolation "chumpy==0.70" \
    && python -m pip install -e . "gradio==5.49.1" "numpy>=1.26,<2"

COPY app.py ./

EXPOSE 7860

CMD ["python", "app.py"]
