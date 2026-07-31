FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# System deps for scientific python
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps — install first for better layer caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Project code (heavy data files are excluded via .dockerignore)
COPY . /workspace

# Ensure src/ is importable as a package
ENV PYTHONPATH=/workspace

# Default command: run baselines. Override with `docker run ... python -m src.train`
CMD ["python", "scripts/run_baselines.py"]
