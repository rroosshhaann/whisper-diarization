# CHANGE: Use 'base' or 'runtime' WITHOUT cuDNN. 
# This prevents the system from having a conflicting version pre-installed.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# Added 'git' and build tools which are often needed for compiling extensions
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    cython3 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

COPY constraints.txt requirements.txt ./

# Install Python dependencies
# Upgrade pip to ensure it handles modern wheels correctly
RUN pip install --upgrade pip \
    && pip install cython \
    && pip install -c constraints.txt -r requirements.txt

# ENVIRONMENT VARIABLE FIX
# We still keep this! Even without the system conflict, 
# this ensures `ctranslate2` (used by faster-whisper) finds the specific 
# cuDNN libraries installed by PyTorch/Pip.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

COPY . .

# Create directories
RUN mkdir -p /data/input /data/output /tmp/diarization_uploads

# Expose API port
EXPOSE 8001

# Default: run API server
# Override with: docker run ... python diarize.py -a /data/input/file.wav
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
CMD []