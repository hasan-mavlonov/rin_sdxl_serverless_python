FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# ---- Install GPU PyTorch ----
# cu121 is a good default for modern RunPod GPUs (A10, A40, etc.)
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- Install your Python deps ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project ----
COPY . /app

# (Optional) verify LoRA exists at build time
#   lora/rinxl_lora.safetensors should be in your repo
#   or mounted into the container's /app/lora
RUN test -f /app/lora/rinxl_lora.safetensors || echo "WARNING: LoRA file missing at build time."

CMD ["python", "-u", "handler.py"]
