FROM runpod/pytorch:2.1.2-py3.10-cuda12.1-devel

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy project files
# -----------------------------
COPY requirements.txt .
COPY handler.py .
COPY entrypoint.sh .
COPY sdxl_inference ./sdxl_inference
COPY lora ./lora

# -----------------------------
# Install Python dependencies
# -----------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Permissions
# -----------------------------
RUN chmod +x entrypoint.sh

# -----------------------------
# Entrypoint for RunPod Serverless
# -----------------------------
ENTRYPOINT ["./entrypoint.sh"]
