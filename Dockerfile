FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# --------- IMPORTANT ---------
# Install CPU PyTorch – RunPod injects GPU libs automatically.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# ---------------------------------

# Install diffusers + accelerate + transformers
RUN pip install diffusers==0.27.2 transformers accelerate safetensors pillow

# Install any extra Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt || true

# Copy project files
COPY . /app

CMD ["python3", "handler.py"]
