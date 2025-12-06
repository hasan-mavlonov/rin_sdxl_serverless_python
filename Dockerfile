FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git wget libgl1 libglib2.0-0

WORKDIR /app

# Install PyTorch with CUDA 12.1 (CPU fallback is automatic on serverless)
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install diffusers + accelerate + transformers
RUN pip install diffusers==0.27.2 transformers accelerate safetensors pillow

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

CMD ["python3", "handler.py"]
