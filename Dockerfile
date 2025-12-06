FROM runpod/pytorch:2.1.2-py3.10-cuda12.1.0

# Install OS dependencies
RUN apt-get update && apt-get install -y git

# Set working directory
WORKDIR /app

# Copy requirements first for cached builds
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /app

# Start the handler
CMD ["python3", "handler.py"]
