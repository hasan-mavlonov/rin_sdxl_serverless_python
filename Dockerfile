FROM runpod/base:0-cuda12.1

# Install git (required for commits + some pip deps)
RUN apt-get update && apt-get install -y git

# Set working directory
WORKDIR /app

# Copy requirements earlier for caching
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
COPY . /app

CMD ["python3", "handler.py"]
