FROM registry.runpod.io/pytorch/pytorch:2.1.2-py3.10-cuda12.1

# Install OS dependencies
RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python3", "handler.py"]
