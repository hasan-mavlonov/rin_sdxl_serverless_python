FROM runpod/pytorch:2.1.2-py3.10-cuda12.1

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "handler.py"]
