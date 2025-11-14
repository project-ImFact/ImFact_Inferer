FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y git wget unzip curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python download_onnx_from_s3.py"]

CMD ["python imfact_batch_infer.py"]
