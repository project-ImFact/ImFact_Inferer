# NVIDIA PyTorch 공식 이미지 기반 (GPU + CUDA + Python)
FROM nvcr.io/nvidia/pytorch:23.04-py3

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 기본 패키지 업데이트
RUN apt-get update && apt-get install -y \
    git wget unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 복사
COPY . .

# 환경변수 설정 (AWS region, cache 경로 등)
ENV TORCH_HOME=/workspace/.cache
ENV TRANSFORMERS_CACHE=/workspace/.cache
ENV AWS_DEFAULT_REGION=ap-northeast-2

# 모델 캐시 폴더 생성
RUN mkdir -p /workspace/.cache

# GPU 버전 torch 확인
RUN python3 -c "import torch; print('Torch CUDA:', torch.cuda.is_available())"

# 실행 명령어
ENTRYPOINT ["python3", "imfact_batch_infer.py"]
