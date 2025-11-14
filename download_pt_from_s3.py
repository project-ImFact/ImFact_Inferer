import os
import boto3
import logging
from botocore.exceptions import ClientError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def download_pt_if_needed(
    bucket: str,
    key: str,
    local_path: str,
):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        logger.info(f"[Model Cache] 이미 존재하여 다운로드 스킵: {local_path}")
        return local_path

    logger.info(f"[Model Download] S3: s3://{bucket}/{key} → {local_path}")

    s3 = boto3.client("s3")

    try:
        s3.download_file(bucket, key, local_path)
        logger.info(f"[Model Download] 완료: {local_path}")
    except ClientError as e:
        logger.error(f"[Model Download] 실패: {e}")
        raise

    return local_path


def download_all_imfact_pt(
    bucket_model="imfact-model",
    cache_dir="./saved_model"
):
    os.makedirs(cache_dir, exist_ok=True)

    title_pt_path = os.path.join(cache_dir, "Part1/BERT/best_model.pt")
    body_pt_path = os.path.join(cache_dir, "Part2/KoBERTSeg/best_model.pt")

    title_key = "saved_model/Part1/BERT/best_model.pt"
    body_key = "saved_model/Part2/KoBERTSeg/best_model.pt"

    logger.info("[PT Download] BERT 모델 다운로드 중...")
    download_pt_if_needed(bucket_model, title_key, title_pt_path)

    logger.info("[PT Download] KoBERTSeg 모델 다운로드 중...")
    download_pt_if_needed(bucket_model, body_key, body_pt_path)

    logger.info("[PT Download] 모든 모델 다운로드 완료")

    return {
        "title_pt": title_pt_path,
        "body_pt": body_pt_path,
    }


if __name__ == "__main__":
    models = download_all_imfact_pt(
        bucket_model="imfact-model",
        cache_dir="./saved_model"
    )
    print("[Done] 다운로드 파일:")
    print(models)
