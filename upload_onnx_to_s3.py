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


def upload_file_to_s3(local_path: str, bucket: str, key: str):
    s3 = boto3.client("s3")

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {local_path}")

    logger.info(f"[S3 Upload] 업로드 시작 → {local_path} → s3://{bucket}/{key}")

    try:
        s3.upload_file(local_path, bucket, key)
        logger.info(f"[S3 Upload] 완료 → s3://{bucket}/{key}")
    except ClientError as e:
        logger.error(f"[S3 Upload] 실패: {e}")
        raise


if __name__ == "__main__":
    TITLE_ONNX_LOCAL = "./onnx_models/title_model.onnx"
    BODY_ONNX_LOCAL = "./onnx_models/body_model.onnx"

    BUCKET = "imfact-model"
    TITLE_ONNX_KEY = "onnx_models/title_model.onnx"
    BODY_ONNX_KEY = "onnx_models/body_model.onnx"

    upload_file_to_s3(TITLE_ONNX_LOCAL, BUCKET, TITLE_ONNX_KEY)
    upload_file_to_s3(BODY_ONNX_LOCAL, BUCKET, BODY_ONNX_KEY)

    logger.info("[S3 Upload] 모든 ONNX 파일 업로드 완료")
