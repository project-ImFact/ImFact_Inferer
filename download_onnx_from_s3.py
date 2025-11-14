import os
import boto3
import logging
from botocore.exceptions import ClientError

from imfact_infer import ImFactInferer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def download_model_if_needed(
    bucket: str, 
    key: str, 
    local_path: str
):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        logger.info(f"[Model Download] ONNX 이미 다운로드됨: {local_path}")
        return

    logger.info(f"[Model Download] S3 → {local_path} 다운로드 시작")
    s3 = boto3.client("s3")

    try:
        s3.download_file(bucket, key, local_path)
        logger.info(f"[Model Download] 완료: {local_path}")
    except ClientError as e:
        logger.error(f"[Model Download] 실패: {e}")
        raise


def load_fast_inferer_from_s3(
    bucket_model: str = "imfact-model",
    title_key: str = "onnx_models/title_model.onnx",
    body_key: str = "onnx_models/body_model.onnx",
    cache_dir: str = "./onnx_models",
):
    os.makedirs(cache_dir, exist_ok=True)

    title_local = os.path.join(cache_dir, "title_model.onnx")
    body_local = os.path.join(cache_dir, "body_model.onnx")

    download_model_if_needed(bucket_model, title_key, title_local)
    download_model_if_needed(bucket_model, body_key, body_local)

    logger.info("[ImfactInferer] ONNX 모델 로드 중...")

    inferer = ImFactInferer(
        title_onnx_path=title_local,
        body_onnx_path=body_local,
        title_max_length=128,
        body_max_length=512,
    )
    logger.info("[ImfactInferer] Inference 엔진 준비 완료")

    return inferer


if __name__ == "__main__":
    inferer = load_fast_inferer_from_s3(
        bucket_model="imfact-model",
        title_key="onnx_models/title_model.onnx",
        body_key="onnx_models/body_model.onnx",
        cache_dir="./onnx_models",
    )

    title = "“당신의 휴대폰, 지금도 누군가 보고 있다? 경찰이 밝힌 충격 진실!”"
    body = (
        "온라인 커뮤니티에서는 최근 스마트폰 카메라가 사용자가 모르는 사이 자동으로 켜진다는 글이 급속히 퍼졌다. "
        "몇몇 이용자는 “폰에 검은 점이 생겼다”고 주장하며 해킹 의혹을 제기했다. "
        "하지만 실제로는 특정 사진 앱 업데이트 과정에서 발생한 단순 버그로 확인됐다. "
        "경찰 관계자는 “현재까지 해킹 정황은 전혀 발견되지 않았다”고 공식 입장을 밝혔다. "
        "전문가들은 “과도한 공포를 조장하며 조회수를 노리는 정보에 주의해야 한다”고 조언한다."
    )

    result = inferer.compute_news_reliability(title, body)
    print(result)
