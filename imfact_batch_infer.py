import json
import io
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import boto3
from tqdm import tqdm

from imfact_infer import ImFactInferer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


_inferer = None

def init_inferer():
    global _inferer
    if _inferer is None:
        _inferer = ImFactInferer(
            title_onnx_path="./onnx_models/title_model.onnx",
            body_onnx_path="./onnx_models/body_model.onnx",
        )
    return _inferer


def process_single_article(article: dict):
    inferer = init_inferer()
    result = inferer.compute_news_reliability(article["title"], article["body"])
    article["reliability_score"] = result["final_score"]
    return article


async def run_batch(
    input_bucket: str,
    input_key: str,
    output_bucket: str,
    output_key: str,
    max_workers: int = 4,
):
    s3 = boto3.client("s3")

    logger.info(f"[JSONL Download] S3에서 입력 파일 다운로드 중: s3://{input_bucket}/{input_key}")

    buf = io.BytesIO()
    s3.download_fileobj(input_bucket, input_key, buf)
    buf.seek(0)

    articles = [json.loads(line) for line in buf.getvalue().decode("utf-8").splitlines()]
    total = len(articles)
    logger.info(f"[JSONL Download] 총 {total}개 기사 로드 완료")

    results = []

    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [loop.run_in_executor(executor, process_single_article, art) for art in articles]

        for f in tqdm(asyncio.as_completed(futures), total=len(futures), desc="Fast Batch Inference"):
            try:
                res = await f
                results.append(res)
                if len(results) % 50 == 0:
                    logger.info(f"[Batch] {len(results)}/{total} 처리 완료")
            except Exception as e:
                logger.exception(f"[Batch] 에러 발생: {e}")

    logger.info("[Batch] 모든 기사 추론 완료 → S3 업로드 준비 중")

    out_buf = io.StringIO()
    for art in results:
        out_buf.write(json.dumps(art, ensure_ascii=False) + "\n")

    s3.put_object(
        Bucket=output_bucket,
        Key=output_key,
        Body=out_buf.getvalue().encode("utf-8")
    )

    logger.info(f"[JSONL Upload] 업로드 완료 → s3://{output_bucket}/{output_key}")
    print(f"완료! 총 {total}개 중 {len(results)}개 기사 처리됨.")


if __name__ == "__main__":
    asyncio.run(
        run_batch(
            input_bucket="imfact-news",
            input_key="summarized/news_summarized.jsonl",
            output_bucket="imfact-news",
            output_key="inference/news_inference.jsonl",
            max_workers=4,
        )
    )
