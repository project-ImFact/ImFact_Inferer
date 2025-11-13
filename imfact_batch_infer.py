import torch
from tqdm import tqdm
import asyncio, io, json, boto3, logging
from concurrent.futures import ThreadPoolExecutor
from imfact_inferer import compute_news_reliability, load_models

# ---------------------------------
# 로그 설정
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,  # INFO 이상 레벨만 출력
    format="%(asctime)s [%(levelname)s] %(message)s",  # 시간, 레벨, 메시지
    handlers=[
        logging.StreamHandler()  # 콘솔에도 동시에 출력
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------
# 모델 로드
# ---------------------------------
tokenizer, title_model, body_model = load_models()

# ---------------------------------
# S3 클라이언트 생성
# ---------------------------------
s3 = boto3.client('s3')

# -------------------------------
# 비동기 처리 함수
# -------------------------------
async def infer_article(article, executor):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, compute_news_reliability,
                                        article["title"], article["body"],
                                        title_model, body_model, tokenizer)
    article["reliability_score"] = result["final_score"]
    return article

# -------------------------------
# 전체 배치 실행 함수
# -------------------------------
async def run_batch(input_bucket, input_key, output_bucket, output_key):
    logger.info(f"S3에서 {input_key} 다운로드 중...")
    buf = io.BytesIO()
    s3.download_fileobj(input_bucket, input_key, buf)
    buf.seek(0)
    articles = [json.loads(line) for line in buf.getvalue().decode("utf-8").splitlines()]
    logger.info(f"총 {len(articles)}개 기사 로드 완료")

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = [infer_article(a, executor) for a in articles]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Batch Inference"):
            try:
                res = await f
                results.append(res)
                if len(results) % 50 == 0:
                    torch.cuda.empty_cache()
                    logger.info(f"[진행상황] {len(results)}/{len(articles)}개 완료, 최근 점수={res['reliability_score']:.4f}")
            except Exception as e:
                logger.exception(f"[오류] 기사 처리 실패: {e}")

    logger.info("추론 완료, 업로드 준비 중...")
    with io.StringIO() as out_buf:
        for art in results:
            out_buf.write(json.dumps(art, ensure_ascii=False) + "\n")
        s3.put_object(Bucket=output_bucket, Key=output_key, Body=out_buf.getvalue().encode("utf-8"))

    print(f"{len(results)}개 기사 업로드 완료 s3://{output_bucket}/{output_key}")

if __name__ == "__main__":
    asyncio.run(run_batch("imfact-news", "summarized/news_summarized.jsonl",
                          "imfact-news", "inference/news_inference.jsonl"))