import torch
import torch.nn as nn
import torch.nn.functional as F
import re, textwrap, io, boto3, logging
from kobert_tokenizer.kobert_tokenizer import KoBERTTokenizer
from transformers import AutoConfig, BertModel

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

# ===============================================================
# BERT 제목 모델 정의
# ===============================================================
class BERT(nn.Module):
    def __init__(self, pretrained_name: str, num_classes: int):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_name)
        self.bert = BertModel.from_pretrained(pretrained_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ===============================================================
# KoBERTSeg 본문 모델 정의
# ===============================================================
from kobert.pytorch_kobert import get_pytorch_kobert_model

class Bert(nn.Module):
    def __init__(self, finetune_bert=False):
        super().__init__()
        self.model, vocab = get_pytorch_kobert_model(cachedir=".cache")
        self.model.resize_token_embeddings(len(vocab))
        self.finetune = finetune_bert
        if not self.finetune:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        return top_vec


class Classifier(nn.Module):
    def __init__(self, window_size):
        super().__init__()

        if window_size == 1:
            conv_kernel_size = 2
            flat_size = 256
        else:
            conv_kernel_size = window_size * 2 - 2
            flat_size = 256 * 3

        ln_size = 1 if window_size == 1 else 3

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=256, kernel_size=conv_kernel_size),
            nn.LayerNorm([256, ln_size]),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(nn.Linear(flat_size, 2))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous()
        out = self.block1(x)
        out = out.view(batch_size, -1)
        out = self.block2(out)
        return out


class KoBERTSeg(nn.Module):
    def __init__(self, finetune_bert=False, window_size=3):
        super().__init__()
        self.bert = Bert(finetune_bert=finetune_bert)
        self.classifier = Classifier(window_size=window_size)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        classified = self.classifier(sents_vec)
        return classified


# ===============================================================
# 모델 로드 함수
# ===============================================================
def load_models(device="cuda"):
    logger.info("[Info] 모델 및 토크나이저 로드 중...")

    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    s3 = boto3.client('s3', region_name='ap-northeast-2')
    bucket = "imfact-model"
    title_key = "saved_model/Part1/BERT/best_model.pt"
    body_key = "saved_model/Part2/KoBERTSeg/best_model.pt"

    # 제목 모델
    title_buf = io.BytesIO()
    s3.download_fileobj(bucket, title_key, title_buf)
    title_buf.seek(0)
    state_title = torch.load(title_buf, map_location=device)
    title_model = BERT("skt/kobert-base-v1", 2)
    clean_state_title = {k: v for k, v in state_title.items() if "position_ids" not in k}
    title_model.load_state_dict(clean_state_title, strict=True)
    title_model.to(device)
    title_model.eval()

    # 본문 모델
    body_buf = io.BytesIO()
    s3.download_fileobj(bucket, body_key, body_buf)
    body_buf.seek(0)
    state_body = torch.load(body_buf, map_location=device)
    body_model = KoBERTSeg(finetune_bert=False, window_size=3)
    clean_state_body = {k: v for k, v in state_body.items() if "position_ids" not in k}
    body_model.load_state_dict(clean_state_body, strict=True)
    body_model.to(device)
    body_model.eval()

    logger.info("[Info] 모델 로드 완료")
    return tokenizer, title_model, body_model


# ===============================================================
# 단일 뉴스 기사 신뢰도 추론 함수
# ===============================================================
def compute_news_reliability(title, body, title_model, body_model, tokenizer, device="cuda"):
    def split_sentences(text):
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sents if s]

    # 제목 신뢰도
    title_inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        title_logits = title_model(**title_inputs)
        probs = F.softmax(title_logits, dim=-1).squeeze(0)
        title_score = 1 - probs[0].item()  # class0 = clickbait
    
    # 본문 신뢰도
    sentences = split_sentences(body)
    if len(sentences) < 6:
        sentences += [sentences[-1]] * (6 - len(sentences))
    window_size = 6
    segment_scores = []
    for i in range(len(sentences) - window_size + 1):
        segment = sentences[i:i+window_size]
        inputs = tokenizer(segment, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        src, segs, mask_src = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
        clss = torch.arange(len(segment)).unsqueeze(0).to(device)
        mask_cls = torch.ones((1, len(segment)), dtype=torch.long).to(device)
        with torch.no_grad():
            logits = body_model(src, segs, clss, mask_src, mask_cls)
            probs = F.softmax(logits, dim=-1)
            score = 1 - probs[:, 1].mean().item()
            segment_scores.append(score)
    body_score = sum(segment_scores) / len(segment_scores)

    final_score = round((title_score * 0.6 + body_score * 0.4), 4)

    return {"title_score": title_score, "body_score": body_score, "final_score": final_score}
