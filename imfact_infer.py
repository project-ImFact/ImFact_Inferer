import logging
import numpy as np
import onnxruntime as ort
import re
import torch
import torch.nn.functional as F

import gluonnlp as nlp
from kobert.utils.utils  import get_tokenizer_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class KoBERTSegTokenizer:
    def __init__(self, sp_tokenizer, vocab, window_size: int = 3, max_word_len: int = 512):
        self.window_size = window_size
        self.max_word_len = max_word_len

        self.tokenizer = sp_tokenizer
        self.vocab = vocab

        self.pad_idx = self.vocab[self.vocab.padding_token]
        self.cls_idx = self.vocab[self.vocab.cls_token]
        self.cls_token = self.vocab.cls_token
        self.sep_token = self.vocab.sep_token

    def _length_processing(self, src_tokens_per_sent):
        special_token_num = (self.window_size * 2) * 2
        avg_len = (self.max_word_len - special_token_num) // (self.window_size * 2)
        return [sent[:avg_len] for sent in src_tokens_per_sent]

    def _get_token_type_ids(self, per_sent_ids):
        segs = []
        for i, sent_ids in enumerate(per_sent_ids):
            if i % 2 == 0:
                segs.append([0] * len(sent_ids))
            else:
                segs.append([1] * len(sent_ids))
        return segs

    def _get_cls_index(self, flat_ids):
        return [i for i, x in enumerate(flat_ids) if x == self.cls_idx]

    def _pad(self, ids_list):
        """
        ids_list: List[int]
        """
        if len(ids_list) >= self.max_word_len:
            return ids_list[: self.max_word_len]
        return ids_list + [self.pad_idx] * (self.max_word_len - len(ids_list))

    def encode(self, sentences):
        # 실문장만 토크나이즈 후 [PAD] 토큰 문장으로 양옆 패딩
        src = [self.tokenizer(s) for s in sentences]
        pad_sents = [[self.vocab.padding_token]] * (self.window_size - 1)
        src = pad_sents + src + pad_sents

        total_window = self.window_size * 2

        batch = {"src": [], "segs": [], "clss": [], "mask_src": [], "mask_cls": []}

        # stride 1 전체 슬라이딩
        for i in range(0, len(src) - total_window + 1):
            window = self._length_processing(src[i : i + total_window])

            per_sent_tokens = [
                [self.cls_token] + sent_tokens + [self.sep_token]
                for sent_tokens in window
            ]
            per_sent_ids = [self.vocab.to_indices(tokens) for tokens in per_sent_tokens]

            flat_ids = [tok_id for sent_ids in per_sent_ids for tok_id in sent_ids]
            segs_flat = [v for s in self._get_token_type_ids(per_sent_ids) for v in s]
            cls_positions = self._get_cls_index(flat_ids)

            src_padded = self._pad(flat_ids)
            segs_padded = self._pad(segs_flat)

            batch["src"].append(src_padded)
            batch["segs"].append(segs_padded)
            batch["clss"].append(cls_positions)
            batch["mask_src"].append([1 if x != self.pad_idx else 0 for x in src_padded])
            batch["mask_cls"].append([1] * len(cls_positions))

        return {
            "src":      np.array(batch["src"], dtype=np.int64),
            "segs":     np.array(batch["segs"], dtype=np.int64),
            "clss":     np.array(batch["clss"], dtype=np.int64),
            "mask_src": np.array(batch["mask_src"], dtype=np.int64),
            "mask_cls": np.array(batch["mask_cls"], dtype=np.int64),
        }


class ImFactInferer:
    def __init__(
        self,
        title_onnx_path: str = "./onnx_models/title_model.onnx",
        body_onnx_path: str = "./onnx_models/body_model.onnx",
        title_max_length: int = 512,
        body_max_length: int = 512,
    ):
        logger.info("[ImfactInferer] Initializing tokenizers & vocab...")

        sp_model_path = get_tokenizer_path()
        vocab = nlp.vocab.BERTVocab.from_sentencepiece(
                    sp_model_path, padding_token="[PAD]"
                )
        sp_tokenizer = nlp.data.BERTSPTokenizer(sp_model_path, vocab, lower=False)

        self.vocab = vocab
        self.sp_tokenizer = sp_tokenizer

        self.pad_idx = self.vocab[self.vocab.padding_token]
        self.cls_token = self.vocab.cls_token
        self.sep_token = self.vocab.sep_token

        self.title_max_length = title_max_length

        self.tokenizer_body = KoBERTSegTokenizer(
            sp_tokenizer=self.sp_tokenizer,
            vocab=self.vocab,
            window_size=3,
            max_word_len=body_max_length,
        )

        logger.info("[ImfactInferer] Initializing ONNX Runtime sessions...")
        so = ort.SessionOptions()

        self.title_sess = ort.InferenceSession(
            title_onnx_path,
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )
        self.title_inputs_names = [i.name for i in self.title_sess.get_inputs()]
        self.title_output_name = self.title_sess.get_outputs()[0].name

        self.body_sess = ort.InferenceSession(
            body_onnx_path,
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )
        self.body_inputs_names = [i.name for i in self.body_sess.get_inputs()]
        self.body_output_name = self.body_sess.get_outputs()[0].name

        logger.info("[ImfactInferer] ONNX Runtime models ready.")

    def _encode_title(self, title: str, body: str):
        # 학습 소스 코드의 BERTDataset.transform/length_processing/tokenize와 동일
        sent_list = [title] + body.split("\n")
        src = [self.sp_tokenizer(s) for s in sent_list]

        # length_processing: 누적 길이를 max_len-3([CLS][SEP][SEP])로 컷, title이 항상 먼저
        max_content_len = self.title_max_length - 3
        cnt = 0
        processed = []
        for sent in src:
            cnt += len(sent)
            if cnt > max_content_len:
                processed.append(sent[: len(sent) - (cnt - max_content_len)])
                break
            processed.append(sent)
        src = processed

        # 2세그먼트: [CLS] title [SEP] / concat(body줄들) [SEP]
        title_block = [self.cls_token] + src[0] + [self.sep_token]
        body_tokens = [t for sent in src[1:] for t in sent]
        body_block = body_tokens + [self.sep_token]

        title_ids = self.vocab.to_indices(title_block)
        body_ids = self.vocab.to_indices(body_block)

        input_ids = title_ids + body_ids
        token_type_ids = [0] * len(title_ids) + [1] * len(body_ids)

        pad_len = max(0, self.title_max_length - len(input_ids))
        input_ids = input_ids + [self.pad_idx] * pad_len
        token_type_ids = token_type_ids + [self.pad_idx] * pad_len

        attention_mask = [1 if x != self.pad_idx else 0 for x in input_ids]

        return {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.array([token_type_ids], dtype=np.int64),
        }

    def _infer_title_score(self, title: str, body: str) -> float:
        inputs = self._encode_title(title, body)
        ort_inputs = {name: inputs[name] for name in self.title_inputs_names}

        logits = self.title_sess.run([self.title_output_name], ort_inputs)[0]
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

        clickbait_prob = float(probs[0, 1])
        return 1.0 - clickbait_prob

    @staticmethod
    def split_sentences(text: str):
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sents if s]

    def _infer_body_score(self, body: str) -> float:
        sentences = self.split_sentences(body)

        # 문장 < 2개면 슬라이딩 윈도우가 0개 → 본문 신호 없음으로 처리 (인덱싱은 유지)
        if len(sentences) < 2:
            logger.warning(
                "[body] 문장 수 부족(%d) → body_score=1.0 처리", len(sentences)
            )
            return 1.0

        encoded = self.tokenizer_body.encode(sentences)

        ort_inputs = {name: encoded[name] for name in self.body_inputs_names}
        logits = self.body_sess.run([self.body_output_name], ort_inputs)[0]
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

        if probs.ndim == 3:
            clickbait_prob = float(probs[:, :, 1].max())

        elif probs.ndim == 2:
            clickbait_prob = float(probs[:, 1].max())

        else:
            raise ValueError(f"Unexpected logits shape: {probs.shape}")
        
        return 1.0 - clickbait_prob

    def compute_news_reliability(self, title: str, body: str):
        title_score = self._infer_title_score(title, body)
        body_score = self._infer_body_score(body)
        final_score = round(title_score * 0.6 + body_score * 0.4, 4)

        return {
            "title_score": round(title_score, 4),
            "body_score": round(body_score, 4),
            "final_score": final_score,
        }


if __name__ == "__main__":
    inferer = ImFactInferer(
        title_onnx_path="./onnx_models/title_model.onnx",
        body_onnx_path="./onnx_models/body_model.onnx",
    )

    title = "LH 국가유공자에 '명품집' 지원...\"나라에 보상받은 느낌\""
    body = (
        "최근 금융시장에서 개인 투자자들이 증가하고 있다. "
        "하지만 자극적인 광고성 콘텐츠가 넘쳐나면서 잘못된 정보를 접하기 쉽다. "
        "전문가들은 데이터 기반의 검증된 정보를 참고해야 한다고 조언한다."
    )

    result = inferer.compute_news_reliability(title, body)
    print(result)
