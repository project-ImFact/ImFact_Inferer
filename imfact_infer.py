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
        pad_txt = ["blank."] * (self.window_size - 1)
        doc_txt = pad_txt + sentences + pad_txt

        src_txt = doc_txt[: self.window_size * 2]

        src_tokens_per_sent = [self.tokenizer(s) for s in src_txt]

        src_tokens_per_sent = self._length_processing(src_tokens_per_sent)

        src_subtokens_per_sent = [
            [self.cls_token] + sent_tokens + [self.sep_token]
            for sent_tokens in src_tokens_per_sent
        ]

        per_sent_ids = [self.vocab.to_indices(tokens) for tokens in src_subtokens_per_sent]

        flat_ids = [tok_id for sent_ids in per_sent_ids for tok_id in sent_ids]

        segs_per_sent = self._get_token_type_ids(per_sent_ids)
        segs_flat = [v for s in segs_per_sent for v in s]

        cls_positions = self._get_cls_index(flat_ids)

        src_padded = self._pad(flat_ids)
        segs_padded = self._pad(segs_flat)

        mask_src = [1 if x != self.pad_idx else 0 for x in src_padded]
        mask_cls = [1] * len(cls_positions)

        return {
            "src":      np.array([src_padded], dtype=np.int64),
            "segs":     np.array([segs_padded], dtype=np.int64),
            "clss":     np.array([cls_positions], dtype=np.int64),
            "mask_src": np.array([mask_src], dtype=np.int64),
            "mask_cls": np.array([mask_cls], dtype=np.int64),
        }


class ImFactInferer:
    def __init__(
        self,
        title_onnx_path: str = "./onnx_models/title_model.onnx",
        body_onnx_path: str = "./onnx_models/body_model.onnx",
        title_max_length: int = 128,
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

    def _encode_title(self, title: str):
        tokens = self.sp_tokenizer(title)

        max_sub_len = self.title_max_length - 2
        tokens = tokens[:max_sub_len]

        tokens = [self.cls_token] + tokens + [self.sep_token]

        input_ids = self.vocab.to_indices(tokens)

        if len(input_ids) < self.title_max_length:
            input_ids = input_ids + [self.pad_idx] * (self.title_max_length - len(input_ids))
        else:
            input_ids = input_ids[: self.title_max_length]

        attention_mask = [1 if x != self.pad_idx else 0 for x in input_ids]
        token_type_ids = [0] * self.title_max_length

        return {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.array([token_type_ids], dtype=np.int64),
        }

    def _infer_title_score(self, title: str) -> float:
        inputs = self._encode_title(title)
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
        if len(sentences) < 3:
            sentences += [sentences[-1]] * (3 - len(sentences))

        encoded = self.tokenizer_body.encode(sentences)

        ort_inputs = {name: encoded[name] for name in self.body_inputs_names}
        logits = self.body_sess.run([self.body_output_name], ort_inputs)[0]
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

        if probs.ndim == 3:
            clickbait_prob = float(probs[:, :, 1].mean())

        elif probs.ndim == 2:
            clickbait_prob = float(probs[:, 1].mean())

        else:
            raise ValueError(f"Unexpected logits shape: {probs.shape}")
        
        return 1.0 - clickbait_prob

    def compute_news_reliability(self, title: str, body: str):
        title_score = self._infer_title_score(title)
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
