# MXNet 호환 패치
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool  


import torch
import torch.nn as nn
from transformers import AutoConfig, BertModel
from kobert.pytorch_kobert import get_pytorch_kobert_model


class TitleBERT(nn.Module):
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
        pooled = self.dropout(outputs.pooler_output)
        return self.classifier(pooled)


class BertForSeg(nn.Module):
    def __init__(self, finetune=False):
        super().__init__()
        self.model, vocab = get_pytorch_kobert_model(cachedir=".cache")
        self.model.resize_token_embeddings(len(vocab))

        self.finetune = finetune
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x, segs, mask):
        top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        return top_vec


class SentenceClassifier(nn.Module):
    def __init__(self, window_size=3):
        super().__init__()
        conv_k = window_size * 2 - 2
        flat = 256 * 3

        ln_size = 1 if window_size == 1 else 3
        self.block1 = nn.Sequential(
            nn.Conv1d(768, 256, conv_k),
            nn.LayerNorm([256, ln_size]),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(flat, 2)
        )

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.block1(x)
        x = x.view(x.size(0), -1)
        return self.block2(x)


class KoBERTSeg(nn.Module):
    def __init__(self, finetune=False, window_size=3):
        super().__init__()
        self.bert = BertForSeg(finetune)
        self.classifier = SentenceClassifier(window_size)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]

        sents_vec = sents_vec * mask_cls[:, :, None].float()

        return self.classifier(sents_vec)


def export_title_onnx(pt_path: str, onnx_path: str):
    print("[Export] TitleBERT → ONNX 변환 시작")

    model = TitleBERT("skt/kobert-base-v1", 2)
    state = torch.load(pt_path, map_location="cpu")

    state = {k: v for k, v in state.items() if "position_ids" not in k}
    model.load_state_dict(state, strict=True)

    model.eval()

    dummy_input_ids = torch.zeros((1, 128), dtype=torch.long)
    dummy_attention = torch.ones((1, 128), dtype=torch.long)
    dummy_token_type = torch.zeros((1, 128), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention, dummy_token_type),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "token_type_ids": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"[Export] TitleBERT ONNX 저장됨: {onnx_path}")


def export_body_onnx(pt_path: str, onnx_path: str):
    print("[Export] KoBERTSeg → ONNX 변환 시작")

    model = KoBERTSeg(finetune=False, window_size=3)
    state = torch.load(pt_path, map_location="cpu")

    state = {k: v for k, v in state.items() if "position_ids" not in k}
    model.load_state_dict(state, strict=True)

    model.eval()

    dummy_src = torch.zeros((1, 512), dtype=torch.long)
    dummy_segs = torch.zeros((1, 512), dtype=torch.long)
    dummy_mask_src = torch.ones((1, 512), dtype=torch.long)

    dummy_clss = torch.tensor([[0, 10, 20, 30, 40, 50]], dtype=torch.long)
    dummy_mask_cls = torch.ones((1, 6), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_src, dummy_segs, dummy_clss, dummy_mask_src, dummy_mask_cls),
        onnx_path,
        input_names=["src", "segs", "clss", "mask_src", "mask_cls"],
        output_names=["logits"],
        dynamic_axes={
            "src": {0: "batch"},
            "segs": {0: "batch"},
            "mask_src": {0: "batch"},
            "clss": {0: "batch", 1: "num_sent"},
            "mask_cls": {0: "batch", 1: "num_sent"},
        },
        opset_version=17,
    )

    print(f"[Export] KoBERTSeg ONNX 저장됨: {onnx_path}")


# ------------------------------------------------------
# Main
# ------------------------------------------------------

if __name__ == "__main__":
    TITLE_PT = "./saved_model/Part1/BERT/best_model.pt"
    BODY_PT = "./saved_model/Part2/KoBERTSeg/best_model.pt"

    TITLE_ONNX = "./onnx_models/title_model.onnx"
    BODY_ONNX = "./onnx_models/body_model.onnx"

    export_title_onnx(TITLE_PT, TITLE_ONNX)
    export_body_onnx(BODY_PT, BODY_ONNX)

    print("[Export] 모든 ONNX 변환 완료")
