"""Convert a ModernBERT-Ja token classifier (label 1 = part of a typo) to Core ML.

The model returns P(label 1) for every token, so one pass over a sentence ranks its tokens.
It reuses the scoring model's encoder (additive padding mask, torch.export).

usage: python scripts/convert_detector.py <checkpoint> <out.mlpackage> [--batch 32] [--length 64]
"""
import argparse
import sys

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForTokenClassification

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from convert import Scoring  # noqa: E402  (also registers the new_ones converter)


class Detector(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.encoder = Scoring(model)  # Scoring.encode only touches model.model

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder.encode(input_ids, attention_mask)
        logits = self.model.classifier(self.model.drop(self.model.head(hidden)))
        return torch.softmax(logits.float(), dim=-1)[..., 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--length", type=int, default=64)
    a = p.parse_args()
    model = AutoModelForTokenClassification.from_pretrained(a.checkpoint, dtype=torch.float32, attn_implementation="eager").eval()
    b, n = a.batch, a.length
    if n > model.config.local_attention // 2:
        raise SystemExit(f"--length must be at most {model.config.local_attention // 2}")
    det = Detector(model).eval()
    example = (torch.randint(5, 1000, (b, n)), torch.ones(b, n, dtype=torch.long))
    with torch.no_grad():
        exported = torch.export.export(det, example).run_decompositions({})
    ml = ct.convert(
        exported,
        inputs=[ct.TensorType("input_ids", shape=(b, n), dtype=np.int32),
                ct.TensorType("attention_mask", shape=(b, n), dtype=np.int32)],
        outputs=[ct.TensorType("p_typo", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    ml.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
