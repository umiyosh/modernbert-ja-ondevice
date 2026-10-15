"""Check a converted scoring model against PyTorch and time it on each compute unit.

usage: python scripts/verify.py <hf model> <model.mlpackage> [--batch 32] [--length 64]
"""
import argparse
import sys
import time

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from convert import Scoring  # noqa: E402

SENTENCES = [
    "新しいエディタは思ったより軽くて、以外と使いやすかった。",
    "お盆は実家に規制する予定なので、その週は休みをもらいます。",
    "法律の規制が厳しくなって、届け出の手間が増えた。",
    "一方で、耳読してから目視による正読に移行したいけど、それは有料なんだよね。",
    "バータリ的な対応はやはりやめよう。",
    "交通費の精算は月末までに経費システムで申請してね。",
]


def rows(tok, batch, length):
    """Masked rows: every real token of each sentence, masked in turn."""
    out = []
    for s in SENTENCES:
        ids = tok(s, truncation=True, max_length=length)["input_ids"]
        for p in range(1, len(ids) - 1):
            masked = list(ids)
            masked[p] = tok.mask_token_id
            out.append((masked, p, ids[p]))
    return out[: batch * (len(out) // batch)] or out


def batches(items, tok, batch, length):
    for k in range(0, len(items), batch):
        chunk = items[k:k + batch]
        ids = np.full((batch, length), tok.pad_token_id, dtype=np.int32)
        att = np.zeros((batch, length), dtype=np.int32)
        pos = np.zeros(batch, dtype=np.int32)
        tgt = np.zeros(batch, dtype=np.int32)
        for i, (x, p, t) in enumerate(chunk):
            ids[i, :len(x)] = x
            att[i, :len(x)] = 1
            pos[i], tgt[i] = p, t
        yield len(chunk), {"input_ids": ids, "attention_mask": att, "position": pos, "target": tgt}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("model")
    a.add_argument("package")
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--length", type=int, default=64)
    a = a.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    ref = Scoring(AutoModelForMaskedLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager")).eval()
    items = rows(tok, a.batch, a.length)
    feeds = list(batches(items, tok, a.batch, a.length))
    expected = []
    with torch.no_grad():
        for n, f in feeds:
            expected += ref(*(torch.from_numpy(f[k]).long() for k in ("input_ids", "attention_mask", "position", "target"))).tolist()[:n]
    for unit in ("CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"):
        model = ct.models.MLModel(a.package, compute_units=getattr(ct.ComputeUnit, unit))
        model.predict(feeds[0][1])  # load and compile before timing
        got, started = [], time.time()
        for n, f in feeds:
            got += model.predict(f)["logp"].tolist()[:n]
        secs = (time.time() - started) / len(feeds)
        diff = np.abs(np.array(got) - np.array(expected))
        print(f"{unit:12} {secs * 1000:7.1f} ms per batch of {a.batch}  |Δlogp| max {diff.max():.3f} mean {diff.mean():.3f}")


if __name__ == "__main__":
    main()
