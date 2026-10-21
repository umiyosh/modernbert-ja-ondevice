"""Fine-tune a ModernBERT-Ja for token classification (TYPO = part of a typo) on the rows of
build_data.py and augment.py, mixed and shuffled together. Saves a checkpoint every 10,000 steps.

The published detector is step 30,000 of one epoch over build_data.py + augment.py rows
(64,332 steps), chosen on the development set.

usage: python training/train.py <hf model> <data.npz[,more.npz]> <out dir> [--epochs 1.0]
"""
import argparse
import math
import os
import time

# On a 48GB Mac the 130m run needed 0.25 / 0.2 here to leave memory for the rest of the system.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

B, LR, WARMUP = 32, 5e-5, 1000
LABELS = {0: "OK", 1: "TYPO"}


class Rows:
    """Flat arrays with row offsets from one or more npz files."""

    def __init__(self, paths: list[str]):
        parts = [np.load(p) for p in paths]
        self.ids = np.concatenate([d["ids"] for d in parts])
        self.labels = np.concatenate([d["labels"] for d in parts])
        base = np.cumsum([0] + [d["offsets"][-1] for d in parts[:-1]])
        self.offsets = np.concatenate([parts[0]["offsets"][:1]] + [d["offsets"][1:] + b for d, b in zip(parts, base)])

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.ids[a:b], self.labels[a:b]


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=float, default=1.0)
    args = ap.parse_args()
    dev = device()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = Rows(args.data.split(","))
    print(f"{len(rows)} rows from {args.data}", flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()}).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = int(math.ceil(len(rows) / B) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, WARMUP, steps)
    os.makedirs(args.out, exist_ok=True)

    def batch(chunk: list[tuple[np.ndarray, np.ndarray]]):
        n = max(len(ids) for ids, _ in chunk)
        x = torch.full((len(chunk), n), tok.pad_token_id)
        y = torch.full((len(chunk), n), -100)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for i, (ids, lab) in enumerate(chunk):
            x[i, :len(ids)] = torch.from_numpy(ids.astype(np.int64))
            y[i, :len(lab)] = torch.from_numpy(lab.astype(np.int64))
            att[i, :len(ids)] = 1
        return x.to(dev), att.to(dev), y.to(dev)

    started, running, step = time.time(), 0.0, 0
    while step < steps:
        order = torch.randperm(len(rows)).tolist()
        for k in range(0, len(order), B):
            if step >= steps:
                break
            x, att, y = batch([rows[i] for i in order[k:k + B]])
            loss = model(input_ids=x, attention_mask=att, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            step += 1
            if step % 500 == 0:
                el = time.time() - started
                print(f"step {step}/{steps} loss {running / 500:.4f} elapsed {el / 60:.1f}m "
                      f"eta {(steps - step) * el / step / 60:.1f}m", flush=True)
                running = 0.0
            if step % 10000 == 0 or step == steps:
                model.save_pretrained(f"{args.out}/step{step}")
                tok.save_pretrained(f"{args.out}/step{step}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
