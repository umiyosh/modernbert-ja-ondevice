"""Fine-tune Kotobae's detector (D) on speech-recognition error rows: the candidate C of PROTOCOL.md.

The loop is modernbert-ja-ondevice's training/train.py with what this work changes: it starts from the
published detector at a pinned revision instead of the base model, takes the learning rate and seed,
and saves a checkpoint every --save-every steps so val picks the step (step 8).

usage: python asr/train.py <data.npz[,more.npz]> <out dir> [--init M] [--revision R] [--epochs 2]
                           [--lr 2e-5] [--seed 0] [--save-every 1000] [--save-from 0] [--stop-at N]

--stop-at ends the run after step N while the schedule stays that of all --epochs, so a checkpoint val
picked can be trained again under other seeds without the steps after it. --save-from skips the
checkpoints before step N (each is 530 MB, and seeds trained for a soup are averaged at known steps).
"""

import argparse
import math
import os
import random
import time

# Leave memory for the rest of a 48GB Mac, as the 130m run of training/train.py needed.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

DETECTOR = "umiyosh/modernbert-ja-130m-typo-detector"
REVISION = "2bf8db36e6fa69b1eb11634b6c7ca00ebb7f9dff"  # the detector Kotobae ships (scripts/fetch-typo-models.sh)
B = 32
WARMUP = 0.06


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--init", default=DETECTOR)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--save-from", type=int, default=0)
    ap.add_argument("--stop-at", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rev = a.revision if a.init == DETECTOR else None
    tok = AutoTokenizer.from_pretrained(a.init, revision=rev)
    model = AutoModelForTokenClassification.from_pretrained(a.init, revision=rev).to(dev).train()
    rows = Rows(a.data.split(","))
    steps = int(math.ceil(len(rows) / B) * a.epochs)
    print(f"{len(rows)} rows from {a.data}; {steps} steps, lr {a.lr}, seed {a.seed}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(steps * WARMUP), steps)
    os.makedirs(a.out, exist_ok=True)

    def batch(chunk: list[tuple[np.ndarray, np.ndarray]]):
        n = max(len(ids) for ids, _ in chunk)
        x = torch.full((len(chunk), n), tok.pad_token_id)
        y = torch.full((len(chunk), n), -100)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for i, (ids, lab) in enumerate(chunk):
            x[i, : len(ids)] = torch.from_numpy(ids.astype(np.int64))
            y[i, : len(lab)] = torch.from_numpy(lab.astype(np.int64))
            att[i, : len(ids)] = 1
        return x.to(dev), att.to(dev), y.to(dev)

    def save(step: int) -> None:
        model.save_pretrained(f"{a.out}/step{step}")
        tok.save_pretrained(f"{a.out}/step{step}")

    gen = torch.Generator().manual_seed(a.seed)
    started, running, step = time.time(), 0.0, 0
    while step < steps:
        order = torch.randperm(len(rows), generator=gen).tolist()
        for k in range(0, len(order), B):
            if step >= steps:
                break
            x, att, y = batch([rows[i] for i in order[k : k + B]])
            loss = model(input_ids=x, attention_mask=att, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            step += 1
            if step % 100 == 0:
                el = time.time() - started
                print(f"step {step}/{steps} loss {running / 100:.4f} elapsed {el / 60:.1f}m "
                      f"eta {(steps - step) * el / step / 60:.1f}m", flush=True)
                running = 0.0
            if step >= a.save_from and (step % a.save_every == 0 or step == steps or step == a.stop_at):
                save(step)
            if step == a.stop_at:
                steps = step
    print("done", flush=True)


if __name__ == "__main__":
    main()
