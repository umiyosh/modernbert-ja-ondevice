"""Token-classification rows from JWTD v2 train: each erroneous sentence with its error characters
labelled 1, and its correction with no error. Character spans come from diffing the pair; a
missing character (an insertion in the correction) labels the character before the gap.

ModernBERT-Ja 30m, 70m and 130m share one tokenizer, so the rows fit any of them.

usage: python training/build_data.py <jwtd train.jsonl> <out.npz> [--exclude sentences.txt] [--limit N]
"""
import argparse
import difflib
import json
import random

import numpy as np
from transformers import AutoTokenizer

L = 64
tok = AutoTokenizer.from_pretrained("sbintuitions/modernbert-ja-130m")


def error_chars(pre: str, post: str) -> set[int]:
    bad = set()
    for op, i1, i2, _, _ in difflib.SequenceMatcher(None, pre, post, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            bad.update(range(i1, i2))
        elif op == "insert":
            bad.add(max(i1 - 1, 0))
    return bad


def encode(text: str, bad: set[int]) -> tuple[list[int], list[int]]:
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=L)
    labels = [-100 if b == a else int(any(c in bad for c in range(a, b))) for a, b in enc["offset_mapping"]]
    return enc["input_ids"], labels


def excluded(path: str | None) -> set[str]:
    """Sentences to keep out of training (an evaluation set), one per line."""
    return {s.rstrip("\n") for s in open(path)} if path else set()


def save(rows: list[tuple[list[int], list[int]]], path: str) -> None:
    """Flat ids and labels with row offsets: a Python list of 2M rows costs ~4GB, this ~0.5GB."""
    lens = np.fromiter((len(ids) for ids, _ in rows), np.int64, len(rows))
    offsets = np.concatenate([[0], np.cumsum(lens)])
    ids = np.fromiter((x for r, _ in rows for x in r), np.int32, offsets[-1])
    labels = np.fromiter((x for _, l in rows for x in l), np.int8, offsets[-1])
    np.savez(path, ids=ids, labels=labels, offsets=offsets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("out")
    ap.add_argument("--exclude")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    rows = []
    for line in open(args.train):
        d = json.loads(line)
        pre, post = d["pre_text"], d["post_text"]
        bad = error_chars(pre, post)
        if not bad or len(bad) > 12:
            continue
        ids, lab = encode(pre, bad)
        if 1 not in lab:
            continue  # the error fell past the truncation
        rows.append((ids, lab))
        rows.append(encode(post, set()))
        if args.limit and len(rows) >= args.limit:
            break
    # Shuffled before the exclusion, the order the published model was trained on.
    random.seed(0)
    random.shuffle(rows)
    # JWTD train and test share some Wikipedia sentences, so either side of a pair can be excluded.
    skip = {tuple(encode(s, set())[0]) for s in excluded(args.exclude) if s.strip("。")}
    kept = [r for r in rows if tuple(r[0]) not in skip]
    save(kept, args.out)
    pos = sum(l.count(1) for _, l in kept)
    tot = sum(sum(x >= 0 for x in l) for _, l in kept)
    print(f"{len(kept)} rows (dropped {len(rows) - len(kept)} excluded), error tokens {pos}/{tot} ({pos / tot:.1%})")


if __name__ == "__main__":
    main()
