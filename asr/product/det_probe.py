"""Detector-only yardsticks on the dev half (Kotobae #98, the detector's pretraining): for each run, the
share of real-voice errors whose span the detector would send to the scorer (max P over the span at
or above the screening line) or mark on its own, and how many tokens of clean clips cross the same
lines per 1,000 characters (the scorer's load and the detector's false marks). Only the dev half of
split-calib.json is read.

usage (from asr/): python tools/det_probe.py <run dir>... [--cpu]
"""
import collections
import json
import os
import sys

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
CPU = "--cpu" in sys.argv
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

import prod  # noqa: E402

es = prod.eval_spans
half = json.load(open("split-calib.json"))
LINES = (0.005, 0.018, 0.4707, 0.68)
B = 64


@torch.no_grad()
def token_p(run, texts):
    dev = "cpu" if CPU or not torch.backends.mps.is_available() else "mps"
    tok = AutoTokenizer.from_pretrained(run)
    model = AutoModelForTokenClassification.from_pretrained(run).to(dev).eval()
    out = []
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], return_offsets_mapping=True, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        offs = enc.pop("offset_mapping").tolist()
        p = model(**{k: v.to(dev) for k, v in enc.items()}).logits.float().softmax(-1)[..., 1].cpu().tolist()
        out += [[(a, b, q) for (a, b), q in zip(o, row) if b > a] for o, row in zip(offs, p)]
    return out


def spans(r):
    """Each error of a dev clip as (kind, raw start, raw end)."""
    _, at = es.norm_map(r["hyp"])
    out = []
    for e in r["edits"] or []:
        x, y = e["hyp_chars"]
        if y > x:
            a, b = es.raw_span(at, x, y)
            out.append((e["kind"], a, b))
    return out


def probe(run, rs):
    found, total = collections.Counter(), collections.Counter()
    clean_hits, chars = collections.Counter(), 0
    for r, toks in zip(rs, token_p(run, [r["hyp"] for r in rs])):
        if r["status"] == "clean":
            chars += len(r["hyp"])
            for line in LINES:
                clean_hits[line] += sum(q >= line for _, _, q in toks)
            continue
        for kind, a, b in spans(r):
            p = max((q for s, e, q in toks if s < b and a < e), default=0.0)
            total[kind] += 1
            for line in LINES:
                found[kind, line] += p >= line
    return found, total, clean_hits, chars


def main():
    rs = [r for r in prod.VAL if half[r["path"]] == "dev"]
    for run in ARGS:
        found, total, clean_hits, chars = probe(run, rs)
        print(run)
        for kind in sorted(total, key=lambda k: -total[k]):
            print(f"  {kind:10s} {total[kind]:4d}: " + " ".join(
                f"≥{line} {100 * found[kind, line] / total[kind]:5.1f}%" for line in LINES))
        print("  clean tokens per 1,000 chars: " + " ".join(
            f"≥{line} {1000 * clean_hits[line] / chars:6.2f}" for line in LINES), flush=True)


if __name__ == "__main__":
    main()
