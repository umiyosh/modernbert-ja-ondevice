"""Scores every candidate of a deltas file with kanakanji-edge-33m: for written word w and alternative a
at s..e, log P(a | left context, reading of a) - log P(w | same), with a few characters of the right
context on both sides. Writes kk-<name>.json: {key: [[s, e, alt, lead], ...]}. Run from asr/.

usage: python kk_score.py <eval dir> <name> <texts.json or labels.jsonl>
"""

import json
import os
import sys

import fugashi
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "yuuki14202028/kanakanji-edge-33m"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO).to(DEV).eval()
tagger = fugashi.Tagger()
CACHE = {}


def reading(text):
    if text not in CACHE:
        out = []
        for w in tagger(text):
            k = getattr(w.feature, "kana", None)
            out.append(k if k and k != "*" else w.surface)
        CACHE[text] = "".join(out)
    return CACHE[text]


@torch.no_grad()
def batch_logp(items):
    """items: (prompt ids, target ids) -> sum of log P(target | prompt)."""
    seqs = [p + t for p, t in items]
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    lp = model(ids.to(DEV), attention_mask=mask.to(DEV)).logits.float().log_softmax(-1).cpu()
    out = []
    for i, (p, t) in enumerate(items):
        pos = torch.arange(len(p) - 1, len(p) + len(t) - 1)
        out.append(float(lp[i, pos, torch.tensor(t)].sum()))
    return out


def texts(path):
    if path.endswith(".jsonl"):
        return {r["path"]: r["hyp"] for r in map(json.loads, open(path, encoding="utf-8"))}
    return {t["id"]: t["text"] for t in json.load(open(path, encoding="utf-8"))}


def main():
    d, name, src = sys.argv[1], sys.argv[2], sys.argv[3]
    T = texts(src)
    deltas = json.load(open(os.path.join(d, f"deltas-{name}.json"), encoding="utf-8"))
    jobs = []
    for key, rows in deltas.items():
        text = T.get(key)
        if text is None:
            continue
        for s, e, w, alts in rows:
            left, right = text[max(0, s - 40):s], text[e:e + 6]
            rk, rt = reading(right)[:6], right[:3]
            for a, _ in alts:
                # The model card's format: context marker, left context, input marker, katakana, output marker.
                prompt = tok.encode(f"{left}{reading(a)}{rk}", add_special_tokens=False)
                jobs.append((key, s, e, a, prompt, tok.encode(w + rt, add_special_tokens=False), tok.encode(a + rt, add_special_tokens=False)))
    print(len(jobs), "pairs", flush=True)
    out = {}
    B = 64
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        items = [(j[4], j[5]) for j in chunk] + [(j[4], j[6]) for j in chunk]
        lp = batch_logp(items)
        for k, j in enumerate(chunk):
            out.setdefault(j[0], []).append([j[1], j[2], j[3], round(lp[len(chunk) + k] - lp[k], 3)])
        if i % (B * 200) == 0:
            print(i, flush=True)
    json.dump(out, open(os.path.join(d, f"kk-{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("done", len(out))


if __name__ == "__main__":
    main()
