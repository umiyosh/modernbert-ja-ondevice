"""Sentence-level causal-LM score for every candidate a policy can score (as pll.py selects them):
log p(sentence with alt) - log p(sentence as written), summed over all tokens after BOS, with up to
PREV characters of the preceding text as an unscored prefix. Writes deltas-<out>.json (same rows, new
leads) for prodforce's ALT. Run from asr/, under ../guard.py.
2410.04027 found a GPT-2 doubled BERT's correction F1 in the same noisy-channel framework.

usage: python clm.py <eval dir> <name> <texts.json or labels.jsonl> <model> <out> [prev chars]
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from pll import sentence, texts  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
B = 32


@torch.no_grad()
def logp(model, tok, items):
    """items: (prefix ids, body ids) -> sum of log p(body | BOS + prefix)."""
    seqs = [[tok.bos_token_id] + p + b for p, b in items]
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), tok.pad_token_id if tok.pad_token_id is not None else 0)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    logits = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits
    out = []
    for i, (p, b) in enumerate(items):
        # Only the body's positions go to float32: the full batch x length x 99k logits did not fit the guard.
        start = 1 + len(p)
        lp = logits[i, start - 1:start - 1 + len(b)].float().log_softmax(-1)
        out.append(float(lp[torch.arange(len(b), device=DEV), torch.tensor(b, device=DEV)].sum()))
    return out


def main():
    d, name, src, repo, out_name = sys.argv[1:6]
    prev = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    T = texts(src)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).to(DEV).eval()
    deltas = json.load(open(os.path.join(d, f"deltas-{name}.json"), encoding="utf-8"))
    probes = json.load(open(os.path.join(d, f"res-{name}det-r0.json"), encoding="utf-8"))
    voice = json.load(open(os.environ.get("VT", "voice-table") + "/voice.json", encoding="utf-8"))
    sus = lambda key, s, e: max((p for x, y, p, _ in probes.get(key, []) if x < e and s < y), default=0.0)  # noqa: E731
    jobs = []  # (key, row, variant, prefix ids, body ids)
    for key, rs in deltas.items():
        text = T.get(key)
        if text is None:
            continue
        for k, (s, e, w, alts) in enumerate(rs):
            if not alts or (sus(key, s, e) < 0.018 and w not in voice):
                continue
            lo, hi = sentence(text, s, e)
            pre = tok.encode(text[max(0, lo - prev):lo], add_special_tokens=False) if prev else []
            for n, word in enumerate([w] + [a for a, _ in alts]):
                jobs.append((key, k, n, pre, tok.encode(text[lo:s] + word + text[e:hi], add_special_tokens=False)))
    jobs.sort(key=lambda j: len(j[3]) + len(j[4]))
    print(len(jobs), "sentences", flush=True)
    got = {}
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        for j, v in zip(chunk, logp(model, tok, [(j[3], j[4]) for j in chunk])):
            got[j[:3]] = v
        if i % (B * 100) == 0:
            print(i, flush=True)
    new = {}
    for key, rs in deltas.items():
        new[key] = [[s, e, w, [[a, round(got[(key, k, n + 1)] - got[(key, k, 0)], 3)] for n, (a, _) in enumerate(alts)
                               if (key, k, n + 1) in got]] for k, (s, e, w, alts) in enumerate(rs)]
    json.dump(new, open(os.path.join(d, f"deltas-{out_name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("done", len(new))


if __name__ == "__main__":
    main()
