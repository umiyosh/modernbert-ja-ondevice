"""Fujii & Shinnou (NLP 2021, P8-20) as a scorer for the paper's comparison (Kotobae #98, plan step 3): the
untrained Tohoku BERT (whole-word masking) sees the sentence with the word replaced by one [MASK], and the lead
of an alternative is its log P at the mask minus the written word's. As in the paper, only words that are one
vocabulary entry can be compared; other alternatives are dropped, and a row whose written word is not one entry
keeps no alternatives. Same rows and gate as pll.py (detector 0.018 or a voice-table word), so the comparison
changes only the scorer. Run from asr/, under ../guard.py.

usage: python pll_fs.py <eval dir> <name> <texts.json or labels.jsonl> <model> <out>
"""

import json
import os
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from pll import sentence, texts

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LEN, B = 128, 64


def masked(tok, left, right):
    l_ids, r_ids = tok.encode(left, add_special_tokens=False), tok.encode(right, add_special_tokens=False)
    keep = (LEN - 3) // 2
    l_ids, r_ids = l_ids[-keep:], r_ids[:LEN - 3 - min(keep, len(l_ids))]
    ids = [tok.cls_token_id] + l_ids + [tok.mask_token_id] + r_ids + [tok.sep_token_id]
    return ids, len(l_ids) + 1


def main():
    d, name, src, repo, out_name = sys.argv[1:6]
    T = texts(src)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForMaskedLM.from_pretrained(repo).to(DEV).eval()
    vocab = tok.get_vocab()
    deltas = json.load(open(os.path.join(d, f"deltas-{name}.json"), encoding="utf-8"))
    probes = json.load(open(os.path.join(d, f"res-{name}det-r0.json"), encoding="utf-8"))
    voice = json.load(open(os.environ.get("VT", "voice-table") + "/voice.json", encoding="utf-8"))
    sus = lambda key, s, e: max((p for x, y, p, _ in probes.get(key, []) if x < e and s < y), default=0.0)  # noqa: E731
    jobs = []  # (key, row index, ids, mask position)
    for key, rs in deltas.items():
        text = T.get(key)
        if text is None:
            continue
        for k, (s, e, w, alts) in enumerate(rs):
            if (sus(key, s, e) >= 0.018 or w in voice) and w in vocab:
                lo, hi = sentence(text, s, e)
                jobs.append((key, k, *masked(tok, text[lo:s], text[e:hi])))
    print(len(jobs), "masked rows", flush=True)
    lps = {}
    with torch.no_grad():
        for i in range(0, len(jobs), B):
            chunk = jobs[i:i + B]
            x = torch.full((len(chunk), LEN), tok.pad_token_id)
            att = torch.zeros((len(chunk), LEN), dtype=torch.long)
            for j, (_, _, ids, _) in enumerate(chunk):
                x[j, :len(ids)] = torch.tensor(ids)
                att[j, :len(ids)] = 1
            logits = model(input_ids=x.to(DEV), attention_mask=att.to(DEV)).logits
            pos = torch.tensor([at for _, _, _, at in chunk])
            lp = logits[torch.arange(len(chunk)), pos.to(DEV)].float().log_softmax(-1).cpu()
            for (key, k, _, _), row in zip(chunk, lp):
                lps[(key, k)] = row
    new = {}
    for key, rs in deltas.items():
        out = []
        for k, (s, e, w, alts) in enumerate(rs):
            row = lps.get((key, k))
            scored = [] if row is None else [[alt, round(float(row[vocab[alt]] - row[vocab[w]]), 3)]
                                              for alt, _ in alts if alt in vocab]
            out.append([s, e, w, scored])
        new[key] = out
    json.dump(new, open(os.path.join(d, f"deltas-{out_name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for pre, suf in (("res-", "det-r0.json"), ("kk-", ".json")):
        src_p, dst = f"{pre}{name}{suf}", os.path.join(d, f"{pre}{out_name}{suf}")
        if os.path.exists(os.path.join(d, src_p)) and not os.path.lexists(dst):
            os.symlink(src_p, dst)
    print("done", len(new), "rows scored", len(lps))


if __name__ == "__main__":
    main()
