"""Rescores every candidate of a deltas file with another masked LM, the way typo::pipeline::score does:
the sentence around the word, cut to 32 tokens centred on it, and the summed log P of the word's tokens
and two on each side, each masked in turn; the lead is alt's sum minus the written word's.
Writes deltas-<out>.json (same rows, new leads) and links res-/kk- of <name> to <out>. Run from asr/,
under ../guard.py.

usage: python pll.py <eval dir> <name> <texts.json or labels.jsonl> <model> <out>
"""

import json
import os
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LEN, NEIGHBOURS, B = 32, 2, 256
ENDS = "。！？\n"


def texts(path):
    if path.endswith(".jsonl"):
        return {r["path"]: r["hyp"] for r in map(json.loads, open(path, encoding="utf-8"))}
    return {t["id"]: t["text"] for t in json.load(open(path, encoding="utf-8"))}


def sentence(text, s, e):
    lo = max((i + 1 for i in range(s) if text[i] in ENDS), default=0)
    hi = next((i + 1 for i in range(e, len(text)) if text[i] in ENDS), len(text))
    return lo, hi


def around(tok, text, s, e):
    enc = tok(text, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    inside = [i for i, (a, b) in enumerate(offs) if b > a and a < e and s < b]
    if not inside:
        return None
    if len(ids) > LEN:
        body = LEN - 2
        mid = (inside[0] + inside[-1]) // 2
        lo = max(1, min(max(0, mid - body // 2), len(ids) - 1 - body))
        ids = [ids[0]] + ids[lo:lo + body] + [ids[-1]]
        inside = [i - lo + 1 for i in inside if i >= lo and i - lo + 1 < LEN - 1]
        if not inside:
            return None
    lo = max(1, inside[0] - NEIGHBOURS)
    hi = min(inside[-1] + NEIGHBOURS, len(ids) - 2)
    return ids, list(range(lo, hi + 1))


def main():
    d, name, src, repo, out_name = sys.argv[1:6]
    T = texts(src)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForMaskedLM.from_pretrained(repo, dtype=torch.float16).to(DEV).eval()
    deltas = json.load(open(os.path.join(d, f"deltas-{name}.json"), encoding="utf-8"))
    # Only the rows a policy can score: the detector suspects them, or they are voice-table words.
    probes = json.load(open(os.path.join(d, f"res-{name}det-r0.json"), encoding="utf-8"))
    voice = json.load(open(os.environ.get("VT", "voice-table") + "/voice.json", encoding="utf-8"))
    sus = lambda key, s, e: max((p for x, y, p, _ in probes.get(key, []) if x < e and s < y), default=0.0)  # noqa: E731
    rows, owner = [], []  # (ids, at), (key, row index, variant index)
    plan = {}
    for key, rs in deltas.items():
        text = T.get(key)
        if text is None:
            continue
        for k, (s, e, w, alts) in enumerate(rs):
            if sus(key, s, e) < 0.018 and w not in voice:
                continue
            lo, hi = sentence(text, s, e)
            sent, a, b = text[lo:hi], s - lo, e - lo
            for n, word in enumerate([w] + [alt for alt, _ in alts]):
                v = around(tok, sent[:a] + word + sent[b:], a, a + len(word))
                if v is None:
                    continue
                plan[(key, k, n)] = True
                for at in v[1]:
                    rows.append((v[0], at))
                    owner.append((key, k, n))
    print(len(rows), "masked rows", flush=True)
    sums = {}
    with torch.no_grad():
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            # One padded length for every batch: MPS keeps buffers per input shape, and varying shapes
            # let them pile up (the 2026-10-21 run reached 25 GB).
            n = LEN
            x = torch.full((len(chunk), n), tok.pad_token_id)
            att = torch.zeros((len(chunk), n), dtype=torch.long)
            for j, (ids, at) in enumerate(chunk):
                x[j, :len(ids)] = torch.tensor(ids)
                x[j, at] = tok.mask_token_id
                att[j, :len(ids)] = 1
            pos = torch.tensor([at for _, at in chunk])
            tgt = torch.tensor([ids[at] for ids, at in chunk])
            # Only the masked position goes through the vocabulary head: full logits are
            # batch x length x 102,400 and were most of the memory.
            hidden = model.model(input_ids=x.to(DEV), attention_mask=att.to(DEV)).last_hidden_state
            h = hidden[torch.arange(len(chunk)), pos.to(DEV)]
            lp = model.decoder(model.head(h)).float().log_softmax(-1)
            got = lp[torch.arange(len(chunk)), tgt.to(DEV)].cpu().tolist()
            for o, g in zip(owner[i:i + B], got):
                sums[o] = sums.get(o, 0.0) + g
            if i % (B * 200) == 0:
                print(i, flush=True)
    new = {}
    for key, rs in deltas.items():
        out = []
        for k, (s, e, w, alts) in enumerate(rs):
            base = sums.get((key, k, 0))
            if base is None:
                out.append([s, e, w, alts])
                continue
            out.append([s, e, w, [[alt, round(sums[(key, k, n + 1)] - base, 3)] for n, (alt, _) in enumerate(alts)
                                  if base is not None and (key, k, n + 1) in sums]])
        new[key] = out
    json.dump(new, open(os.path.join(d, f"deltas-{out_name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for pre, suf in (("res-", "det-r0.json"), ("kk-", ".json")):
        src_p, dst = f"{pre}{name}{suf}", os.path.join(d, f"{pre}{out_name}{suf}")
        if os.path.exists(os.path.join(d, src_p)) and not os.path.lexists(dst):
            os.symlink(src_p, dst)
    print("done", len(new))


if __name__ == "__main__":
    main()
