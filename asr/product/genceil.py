"""Ceiling of reading-based candidates: for real-voice homophone spans no candidate row offers the right word
for, is the right word among kanakanji-edge-33m's top-k conversions of the written word's reading?"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
import kk_score as k
import prodforce
prod = prodforce.prod
_, full = prod.load("eval-val", "kaws1")
items = []
for r in prod.VAL:
    if r["status"] == "clean":
        continue
    _, at = prod.eval_spans.norm_map(r["hyp"])
    for e in r["edits"]:
        x, y = e["hyp_chars"]
        if e["kind"] != "homophone" or y <= x:
            continue
        s, t = prod.eval_spans.raw_span(at, x, y)
        rows = [w for w in full.get(r["path"], []) if w[0] < t and s < w[1]]
        if any(alt == e["ref"] for w in rows for alt, _ in w[3]):
            continue
        items.append((r["hyp"], s, t, e["hyp"], e["ref"]))
hit = {1: 0, 5: 0, 10: 0}
eos = k.tok.eos_token_id
for text, s, t, h, ref in items:
    kana = k.reading(text[s:t])
    p = k.tok.encode(f"{text[max(0, s - 40):s]}{kana}", add_special_tokens=False)
    out = k.model.generate(torch.tensor([p]).to(k.DEV), num_beams=10, num_return_sequences=10, max_new_tokens=len(kana) + 4,
                           eos_token_id=eos, pad_token_id=eos, do_sample=False)
    outs = [k.tok.decode(o[len(p):], skip_special_tokens=True) for o in out]
    for n in hit:
        hit[n] += ref in outs[:n]
print(len(items), "spans; right word in top-1/5/10:", hit)
