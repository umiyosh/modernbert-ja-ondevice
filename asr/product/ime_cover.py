"""Would IME-simulated errors cover the real ones? For each real-voice homophone on val, put the right
word back, give kanakanji-edge-33m the left context and the right word's reading in the model card's
format, and see at which beam rank the wrongly written word comes out. Split by whether the pair is in
the voice table (seen in training). usage (from asr/): python tools/ime_cover.py [beams]
"""

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import kk_score as k  # noqa: E402
import prodforce  # noqa: E402
import torch  # noqa: E402

prod = prodforce.prod
BEAMS = int(next((a for a in sys.argv[1:] if a.isdigit()), "10"))
B = 16


@torch.no_grad()
def convert(prompts, limit):
    k.tok.padding_side = "left"
    eos = k.tok.eos_token_id
    n = max(map(len, prompts))
    ids = torch.full((len(prompts), n), eos)
    att = torch.zeros((len(prompts), n), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, n - len(p):] = torch.tensor(p)
        att[i, n - len(p):] = 1
    out = k.model.generate(ids.to(k.DEV), attention_mask=att.to(k.DEV), num_beams=BEAMS, num_return_sequences=BEAMS,
                           max_new_tokens=limit, eos_token_id=eos, pad_token_id=eos, do_sample=False)
    texts = [k.tok.decode(o[n:], skip_special_tokens=True) for o in out]
    return [texts[i * BEAMS:(i + 1) * BEAMS] for i in range(len(prompts))]


def main():
    jobs = []
    for r in prod.VAL:
        if r["status"] == "clean":
            continue
        _, at = prod.eval_spans.norm_map(r["hyp"])
        for e in r["edits"]:
            x, y = e["hyp_chars"]
            if e["kind"] != "homophone" or y <= x:
                continue
            s, t = prod.eval_spans.raw_span(at, x, y)
            left = r["hyp"][max(0, s - 40):s]
            kana = k.reading(e["ref"])
            seen = bool(prodforce.COUNTS.get(e["hyp"], {}).get("to", {}).get(e["ref"]))
            prompt = k.tok.encode(f"{left}{kana}", add_special_tokens=False)
            jobs.append((seen, e["hyp"], e["ref"], prompt, len(kana)))
    print(len(jobs), "homophones", flush=True)
    c = collections.Counter()
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        for (seen, hyp, ref, _, _), outs in zip(chunk, convert([j[3] for j in chunk], max(j[4] for j in chunk) + 4)):
            tag = "seen" if seen else "unseen"
            outs = [o.strip() for o in outs]
            c[tag] += 1
            c[tag + " ref top1"] += outs[:1] == [ref]
            rank = outs.index(hyp) + 1 if hyp in outs else 0
            for kk in (2, 5, 10):
                c[f"{tag} hyp in top{kk}"] += 0 < rank <= kk
    for tag in ("seen", "unseen"):
        n = c[tag]
        print(f"{tag}: {n} | converter's top1 is the right word {c[tag + ' ref top1']} | wrong word in top2 {c[tag + ' hyp in top2']}, "
              f"top5 {c[tag + ' hyp in top5']}, top10 {c[tag + ' hyp in top10']}")


if __name__ == "__main__":
    main()
