"""Can the scorer itself offer the right word the candidate rows miss? (Kotobae #98, paper plan step 4:
the candidate stage, DR-CSC style.) On the dev half only: real-voice homophones the product point misses
and whose right word no candidate row offers (as tools/nocand.py finds them). For each, whether the written
and right words share a reading (UniDic), whether UniDic knows the right word as one word, and the rank of
the right word among the masked LM's fills of the written span that read the same, for 1 to 3 masks.
Then, as the cost, how many same-reading fills per word a correct clip would add.

usage (from asr/, under guard.py): python tools/cand_probe.py [name sb0s] [line 0.1] [w 1] [det 0.96] [score 9.5]
"""
import collections
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import fugashi  # noqa: E402
import jaconv  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

import nocand_cases  # noqa: E402

REPO = "sbintuitions/modernbert-ja-130m"
TOP = 200  # fills kept per mask position before the reading filter
BEAM = 40
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tagger = fugashi.Tagger()


def reading(text):
    out = []
    for w in tagger(text):
        k = getattr(w.feature, "kana", None) or getattr(w.feature, "pron", None)
        if not k or k == "*":
            k = w.surface
        out.append(k)
    return jaconv.kata2hira("".join(out))


def one_word(text):
    ws = list(tagger(text))
    return len(ws) == 1 and not ws[0].is_unk


def fills(tok, model, left, right, n):
    """Beam over n masks, left to right, each step re-run with the chosen pieces in place."""
    beams = [([], 0.0)]
    for i in range(n):
        nxt = []
        for pieces, lp in beams:
            ids = tok(left, add_special_tokens=False)["input_ids"] + pieces + [tok.mask_token_id] * (n - i) \
                + tok(right, add_special_tokens=False)["input_ids"]
            ids = [tok.cls_token_id] + ids + [tok.sep_token_id]
            pos = 1 + len(tok(left, add_special_tokens=False)["input_ids"]) + i
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids], device=DEV)).logits[0, pos]
            top = torch.log_softmax(logits.float(), -1).topk(TOP)
            nxt += [(pieces + [t], lp + l) for l, t in zip(top.values.tolist(), top.indices.tolist())]
        beams = sorted(nxt, key=lambda b: -b[1])[:BEAM] if i < n - 1 else nxt
    return [(tok.decode(p).strip(), lp) for p, lp in beams]


def ranked(tok, model, text, a, b):
    want = reading(text[a:b])
    out = {}
    for n in (1, 2, 3):
        for w, lp in fills(tok, model, text[:a], text[b:], n):
            if w and w != text[a:b] and reading(w) == want:
                out[w] = max(out.get(w, -1e9), lp)
    return [w for w, _ in sorted(out.items(), key=lambda x: -x[1])]


def main():
    name = ARGS[0] if ARGS else "sb0s"
    line, w, det, score = (float(x) for x in (ARGS[1:5] + ["0.1", "1", "0.96", "9.5"][len(ARGS[1:5]):]))
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForMaskedLM.from_pretrained(REPO, dtype=torch.float16).to(DEV).eval()
    c = collections.Counter()
    ranks = collections.defaultdict(list)
    for _, text, a, b, e in nocand_cases.cases(name, line, w, det, score):
        same = reading(text[a:b]) == reading(e["ref"])
        c["cases"] += 1
        c["same reading"] += same
        c["one UniDic word"] += one_word(e["ref"])
        got = ranked(tok, model, text, a, b) if same else []
        ranks["same" if same else "other"].append(got.index(e["ref"]) + 1 if e["ref"] in got else None)
        c["fills"] += len(got)
        c["fill sets"] += same
    print(f"{name} at det {det} score {score}, line {line}, w {w}: missed homophones with no candidate {c['cases']}")
    print(f"  written and right read the same (UniDic): {c['same reading']} | right word one UniDic word: {c['one UniDic word']}")
    rs = ranks["same"]
    for k in (1, 3, 5, 10, 32):
        print(f"  right word within the top {k} same-reading fills: {sum(1 for x in rs if x and x <= k)} of {len(rs)}")
    print(f"  same-reading fills per case: {c['fills'] / max(c['fill sets'], 1):.1f}")


if __name__ == "__main__":
    main()
