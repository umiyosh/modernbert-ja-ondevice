"""IME-simulated misconversions for pretraining the detector (Kotobae #98, the CSCD-NS recipe, arXiv
2211.08788): correct sentences with one word swapped for what a kana-kanji converter would offer for its
reading, kept only when a language model finds the swap less likely than the word written.

- Sentences: JWTD v2 train's corrected sentences (12-80 characters), leaving out any that shares a
  15-character run with Kotobae's texts and typo sets or the Kokoro held-out lines (the evaluation).
- Words: up to --per sentence content words holding kanji, each one example.
- Swap: kanakanji-edge-33m (Apache-2.0) converts the word's reading after its left context, in the
  model card's format (\\uEE02 left \\uEE00 katakana \\uEE01). If its first conversion is the word itself,
  the next one that differs is taken; otherwise the first (the converter's own mistake). Only a homophone
  written with kanji is kept: kana and kanji alone, and the same reading as the word.
- Filter: kept when the base masked LM's pseudo-log-likelihood (the app's scoring, train_scorer.scores)
  of the swapped word is below the written word's (delta > 0), as CSCD-NS found the filter essential.

Each line: {"text", "s", "e", "orig", "swap", "lead"} (text holds swap at s..e; lead is the written word's
score minus the swap's). Progress is kept in <out>.done, so a stopped run resumes.

usage (from typo-study/asr): python synth_kkc.py <jwtd train.jsonl> <out.jsonl> [--limit N] [--per 3] [--seed 0]
"""

import argparse
import json
import os
import random
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(__file__))
import fugashi  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

import train_scorer as ts  # noqa: E402

KKC = "yuuki14202028/kanakanji-edge-33m"
RUN, LEFT, BEAMS = 15, 40, 4
B = int(os.environ.get("KKC_BATCH", "64"))  # prompts per generate call
KANJI = re.compile("[㐀-鿿]")
WORDLIKE = re.compile("[぀-ヿ㐀-鿿]+")
CONTENT = {"名詞", "動詞", "形容詞", "副詞", "形状詞"}
DEV = ts.DEV


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def shingles(s: str) -> set[str]:
    return {s[i:i + RUN] for i in range(len(s) - RUN + 1)}


def held_runs() -> set[str]:
    held = set()
    for t in json.load(open("../texts.json", encoding="utf-8")):
        held |= shingles(nfkc(t["text"]))
    for c in json.load(open("../set.json", encoding="utf-8")):
        held |= shingles(nfkc(c["text"])) | shingles(nfkc(c["fixed"]))
    for line in open("eval-kokoro/labels.jsonl", encoding="utf-8"):
        held |= shingles(nfkc(json.loads(line)["ref"]))
    return held


def sentences(path: str, seed: int) -> list[str]:
    held = held_runs()
    pool = set()
    for line in open(path, encoding="utf-8"):
        s = nfkc(json.loads(line)["post_text"]).strip()
        if 12 <= len(s) <= 80 and "\n" not in s and not shingles(s) & held:
            pool.add(s)
    pool = sorted(pool)
    random.Random(seed).shuffle(pool)
    return pool


def targets(tagger, text: str, per: int, rng: random.Random):
    """Up to `per` content words with kanji: (start, end, surface, katakana reading)."""
    out, at = [], 0
    for w in tagger(text):
        s = text.find(w.surface, at)
        if s < 0:
            continue
        at = s + len(w.surface)
        kana = getattr(w.feature, "kana", None)
        if (w.feature.pos1 in CONTENT and KANJI.search(w.surface) and kana and kana != "*"
                and 1 <= len(w.surface) <= 6):
            out.append((s, at, w.surface, kana))
    rng.shuffle(out)
    return out[:per]


@torch.no_grad()
def convert(model, tok, prompts, limit):
    tok.padding_side = "left"
    eos = tok.eos_token_id
    ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
    n = max(map(len, ids))
    x = torch.full((len(ids), n), eos)
    att = torch.zeros((len(ids), n), dtype=torch.long)
    for i, p in enumerate(ids):
        x[i, n - len(p):] = torch.tensor(p)
        att[i, n - len(p):] = 1
    out = model.generate(x.to(DEV), attention_mask=att.to(DEV), num_beams=BEAMS, num_return_sequences=BEAMS,
                         max_new_tokens=limit, eos_token_id=eos, pad_token_id=eos, do_sample=False)
    texts = [tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]
    return [texts[i * BEAMS:(i + 1) * BEAMS] for i in range(len(prompts))]


def reading(tagger, text: str) -> str:
    return "".join(k if (k := getattr(w.feature, "kana", None)) and k != "*" else w.surface for w in tagger(text))


def swap_of(tagger, word: str, kana: str, outs: list[str]) -> str | None:
    """The converter's pick for the reading, kept only when it is a homophone written with kanji: the
    converter also answers in plain kana and drags in punctuation, which are not the errors sought."""
    if not outs or not outs[0]:
        return None
    swap = outs[0] if outs[0] != word else next((o for o in outs[1:] if o and o != word), None)
    if not swap or not WORDLIKE.fullmatch(swap) or reading(tagger, swap) != kana:
        return None
    # A new kanji is required: the same kanji with more kana (練り直し, 練りなおし) is a spelling variant.
    if not set(KANJI.findall(swap)) - set(word):
        return None
    return swap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jwtd")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pool = sentences(a.jwtd, a.seed)
    if a.limit:
        pool = pool[: a.limit]
    done_path = a.out + ".done"
    start = int(open(done_path).read()) if os.path.exists(done_path) else 0
    print(json.dumps({"pool": len(pool), "from": start}), flush=True)
    rng = random.Random(a.seed)
    tagger = fugashi.Tagger()
    ktok = AutoTokenizer.from_pretrained(KKC)
    kkc = AutoModelForCausalLM.from_pretrained(KKC).to(DEV).eval()
    stok = AutoTokenizer.from_pretrained(ts.BASE)
    lm = AutoModelForMaskedLM.from_pretrained(ts.BASE, dtype=torch.float16).to(DEV).eval()
    kept = tried = 0
    with open(a.out, "a", encoding="utf-8") as f:
        for i in range(start, len(pool), B):
            jobs = []
            for text in pool[i:i + B]:
                for s, e, word, kana in targets(tagger, text, a.per, rng):
                    jobs.append((text, s, e, word, f"{text[max(0, s - LEFT):s]}{kana}", len(kana), kana))
            rows = []
            for j in range(0, len(jobs), B):
                chunk = jobs[j:j + B]
                for (text, s, e, word, _, _, kana), outs in zip(chunk, convert(kkc, ktok, [c[4] for c in chunk],
                                                                                 max(c[5] for c in chunk) + 4)):
                    swap = swap_of(tagger, word, kana, outs)
                    if swap:
                        rows.append({"text": text, "s": s, "e": e, "good": word, "bad": swap})
            tried += len(jobs)
            # The written word (good) against the swap, in the sentence as written: delta > 0 keeps it.
            pairs = [(r, v) for r in rows if (v := ts.variants(stok, r)) is not None]
            for k in range(0, len(pairs), 64):
                part = pairs[k:k + 64]
                with torch.no_grad():
                    sc = ts.scores(lm, stok, [w for _, v in part for w in v]).view(-1, 2).tolist()
                for (r, _), (good, bad) in zip(part, sc):
                    if good - bad > 0:
                        t, s, e = r["text"], r["s"], r["e"]
                        f.write(json.dumps({"text": t[:s] + r["bad"] + t[e:], "s": s, "e": s + len(r["bad"]),
                                            "orig": r["good"], "swap": r["bad"], "lead": round(good - bad, 2)},
                                           ensure_ascii=False) + "\n")
                        kept += 1
            f.flush()
            with open(done_path, "w") as g:
                g.write(str(min(i + B, len(pool))))
            if (i // B) % 50 == 0:
                print(json.dumps({"done": min(i + B, len(pool)), "tried": tried, "kept": kept}), flush=True)
    print(json.dumps({"done": len(pool), "tried": tried, "kept": kept}), flush=True)


if __name__ == "__main__":
    main()
