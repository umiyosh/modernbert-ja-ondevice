"""How many missed homophones with the right word in no row a corpus vocabulary would offer, and at what cost
(Kotobae #98, plan step 2c). The vocabulary is every run of one to three UniDic words with a kanji that the
training text uses at least --min times (Common Voice train references and JWTD v2 train's corrected text),
keyed by its reading. A miss counts as offered when the right word is in the vocabulary under the reading
of the written span. A training text sharing a 15-character run with any evaluation text is left out, as in
jwtd_pairs.py: Common Voice gives one sentence to many speakers, so a dev sentence can sit in train. The cost is the alternatives such a lookup adds per 1,000 characters of the dev half's
correct clips, over the same one-to-three-word runs. Counts only, never text.

usage (from asr/): python tools/cand_vocab.py <jwtd train.jsonl> [name] [line] [w] [det] [score] [--min 3] [--jwtd 300000]
                                 [--out vocab.json]  (writes reading -> words, the most frequent first, for KOTOBAE_TYPO_VOCAB)
"""
import collections
import json
import os
import sys

ARGS = [a for a in sys.argv[1:]]
MIN = int(ARGS[ARGS.index("--min") + 1]) if "--min" in ARGS else 3
JW = int(ARGS[ARGS.index("--jwtd") + 1]) if "--jwtd" in ARGS else 300000
OUT = ARGS[ARGS.index("--out") + 1] if "--out" in ARGS else None
POS = [a for i, a in enumerate(ARGS) if not a.startswith("--") and (i == 0 or not ARGS[i - 1].startswith("--"))]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import fugashi  # noqa: E402
import jaconv  # noqa: E402

import nocand_cases  # noqa: E402
import prod  # noqa: E402
from jwtd_pairs import TEXTS, runs as grams, texts_of  # noqa: E402

tagger = fugashi.Tagger()


def kanji(t):
    return any("一" <= ch <= "鿿" for ch in t)


def words(text):
    return [(w.surface, jaconv.hira2kata(getattr(w.feature, "kana", None) or w.surface)) for w in tagger(text)]


def runs(ws, n=3):
    for i in range(len(ws)):
        for k in range(1, n + 1):
            if i + k <= len(ws):
                part = ws[i:i + k]
                yield "".join(s for s, _ in part), "".join(r for _, r in part)


def reading(text):
    return "".join(r for _, r in words(text))


def main():
    count = collections.Counter()
    texts = [json.loads(line)["ref"] for line in open("labels.jsonl", encoding="utf-8")
             if json.loads(line)["split"] == "train"]
    for i, line in enumerate(open(POS[0], encoding="utf-8")):
        if i >= JW:
            break
        texts.append(json.loads(line)["post_text"])
    seen = set().union(*(grams(t) for p in TEXTS.split(",") for t in texts_of(p)))
    kept = [t for t in texts if not grams(t) & seen]
    print(f"left out {len(texts) - len(kept)} of {len(texts)} training texts sharing 15 characters with evaluation")
    texts = kept
    for t in texts:
        count.update(x for x in runs(words(t)) if kanji(x[0]))
    vocab = collections.defaultdict(set)
    for (sf, rd), n in count.items():
        if n >= MIN and 2 <= len(sf) <= 8:
            vocab[rd].add(sf)
    print(f"vocabulary: {sum(map(len, vocab.values()))} words under {len(vocab)} readings (min {MIN}, {len(texts)} texts)")
    if OUT:
        out = {rd: sorted(ws, key=lambda w: -count[(w, rd)]) for rd, ws in vocab.items()}
        json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    name, line, w, det, score = (POS[1:] + ["sb2k5m45", "0.05", "1", "0.98", "7.0"][len(POS[1:]):])[:5]
    misses = list(nocand_cases.cases(name, float(line), float(w), float(det), float(score)))
    got = sum(e["ref"] in vocab.get(reading(text[a:b]), ()) for _, text, a, b, e in misses)
    print(f"{name}: dev misses with no row offering the right word {len(misses)}, offered by the vocabulary {got}")
    half = json.load(open("split-calib.json"))
    chars = alts = 0
    for r in prod.VAL:
        if half[r["path"]] == "dev" and r["status"] == "clean":
            chars += len(r["hyp"])
            alts += sum(len(vocab.get(rd, set()) - {sf}) for sf, rd in runs(words(r["hyp"])) if kanji(sf))
    print(f"cost: {1000 * alts / max(1, chars):.0f} alternatives per 1,000 characters of correct dev clips")


if __name__ == "__main__":
    main()
