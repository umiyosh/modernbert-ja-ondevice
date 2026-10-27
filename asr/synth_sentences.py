"""More written sentences for synth_tts.py: corrected sentences (post_text) of JWTD's train split, appended
to the sentence file so the lines already read keep their numbers. Left out: any sentence sharing a
15-character run with a line already there or with Kotobae's texts or typo sets (the product's evaluation). Kept:
NFKC, 12-80 characters, under 64 detector tokens, picked at random.

usage: python asr/synth_sentences.py <jwtd train.jsonl> <sentences.txt> <kotobae texts.json> <set.json> <n> [--seed 0]
"""

import json
import random
import sys
import unicodedata

from transformers import AutoTokenizer

from train import DETECTOR, REVISION

RUN = 15


def shingles(s: str) -> set[str]:
    return {s[i:i + RUN] for i in range(len(s) - RUN + 1)}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jwtd, out, texts, typo_set, n = args[0], args[1], args[2], args[3], int(args[4])
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    nfkc = lambda s: unicodedata.normalize("NFKC", s)  # noqa: E731
    held = set()
    for t in json.load(open(texts, encoding="utf-8")):
        held |= shingles(nfkc(t["text"]))
    for c in json.load(open(typo_set, encoding="utf-8")):
        held |= shingles(nfkc(c["text"])) | shingles(nfkc(c["fixed"]))
    lines = open(out, encoding="utf-8").read().splitlines()
    have = set(lines)
    # Another revision of a line already there would leak into its held-out lines.
    for s in lines:
        held |= shingles(s)
    pool = set()
    for line in open(jwtd, encoding="utf-8"):
        s = nfkc(json.loads(line)["post_text"]).strip()
        if 12 <= len(s) <= 80 and "\n" not in s and s not in have and not shingles(s) & held:
            pool.add(s)
    pool = sorted(pool)
    random.Random(seed).shuffle(pool)
    tok = AutoTokenizer.from_pretrained(DETECTOR, revision=REVISION)
    picked = []
    for s in pool:
        if len(tok(s)["input_ids"]) < 64:
            picked.append(s)
            if len(picked) == n:
                break
    with open(out, "a", encoding="utf-8") as f:
        f.writelines(s + "\n" for s in picked)
    print(json.dumps({"pool": len(pool), "appended": len(picked), "lines": len(lines) + len(picked)}))


if __name__ == "__main__":
    main()
