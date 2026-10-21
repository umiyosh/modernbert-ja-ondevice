"""Count the misconversion pairs in JWTD v2 train: how often editors corrected each written word
(pre_str) to each word (post_str), over the kanji-conversion and substitution edits.

augment.py draws its "table" errors from this, in the direction the real edit history goes.

usage: python training/pairs.py <jwtd train.jsonl> <pairs.json>
"""
import argparse
import json
from collections import Counter

CATEGORIES = {"kanji-conversion_a", "kanji-conversion_b", "substitution"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("out")
    args = ap.parse_args()
    pairs = Counter()
    for line in open(args.train):
        for x in json.loads(line)["diffs"]:
            if x["category"] in CATEGORIES:
                pairs[f'{x["pre_str"]}\t{x["post_str"]}'] += 1
    json.dump(dict(pairs), open(args.out, "w"), ensure_ascii=False)
    print(f"{len(pairs)} pairs")


if __name__ == "__main__":
    main()
