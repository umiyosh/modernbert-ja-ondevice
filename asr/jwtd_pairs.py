"""Rehearsal pairs for the on-device check's scorer from JWTD v2 train (Kotobae #98): each kanji misconversion
read the same way (kanji-conversion_a, 解放→開放) becomes a `fix` pair in scorer_pairs.py's format, so the
scorer keeps the written-text homophones it already ranked well while it learns voice input's.

JWTD records the edit by word; the pair's span is the occurrence of the wrong word whose replacement gives
the corrected text (as tools/jwtd_eval.py does). A sentence sharing a 15-character run with any evaluation
text (dev, calib, Kotobae's texts, correct JWTD text, the JWTD test set) is left out, so no measured
sentence is trained on. A sample of --count pairs is written, drawn evenly from the shuffled pool.

usage (from typo-study/asr): python jwtd_pairs.py <jwtd train.jsonl> <out.jsonl> --count N
                             [--texts eval-dev/texts.json,...] [--seed 0]
"""

import argparse
import json
import random

from scorer_pairs import sentence

K = 15
TEXTS = "eval-dev/texts.json,eval-calib/texts.json,../texts.json,eval-clean/texts.json,eval-jwtd/texts.json"


def runs(text: str) -> set[str]:
    return {text[i:i + K] for i in range(len(text) - K + 1)}


def texts_of(path: str) -> list[str]:
    data = json.load(open(path, encoding="utf-8"))
    return [t["text"] for t in data] if isinstance(data, list) else list(data.values())


def span(pre: str, post: str, wrong: str, right: str) -> tuple[int, int] | None:
    at = [k for k in range(len(pre)) if pre.startswith(wrong, k) and pre[:k] + right + pre[k + len(wrong):] == post]
    return (at[0], at[0] + len(wrong)) if len(at) == 1 and wrong else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("out")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--texts", default=TEXTS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seen = set().union(*(runs(t) for p in a.texts.split(",") for t in texts_of(p)))
    pool, dropped = [], {"span": 0, "overlap": 0}
    for line in open(a.train, encoding="utf-8"):
        r = json.loads(line)
        for d in r["diffs"]:
            if d.get("category") != "kanji-conversion_a":
                continue
            wrong, right = d.get("pre_str", ""), d.get("post_str", "")
            at = span(r["pre_text"], r["post_text"], wrong, right)
            if at is None:
                dropped["span"] += 1
                continue
            lo, hi = sentence(r["pre_text"], *at)
            text = r["pre_text"][lo:hi]
            if runs(text) & seen:
                dropped["overlap"] += 1
                continue
            pool.append({"text": text, "s": at[0] - lo, "e": at[1] - lo, "good": right, "bad": wrong,
                         "kind": "fix", "group": f"jwtd:{wrong}>{right}"})
    random.Random(a.seed).shuffle(pool)
    rows = pool[: a.count]
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"pool": len(pool), "written": len(rows), "dropped": dropped,
                      "groups": len({r["group"] for r in rows})}))


if __name__ == "__main__":
    main()
