"""Hard negatives for the scorer (Kotobae #98, step 2b): on correct training-side sentences run through the
shipped pipeline (eval-mine), every scored alternative the scorer favours over the written word by at
least --lead. These are the false marks in waiting; a `hard` pair asks the scorer to keep the written word.
Written in scorer_pairs.py's format, grouped by pair.

usage (from typo-study/asr): python hard_negatives.py <eval dir> <name> <out.jsonl> [--lead 3.0]
"""

import argparse
import json
import os

ENDS = "。！？\n"


def sentence(text: str, s: int, e: int) -> tuple[int, int]:
    lo = max((i + 1 for i in range(s) if text[i] in ENDS), default=0)
    hi = next((i + 1 for i in range(e, len(text)) if text[i] in ENDS), len(text))
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("name")
    ap.add_argument("out")
    ap.add_argument("--lead", type=float, default=3.0)
    a = ap.parse_args()
    texts = {t["id"]: t["text"] for t in json.load(open(os.path.join(a.dir, "texts.json"), encoding="utf-8"))}
    deltas = json.load(open(os.path.join(a.dir, f"deltas-{a.name}.json"), encoding="utf-8"))
    rows, scored = [], 0
    for key, rs in deltas.items():
        text = texts[key]
        for s, e, w, alts in rs:
            scored += 1
            for alt, lead in alts:
                if lead >= a.lead:
                    lo, hi = sentence(text, s, e)
                    rows.append({"text": text[lo:hi], "s": s - lo, "e": e - lo, "good": w, "bad": alt,
                                 "kind": "hard", "group": f"{w}>{alt}", "lead": lead})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"sentences": len(texts), "scored rows": scored, "hard": len(rows),
                      "groups": len({r["group"] for r in rows})}))


if __name__ == "__main__":
    main()
