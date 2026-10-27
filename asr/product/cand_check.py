"""A candidate-search switch measured before any scoring (Kotobae #98, paper plan step 4): the Rust test
typo_cands writes cands-<name>.json and tokens-<name>.json for eval-dev. Of the dev half's homophones the
product point misses with the right word in no row (nocand_cases.py), how many the new rows offer; how
many alternatives the new rows add per 1,000 characters of the dev half's correct clips (the scoring
cost); and, for the ones still missing with no row on the span, how the segmenter tagged the tokens there.

usage (from asr/): python tools/cand_check.py <name> [base c0] [system line w det score, default sb0s 0.1 1 0.96 9.5]
"""
import collections
import json
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import nocand_cases  # noqa: E402
import prod  # noqa: E402


def load(name):
    return (json.load(open(f"eval-dev/cands-{name}.json", encoding="utf-8")),
            json.load(open(f"eval-dev/tokens-{name}.json", encoding="utf-8")))


def alts_per_1k(cands, half):
    n = chars = 0
    for r in prod.VAL:
        if half[r["path"]] == "dev" and r["status"] == "clean":
            chars += len(r["hyp"])
            n += sum(len(alts) for _, _, alts in cands.get(r["path"], []))
    return 1000 * n / chars


def main():
    name, base = ARGS[0], ARGS[1] if len(ARGS) > 1 else "c0"
    cands, toks = load(name)
    half = json.load(open("split-calib.json"))
    c, tags = collections.Counter(), collections.Counter()
    system = (ARGS[2], *(float(x) for x in ARGS[3:7])) if len(ARGS) > 6 else ()
    for r, text, a, b, e in nocand_cases.cases(*system):
        c["cases"] += 1
        right = text[:a] + e["ref"] + text[b:]
        rows = cands.get(r["path"], [])
        if any(text[:p] + alt + text[q:] == right for p, q, alts in rows for alt in alts):
            c["offered now"] += 1
            continue
        if not any(p < b and a < q for p, q, _ in rows):
            over = [t for t in toks.get(r["path"], []) if t[0] < b and a < t[1]]
            tags[" + ".join(f"{t[2]}/{t[3]}{'' if t[4] else '/unknown'}" for t in over)] += 1
    print(f"{name}: of {c['cases']} missed homophones with no candidate in the product run, offered now {c['offered now']}")
    print(f"  alternatives per 1,000 chars of correct dev clips: {base} {alts_per_1k(load(base)[0], half):.0f}, "
          f"{name} {alts_per_1k(cands, half):.0f}")
    print("  still no row on the span, tokens there:")
    for k, v in tags.most_common():
        print(f"    {v} {k}")


if __name__ == "__main__":
    main()
