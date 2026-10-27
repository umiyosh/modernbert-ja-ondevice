"""Where a candidate-search switch's gain and cost come from (Kotobae #98, plan step 2c): of the dev half's
homophones the product point misses with the right word in no row, the ones the new rows offer, by how many
tokens the offering span covers; and the alternatives the new rows add per 1,000 characters of the dev
half's correct clips, by the same span length. Both runs come from the Rust test typo_cands. Counts only.

usage (from asr/): python tools/cand_cost.py <name> <base> [system line w det score]
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


def width(toks, p, q):
    return sum(1 for t in toks if p <= t[0] and t[1] <= q) or 1


def added(rows, old):
    before = {(p, q): set(alts) for p, q, alts in old}
    for p, q, alts in rows:
        new = set(alts) - before.get((p, q), set())
        if new:
            yield p, q, new


def main():
    name, base = ARGS[0], ARGS[1]
    (cands, toks), (old, _) = load(name), load(base)
    system = (ARGS[2], *(float(x) for x in ARGS[3:7])) if len(ARGS) > 6 else ()
    gain = collections.Counter()
    for r, text, a, b, e in nocand_cases.cases(*system):
        right = text[:a] + e["ref"] + text[b:]
        t = toks.get(r["path"], [])
        hits = [width(t, p, q) for p, q, alts in added(cands.get(r["path"], []), old.get(r["path"], []))
                for alt in alts if text[:p] + alt + text[q:] == right]
        gain[min(hits) if hits else 0] += 1
    half = json.load(open("split-calib.json"))
    cost, chars = collections.Counter(), 0
    for r in prod.VAL:
        if half[r["path"]] == "dev" and r["status"] == "clean":
            chars += len(r["hyp"])
            t = toks.get(r["path"], [])
            for p, q, new in added(cands.get(r["path"], []), old.get(r["path"], [])):
                cost[width(t, p, q)] += len(new)
    print(f"{name} over {base}: offered by span width {dict(sorted(gain.items()))} (0 = still not offered)")
    print("  added alternatives per 1,000 chars by span width:",
          {k: round(1000 * v / chars) for k, v in sorted(cost.items())})


if __name__ == "__main__":
    main()
