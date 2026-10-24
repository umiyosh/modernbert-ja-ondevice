"""Shapes of two miss stages on the dev half only (the half development may look at; calib reports):
real-voice homophones whose right word no candidate row offers, by how the candidate rows sit on the error
span, and those the detector never sent to scoring, by how far its P stayed below the screening line.

usage (from asr/): python tools/nocand.py <name> <det> <score>
"""
import collections
import json
import os
import sys

import numpy as np

sys.argv, ARGS = [sys.argv[0], "--gate"], sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402

es = prod.eval_spans
SCREEN = 0.018


def main():
    name, det, score = ARGS[0], float(ARGS[1]), float(ARGS[2])
    half = json.load(open("split-calib.json"))
    src = prod.load("eval-val", name)
    cands = json.load(open(f"eval-val/cands-{name}.json", encoding="utf-8"))
    deltas = src[1]
    shape, low, sizes = collections.Counter(), [], collections.Counter()
    for r in prod.VAL:
        if r["status"] == "clean" or half[r["path"]] != "dev":
            continue
        text = r["hyp"]
        _, at = es.norm_map(text)
        marks = prod.mk(src, r["path"], det, score)
        probes = src[0].get(r["path"], [])
        rows = cands.get(r["path"], [])
        for e in r["edits"] or []:
            x, y = e["hyp_chars"]
            if e["kind"] != "homophone" or y <= x:
                continue
            a, b = es.raw_span(at, x, y)
            if any(s < b and a < t for s, t in marks):
                continue
            right = text[:a] + e["ref"] + text[b:]
            fits = lambda s, t, alt: text[:s] + alt + text[t:] == right  # noqa: E731
            if not any(fits(s, t, alt) for s, t, alts in rows for alt in alts):
                over = [(s, t) for s, t, _ in rows if s < b and a < t]
                if not over:
                    k = "no row touches the span"
                elif (a, b) in over:
                    k = "a row on the exact span, right word not offered"
                elif any(s <= a and b <= t for s, t in over):
                    k = "rows only on wider spans"
                else:
                    k = "rows only on narrower or shifted spans"
                shape[k] += 1
                sizes[(len(e["hyp"]), len(e["ref"]))] += 1
                continue
            if not any(fits(s, t, alt) for s, t, _, alts in deltas.get(r["path"], []) for alt, _ in alts):
                p = max((q for s, t, q, _ in probes if s < b and a < t), default=0.0)
                low.append(p)
    n = sum(shape.values())
    print(f"dev half, right word never a candidate: {n}")
    for k, v in shape.most_common():
        print(f"  {k}: {v} ({100 * v / n:.0f}%)")
    print("  written/right lengths:", ", ".join(f"{h}/{r}: {v}" for (h, r), v in sizes.most_common(6)))
    low = np.array(low)
    print(f"dev half, right word offered but never scored: {len(low)}")
    for q in (0.25, 0.5, 0.75, 0.9):
        print(f"  detector P quantile {q:.2f}: {np.quantile(low, q):.4f}")
    for t in (0.005, 0.01):
        print(f"  share with P >= {t}: {np.mean(low >= t):.0%}")


if __name__ == "__main__":
    main()
