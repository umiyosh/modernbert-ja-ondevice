"""Paired comparison of two systems on the JWTD test homophones (Kotobae #98): which sentences each one
catches at its own dev-half product point, and the exact McNemar test on the discordant ones. JWTD only
reports; it never picks a point.

usage (from asr/): python tools/jwtd_pair.py <name> <det> <score> <name> <det> <score> [line 0.1] [w 1]
"""
import json
import os
import sys
from math import comb

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402
import screen_sweep as s  # noqa: E402
import verify_sweep as v  # noqa: E402

D = "eval-jwtd"


def hits(name, det, sc, labels, line, w):
    src = prod.load(D, name)
    if w:
        src = v.verified(src, json.load(open(os.path.join(D, f"verify-{name}.json"), encoding="utf-8")), w)
    src = s.screened(src, line)
    out = {}
    for err, lab in labels.items():
        if lab["kind"] != "homophone":
            continue
        a, b = lab["span"]
        out[err] = any(x < b and a < y for x, y in prod.mk(src, err, det, sc))
    return out


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def main():
    line = float(ARGS[6]) if len(ARGS) > 6 else 0.1
    w = float(ARGS[7]) if len(ARGS) > 7 else 1.0
    labels = json.load(open(os.path.join(D, "labels.json"), encoding="utf-8"))
    prod.TEXT.update({t["id"]: t["text"] for t in json.load(open(os.path.join(D, "texts.json"), encoding="utf-8"))})
    x = hits(ARGS[0], float(ARGS[1]), float(ARGS[2]), labels, line, w)
    y = hits(ARGS[3], float(ARGS[4]), float(ARGS[5]), labels, line, w)
    only_x = sum(x[k] and not y[k] for k in x)
    only_y = sum(y[k] and not x[k] for k in x)
    print(f"{ARGS[0]} {sum(x.values())}/{len(x)} vs {ARGS[3]} {sum(y.values())}/{len(y)}: "
          f"only {ARGS[0]} {only_x}, only {ARGS[3]} {only_y}, exact McNemar p={mcnemar(only_x, only_y):.4g}")


if __name__ == "__main__":
    main()
