"""False marks per 1,000 characters on eval-clean (correct written text from JWTD test, outside the typo
sets) for runs at given thresholds, with a 95% interval from the Poisson count. The check before #98 had
no kana/kanji gate, so a run named `d` is counted without it.

usage (from asr/): python tools/clean_fm.py <name>:<det>:<score> ...
"""

import json
import math
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402

TEXTS = {t["id"]: t["text"] for t in json.load(open("eval-clean/texts.json", encoding="utf-8"))}
prod.TEXT.update(TEXTS)
CHARS = sum(map(len, TEXTS.values()))


def interval(k: int) -> tuple[float, float]:
    """Approximate 95% bounds of a Poisson count (Wilson-Hilferty)."""
    lo = 0.0 if k == 0 else k * (1 - 1 / (9 * k) - 1.96 / (3 * math.sqrt(k))) ** 3
    k1 = k + 1
    hi = k1 * (1 - 1 / (9 * k1) + 1.96 / (3 * math.sqrt(k1))) ** 3
    return lo, hi


def main() -> None:
    for spec in ARGS:
        name, det, score = spec.split(":")
        src = prod.load("eval-clean", name)
        prod.GATE = name != "d"
        k = sum(len(prod.mk(src, key, float(det), float(score))) for key in TEXTS)
        lo, hi = interval(k)
        per = 1000 / CHARS
        print(f"{name:8s} det {det} score {score}: {k} marks on {CHARS} chars = {k * per:.3f} per 1,000 "
              f"(95% {lo * per:.3f}-{hi * per:.3f})")


if __name__ == "__main__":
    main()
