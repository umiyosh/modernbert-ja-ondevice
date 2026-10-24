"""screen_sweep.py over a wider grid for detectors whose P sits higher (Kotobae #98: the detector given
the sentence marks it was trained with): screening lines up to 0.3 and detector marks up to 0.995,
optionally with the verify lead of verify_sweep.py. Only the dev half is read.

usage (from asr/): python tools/marks_sweep.py <name> [lines] [weights]
"""
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import json  # noqa: E402

import prod  # noqa: E402
import screen_sweep as s  # noqa: E402
import verify_sweep as v  # noqa: E402

DIRS = ("eval-dev", "..", "eval-clean")
DETS = [round(0.5 + 0.02 * i, 2) for i in range(23)] + [0.96, 0.97, 0.98, 0.99, 0.995, 2.0]
SCORES = [round(4.0 + 0.5 * i, 2) for i in range(25)]


def main():
    name = ARGS[0]
    lines = [float(x) for x in (ARGS[1] if len(ARGS) > 1 else "0.018,0.05,0.1,0.2,0.3").split(",")]
    weights = [float(x) for x in (ARGS[2] if len(ARGS) > 2 else "0").split(",")]
    runs = {d: prod.load(d, name) for d in DIRS}
    vers = {d: json.load(open(os.path.join(d, f"verify-{name}.json"), encoding="utf-8")) for d in DIRS} if weights != [0] else {}
    base, base_sets = s.limits()
    for line in lines:
        for w in weights:
            src = {d: v.verified(runs[d], vers[d], w) if w else runs[d] for d in DIRS}
            V, K, C = (s.screened(src[d], line) for d in DIRS)
            print(f"line {line} w {w}: scored {s.scored_per_1k(V):.1f}/1k | {s.show(s.search(V, K, C, base, base_sets, SCORES, DETS))}", flush=True)


if __name__ == "__main__":
    main()
