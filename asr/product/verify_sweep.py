"""Experiment 3 of the 60% plan (Kotobae #98): add the detector's verdict on the candidate-substituted
sentence (tools/verify_det.py) to the lead, and re-choose the two marks within every limit on the dev half,
for several screening lines (tools/screen_sweep.py). lead' = lead + w * (logit p_written - logit p_alt):
an alternative the detector finds cleaner than the written word gains, one it finds worse loses.

usage (from asr/): python tools/verify_sweep.py <name> [lines] [weights]
"""
import math
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import json  # noqa: E402

import prod  # noqa: E402
import screen_sweep as s  # noqa: E402

DIRS = ("eval-dev", "..", "eval-clean")


def logit(p):
    p = min(max(p, 1e-5), 1 - 1e-5)
    return math.log(p / (1 - p))


def verified(src, ver, w):
    """The run with each verified alternative's lead moved by w times the detector's verdict."""
    P, D = src
    out = {}
    for key, rows in D.items():
        v = {(a, b, alt): logit(pw) - logit(pa) for a, b, alt, pw, pa in ver.get(key, [])}
        out[key] = [[a, b, wr, [[alt, lead + w * v.get((a, b, alt), 0.0)] for alt, lead in alts]] for a, b, wr, alts in rows]
    return P, out


def main():
    name = ARGS[0]
    lines = [float(x) for x in (ARGS[1] if len(ARGS) > 1 else "0.018,0.005,0").split(",")]
    weights = [float(x) for x in (ARGS[2] if len(ARGS) > 2 else "0,0.5,1,2,4").split(",")]
    runs = {d: prod.load(d, name) for d in DIRS}
    vers = {d: json.load(open(os.path.join(d, f"verify-{name}.json"), encoding="utf-8")) for d in DIRS}
    base, base_sets = s.limits()
    scores = [round(4.0 + 0.5 * i, 2) for i in range(25)]
    for line in lines:
        for w in weights:
            V, K, C = (s.screened(verified(runs[d], vers[d], w), line) for d in DIRS)
            print(f"line {line} w {w}: {s.show(s.search(V, K, C, base, base_sets, scores))}", flush=True)


if __name__ == "__main__":
    main()
