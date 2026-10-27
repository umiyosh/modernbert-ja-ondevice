"""The candidate mark judged on lead plus a times the logit of the detector's P on the written word (Kotobae #98,
10-24 night). Today the detector's suspicion only screens which words are scored, so a word the detector barely
suspects needs the same lead as one it is nearly sure of. Same limits and grid as marks_sweep.py.

usage (from asr/): python tools/plead_sweep.py <name> [line 0.1] [w 1] [a,a,... default 0,0.25,0.5,1,2]
"""
import json
import math
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import marks_sweep as m  # noqa: E402
import prod  # noqa: E402
import screen_sweep as s  # noqa: E402
import verify_sweep as v  # noqa: E402

EPS = 1e-4


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def shifted(src, a):
    """Each alternative's lead moved by a times the logit of the written word's highest token P."""
    P, D = src
    out = {}
    for key, rows in D.items():
        probes = P.get(key, [])
        new = []
        for s0, e0, wr, alts in rows:
            sus = max((p for x, y, p, _ in probes if x < e0 and s0 < y), default=0.0)
            new.append([s0, e0, wr, [[alt, lead + a * logit(sus)] for alt, lead in alts]])
        out[key] = new
    return P, out


def main():
    name = ARGS[0]
    line = float(ARGS[1]) if len(ARGS) > 1 else 0.1
    w = float(ARGS[2]) if len(ARGS) > 2 else 1.0
    alphas = [float(x) for x in (ARGS[3] if len(ARGS) > 3 else "0,0.25,0.5,1,2").split(",")]
    base, base_sets = s.limits()
    src = {}
    for d in m.DIRS:
        run = prod.load(d, name)
        if w:
            run = v.verified(run, json.load(open(os.path.join(d, f"verify-{name}.json"), encoding="utf-8")), w)
        src[d] = s.screened(run, line)
    scores = [round(x * 0.5, 2) for x in range(0, 41)]
    for a in alphas:
        V, K, C = (shifted(src[d], a) for d in m.DIRS)
        print(f"a {a}: {s.show(s.search(V, K, C, base, base_sets, scores, m.DETS))}", flush=True)


if __name__ == "__main__":
    main()
