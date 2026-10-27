"""Where one system's false marks come from at one operating point (Kotobae #98, 10-24 night): on the dev
half's correct clips and on the 60 correct Kotobae texts, each mark counted as a candidate mark (an
alternative leads by score or more) or a detector mark (a run of tokens at det or more), with the
detector's P and the best lead on the marked word. Counts only; no text is printed.

usage (from asr/): python tools/fm_origin.py <name> <line> <w> <det> <score>
"""
import collections
import json
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402
import screen_sweep as s  # noqa: E402
import verify_sweep as v  # noqa: E402


def load(d, name, line, w):
    run = prod.load(d, name)
    if w:
        run = v.verified(run, json.load(open(os.path.join(d, f"verify-{name}.json"), encoding="utf-8")), w)
    return s.screened(run, line)


def origin(src, keys, det, score):
    P, D = src
    c, ps, leads = collections.Counter(), [], []
    for k in keys:
        rows = D.get(k, [])
        for a, b in prod.mk(src, k, det, score):
            lead = max((x for s0, e0, _, alts in rows if s0 < b and a < e0 for _, x in alts), default=None)
            p = max((q for x, y, q, _ in P.get(k, []) if x < b and a < y), default=0.0)
            cand = lead is not None and lead >= score
            c["candidate" if cand else "detector"] += 1
            c["candidate, detector also >= det" if cand and p >= det else ""] += cand and p >= det
            (leads if cand else ps).append(lead if cand else p)
    c.pop("", None)
    return c


def main():
    name, line, w, det, score = ARGS[0], *(float(x) for x in ARGS[1:5])
    half = json.load(open("split-calib.json"))
    V = load("eval-dev", name, line, w)
    keys = [r["path"] for r in prod.VAL if half[r["path"]] == "dev" and r["status"] == "clean"]
    print(f"{name} det {det} score {score}, dev half correct clips: {dict(origin(V, keys, det, score))}")
    K = load("..", name, line, w)
    print(f"  Kotobae correct texts: {dict(origin(K, prod.CLEAN, det, score))}")
    C = load("eval-clean", name, line, w)
    print(f"  JWTD correct sentences: {dict(origin(C, list(C[0].keys()), det, score))}")


if __name__ == "__main__":
    main()
