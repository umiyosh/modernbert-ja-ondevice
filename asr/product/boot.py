"""Speaker-level paired test of two runs on real voice-input homophones (Kotobae #98, PROTOCOL step 10):
homophone recall and clean-clip false marks per 1,000 characters on one half of val, each run at its
own thresholds. Bootstrap resamples speakers (10,000 times) for the 95% interval of the difference; the
p-value is a sign-flip permutation over speakers (swap the two runs within a speaker).

usage (from asr/): python tools/boot.py <half: dev|calib|all> <nameA> <detA> <scoreA> <nameB> <detB> <scoreB>
A name written name:line:w reads eval-dev's run with the detector's checks at weight w and screening at line.
"""
import collections
import csv
import json
import os
import sys

import numpy as np

sys.argv, ARGS = [sys.argv[0], "--gate"], sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
import learned  # noqa: E402
import prod  # noqa: E402
import screen_sweep as ss  # noqa: E402
import verify_sweep as vs  # noqa: E402

es = prod.eval_spans
N = 10_000


def per_speaker(src, det, score, rs, who):
    """speaker -> [homophones found, homophones, clean marks, clean chars]"""
    out = collections.defaultdict(lambda: np.zeros(4))
    for r in rs:
        marks = prod.mk(src, r["path"], det, score)
        v = out[who.get(r["path"], r["path"])]
        if r["status"] == "clean":
            v[2] += len(marks)
            v[3] += len(r["hyp"])
            continue
        _, at = es.norm_map(r["hyp"])
        for e in r["edits"] or []:
            x, y = e["hyp_chars"]
            if e["kind"] != "homophone" or y <= x:
                continue
            a, b = es.raw_span(at, x, y)
            v[0] += any(s < b and a < t for s, t in marks)
            v[1] += 1
    return out


# The calib half is scored in its own directory (Kotobae #98, 10-24): eval-dev holds only the dev half.
RUNS = "eval-calib" if ARGS and ARGS[0] == "calib" else "eval-dev"


def load(spec):
    if ":" not in spec:
        return prod.load("eval-val", spec)
    name, line, w = spec.split(":")
    src = prod.load(RUNS, name)
    if float(w):
        src = vs.verified(src, json.load(open(f"{RUNS}/verify-{name}.json", encoding="utf-8")), float(w))
    return ss.screened(src, float(line))


def stats(m):
    """m: speakers x 4 -> (recall, false marks per 1,000 characters)"""
    t = m.sum(0)
    return t[0] / t[1], 1000 * t[2] / t[3]


def main():
    half = ARGS[0]
    runs = [(ARGS[1], float(ARGS[2]), float(ARGS[3])), (ARGS[4], float(ARGS[5]), float(ARGS[6]))]
    who = {r["path"]: r["client_id"] for r in csv.DictReader(open(learned.CV, encoding="utf-8"), delimiter="\t")}
    split = json.load(open("split-calib.json"))
    rs = [r for r in prod.VAL if half == "all" or split[r["path"]] == half]
    a, b = (per_speaker(load(n), d, s, rs, who) for n, d, s in runs)
    spk = sorted(set(a) | set(b))
    A = np.array([a[k] for k in spk])
    B = np.array([b[k] for k in spk])
    (ra, fa), (rb, fb) = stats(A), stats(B)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N):
        i = rng.integers(0, len(spk), len(spk))
        (x, u), (y, w) = stats(A[i]), stats(B[i])
        boot.append((y - x, w - u))
    boot = np.array(boot)
    perm = []
    for _ in range(N):
        f = rng.integers(0, 2, len(spk)).astype(bool)[:, None]
        (x, u), (y, w) = stats(np.where(f, B, A)), stats(np.where(f, A, B))
        perm.append((y - x, w - u))
    perm = np.abs(np.array(perm))
    obs = np.array([rb - ra, fb - fa])
    p = ((perm >= np.abs(obs) - 1e-12).sum(0) + 1) / (N + 1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print(f"half {half}: speakers {len(spk)}, homophones {int(A[:, 1].sum())}, clean chars {int(A[:, 3].sum())}")
    print(f"recall      {runs[0][0]} {ra:.3f} -> {runs[1][0]} {rb:.3f}: diff {obs[0]:+.3f} "
          f"(95% CI {lo[0]:+.3f} to {hi[0]:+.3f}), p = {p[0]:.4f}")
    print(f"false marks {fa:.3f} -> {fb:.3f} per 1,000 chars: diff {obs[1]:+.3f} "
          f"(95% CI {lo[1]:+.3f} to {hi[1]:+.3f}), p = {p[1]:.4f}")


if __name__ == "__main__":
    main()
