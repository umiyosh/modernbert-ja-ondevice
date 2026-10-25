"""Which limit keeps a version out when marks_sweep.py finds nothing within limits (Kotobae #98): over the same
grid, the points that break exactly one limit, counted per limit, with the best dev-half homophone recall each
would have reached. Only the dev half is read.

usage (from asr/): python tools/limit_why.py <name> <line>
"""
import collections
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import marks_sweep as m  # noqa: E402
import prod  # noqa: E402
import screen_sweep as s  # noqa: E402


def broken(V, K, C, base, base_sets, det, score):
    sets = prod.sets(K, det, score)
    rec, fm = s.hom(V, det, score)
    out = [k for k, ok in (("kotobae", prod.kotobae_fm(K, det, score) <= max(1.0, base["kotobae"])),
                           ("jwtd clean", s.clean(C, det, score) <= base["clean"]),
                           ("dev fm", fm <= base["dev_fm"]),
                           ("right_marked", sets["right_marked"] <= 2)) if not ok]
    out += [k for k, v in base_sets.items() if k != "right_marked" and sets[k] < v]
    return rec, out


def main():
    name, line = ARGS[0], float(ARGS[1])
    runs = {d: prod.load(d, name) for d in m.DIRS}
    base, base_sets = s.limits()
    V, K, C = (s.screened(runs[d], line) for d in m.DIRS)
    best, count = {}, collections.Counter()
    for det in m.DETS:
        for score in m.SCORES:
            rec, out = broken(V, K, C, base, base_sets, det, score)
            if len(out) == 1:
                count[out[0]] += 1
                if rec > best.get(out[0], (0,))[0]:
                    best[out[0]] = (rec, det, score)
    for k, n in count.most_common():
        rec, det, score = best[k]
        print(f"{name} line {line}: only {k} broken at {n} points; best there dev {rec:.3f} (det {det} score {score})")


if __name__ == "__main__":
    main()
