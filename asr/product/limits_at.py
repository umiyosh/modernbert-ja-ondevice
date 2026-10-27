"""Every product limit at chosen operating points (Kotobae #98: limits_why.py says which limit is never
met; this says by how much a promising point misses each one).

usage (from asr/): python tools/limits_at.py <name> <line> <w> <det>:<score> [<det>:<score> ...]
"""
import json
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import marks_sweep as m  # noqa: E402
import prod  # noqa: E402
import screen_sweep as s  # noqa: E402
import verify_sweep as v  # noqa: E402


def main():
    name, line, w = ARGS[0], float(ARGS[1]), float(ARGS[2])
    src = {}
    for d in m.DIRS:
        run = prod.load(d, name)
        if w:
            run = v.verified(run, json.load(open(os.path.join(d, f"verify-{name}.json"), encoding="utf-8")), w)
        src[d] = run
    V, K, C = (s.screened(src[d], line) for d in m.DIRS)
    base, base_sets = s.limits()
    print(f"limits: kotobae {max(1.0, base['kotobae']):.3f}/1k clean {base['clean']} dev_fm {base['dev_fm']:.3f}/1k "
          f"right_marked <= 2 sets {base_sets}", flush=True)
    for point in ARGS[3:]:
        det, score = map(float, point.split(":"))
        rec, fm = s.hom(V, det, score)
        sets = prod.sets(K, det, score)
        print(f"det {det} score {score}: dev hom {rec:.3f} dev_fm {fm:.3f} kotobae {prod.kotobae_fm(K, det, score):.3f} "
              f"clean {s.clean(C, det, score)} sets {sets}", flush=True)


if __name__ == "__main__":
    main()
