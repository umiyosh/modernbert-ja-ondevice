"""Which product limit keeps a system out (Kotobae #98: marks_sweep.py finding nothing within limits).
Over marks_sweep.py's grid, for each limit: how many points pass it, and the best dev-half homophone
recall when every other limit holds, with the failing limit's value there.

usage (from asr/): python tools/limits_why.py <name> [line 0.1] [w 1]
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
    name = ARGS[0]
    line = float(ARGS[1]) if len(ARGS) > 1 else 0.1
    w = float(ARGS[2]) if len(ARGS) > 2 else 1.0
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
    rows = []
    for det in m.DETS:
        for score in m.SCORES:
            sets = prod.sets(K, det, score)
            rec, fm = s.hom(V, det, score)
            kf, cl = prod.kotobae_fm(K, det, score), s.clean(C, det, score)
            checks = {"kotobae": (kf <= max(1.0, base["kotobae"]), round(kf, 3)),
                      "clean": (cl <= base["clean"], cl),
                      "dev_fm": (fm <= base["dev_fm"], round(fm, 3)),
                      "right_marked": (sets["right_marked"] <= 2, sets["right_marked"])}
            for k, floor in base_sets.items():
                if k != "right_marked":
                    checks[k] = (sets[k] >= floor, sets[k])
            rows.append((det, score, rec, checks))
    for k in rows[0][3]:
        passing = sum(r[3][k][0] for r in rows)
        others = [r for r in rows if all(ok for kk, (ok, _) in r[3].items() if kk != k)]
        best = max(others, key=lambda r: r[2]) if others else None
        tail = (f"best without it det {best[0]} score {best[1]}: dev {best[2]:.3f}, {k} {best[3][k][1]}"
                if best else "nothing even without it")
        print(f"{k}: {passing}/{len(rows)} pass | {tail}", flush=True)


if __name__ == "__main__":
    main()
