"""Paired comparison of two systems under crossfit.py's protocol (Kotobae #98, 10-25): each system's point
is picked on one val half and scored on the other, both ways, and the val homophones each one catches are
paired with the exact McNemar test. Counts only, never text.

usage (from asr/): python tools/crossfit_pair.py <name> <name> [line 0.1] [w 1] [margin 0.85] [line2]
  line2, when given, is the second system's screening line (default: the same as the first's)
"""
import sys

ARGS = sys.argv[1:]
import crossfit as cf  # noqa: E402  (crossfit resets sys.argv for the modules it imports)
from jwtd_pair import mcnemar  # noqa: E402

m, prod, s = cf.m, cf.prod, cf.s
es = prod.eval_spans


def hits(name, line, w, k):
    """{(path, edit index): caught} on both halves, each at the point picked on the other."""
    K, C = cf.src("..", name, line, w), cf.src("eval-clean", name, line, w)
    V = {h: cf.src(d, name, line, w) for h, d in cf.RUNS.items()}
    out = {}
    for pick, score in (("dev", "calib"), ("calib", "dev")):
        s.half = cf.as_dev(pick)
        base, base_sets = s.limits()
        best = s.search(V[pick], K, C, {**base, "dev_fm": base["dev_fm"] * k}, base_sets, m.SCORES, m.DETS)
        s.half = cf.as_dev(score)
        print(f"{name} pick {pick}: det {best[1]} score {best[2]}", flush=True)
        for r in prod.VAL:
            if cf.SPLIT[r["path"]] != score:
                continue
            marks = prod.mk(V[score], r["path"], best[1], best[2])
            _, at = es.norm_map(r["hyp"])
            for i, e in enumerate(r["edits"] or []):
                x, y = e["hyp_chars"]
                if e["kind"] == "homophone" and y > x:
                    a, b = es.raw_span(at, x, y)
                    out[(r["path"], i)] = any(p < b and a < q for p, q in marks)
    return out


def main():
    line = float(ARGS[2]) if len(ARGS) > 2 else 0.1
    w = float(ARGS[3]) if len(ARGS) > 3 else 1.0
    k = float(ARGS[4]) if len(ARGS) > 4 else 0.85
    line2 = float(ARGS[5]) if len(ARGS) > 5 else line
    x, y = hits(ARGS[0], line, w, k), hits(ARGS[1], line2, w, k)
    only_x = sum(x[e] and not y[e] for e in x)
    only_y = sum(y[e] and not x[e] for e in x)
    print(f"{ARGS[0]} {sum(x.values())}/{len(x)} vs {ARGS[1]} {sum(y.values())}/{len(y)}: "
          f"only {ARGS[0]} {only_x}, only {ARGS[1]} {only_y}, exact McNemar p={mcnemar(only_x, only_y):.4g}")


if __name__ == "__main__":
    main()
