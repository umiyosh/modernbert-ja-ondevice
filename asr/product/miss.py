"""Where the real-voice homophone spans go under one prodforce policy: found, or missed because
A  no candidate row overlapping the span was screened in (split by whether the right word was a candidate),
B1 no candidate row overlaps the span at all,
B2 rows were scored but none offered the right word,
C  the right word was offered and lost to the score threshold.
usage (from asr/): python miss.py <name> a,b,c[,rate] det score
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import prodforce  # noqa: E402

prod = prodforce.prod


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    name, spec, det, score = args[0], args[1], float(args[2]), float(args[3])
    parts = spec.split(",")
    a, b, c = map(float, parts[:3])
    force = len(parts) > 3
    prodforce.RATE = float(parts[3]) if force else 0.0
    prodforce.GENPEN = float(os.environ.get("GENPEN", "0"))
    V = prodforce.policy("eval-val", name, a, b, c, force)
    _, full = prod.load("eval-val", name)
    out = collections.Counter()
    lost = []
    for r in prod.VAL:
        if r["status"] == "clean":
            continue
        ms = prod.mk(V, r["path"], det, score)
        _, at = prod.eval_spans.norm_map(r["hyp"])
        for e in r["edits"]:
            x, y = e["hyp_chars"]
            if e["kind"] != "homophone" or y <= x:
                continue
            s, t = prod.eval_spans.raw_span(at, x, y)
            if any(p < t and s < q for p, q in ms):
                out["found"] += 1
                continue
            rows = [w for w in full.get(r["path"], []) if w[0] < t and s < w[1]]
            kept = [w for w in V[1].get(r["path"], []) if w[0] < t and s < w[1]]
            has = lambda ws: any(alt == e["ref"] for w in ws for alt, _ in w[3])  # noqa: E731
            if not rows:
                out["B1 no candidates"] += 1
            elif not kept:
                out["A not screened, right word offered" if has(rows) else "A not screened, right word absent"] += 1
            elif not has(kept):
                out["B2 right word absent"] += 1
            else:
                v = max(val for w in kept for alt, val in w[3] if alt == e["ref"])
                out["C lost on score"] += 1
                lost.append((round(v, 1), e["hyp"], e["ref"]))
    n = sum(out.values())
    for k, v in sorted(out.items()):
        print(f"{k:38s} {v:4d}  {100 * v / n:5.1f}")
    print("total", n)
    lost.sort()
    print("C values:", collections.Counter(int(v) for v, _, _ in lost).most_common())
    print("C examples:", lost[-15:])


if __name__ == "__main__":
    main()
