"""Cross-fitted thresholds over the two val halves (Kotobae #98, 10-25): the product point is searched on
one half, as marks_sweep.py does on dev, and scored on the other, both ways. The calib half had its one
look (sb2k5q7: dev 65.8% -> calib 60.3%, false marks 1.09 -> 1.61 per 1,000 characters), so it now serves
to measure how far a point chosen at the edge of the false-mark limit carries to unseen speakers. The
cap on the searching half is D's false marks there times a margin.

usage (from asr/): python tools/crossfit.py <name> [line 0.1] [w 1] [margins 1,0.85,0.7]
"""
import collections
import json
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import marks_sweep as m  # noqa: E402
import prod  # noqa: E402
import screen_sweep as s  # noqa: E402
import verify_sweep as v  # noqa: E402

SPLIT = dict(s.half)
RUNS = {"dev": "eval-dev", "calib": "eval-calib"}


def as_dev(h):
    """The split with half h labelled dev, which is the half screen_sweep reads."""
    return {k: "dev" if x == h else "other" for k, x in SPLIT.items()}


def src(d, name, line, w):
    run = prod.load(d, name)
    if w:
        run = v.verified(run, json.load(open(os.path.join(d, f"verify-{name}.json"), encoding="utf-8")), w)
    return s.screened(run, line)


def main():
    name = ARGS[0]
    line = float(ARGS[1]) if len(ARGS) > 1 else 0.1
    w = float(ARGS[2]) if len(ARGS) > 2 else 1.0
    margins = [float(x) for x in (ARGS[3] if len(ARGS) > 3 else "1,0.85,0.7").split(",")]
    K, C = src("..", name, line, w), src("eval-clean", name, line, w)
    V = {h: src(d, name, line, w) for h, d in RUNS.items()}
    pooled = {k: collections.Counter() for k in margins}
    for pick, score in (("dev", "calib"), ("calib", "dev")):
        s.half = as_dev(pick)
        base, base_sets = s.limits()
        cap = base["dev_fm"]
        s.half = as_dev(score)
        d_fm = s.limits()[0]["dev_fm"]
        for k in margins:
            s.half = as_dev(pick)
            best = s.search(V[pick], K, C, {**base, "dev_fm": cap * k}, base_sets, m.SCORES, m.DETS)
            if not best:
                print(f"pick {pick} cap {cap * k:.2f} ({k}x D): nothing within limits", flush=True)
                continue
            s.half = as_dev(score)
            c = counts(V[score], best[1], best[2], score)
            pooled[k].update(c)
            rec, fm = prod.summary(c)["homophone_recall"], prod.summary(c)["false_marks_per_1k_chars"]
            print(f"pick {pick} cap {cap * k:.2f} ({k}x D {cap:.2f}): det {best[1]} score {best[2]} "
                  f"{pick} {100 * best[0]:.1f}% fm {best[3]:.2f} -> {score} {100 * rec:.1f}% fm {fm:.2f} (D {d_fm:.2f})",
                  flush=True)
    for k, c in pooled.items():
        t = prod.summary(c)
        print(f"pooled {k}x D: homophones {100 * t['homophone_recall']:.1f}% ({c['homophone_found']}/{c['homophone_spans']}) "
              f"fm {t['false_marks_per_1k_chars']:.2f}/1k ({c['false_marks']} marks, {c['clean_chars']} chars)", flush=True)


def counts(src_, det, sc, h):
    """screen_sweep.hom's counter on half h, so the two scored halves can be added up."""
    c = collections.Counter()
    for r in prod.VAL:
        if SPLIT[r["path"]] == h:
            c.update(prod.clip_counts(r, prod.mk(src_, r["path"], det, sc)))
    return c


if __name__ == "__main__":
    main()
