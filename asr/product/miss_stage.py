"""Where the val homophones are lost, stage by stage, at one operating point (Kotobae #98, 10-25): no
candidate row on the span; a row without the reference word among its alternatives; the reference word
offered but its row left unscored by the screening line; scored but its lead under the mark (by how far).
Both halves, each from its own run directory. Counts only, never text.

usage (from asr/): python tools/miss_stage.py <name> <line> <w> <det> <score>
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

es = prod.eval_spans
RUNS = {"dev": "eval-dev", "calib": "eval-calib"}


def stage(r, e, full, kept, det, sc):
    """The stage at which one homophone edit is lost, or 'found'."""
    _, at = es.norm_map(r["hyp"])
    a, b = es.raw_span(at, *e["hyp_chars"])
    if any(x < b and a < y for x, y in prod.mk(kept, r["path"], det, sc)):
        return "found"
    over = lambda rows: [row for row in rows if row[0] < b and a < row[1]]  # noqa: E731
    rows = over(full[1].get(r["path"], []))
    if not rows:
        return "no row on the span"
    right = [row for row in rows if any(alt == e["ref"] for alt, _ in row[3])]
    if not right:
        return "reference word not offered"
    scored = [row for row in over(kept[1].get(r["path"], [])) if any(alt == e["ref"] for alt, _ in row[3])]
    if not scored:
        return "offered, not scored (screening line)"
    lead = max(x for row in scored for alt, x in row[3] if alt == e["ref"])
    gap = sc - lead
    return ("scored, lead short by < 1" if gap < 1 else "scored, lead short by 1-3" if gap < 3
            else "scored, lead short by >= 3")


def main():
    name, line, w, det, sc = ARGS[0], float(ARGS[1]), float(ARGS[2]), float(ARGS[3]), float(ARGS[4])
    total = collections.Counter()
    for h, d in RUNS.items():
        run = prod.load(d, name)
        if w:
            run = v.verified(run, json.load(open(os.path.join(d, f"verify-{name}.json"), encoding="utf-8")), w)
        kept = s.screened(run, line)
        c = collections.Counter()
        for r in prod.VAL:
            if s.half[r["path"]] != h:
                continue
            for e in r["edits"] or []:
                x, y = e["hyp_chars"]
                if e["kind"] == "homophone" and y > x:
                    c[stage(r, e, run, kept, det, sc)] += 1
        total.update(c)
        print(h, dict(c.most_common()), flush=True)
    n = sum(total.values())
    for k, m in total.most_common():
        print(f"  {k:40s} {m:4d}  {100 * m / n:5.1f}%")


if __name__ == "__main__":
    main()
