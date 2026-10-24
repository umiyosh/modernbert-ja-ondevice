"""Where the dev half's missed homophones fall through (Kotobae #98, paper plan: the unseen-pair weakness),
split by whether training corrected the same (written, right) pair. For one system at one operating point:
found; the right word is no candidate of any row over the span; a row offers it but screening drops the
row (the detector does not suspect the word); the row is scored but the right word's lead stays below the
mark and the detector below its own. For the last two, the distance left (P, lead) is shown as quantiles.

usage (from asr/): python tools/miss_stage.py [name sb0s] [line 0.1] [w 1] [det 0.96] [score 9.5]
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
half = json.load(open("split-calib.json"))
SEEN = {(e["hyp"], e["ref"]) for r in map(json.loads, open("labels.jsonl", encoding="utf-8"))
        if r["split"] == "train" for e in r["edits"] or [] if e["kind"] == "homophone"}


def quart(xs):
    xs = sorted(xs)
    return [round(xs[int(q * (len(xs) - 1))], 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)] if xs else []


def main():
    name = ARGS[0] if ARGS else "sb0s"
    line, w, det, score = (float(x) for x in (ARGS[1:5] + ["0.1", "1", "0.96", "9.5"][len(ARGS[1:5]):]))
    src = prod.load("eval-dev", name)
    if w:
        src = v.verified(src, json.load(open(f"eval-dev/verify-{name}.json", encoding="utf-8")), w)
    P, rows = src
    kept = s.screened(src, line)[1]
    stage = collections.Counter()
    gap = collections.defaultdict(list)
    for r in prod.VAL:
        if half[r["path"]] != "dev" or r["status"] == "clean":
            continue
        ms = prod.mk((P, kept), r["path"], det, score)
        _, at = es.norm_map(r["hyp"])
        for e in r["edits"]:
            if e["kind"] != "homophone" or e["hyp_chars"][1] <= e["hyp_chars"][0]:
                continue
            a, b = es.raw_span(at, *e["hyp_chars"])
            k = "seen" if (e["hyp"], e["ref"]) in SEEN else "unseen"
            over = lambda x, y: x < b and a < y  # noqa: E731
            if any(over(x, y) for x, y in ms):
                stage[k, "found"] += 1
                continue
            right = [row for row in rows.get(r["path"], []) if over(row[0], row[1]) and any(alt == e["ref"] for alt, _ in row[3])]
            p = max((q for x, y, q, _ in P.get(r["path"], []) if over(x, y)), default=0.0)
            if not right:
                stage[k, "no candidate"] += 1
                continue
            scored = [row for row in kept.get(r["path"], []) if row in right]
            lead = max(l for row in right for alt, l in row[3] if alt == e["ref"])
            if not scored:
                stage[k, "not suspected"] += 1
                gap[k, "not suspected: P"].append(p)
            else:
                stage[k, "lead short"] += 1
                gap[k, "lead short: lead"].append(lead)
                gap[k, "lead short: P"].append(p)
    for k in ("seen", "unseen"):
        n = sum(c for (kk, _), c in stage.items() if kk == k)
        print(f"{k} ({n}): " + " | ".join(f"{st} {stage[k, st]} ({100 * stage[k, st] / n:.1f}%)"
                                          for st in ("found", "no candidate", "not suspected", "lead short")))
        for (kk, what), xs in sorted(gap.items()):
            if kk == k:
                print(f"  {what} quantiles 10/25/50/75/90%: {quart(xs)}")


if __name__ == "__main__":
    main()
