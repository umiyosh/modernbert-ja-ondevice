"""Val exploration: the full system plus rule spans (unknown kanji/katakana runs, table hints) that the
detector backs with P >= r somewhere inside the span. Same budget as select_full.

usage: python rules_gate.py <labels.jsonl> <eval dir> <name>...
"""

import collections
import functools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
import eval_spans  # noqa: E402
from eval_spans import clip_counts, marks, summary  # noqa: E402

eval_spans.norm_map = functools.lru_cache(maxsize=None)(eval_spans.norm_map)

BUDGET = 1.0
DET = [round(x, 3) for x in np.arange(0.70, 1.0, 0.01)]
SCORE = [7.0, 8.0, 9.0, 10.0, float("inf")]
GATE = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)] + [2.0]


def gated(probes, rules, r):
    out = []
    for s, e in rules:
        p = max((q for a, b, q, _ in probes if a < e and s < b), default=0.0)
        if p >= r:
            out.append((s, e))
    return out


def main() -> None:
    labels, eval_dir, *names = sys.argv[1:]
    with open(labels, encoding="utf-8") as f:
        recs = [r for r in map(json.loads, f) if r["split"] == "val" and r["status"] != "quarantine" and not r["jwtd"]]
    for name in names:
        load = lambda k: json.load(open(os.path.join(eval_dir, k), encoding="utf-8"))  # noqa: E731
        probes, deltas, rules = load(f"res-{name}det-r0.json"), load(f"deltas-{name}.json"), load(f"rules-{name}.json")
        per_gate = {g: {r["path"]: gated(probes[r["path"]], rules[r["path"]], g) for r in recs} for g in GATE}
        best = None
        for g in GATE:
            for s in SCORE:
                for t in DET:
                    c = collections.Counter()
                    for rec in recs:
                        ms = marks(probes[rec["path"]], deltas[rec["path"]], t, s)
                        ms += [x for x in per_gate[g][rec["path"]] if not any(a < x[1] and x[0] < b for a, b in ms)]
                        c.update(clip_counts(rec, ms))
                    m = summary(c)
                    if m["false_marks_per_1k_chars"] <= BUDGET:
                        if best is None or m["span_recall"] > best["span_recall"]:
                            best = {"gate": g, "detector": t, "score": s, **m}
                        break
            print(name, "gate", g, "best so far", json.dumps(best), flush=True)
        print(name, "BEST", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
