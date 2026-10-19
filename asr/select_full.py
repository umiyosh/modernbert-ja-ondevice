"""Refit both thresholds of a full system (detector + candidate scoring, Kotobae's typo_eval output) on
val: among the pairs whose false marks on error-free clips stay within 1 per 1,000 characters (the
operating point of PROTOCOL.md), the one with the highest span recall. JWTD-overlap clips are left out
as in the primary metric. D keeps its shipped thresholds; this is for C.

usage: python asr/select_full.py <labels.jsonl> <eval dir> <name> [<name>...]
"""

import collections
import functools
import json
import os
import sys

import numpy as np

import eval_spans
from eval_spans import clip_counts, marks, summary

# clip_counts maps each transcript again for every pair of thresholds.
eval_spans.norm_map = functools.lru_cache(maxsize=None)(eval_spans.norm_map)

BUDGET = 1.0
DET = [round(x, 3) for x in np.arange(0.30, 1.0, 0.01)]
SCORE = [round(x, 2) for x in np.arange(2.0, 12.01, 0.5)] + [float("inf")]


def counts(recs, probes, deltas, det, score) -> collections.Counter:
    c = collections.Counter()
    for rec in recs:
        c.update(clip_counts(rec, marks(probes[rec["path"]], deltas[rec["path"]], det, score)))
    return c


def main() -> None:
    labels, eval_dir, *names = sys.argv[1:]
    with open(labels, encoding="utf-8") as f:
        recs = [r for r in map(json.loads, f) if r["split"] == "val" and r["status"] != "quarantine" and not r["jwtd"]]
    for name in names:
        probes = json.load(open(os.path.join(eval_dir, f"res-{name}det-r0.json"), encoding="utf-8"))
        deltas = json.load(open(os.path.join(eval_dir, f"deltas-{name}.json"), encoding="utf-8"))
        best = None
        for s in SCORE:
            # False marks fall as the detector threshold rises, so the first that fits is the best
            # detector threshold for this score threshold.
            for t in DET:
                m = summary(counts(recs, probes, deltas, t, s))
                if m["false_marks_per_1k_chars"] <= BUDGET:
                    if best is None or m["span_recall"] > best["span_recall"]:
                        best = {"detector": t, "score": s, **m}
                    break
        print(name, json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
