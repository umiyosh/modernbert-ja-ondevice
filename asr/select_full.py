"""Refit both thresholds of a full system (detector + candidate scoring, Kotobae's typo_eval output) on
val: among the pairs whose false marks on error-free clips stay within 1 per 1,000 characters (the
operating point of PROTOCOL.md), the one with the highest span recall. JWTD-overlap clips are left out
as in the primary metric. D keeps its shipped thresholds; this is for C.

With --written <dir>, a pair must also keep within the budget on the 60 error-free Kotobae texts
(clean~ ids of <dir>/texts.json, run through typo_eval under the same name), as D's shipped thresholds
were chosen to: on val alone C's detector threshold fell to where it marked 3.5 per 1,000
characters of those texts (PROTOCOL.md, operating point).

usage: python asr/select_full.py <labels.jsonl> <eval dir> [--written <dir>] <name> [<name>...]
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


def written_budget(d: str, name: str):
    texts = {t["id"]: t["text"] for t in json.load(open(os.path.join(d, "texts.json"), encoding="utf-8"))}
    clean = [t for t in texts if t.startswith("clean~")]
    chars = sum(len(texts[t]) for t in clean)
    probes = json.load(open(os.path.join(d, f"res-{name}det-r0.json"), encoding="utf-8"))
    deltas = json.load(open(os.path.join(d, f"deltas-{name}.json"), encoding="utf-8"))
    return lambda det, score: 1000 * sum(len(marks(probes[t], deltas[t], det, score)) for t in clean) / chars <= BUDGET


def main() -> None:
    args = sys.argv[1:]
    written = None
    if "--written" in args:
        i = args.index("--written")
        written = args[i + 1]
        del args[i : i + 2]
    labels, eval_dir, *names = args
    with open(labels, encoding="utf-8") as f:
        recs = [r for r in map(json.loads, f) if r["split"] == "val" and r["status"] != "quarantine" and not r["jwtd"]]
    for name in names:
        probes = json.load(open(os.path.join(eval_dir, f"res-{name}det-r0.json"), encoding="utf-8"))
        deltas = json.load(open(os.path.join(eval_dir, f"deltas-{name}.json"), encoding="utf-8"))
        within = written_budget(written, name) if written else (lambda t, s: True)
        best = None
        for s in SCORE:
            # False marks fall as the detector threshold rises, so the first that fits is the best
            # detector threshold for this score threshold.
            for t in DET:
                m = summary(counts(recs, probes, deltas, t, s))
                if m["false_marks_per_1k_chars"] <= BUDGET and within(t, s):
                    if best is None or m["span_recall"] > best["span_recall"]:
                        best = {"detector": t, "score": s, **m}
                    break
        print(name, json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
