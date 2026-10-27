"""Recall, precision and sentence accuracy of representative settings on the dev half (Kotobae #98).
Recall: homophones (and all error spans) marked. Precision: marks overlapping a real error span, over all
marks on dev clips (clean clips included). Accuracy: clips judged right (an error clip with a mark on an
error, a clean clip with no mark).

usage (from asr/): python tools/prf.py
"""
import collections
import json
import os
import sys

sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402
import screen_sweep as s  # noqa: E402
import verify_sweep as v  # noqa: E402

es = prod.eval_spans
half = json.load(open("split-calib.json"))
RS = [r for r in prod.VAL if half[r["path"]] == "dev"]


def run(d, name, line=None, w=0.0):
    src = prod.load(d, name)
    if w:
        src = v.verified(src, json.load(open(os.path.join(d, f"verify-{name}.json"), encoding="utf-8")), w)
    return s.screened(src, line) if line is not None else src


def prf(src, det, score, gate=True):
    prod.GATE = gate
    c = collections.Counter()
    for r in RS:
        ms = prod.mk(src, r["path"], det, score)
        c.update(prod.clip_counts(r, ms))
        if r["status"] == "clean":
            c["right_clips"] += not ms
        else:
            _, at = es.norm_map(r["hyp"])
            spans = [es.raw_span(at, *e["hyp_chars"]) for e in r["edits"] if e["kind"] in es.SPAN_KINDS and e["hyp_chars"][1] > e["hyp_chars"][0]]
            c["right_clips"] += any(a < t and s_ < b for a, b in ms for s_, t in spans)
    prod.GATE = True
    marks = c["marks_on_error_clips"] + c["false_marks"]
    return {"hom": c["homophone_found"] / c["homophone_spans"], "all": c["found"] / c["spans"],
            "prec": c["marks_on_spans"] / marks, "marks": marks, "acc": c["right_clips"] / len(RS),
            "fm": 1000 * c["false_marks"] / c["clean_chars"], "n": (c["homophone_spans"], c["spans"], c["error_clips"], len(RS))}


ROWS = [
    ("D", run("eval-val", "d"), 0.4707, 5.13, False),
    ("ship3", run("eval-val", "ship3"), 0.68, 6.75, True),
    ("free", run("eval-dev", "sb0", 0.005, 1.0), 0.64, 9.5, True),
    ("marks", run("eval-dev", "sb0s", 0.1, 1.0), 0.96, 9.5, True),
    ("trial", run("eval-dev", "sb0kk", 0.05), 0.9, 9.5, True),
]
for name, src, det, score, gate in ROWS:
    x = prf(src, det, score, gate)
    print(name, {k: (round(val, 4) if isinstance(val, float) else val) for k, val in x.items()}, flush=True)
