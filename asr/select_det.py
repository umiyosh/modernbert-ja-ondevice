"""Pick a detector checkpoint and its threshold on val (step 8): for each checkpoint, the lowest
detector threshold at which false marks on val error-free clips stay within 1 per 1,000 characters
(the operating point of PROTOCOL.md), and the span recall there. Detector marks only (C-det / D-det);
JWTD-overlap clips are left out as in the primary metric.

usage: python asr/select_det.py <cv ja dir> <labels.jsonl> <eval dir> <out.json> <name=model dir>...
       a name whose res-<name>det-r0.json already exists in the eval dir is not run again
"""

import collections
import json
import os
import sys

import numpy as np

from detect import load, probes
from eval_spans import clip_counts, marks, summary

BUDGET = 1.0  # false marks per 1,000 characters of error-free clips
GRID = [round(x, 3) for x in np.arange(0.05, 0.995, 0.005)]


def sweep(recs: list[dict], res: dict) -> dict:
    best = None
    for t in GRID:
        c = collections.Counter()
        for rec in recs:
            c.update(clip_counts(rec, marks(res[rec["path"]], [], t, float("inf"))))
        m = summary(c)
        if m["false_marks_per_1k_chars"] <= BUDGET:
            best = {"threshold": t, **m, "spans": c["spans"]}
            break
    return best


def main() -> None:
    root, labels, eval_dir, out, *pairs = sys.argv[1:]
    with open(labels, encoding="utf-8") as f:
        recs = [r for r in map(json.loads, f) if r["split"] == "val" and r["status"] != "quarantine" and not r["jwtd"]]
    texts = {r["id"]: r["text"] for r in json.load(open(os.path.join(eval_dir, "texts.json"), encoding="utf-8"))}
    report = {}
    for pair in pairs:
        name, model = pair.split("=", 1)
        path = os.path.join(eval_dir, f"res-{name}det-r0.json")
        if not os.path.exists(path):
            m, tok = load(model, None)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(probes(m, tok, texts), f)
            del m
        res = json.load(open(path, encoding="utf-8"))
        report[name] = sweep(recs, res)
        print(name, json.dumps(report[name]), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
