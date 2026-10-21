"""Splits val into two halves by speaker (Kotobae #98, step 2a): `dev` to choose weights and rules on,
`calib` to fix thresholds under the false-mark limits, so that the reported recall comes from data no
choice was made on. Writes split-calib.json ({path: "dev"|"calib"}) and prints one run's numbers on each
half at given thresholds, with clean-clip false marks per 1,000 characters.

usage (from asr/): python tools/halves.py <name> <det> <score>
"""
import collections
import csv
import hashlib
import json
import os
import sys

sys.argv, ARGS = [sys.argv[0], "--gate"], sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
import learned  # noqa: E402
import prod  # noqa: E402

who = {r["path"]: r["client_id"] for r in csv.DictReader(open(learned.CV, encoding="utf-8"), delimiter="\t")}
half = {r["path"]: "dev" if int(hashlib.sha256(who.get(r["path"], r["path"]).encode()).hexdigest(), 16) % 2 == 0 else "calib"
        for r in prod.VAL}
json.dump(half, open("split-calib.json", "w"), indent=0)
name, det, score = ARGS[0], float(ARGS[1]), float(ARGS[2])
src = prod.load("eval-val", name)
for h in ("dev", "calib"):
    rs = [r for r in prod.VAL if half[r["path"]] == h]
    clean = [r for r in rs if r["status"] == "clean"]
    fm = 1000 * sum(len(prod.mk(src, r["path"], det, score)) for r in clean) / sum(len(r["hyp"]) for r in clean)
    c = collections.Counter()
    for r in rs:
        c.update(prod.clip_counts(r, prod.mk(src, r["path"], det, score)))
    s = prod.summary(c)
    spk = len({who.get(r["path"]) for r in rs})
    print(f"{h}: clips {len(rs)} speakers {spk} clean chars {sum(len(r['hyp']) for r in clean)} | {json.dumps(s)} | fm {fm:.3f}")
