"""Average the probes of several detector runs (same spans) and sweep as select_det does."""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
from select_det import sweep
names = sys.argv[1:]
recs = [r for r in map(json.loads, open("labels.jsonl")) if r["split"] == "val" and r["status"] != "quarantine" and not r["jwtd"]]
res = [json.load(open(f"eval-val/res-{n}det-r0.json", encoding="utf-8")) for n in names]
avg = {p: [[s, e, sum(r[p][i][2] for r in res) / len(res), t] for i, (s, e, _, t) in enumerate(res[0][p])] for p in res[0]}
b = sweep(recs, avg)
print("+".join(names), {k: round(v, 3) if isinstance(v, float) else v for k, v in b.items()})
