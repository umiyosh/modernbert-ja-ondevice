# Is Whisper better on clips in Common Voice's official train.tsv (possibly in its pretraining)? Counts only.
import collections, json, os, sys
sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
from near_dup import read
CV = os.path.expanduser("~/dvlp/kotobae/.jev-eval/cv-ja-27/cv-corpus-27.0-2026-09-11/ja")
D = os.path.expanduser("~/dvlp/kotobae/.jev-eval/typo-study/asr")
off = {}
for name in ("train", "dev", "test", "other", "invalidated"):
    p = os.path.join(CV, name + ".tsv")
    if os.path.exists(p):
        for r in read(p):
            off.setdefault(r["path"], name)
ver = {r["path"]: r.get("version", "") for r in read(os.path.join(CV, "validated.tsv"))}
agg = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in map(json.loads, open(os.path.join(D, "labels.jsonl"), encoding="utf-8")):
    for k in ((r["split"], "official " + off.get(r["path"], "none")),):
        a = agg[k]; a[0] += 1; a[1] += "broken" in r["reasons"]
        if r["status"] != "quarantine":
            a[2] += 1; a[3] += r["status"] == "error"
for (s, b), (n, k, q, e) in sorted(agg.items()):
    print(f"{s:5} {b:18} clips {n:6}  broken {100*k/n:5.1f}%  error {100*e/max(q,1):5.1f}%")
print("validated.tsv columns:", list(read(os.path.join(CV, "validated.tsv"))[0].keys()))
