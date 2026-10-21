# Broken / error rates by script length and by how often the sentence is read in train. Counts only.
import collections, json, os, sys
sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
from classify import norm, LATIN
D = os.path.expanduser("~/dvlp/kotobae/.jev-eval/typo-study/asr")
recs = [json.loads(l) for l in open(os.path.join(D, "labels.jsonl"), encoding="utf-8")]
reads = collections.Counter(norm(r["ref"]) for r in recs if r["split"] == "train")
agg = collections.defaultdict(lambda: [0, 0, 0, 0])
lat = collections.Counter()
for r in recs:
    n = len(norm(r["ref"]))
    lb = "<=10" if n <= 10 else "11-20" if n <= 20 else "21-35" if n <= 35 else ">35"
    keys = [(r["split"], "len " + lb)]
    if r["split"] == "train":
        c = reads[norm(r["ref"])]
        keys.append(("train", "reads " + ("1" if c == 1 else "2-5" if c <= 5 else "6+")))
    lat[(r["split"], bool(LATIN.search(r["ref"])))] += 1
    for k in keys:
        a = agg[k]; a[0] += 1; a[1] += "broken" in r["reasons"]
        if r["status"] != "quarantine":
            a[2] += 1; a[3] += r["status"] == "error"
for (s, b), (n, k, q, e) in sorted(agg.items()):
    print(f"{s:5} {b:10} clips {n:6}  broken {100*k/n:5.1f}%  error {100*e/max(q,1):5.1f}%")
print("script has Latin letters:", {f"{s} {x}": v for (s, x), v in sorted(lat.items())})
