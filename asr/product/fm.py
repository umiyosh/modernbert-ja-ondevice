import collections, json, os, sys, re
sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
from eval_spans import marks
name, det, score = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
L = lambda k: json.load(open(f"eval-val/{k}", encoding="utf-8"))
P, Dl = L(f"res-{name}det-r0.json"), L(f"deltas-{name}.json")
recs = [r for r in map(json.loads, open("labels.jsonl")) if r["split"]=="val" and r["status"]=="clean" and not r["jwtd"]]
kind = collections.Counter(); src = collections.Counter(); per = collections.Counter(); chars=0
def cls(s):
    if re.fullmatch(r"[0-9０-９]+", s): return "digits"
    if re.fullmatch(r"[゠-ヿ]+", s): return "katakana"
    if re.fullmatch(r"[一-鿿々]+", s): return "kanji"
    if re.fullmatch(r"[぀-ゟ]+", s): return "hiragana"
    if re.fullmatch(r"[A-Za-zＡ-Ｚａ-ｚ]+", s): return "latin"
    return "mixed/punct"
cand = {}
for r in recs:
    chars += len(r["hyp"])
    ms = marks(P[r["path"]], Dl[r["path"]], det, score)
    cs = {(s,e) for s,e,_,a in Dl[r["path"]] if a and max(v for _,v in a)>=score}
    per[len(ms)] += 1
    for s,e in ms:
        kind[cls(r["hyp"][s:e])] += 1; src["cand" if (s,e) in cs else "det"] += 1
print("clean chars", chars, "false marks", sum(kind.values()))
print("by char class", dict(kind)); print("by source", dict(src)); print("marks per clip", dict(sorted(per.items())))
