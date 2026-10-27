# (1) Which items each 30m model catches at the <=1 mark/1k budget (ft alone), paired.
# (2) Counterfactual: recall if the clean marks on the skewed words were not there.
import json, sys
D = sys.argv[1]
st = {c["id"]: c for c in json.load(open(f"{D}/set.json"))}
texts = {t["id"]: t["text"] for t in json.load(open(f"{D}/texts.json"))}
clean = [t for t in texts if t.startswith("clean~")]
chars = sum(len(texts[t]) for t in clean); k = int(chars / 1000)
ids = [c for c in st if st[c]["set"] in ("dev", "holdout")]
def wmax(r, tid, w):
    t = texts[tid]; a = t.find(w); b = a + len(w)
    return max([p for s, e, p, _ in r.get(tid, []) if s < b and a < e] or [0])
def caught(n, drop=()):
    r = json.load(open(f"{D}/res-{n}-r0.json"))
    cl = sorted((p for t in clean for s, e, p, _ in r.get(t, []) if texts[t][s:e] not in drop), reverse=True)
    th = cl[k] + 1e-6
    return {c for c in ids if wmax(r, c + "~err", st[c]["wrong"]) >= th}, th
print("clean chars", chars, "marks allowed", k)
a, _ = caught("ft30maug"); b, _ = caught("ft30maug2")
print(f"aug {len(a)}  aug2 {len(b)}  both {len(a & b)}  only aug {len(a - b)}  only aug2 {len(b - a)}")
for drop in (("次に",), ("次に", "訪ね", "朝")):
    for n in ("ft30maug", "ft30maug2"):
        got, th = caught(n, drop)
        print(f"without marks on {'/'.join(drop)}: {n} {len(got)}/{len(ids)} th={th:.3f}")
