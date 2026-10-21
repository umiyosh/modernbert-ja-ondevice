# Per-kind recall of ft alone at the threshold giving <=1 clean mark per 1k chars, the two real
# cases, and the clean words each model marks above that threshold.
import json, os, sys
from collections import Counter
D = sys.argv[1]; names = sys.argv[2].split(",")
st = {c["id"]: c for c in json.load(open(f"{D}/set.json"))}
texts = {t["id"]: t["text"] for t in json.load(open(f"{D}/texts.json"))}
clean = [t for t in texts if t.startswith("clean~")]
chars = sum(len(texts[t]) for t in clean)
ids = [c for c in st if st[c]["set"] in ("dev", "holdout")]
def wmax(r, tid, w):
    t = texts[tid]; a = t.find(w); b = a + len(w)
    return max([p for s, e, p, _ in r.get(tid, []) if s < b and a < e] or [0])
for n in names:
    r = json.load(open(f"{D}/res-{n}-r0.json"))
    cl = sorted(((p, t, s, e) for t in clean for s, e, p, _ in r.get(t, [])), reverse=True)
    k = int(chars / 1000)  # marks allowed at budget 1
    th = cl[k][0] + 1e-6
    tot, hit = Counter(), Counter()
    for c in ids:
        key = f"{st[c]['source']}/{st[c]['kind']}"; tot[key] += 1
        hit[key] += wmax(r, c + "~err", st[c]["wrong"]) >= th
    fa = sum(wmax(r, c + "~ok", st[c]["right"]) >= th for c in ids)
    print(f"== {n} th={th:.3f} recall {sum(hit.values())}/{len(ids)} fa_ok {fa}")
    print("  " + "  ".join(f"{k}:{hit[k]}/{tot[k]}" for k in sorted(tot)))
    print("  real: " + "  ".join(f"{c}:{wmax(r, c + '~err', st[c]['wrong']):.4f}" for c in ("t01", "r01")))
    print("  clean top: " + " ".join(f"{texts[t][s:e]}({p:.2f})" for p, t, s, e in cl[:k + 1]))
