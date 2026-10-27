"""deltas-/kk- of <name> plus those of <gen> (scored generated candidates) as <out>; res- linked. Run from asr/."""
import json, os, sys
d, name, gen, out = sys.argv[1:5]
L = lambda p: json.load(open(os.path.join(d, p), encoding="utf-8")) if os.path.exists(os.path.join(d, p)) else {}  # noqa: E731
D, G = L(f"deltas-{name}.json"), L(f"deltas-{gen}.json")
M = {key: [[s, e, w, list(alts)] for s, e, w, alts in rows] for key, rows in D.items()}
for key, rows in G.items():
    for s, e, w, alts in rows:
        row = next((r for r in M.setdefault(key, []) if r[0] == s and r[1] == e), None)
        if row is None:
            M[key].append([s, e, w, alts])
        else:
            have = {a for a, _ in row[3]}
            row[3] += [x for x in alts if x[0] not in have]
json.dump(M, open(os.path.join(d, f"deltas-{out}.json"), "w", encoding="utf-8"), ensure_ascii=False)
K = L(f"kk-{name}.json")
for key, rows in L(f"kk-{gen}.json").items():
    K.setdefault(key, []).extend(rows)
json.dump(K, open(os.path.join(d, f"kk-{out}.json"), "w", encoding="utf-8"), ensure_ascii=False)
dst = os.path.join(d, f"res-{out}det-r0.json")
if not os.path.lexists(dst):
    os.symlink(f"res-{name}det-r0.json", dst)
print("merged", sum(map(len, G.values())), "generated rows into", out)
