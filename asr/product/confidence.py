# Confidence for "no misconversion": how far a fine text's most suspicious signals stay below
# their marks. Chooses the formula on dev (AUC of err vs ok among texts called fine), reports
# holdout+fresh, and the share of erroneous texts among those called fine by confidence.
import json, sys
D = sys.argv[1]
st = {c["id"]: c for c in json.load(open(f"{D}/set.json"))}
texts = {t["id"]: t["text"] for t in json.load(open(f"{D}/texts.json"))}
jev = json.load(open(f"{D}/res-base-r0.json"))
det = json.load(open(f"{D}/res-rustdet-r0.json"))
dl = json.load(open(f"{D}/deltas-rust.json"))
JEV, DET, SC = 0.55, 0.4707, 5.13

def signals(t):
    j = max([p for *_, p, _ in jev.get(t, [])] or [0.0])
    d = max([p for *_, p, _ in det.get(t, [])] or [0.0])
    s = max([v for *_, alts in dl.get(t, []) for _, v in alts] or [-99.0])
    return j, d, s

def fine(sig):
    j, d, s = sig
    return j < JEV and d < DET and s < SC

def parts(sig):
    j, d, s = sig
    return [j / JEV, d / DET, max(s, 0.0) / SC]

FORMULAS = {
    "A nearest": lambda sig: 1 - max(parts(sig)),
    "B mean": lambda sig: 1 - sum(parts(sig)) / 3,
    "C jev only": lambda sig: 1 - parts(sig)[0],
}

def items(sets):
    out = []
    for c, x in st.items():
        if x["set"] in sets:
            out.append((c + "~err", 1)); out.append((c + "~ok", 0))
    return out

def auc(pairs):
    pos = [v for v, y in pairs if y]; neg = [v for v, y in pairs if not y]
    if not pos or not neg: return float("nan")
    # Higher confidence should mean correct: count correct-above-erroneous pairs.
    wins = sum((n > p) + 0.5 * (n == p) for p in pos for n in neg)
    return wins / len(pos) / len(neg)

for name, f in FORMULAS.items():
    for label, sets in (("dev", {"dev"}), ("holdout+fresh", {"holdout", "fresh"})):
        rows = [(f(signals(t)), y) for t, y in items(sets) if fine(signals(t))]
        print(f"{name:12s} {label:14s} fine {len(rows):3d} (err {sum(y for _, y in rows)}) AUC {auc(rows):.3f}")
clean = [t for t in texts if t.startswith("clean~")]
print("clean long texts called fine:", sum(fine(signals(t)) for t in clean), "/", len(clean))
