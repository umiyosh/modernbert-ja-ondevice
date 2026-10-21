# Why did the capped data lose recall? For each dev+holdout item: the training error rate of the
# wrong word's tokens before (base+aug) and after (base-t+augb-t) the cap, and each model's score
# relative to its own <=1-mark threshold (ft alone, as compare.py). If the loss sits on items whose
# wrong word the cap cut, the old model found them by the word's prior, not by the context.
# usage (from typo-study/ft): python3 -I cause.py
import json, os
os.environ.setdefault("HF_HOME", "../hf"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sbintuitions/modernbert-ja-30m")
V = 120000
st = {c["id"]: c for c in json.load(open("../set.json"))}
texts = {t["id"]: t["text"] for t in json.load(open("../texts.json"))}
clean = [t for t in texts if t.startswith("clean~")]
chars = sum(len(texts[t]) for t in clean)
ids = [c for c in st if st[c]["set"] in ("dev", "holdout")]


def counts(paths):
    E = np.zeros(V, np.int64); A = np.zeros(V, np.int64)
    for p in paths:
        d = np.load(p); lab = d["labels"]
        A += np.bincount(d["ids"][lab >= 0], minlength=V); E += np.bincount(d["ids"][lab == 1], minlength=V)
    return E, A


def wrong_toks(c):
    text = texts[c + "~err"]; w = st[c]["wrong"]; a = text.find(w); b = a + len(w)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    return [t for t, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]) if s < b and a < e and e > s]


def model(name):
    r = json.load(open(f"../res-{name}-r0.json"))
    cl = sorted((p for t in clean for s, e, p, _ in r.get(t, [])), reverse=True)
    th = cl[int(chars / 1000)] + 1e-6
    def wmax(tid, w):
        t = texts[tid]; a = t.find(w); b = a + len(w)
        return max([p for s, e, p, _ in r.get(tid, []) if s < b and a < e] or [1e-6])
    return th, {c: wmax(c + "~err", st[c]["wrong"]) for c in ids}


old = counts(["base.npz", "aug.npz"]); new = counts(["base-t.npz", "augb-t.npz"])
toks = {c: wrong_toks(c) for c in ids}
def rate(cnt, c):
    E, A = cnt
    return max((E[t] / A[t] if A[t] else 0.0) for t in toks[c])
r_old = {c: rate(old, c) for c in ids}; r_new = {c: rate(new, c) for c in ids}
th_a, sa = model("ft30maug"); th_b, sb = model("ft30mbound")
print(f"thresholds: aug {th_a:.3f}, bound {th_b:.3f}")
cut = {c: r_old[c] > 0.05 for c in ids}  # the cap bound for this wrong word
groups = {"lost": [], "gained": [], "both": [], "neither": []}
for c in ids:
    ha, hb = sa[c] >= th_a, sb[c] >= th_b
    groups["both" if ha and hb else "lost" if ha else "gained" if hb else "neither"].append(c)
for g, cs in groups.items():
    if cs:
        print(f"{g:8s} n={len(cs):3d}  cap bound (old rate>5%): {sum(cut[c] for c in cs):3d}  "
              f"median old rate {np.median([r_old[c] for c in cs]):.1%}  new {np.median([r_new[c] for c in cs]):.1%}")
# Margin over the model's own threshold, so the general calibration shift does not count as loss.
for label, sel in (("cap bound", [c for c in ids if cut[c]]), ("cap not bound", [c for c in ids if not cut[c]])):
    m = np.array([np.log(sb[c] / th_b) - np.log(sa[c] / th_a) for c in sel])
    ha = sum(sa[c] >= th_a for c in sel); hb = sum(sb[c] >= th_b for c in sel)
    print(f"{label:14s} n={len(sel):3d}  hits aug {ha} -> bound {hb}  median change in log(score/threshold) {np.median(m):+.2f}")
print("lost items: " + "  ".join(f"{st[c]['wrong']}→{st[c]['right']}({r_old[c]:.0%}→{r_new[c]:.0%})" for c in groups["lost"]))
print("gained items: " + "  ".join(f"{st[c]['wrong']}→{st[c]['right']}({r_old[c]:.0%}→{r_new[c]:.0%})" for c in groups["gained"]))
