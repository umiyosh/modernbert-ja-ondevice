# Does a token's error rate in the training data (labels==1 / labelled occurrences) predict the
# model's score on correct text? If so, false marks come from surface priors the data taught,
# a property measurable before training.
import json, os, sys
os.environ.setdefault("HF_HOME", "../hf"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sbintuitions/modernbert-ja-30m")

def counts(paths):
    err = allc = None
    for p in paths:
        d = np.load(p); ids, lab = d["ids"], d["labels"]
        V = 120000
        a = np.bincount(ids[lab >= 0], minlength=V); e = np.bincount(ids[lab == 1], minlength=V)
        err = e if err is None else err + e; allc = a if allc is None else allc + a
    return err, allc

texts = {t["id"]: t["text"] for t in json.load(open("../texts.json"))}
clean = [t for t in texts if t.startswith("clean~")]

def clean_tokens():
    out = {}
    for tid in clean:
        text, start = texts[tid], 0
        for i, c in enumerate(text):
            if c in "。！？\n" or i == len(text) - 1:
                sent = text[start:i + 1]
                if sent.strip():
                    enc = tok(sent, return_offsets_mapping=True, add_special_tokens=False)
                    for t, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
                        if b > a:
                            out[(tid, start + a, start + b)] = t
                start = i + 1
    return out

def rank(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x)); return r

ct = clean_tokens()
base_e, base_a = counts(["base.npz"])
for name, paths in (("ft30m", ["base.npz"]), ("ft30maug2", ["base.npz", "aug.npz"])):
    e, a = counts(paths)
    rate = (e + 0.5) / (a + 50)              # smoothed error rate of the token in the training data
    lift = rate / ((base_e + 0.5) / (base_a + 50))  # how far the synthetic rows raised it
    res = json.load(open(f"../res-{name}-r0.json"))
    rows = [(p, ct[(t, s, e_)]) for t in clean for s, e_, p, _ in res.get(t, []) if (t, s, e_) in ct]
    p = np.array([x for x, _ in rows]); ids = np.array([i for _, i in rows])
    rho = np.corrcoef(rank(p), rank(rate[ids]))[0, 1]
    top = np.argsort(-p)[:28]
    print(f"== {name}: clean tokens {len(p)}, spearman(score, train error rate) = {rho:.2f}")
    print(f"   train error rate: all clean tokens median {np.median(rate[ids]):.4f}, top-28 marks median {np.median(rate[ids[top]]):.4f}")
    if name != "ft30m":
        print(f"   lift by synthetic rows: all clean median {np.median(lift[ids]):.2f}, top-28 median {np.median(lift[ids[top]]):.2f}, top-28 with lift>=2: {(lift[ids[top]] >= 2).sum()}/28")
        print("   top-28: " + " ".join(f"{tok.convert_ids_to_tokens(int(i))}({pp:.2f},r{rate[i]:.3f},x{lift[i]:.1f})" for pp, i in zip(p[top], ids[top])))
for nm, paths in (("aug", ["base.npz", "aug.npz"]), ("augcap", ["base.npz", "augcap.npz"])):
    e, a = counts(paths)
    lift = ((e + 0.5) / (a + 50)) / ((base_e + 0.5) / (base_a + 50))
    seen = a >= 200
    print(f"{nm}: tokens seen>=200: {seen.sum()}, lift>=2: {(lift[seen] >= 2).sum()}, lift>=5: {(lift[seen] >= 5).sum()}, max {lift[seen].max():.1f}")
    for w in ("次に", "朝", "訪", "校正", "当番"):
        i = tok.convert_tokens_to_ids(w)
        if i is not None and i != tok.unk_token_id:
            print(f"   {w}: rate {(e[i] + 0.5) / (a[i] + 50):.4f} lift {lift[i]:.1f} (err {e[i]}/{a[i]})")
