# Why are broken transcripts 4x more common in val than train? Counts only; no ids or text printed.
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
from classify import norm, LATIN, REPEAT
from near_dup import read
CV = os.path.expanduser("~/dvlp/kotobae/.jev-eval/cv-ja-27/cv-corpus-27.0-2026-09-11/ja")
D = os.path.expanduser("~/dvlp/kotobae/.jev-eval/typo-study/asr")
spk = {r["path"]: r["client_id"] for r in read(os.path.join(CV, "validated.tsv"))}
ms = {r["clip"]: int(r["duration[ms]"]) for r in read(os.path.join(CV, "clip_durations.tsv"))}
by = {s: collections.defaultdict(lambda: [0, 0]) for s in ("train", "val")}
why = {s: collections.Counter() for s in ("train", "val")}
dur = {s: [] for s in ("train", "val")}
for rec in map(json.loads, open(os.path.join(D, "labels.jsonl"), encoding="utf-8")):
    s = rec["split"]; b = "broken" in rec["reasons"]
    g = by[s][spk[rec["path"]]]; g[0] += 1; g[1] += b
    dur[s].append(ms.get(rec["path"], 0))
    if b:
        r, h = norm(rec["ref"]), norm(rec["hyp"])
        ratio = len(h) / max(len(r), 1)
        why[s]["short_hyp" if ratio < 0.6 else "long_hyp" if ratio > 1.5 else "latin" if LATIN.search(h) else "repeat"] += 1
rng = np.random.default_rng(0)
for s in ("train", "val"):
    m = np.array(list(by[s].values()), dtype=float)
    n, k = m.sum(0)
    top = m[np.argsort(-m[:, 1])]
    boot = [m[i].sum(0) for i in (rng.integers(0, len(m), len(m)) for _ in range(2000))]
    ci = np.quantile([b[1] / b[0] for b in boot], [0.025, 0.975])
    with_b = (m[:, 1] > 0).sum()
    print(f"{s}: clips {int(n)} broken {int(k)} ({100*k/n:.1f}%, speaker CI {100*ci[0]:.1f}-{100*ci[1]:.1f}) speakers {len(m)} with-broken {with_b}")
    for t in (1, 3, 10):
        rest = m.sum(0) - top[:t].sum(0)
        print(f"   top{t} speakers hold {int(top[:t,1].sum())} broken ({100*top[:t,1].sum()/k:.0f}%); rate without them {100*rest[1]/rest[0]:.2f}%; their clips {int(top[:t,0].sum())}")
    print("   kinds", dict(why[s]), " median dur ms", int(np.median(dur[s])), "p90", int(np.quantile(dur[s], 0.9)))
