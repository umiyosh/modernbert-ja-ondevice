"""Product yardsticks for one detector at a time, each at the lowest detector threshold (per candidate
score) that keeps false marks within D's today: val clean clips <= max(1, D), the 60 Kotobae clean
texts <= max(1, D). Reported there: ASR errors found on val and on the held-out synthetic readings
(all spans, homophones), and Kotobae's typo sets found per subset (with right words marked on ~ok).

usage (from asr/): python prod.py <val name>:<kotobae name>[:<synth name>] ...
D today is measured first from d:rust[:d] at its shipped thresholds.
"""

import collections
import functools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
import eval_spans  # noqa: E402

eval_spans.norm_map = functools.lru_cache(maxsize=None)(eval_spans.norm_map)
from eval_spans import clip_counts, marks, summary  # noqa: E402

INF = float("inf")
SCORES = [5.13, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 14.0, INF]
DETS = [round(x, 3) for x in np.arange(0.3, 1.0, 0.01)] + [INF]


def recs(path):
    return [r for r in map(json.loads, open(path, encoding="utf-8")) if r["split"] == "val" and r["status"] != "quarantine" and not r["jwtd"]]


VAL = recs("labels.jsonl")
VAL_CLEAN = [r for r in VAL if r["status"] == "clean"]
SYN = recs("eval-kokoro/labels.jsonl")
TEXTS = {t["id"]: t["text"] for t in json.load(open("../texts.json", encoding="utf-8"))}
SET = {c["id"]: c for c in json.load(open("../set.json", encoding="utf-8"))}
CLEAN = [t for t in TEXTS if t.startswith("clean~")]
KCHARS = sum(len(TEXTS[t]) for t in CLEAN)
ITEMS = [c for c in SET if c + "~err" in TEXTS]
TEXT = {**TEXTS, **{r["path"]: r["hyp"] for r in VAL + SYN}}


def load(d, name):
    p = os.path.join(d, f"res-{name}det-r0.json")
    q = os.path.join(d, f"deltas-{name}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")), json.load(open(q, encoding="utf-8")) if os.path.exists(q) else {}


JA = re.compile("[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]")
GATE = "--gate" in sys.argv


def mk(src, key, det, score):
    """With --gate, a mark on text without kana or kanji (Latin, digits, symbols) is dropped: the
    product looks for Japanese misconversions, and these were most of the new detectors' false marks."""
    P, Dl = src
    ms = marks(P.get(key, []), Dl.get(key, []), det, score)
    return [(s, e) for s, e in ms if JA.search(TEXT[key][s:e])] if GATE else ms


def kotobae_fm(K, det, score):
    return 1000 * sum(len(mk(K, t, det, score)) for t in CLEAN) / KCHARS


def val_fm(V, det, score):
    marks_n = sum(len(mk(V, r["path"], det, score)) for r in VAL_CLEAN)
    return 1000 * marks_n / sum(len(r["hyp"]) for r in VAL_CLEAN)


def found(src, rs, det, score):
    c = collections.Counter()
    for r in rs:
        c.update(clip_counts(r, mk(src, r["path"], det, score)))
    return summary(c)


def sets(K, det, score):
    out = collections.Counter()
    for c in ITEMS:
        x = SET[c]
        for suffix, word, key in (("~err", x["wrong"], x["set"]), ("~ok", x["right"], "right_marked")):
            a = TEXTS[c + suffix].find(word)
            out[key] += any(s < a + len(word) and a < e for s, e in mk(K, c + suffix, det, score))
    return dict(out)


def row(V, K, S, det, score):
    v = found(V, VAL, det, score)
    s = found(S, SYN, det, score) if S else {}
    return {"det": det, "score": score, "kotobae_fm": round(kotobae_fm(K, det, score), 2), "val_fm": round(v["false_marks_per_1k_chars"], 3),
            "val": round(v["span_recall"], 3), "val_hom": round(v["homophone_recall"], 3),
            "syn": round(s.get("span_recall", 0), 3), "syn_hom": round(s.get("homophone_recall", 0), 3), "sets": sets(K, det, score)}


def show(name, r):
    print(f"{name:22s} det {r['det']:<5} score {r['score']:<5} | val {r['val']:.3f} hom {r['val_hom']:.3f} | syn {r['syn']:.3f} hom {r['syn_hom']:.3f}"
          f" | fm val {r['val_fm']:.3f} kotobae {r['kotobae_fm']:.2f} | {r['sets']}", flush=True)


def main() -> None:
    names = [a.split(":") for a in sys.argv[1:] if not a.startswith("--")]
    global GATE
    gate, GATE = GATE, False
    base = row(load("eval-val", "d"), load("..", "rust"), load("eval-kokoro", "rustd"), 0.4707, 5.13)
    GATE = gate
    show("D today", base)
    kb, vb = max(1.0, base["kotobae_fm"]), max(1.0, base["val_fm"])
    for n in names:
        V, K = load("eval-val", n[0]), load("..", n[1])
        S = load("eval-kokoro", n[2]) if len(n) > 2 else None
        best = []
        for score in SCORES:
            det = next((d for d in DETS if kotobae_fm(K, d, score) <= kb and val_fm(V, d, score) <= vb), None)
            if det is None:
                continue
            r = row(V, K, S, det, score)
            keeps = all(r["sets"].get(k, 0) >= base["sets"].get(k, 0) for k in base["sets"] if k != "right_marked")
            best.append((keeps, r))
        for keeps, r in sorted(best, key=lambda x: (not x[0], -x[1]["val_hom"]))[:6]:
            show(n[0] + (" keeps" if keeps else " DROPS"), r)


if __name__ == "__main__":
    main()
