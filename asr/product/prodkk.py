"""prod.py with the candidate score replaced by a mix of the PLL lead and the kana-kanji lead
(kk-<name>.json from kk_score.py): value = a * PLL + b * kk. Same thresholds rule and yardsticks.
usage (from asr/): python prodkk.py <val>:<kotobae>:<synth> <a>,<b> ...
"""
import os
import sys

sys.argv = [sys.argv[0], "--gate"] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
import json  # noqa: E402

import functools  # noqa: E402

import fugashi  # noqa: E402
import numpy as np  # noqa: E402
import prod  # noqa: E402

prod.SCORES = [round(x, 2) for x in np.arange(2.0, 20.0, 0.5)] + [float("inf")]
TAG = fugashi.Tagger()
SAME = "--same" in sys.argv


@functools.lru_cache(maxsize=None)
def kana(w):
    return "".join((getattr(t.feature, "kana", None) or t.surface) for t in TAG(w))


def mixed(d, name, a, b):
    src = prod.load(d, name)
    if src is None:
        return None
    P, D = src
    kk = {}
    p = os.path.join(d, f"kk-{name}.json")
    for key, rows in (json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}).items():
        for s, e, alt, v in rows:
            kk[(key, s, e, alt)] = v
    D2 = {key: [[s, e, w, [[alt, a * v + b * (kk.get((key, s, e, alt), 0.0) if not SAME or kana(w) == kana(alt) else 0.0)] for alt, v in alts]] for s, e, w, alts in rows]
          for key, rows in D.items()}
    return P, D2


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    names = args[0].split(":")
    prod.GATE = False
    base = prod.row(prod.load("eval-val", "d"), prod.load("..", "rust"), prod.load("eval-kokoro", "rustd"), 0.4707, 5.13)
    prod.GATE = True
    prod.show("D today", base)
    kb, vb = max(1.0, base["kotobae_fm"]), max(1.0, base["val_fm"])
    for mix in args[1:]:
        a, b = map(float, mix.split(","))
        V, K, S = mixed("eval-val", names[0], a, b), mixed("..", names[1], a, b), mixed("eval-kokoro", names[2], a, b)
        best = []
        for score in prod.SCORES:
            det = next((d for d in prod.DETS if prod.kotobae_fm(K, d, score) <= kb and prod.val_fm(V, d, score) <= vb), None)
            if det is None:
                continue
            r = prod.row(V, K, S, det, score)
            keeps = all(r["sets"].get(k, 0) >= base["sets"].get(k, 0) for k in base["sets"] if k != "right_marked")
            best.append((keeps, r))
        for keeps, r in sorted(best, key=lambda x: (not x[0], -x[1]["val_hom"]))[:3]:
            prod.show(f"{mix} " + ("keeps" if keeps else "DROPS"), r)


if __name__ == "__main__":
    main()
