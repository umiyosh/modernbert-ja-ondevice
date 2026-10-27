"""Screening policies on a run that scored every candidate (KOTOBAE_TYPO_SCREEN=0): a span is scored when
the detector's P over it is >= 0.018 or, with --force, when its written word is a voice-table word. Each
alternative's value is a * PLL + b * kana-kanji lead (same-reading pairs only) + c * channel bonus, where
the bonus is log(n(w->alt) + 0.5) - log(kept(w) + 0.5) from the voice table (0 for pairs not in it).
usage (from asr/): python prodforce.py <name> a,b,c[,force] ...
"""
import functools
import json
import math
import os
import sys

sys.argv = [sys.argv[0], "--gate"] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
import fugashi  # noqa: E402
import numpy as np  # noqa: E402
import prod  # noqa: E402

prod.SCORES = [round(x, 2) for x in np.arange(2.0, 20.0, 0.5)] + [float("inf")]
TAG = fugashi.Tagger()
SCREEN = 0.018
VOICE = json.load(open(os.environ.get("VT", "voice-table") + "/voice.json", encoding="utf-8"))
COUNTS = json.load(open(os.environ.get("VT", "voice-table") + "/voice-counts.json", encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def kana(w):
    return "".join((getattr(t.feature, "kana", None) or t.surface) for t in TAG(w))


RATE = 0.0
SUSW = 0.0  # weight of log(P / SCREEN): a suspicious span needs a smaller lead (FASPell-style 2-D cut)
ALT = os.environ.get("ALT")  # a second scorer's deltas (pll.py), weighted by the 6th spec value
ALTW = 0.0
BASE = os.environ.get("BASE")  # candidates not in this run's deltas were generated (gen.py) and pay GENPEN
GENPEN = 0.0


def rate(w):
    e = VOICE[w]
    return e["wrong"] / max(1, e["wrong"] + e["kept"])


def bonus(w, alt):
    n = COUNTS.get(w, {}).get("to", {}).get(alt)
    if not n:
        return 0.0
    return math.log(n + 0.5) - math.log(VOICE[w]["kept"] + 0.5)


def policy(d, name, a, b, c, force):
    P, D = prod.load(d, name)
    p = os.path.join(d, f"kk-{name}.json")
    kk = {}
    for key, rows in (json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}).items():
        for s, e, alt, v in rows:
            kk[(key, s, e, alt)] = v
    alt2 = {}
    if ALT and ALTW:
        for key, rows in json.load(open(os.path.join(d, f"deltas-{ALT}.json"), encoding="utf-8")).items():
            for s, e, w, alts in rows:
                for alt, v in alts:
                    alt2[(key, s, e, alt)] = v
    orig = set()
    if BASE and GENPEN and os.path.exists(os.path.join(d, f"deltas-{BASE}.json")):
        for key, rows in json.load(open(os.path.join(d, f"deltas-{BASE}.json"), encoding="utf-8")).items():
            for s, e, w, alts in rows:
                orig.update((key, s, e, alt) for alt, _ in alts)
    gensus = float(os.environ.get("GENSUS", "0"))  # generated candidates only where the detector is this sure
    pen = lambda key, s, e, alt, sus: (GENPEN if sus >= gensus else 1e9) if orig and (key, s, e, alt) not in orig else 0.0  # noqa: E731
    D2 = {}
    for key, rows in D.items():
        probes = P.get(key, [])
        out = []
        for s, e, w, alts in rows:
            sus = max((pp for x, y, pp, _ in probes if x < e and s < y), default=0.0)
            if sus < SCREEN and not (force and w in VOICE and rate(w) >= RATE):
                continue
            sw = SUSW * math.log(max(sus, 1e-4) / SCREEN)
            out.append([s, e, w, [[alt, sw - pen(key, s, e, alt, sus) + ALTW * alt2.get((key, s, e, alt), 0.0) + a * v + (b * kk.get((key, s, e, alt), 0.0) if kana(w) == kana(alt) else 0.0) + c * bonus(w, alt)]
                                  for alt, v in alts]])
        D2[key] = out
    return P, D2


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    name = args[0]
    prod.GATE = False
    base = prod.row(prod.load("eval-val", "d"), prod.load("..", "rust"), prod.load("eval-kokoro", "rustd"), 0.4707, 5.13)
    prod.GATE = True
    prod.show("D today", base)
    kb, vb = max(1.0, base["kotobae_fm"]), max(1.0, base["val_fm"])
    for spec in args[1:]:
        parts = spec.split(",")
        a, b, c = map(float, parts[:3])
        force = len(parts) > 3
        global RATE
        RATE = float(parts[3]) if force else 0.0
        global SUSW
        SUSW = float(parts[4]) if len(parts) > 4 else 0.0
        global ALTW
        ALTW = float(parts[5]) if len(parts) > 5 else 0.0
        global GENPEN
        GENPEN = float(parts[6]) if len(parts) > 6 else 0.0
        V, K, S = (policy(d, name, a, b, c, force) for d in ("eval-val", "..", "eval-kokoro"))
        best = []
        for score in prod.SCORES:
            det = next((x for x in prod.DETS if prod.kotobae_fm(K, x, score) <= kb and prod.val_fm(V, x, score) <= vb), None)
            if det is None:
                continue
            r = prod.row(V, K, S, det, score)
            keeps = all(r["sets"].get(k, 0) >= base["sets"].get(k, 0) for k in base["sets"] if k != "right_marked")
            best.append((keeps, r))
        for keeps, r in sorted(best, key=lambda x: (not x[0], -x[1]["val_hom"]))[:2]:
            prod.show(f"{spec} " + ("keeps" if keeps else "DROPS"), r)


if __name__ == "__main__":
    main()
