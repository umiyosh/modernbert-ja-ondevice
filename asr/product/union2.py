"""D and a new detector C in one pipeline, each with its own thresholds: a token is marked when
pD >= tD or pC >= tC; a candidate is marked when its lead >= sD on a word D screened or >= sC on a word C
screened. Kept only where false marks stay within D's today (val clean, Kotobae's 60 clean texts) and
every typo subset finds at least what D finds; ranked by val homophones. --gate as in prod.py.

usage (from asr/): python union2.py <C name> [--gate]   (val and Kotobae probes under the same name)
"""

import collections
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod  # noqa: E402
from eval_spans import clip_counts, marks, summary  # noqa: E402

INF = float("inf")


def join(D, C, key):
    """Probes as (s, e, pD, pC); deltas as (s, e, word, [(alt, lead)]) tagged by who screened them."""
    (PD, DD), (PC, DC) = D, C
    pc = {(s, e): p for s, e, p, _ in PC.get(key, [])}
    probes = [(s, e, p, pc.get((s, e), 0.0)) for s, e, p, _ in PD.get(key, [])]
    lead = lambda alts: max((v for _, v in alts), default=-INF)  # noqa: E731
    dd = {(s, e): lead(a) for s, e, _, a in DD.get(key, []) if a}
    dc = {(s, e): lead(a) for s, e, _, a in DC.get(key, []) if a}
    return probes, dd, dc


def umarks(item, key, td, tc, sd, sc):
    probes, dd, dc = item
    deltas = []
    for k in set(dd) | set(dc):
        v = max(dd.get(k, -INF) / sd, dc.get(k, -INF) / sc)
        deltas.append((k[0], k[1], "", [("", v)]))
    scaled = [(s, e, max(a / td, b / tc), "det") for s, e, a, b in probes]
    ms = marks(scaled, deltas, 1.0, 1.0)
    return [(s, e) for s, e in ms if prod.JA.search(prod.TEXT[key][s:e])] if prod.GATE else ms


def main() -> None:
    cname = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    VD, VC = prod.load("eval-val", "d"), prod.load("eval-val", cname)
    KD, KC = prod.load("..", "rust"), prod.load("..", cname)
    V = {r["path"]: join(VD, VC, r["path"]) for r in prod.VAL}
    K = {t: join(KD, KC, t) for t in prod.TEXTS}
    base = {"dev": 72, "holdout": 36, "fresh": 33, "dev-seen": 44, "real": 2}
    kb, vb = 1.0, 1.176

    def kotobae(*th):
        fm = 1000 * sum(len(umarks(K[t], t, *th)) for t in prod.CLEAN) / prod.KCHARS
        found = collections.Counter()
        for c in prod.ITEMS:
            x = prod.SET[c]
            a = prod.TEXTS[c + "~err"].find(x["wrong"])
            found[x["set"]] += any(s < a + len(x["wrong"]) and a < e for s, e in umarks(K[c + "~err"], c + "~err", *th))
        return fm, found

    def val(*th):
        c = collections.Counter()
        for r in prod.VAL:
            c.update(clip_counts(r, umarks(V[r["path"]], r["path"], *th)))
        return summary(c)

    TD = [0.4707, 0.5, 0.55, 0.6, 0.7, INF]
    TC = [0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, INF]
    SD = [5.13, 5.5, 6.0, 7.0]
    SC = [7.0, 8.0, 9.0, 10.0, 12.0, INF]
    rows = []
    for th in itertools.product(TD, TC, SD, SC):
        fm, found = kotobae(*th)
        if fm > kb or any(found[k] < v for k, v in base.items()):
            continue
        m = val(*th)
        if m["false_marks_per_1k_chars"] > vb:
            continue
        rows.append((th, fm, dict(found), m))
    print(len(rows), "keep every typo subset and both budgets")
    for th, fm, found, m in sorted(rows, key=lambda r: -r[3]["homophone_recall"])[:8]:
        print(f"tD {th[0]} tC {th[1]} sD {th[2]} sC {th[3]} | val {m['span_recall']:.3f} hom {m['homophone_recall']:.3f} fm {m['false_marks_per_1k_chars']:.3f} | kotobae {fm:.2f} {found}")


if __name__ == "__main__":
    main()
