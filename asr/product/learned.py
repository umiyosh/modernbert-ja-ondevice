"""A learned decision over each screened candidate row, in place of the hand-set lead threshold.
Features per row (best alternative by PLL + same-reading kana-kanji lead): leads, detector P, voice-table
rate and channel counts, reading match, sizes. Labels: the row overlaps a real error span on a val clip.
Val scores are out-of-fold (5 folds by speaker); Kotobae's texts and the Kokoro readings are scored by a
model fit on all of val, so the Kokoro numbers are an outside check. A row is marked when the model's
probability >= t, or by today's rule (detector P >= det, or lead >= s0); thresholds as in prodforce.

usage (from asr/): python tools/learned.py <name> [rate]   (VT selects the voice table)
"""

import collections
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import prodforce  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

prod = prodforce.prod
CV = os.path.expanduser("~/dvlp/kotobae/.jev-eval/cv-ja-27/cv-corpus-27.0-2026-09-11/ja/validated.tsv")
INF = float("inf")


def rows_of(d, name, rate):
    """(key, s, e, features, best alt, best lead) for every row today's policy would score."""
    prodforce.RATE = rate
    prodforce.ALTW = float(os.environ.get("ALTW", "0"))  # with ALT=<deltas>, fold a second scorer into the lead
    P, D = prodforce.policy(d, name, 1.0, 1.0, 0.0, True)
    _, pll = prodforce.policy(d, name, 1.0, 0.0, 0.0, True)
    out = []
    for key, rs in D.items():
        probes = P.get(key, [])
        plain = {(s, e): alts for s, e, _, alts in pll.get(key, [])}
        for s, e, w, alts in rs:
            if not alts:
                continue
            alt, lead = max(alts, key=lambda x: x[1])
            p_alts = dict(plain.get((s, e), []))
            sus = max((pp for x, y, pp, _ in probes if x < e and s < y), default=0.0)
            v = prodforce.VOICE.get(w)
            n = prodforce.COUNTS.get(w, {}).get("to", {}).get(alt, 0)
            f = [lead, p_alts.get(alt, 0.0), lead - p_alts.get(alt, 0.0), max(p_alts.values()),
                 math.log(max(sus, 1e-4)), prodforce.rate(w) if v else -1.0, math.log1p(v["kept"]) if v else -1.0,
                 math.log1p(n), float(prodforce.kana(w) == prodforce.kana(alt)), len(alts), len(w),
                 sorted((x for _, x in alts), reverse=True)[1] - lead if len(alts) > 1 else -20.0]
            out.append((key, s, e, f, alt, lead))
    return P, out


def labels(rows):
    spans = {}
    for r in prod.VAL:
        if r["status"] == "clean":
            continue
        _, at = prod.eval_spans.norm_map(r["hyp"])
        spans[r["path"]] = [prod.eval_spans.raw_span(at, *e["hyp_chars"]) for e in r["edits"]
                            if e["kind"] in prod.eval_spans.SPAN_KINDS and e["hyp_chars"][1] > e["hyp_chars"][0]]
    return np.array([any(a < e and s < b for a, b in spans.get(k, [])) for k, s, e, *_ in rows], dtype=int)


def speakers(rows):
    who = {r["path"]: r["client_id"] for r in csv.DictReader(open(CV, encoding="utf-8"), delimiter="\t")}
    return [who.get(k, k) for k, *_ in rows]


def model():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0)


def as_src(P, rows, prob, s0, t):
    """Deltas whose single value is >= 1 exactly when the row is marked (lead >= s0 or prob >= t)."""
    D = collections.defaultdict(list)
    for (key, s, e, _, alt, lead), q in zip(rows, prob):
        D[key].append([s, e, "", [[alt, max(lead / s0 if s0 < INF else -INF, q / t)]]])
    return P, D


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]  # prodforce adds --gate
    name = args[0]
    rate = float(args[1]) if len(args) > 1 else 0.2
    (PV, RV), (PK, RK), (PS, RS) = (rows_of(d, name, rate) for d in ("eval-val", "..", "eval-kokoro"))
    X, y, g = np.array([r[3] for r in RV]), labels(RV), speakers(RV)
    print(f"val rows {len(RV)}, positives {y.sum()}, speakers {len(set(g))}", flush=True)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        oof[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    full = model().fit(X, y)
    pk, ps = (full.predict_proba(np.array([r[3] for r in R]))[:, 1] for R in (RK, RS))
    prod.GATE = False
    base = prod.row(prod.load("eval-val", "d"), prod.load("..", "rust"), prod.load("eval-kokoro", "rustd"), 0.4707, 5.13)
    prod.GATE = True
    prod.show("D today", base)
    kb, vb = max(1.0, base["kotobae_fm"]), max(1.0, base["val_fm"])
    best = []
    for s0 in (4.5, 5.0, 6.0, INF):
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            V, K, S = as_src(PV, RV, oof, s0, t), as_src(PK, RK, pk, s0, t), as_src(PS, RS, ps, s0, t)
            det = next((x for x in prod.DETS if prod.kotobae_fm(K, x, 1.0) <= kb and prod.val_fm(V, x, 1.0) <= vb), None)
            if det is None:
                continue
            r = prod.row(V, K, S, det, 1.0)
            keeps = all(r["sets"].get(k, 0) >= base["sets"].get(k, 0) for k in base["sets"] if k != "right_marked")
            keeps = keeps and r["sets"].get("right_marked", 0) <= base["sets"]["right_marked"]
            best.append((keeps, f"s0 {s0} t {t}", r))
    for keeps, label, r in sorted(best, key=lambda x: (not x[0], -x[2]["val_hom"]))[:6]:
        prod.show(f"{label} " + ("keeps" if keeps else "DROPS"), r)


if __name__ == "__main__":
    main()
