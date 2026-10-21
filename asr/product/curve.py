"""How many real-voice homophones the current signals can find at a given false-mark rate: the learned
decision (out-of-fold, learned.py) alone and together with today's rule, swept over its threshold with
no budget, printing val homophone recall against false marks per 1,000 characters (val clean clips and
Kotobae's clean texts). Also splits the homophones by whether the (written, right) pair is in the voice
table, i.e. was seen in training. usage (from asr/): python tools/curve.py <name> [rate]
"""

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import learned  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

prod = learned.prod
INF = float("inf")


def hom_split(V):
    """Homophone recall split by whether the pair is in the voice table (seen) or not."""
    c = collections.Counter()
    for r in prod.VAL:
        if r["status"] == "clean":
            continue
        ms = prod.mk(V, r["path"], 1.0, 1.0)
        _, at = prod.eval_spans.norm_map(r["hyp"])
        for e in r["edits"]:
            x, y = e["hyp_chars"]
            if e["kind"] != "homophone" or y <= x:
                continue
            s, t = prod.eval_spans.raw_span(at, x, y)
            seen = "seen" if learned.prodforce.COUNTS.get(e["hyp"], {}).get("to", {}).get(e["ref"]) else "unseen"
            c[seen] += 1
            c[seen + "_found"] += any(a < t and s < b for a, b in ms)
    return c


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name, rate = args[0], float(args[1]) if len(args) > 1 else 0.2
    (PV, RV), (PK, RK) = (learned.rows_of(d, name, rate) for d in ("eval-val", ".."))
    X, y, g = np.array([r[3] for r in RV]), learned.labels(RV), learned.speakers(RV)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        oof[te] = learned.model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    pk = learned.model().fit(X, y).predict_proba(np.array([r[3] for r in RK]))[:, 1]
    print("rule                   | hom found /100 | false marks per 1,000 chars: val, Kotobae | seen pairs, unseen pairs")
    for s0, t, det in [(4.5, INF, 0.6)] + [(INF, t, INF) for t in (0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05)]:
        V, K = learned.as_src(PV, RV, oof, s0, t), learned.as_src(PK, RK, pk, s0, t)
        # det folds into the same 0/1 cut: probes over det count as marks, as in prod.mk
        Vd, Kd = (V[0], V[1]), (K[0], K[1])
        v = prod.found(Vd, prod.VAL, det, 1.0)
        kf = prod.kotobae_fm(Kd, det, 1.0)
        h = hom_split(Vd) if det == INF else None
        tag = "today's rule (A)" if s0 < INF else f"learned p >= {t}"
        extra = f"{h['seen_found']}/{h['seen']}, {h['unseen_found']}/{h['unseen']}" if h else ""
        print(f"{tag:22s} | {100 * v['homophone_recall']:5.1f} | {v['false_marks_per_1k_chars']:5.2f}, {kf:5.2f} | {extra}", flush=True)


if __name__ == "__main__":
    main()
