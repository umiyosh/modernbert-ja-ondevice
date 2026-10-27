"""Both detectors in one pipeline: a token is marked when D's P >= tD or C's P >= tC; candidates are
scored for words either detector screened, and marked at lead >= score. Product yardsticks only:
false marks on correct text no worse than D today (val clean clips, the 60 Kotobae texts), Kotobae's
typo sets found no fewer than D, and the most ASR errors found on val (all spans, homophones).

usage: python union.py <D name val> <C name val> <D name kotobae> <C name kotobae>
"""

import collections
import functools
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
import eval_spans  # noqa: E402

eval_spans.norm_map = functools.lru_cache(maxsize=None)(eval_spans.norm_map)
from eval_spans import clip_counts, marks, summary  # noqa: E402

ASR = os.path.expanduser("~/dvlp/kotobae/.jev-eval/typo-study/asr")
KS = os.path.expanduser("~/dvlp/kotobae/.jev-eval/typo-study")
OFF = 9.0


def load(d, kind, name):
    return json.load(open(os.path.join(d, f"{kind}-{name}{'det-r0' if kind == 'res' else ''}.json"), encoding="utf-8"))


def merge(pd, pc, dd, dc):
    """Probes as max(pD/tD, pC/tC) need the thresholds, so keep both P per span; deltas joined."""
    cp = {(s, e): p for s, e, p, _ in pc}
    probes = [(s, e, p, cp.get((s, e), 0.0)) for s, e, p, _ in pd]
    seen = {(s, e) for s, e, *_ in dd}
    return probes, dd + [x for x in dc if (x[0], x[1]) not in seen]


def union_marks(item, td, tc, score):
    probes, deltas = item
    scaled = [[s, e, max(a / td, b / tc), "det"] for s, e, a, b in probes]
    return marks(scaled, deltas, 1.0, score)


def main() -> None:
    dv, cv, dk, ck = sys.argv[1:5]
    recs = [r for r in map(json.loads, open(os.path.join(ASR, "labels.jsonl"))) if r["split"] == "val" and r["status"] != "quarantine" and not r["jwtd"]]
    V = {}
    PD, PC, DD, DC = load(f"{ASR}/eval-val", "res", dv), load(f"{ASR}/eval-val", "res", cv), load(f"{ASR}/eval-val", "deltas", dv), load(f"{ASR}/eval-val", "deltas", cv)
    for r in recs:
        p = r["path"]
        V[p] = merge(PD[p], PC[p], DD[p], DC[p])
    texts = {t["id"]: t["text"] for t in json.load(open(f"{KS}/texts.json", encoding="utf-8"))}
    st = {c["id"]: c for c in json.load(open(f"{KS}/set.json", encoding="utf-8"))}
    PD, PC, DD, DC = load(KS, "res", dk), load(KS, "res", ck), load(KS, "deltas", dk), load(KS, "deltas", ck)
    K = {t: merge(PD.get(t, []), PC.get(t, []), DD.get(t, []), DC.get(t, [])) for t in texts}
    clean = [t for t in texts if t.startswith("clean~")]
    kchars = sum(len(texts[t]) for t in clean)
    items = [c for c in st if c + "~err" in texts]

    def kotobae(td, tc, score):
        fm = sum(len(union_marks(K[t], td, tc, score)) for t in clean)
        found = collections.Counter()
        for c in items:
            w = st[c]["wrong"]
            a = texts[c + "~err"].find(w)
            hit = any(s < a + len(w) and a < e for s, e in union_marks(K[c + "~err"], td, tc, score))
            found[st[c]["set"]] += hit
        return 1000 * fm / kchars, found

    def val(td, tc, score):
        c = collections.Counter()
        for r in recs:
            c.update(clip_counts(r, union_marks(V[r["path"]], td, tc, score)))
        return summary(c)

    # D today: its shipped thresholds, C off.
    base_k, base_found = kotobae(0.4707, OFF, 5.13)
    base_v = val(0.4707, OFF, 5.13)
    print("D today", f"kotobae fm {base_k:.2f}", dict(base_found), json.dumps({k: round(v, 3) for k, v in base_v.items()}), flush=True)
    kbudget, vbudget = max(1.0, base_k), max(1.0, base_v["false_marks_per_1k_chars"])
    TD = [0.4707, 0.55, 0.65, 0.8, OFF]
    TC = [0.6, 0.65, 0.7, 0.75, 0.8, 0.84, 0.88, 0.92, 0.96, OFF]
    SC = [5.13, 6.0, 7.0, 7.5, 8.0, 9.0, 10.0]
    rows = []
    for td, tc, sc in itertools.product(TD, TC, SC):
        kf, found = kotobae(td, tc, sc)
        if kf > kbudget or any(found[s] < base_found[s] for s in ("dev", "holdout", "fresh")):
            continue
        m = val(td, tc, sc)
        if m["false_marks_per_1k_chars"] > vbudget:
            continue
        rows.append((td, tc, sc, kf, dict(found), m))
        print(f"ok tD {td} tC {tc} score {sc}: kotobae fm {kf:.2f} {dict(found)} val recall {m['span_recall']:.3f} hom {m['homophone_recall']:.3f} fm {m['false_marks_per_1k_chars']:.2f}", flush=True)
    for key in ("span_recall", "homophone_recall"):
        b = max(rows, key=lambda r: (r[5][key], r[5]["span_recall"]))
        print("BEST", key, b[:5], json.dumps({k: round(v, 3) for k, v in b[5].items()}))


if __name__ == "__main__":
    main()
