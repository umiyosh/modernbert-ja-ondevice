# Ablation: candidate scoring without the JWTD-derived confusion table. Rebuilds each text's
# candidates without the confusion hints, keeps only those alternatives from the hyb15nb2 deltas,
# and writes single-threshold sources (no in/out split) for judge.py.
import json, os, sys
D = sys.argv[1]
sys.path.insert(0, D)
os.chdir(D)
import mlm_study_helpers as h

h.hints = {t: [x for x in v if x["why"] != "confusion"] for t, v in h.hints.items()}
deltas = json.load(open("deltas-hyb15nb2.json"))
OFF = 99.0
for name, keep_all in (("abl-all1", True), ("abl-nojwtd1", False)):
    out = {}
    for tid, rows in deltas.items():
        allowed = h.candidates(tid) if not keep_all else None
        spans = []
        for s, e, w, alts in rows:
            ok = [d for a, d in alts if keep_all or a in allowed.get((s, e), [])]
            if ok:
                spans.append((s, e, max(ok), "cs"))
        out[tid] = spans
    json.dump(out, open(f"res-{name}-r0.json", "w"), ensure_ascii=False)
    print(name, sum(len(v) for v in out.values()), "spans")
