"""Detector-only: lowest threshold within 1 false mark per 1,000 chars on val clean clips and on the
60 Kotobae texts; val span recall there."""
import collections, functools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
import eval_spans
eval_spans.norm_map = functools.lru_cache(maxsize=None)(eval_spans.norm_map)
from eval_spans import clip_counts, marks, summary
recs = [r for r in map(json.loads, open("labels.jsonl")) if r["split"] == "val" and r["status"] != "quarantine" and not r["jwtd"]]
texts = {t["id"]: t["text"] for t in json.load(open("../texts.json"))}
clean = [t for t in texts if t.startswith("clean~")]; chars = sum(len(texts[t]) for t in clean)
for pair in sys.argv[1:]:
    vn, kn = pair.split(":")
    V = json.load(open(f"eval-val/res-{vn}det-r0.json")); K = json.load(open(f"../res-{kn}det-r0.json"))
    for t in [round(x, 3) for x in np.arange(0.05, 0.995, 0.005)]:
        kf = 1000 * sum(len(marks(K.get(i, []), [], t, float("inf"))) for i in clean) / chars
        if kf > 1: continue
        c = collections.Counter()
        for r in recs: c.update(clip_counts(r, marks(V[r["path"]], [], t, float("inf"))))
        m = summary(c)
        if m["false_marks_per_1k_chars"] <= 1:
            print(vn, f"t={t} recall {m['span_recall']:.3f} hom {m['homophone_recall']:.3f} valfm {m['false_marks_per_1k_chars']:.2f} kotobae {kf:.2f}"); break
