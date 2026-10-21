"""select_full with probes on punctuation-only spans dropped (val exploration)."""
import json, os, sys, unicodedata
sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
import select_full
texts = {t["id"]: t["text"] for t in json.load(open("eval-val/texts.json", encoding="utf-8"))}
punct = lambda s: s and all(unicodedata.category(c)[0] in "PZ" for c in s)
orig = json.load
def load(f, *a, **k):
    d = orig(f, *a, **k)
    if getattr(f, "name", "").split("/")[-1].startswith("res-"):
        d = {p: [x for x in v if not punct(texts[p][x[0]:x[1]])] for p, v in d.items()}
    return d
select_full.json.load = load
select_full.main()
