import collections, json, os, sys, unicodedata
sys.path.insert(0, os.path.expanduser("~/dvlp/modernbert-ja-ondevice-asr/asr"))
from eval_spans import marks
texts = {t["id"]: t["text"] for t in json.load(open("texts.json"))}
clean = [t for t in texts if t.startswith("clean~")]
ch = sum(len(texts[t]) for t in clean)
P, Dl = json.load(open("res-csoupdet-r0.json")), json.load(open("deltas-csoup.json"))
src = collections.Counter(); cls = collections.Counter()
for t in clean:
    cs = {(s, e) for s, e, _, a in Dl.get(t, []) if a and max(v for _, v in a) >= 9.5}
    for s, e in marks(P.get(t, []), Dl.get(t, []), 0.67, 9.5):
        src["cand" if (s, e) in cs else "det"] += 1
        w = texts[t][s:e]
        cls["punct-only" if all(unicodedata.category(c)[0] in "PZ" for c in w) else "latin/digit" if any(c.isascii() and c.isalnum() for c in w) else "kana/kanji"] += 1
print(dict(src), dict(cls))
# detector-only budget on the 60 texts: lowest det threshold with candidates off
for det in [0.67, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]:
    for sc in [9.5, 12.0, float("inf")]:
        fm = sum(len(marks(P.get(t, []), Dl.get(t, []), det, sc)) for t in clean)
        print(f"det {det} score {sc}: {1000 * fm / ch:.2f}", end="  ")
    print()
