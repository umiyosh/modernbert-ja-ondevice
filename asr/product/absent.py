"""Real-voice homophone spans whose right word no candidate row offers (B1, B2, A-absent), with shape counts."""
import collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import prodforce  # noqa: E402
prod = prodforce.prod
name = [x for x in sys.argv[1:] if not x.startswith("--")][0]
_, full = prod.load("eval-val", name)
KANA = re.compile("^[぀-ゟ]+$"); KATA = re.compile("^[゠-ヿ]+$")
shape, ex = collections.Counter(), []
for r in prod.VAL:
    if r["status"] == "clean":
        continue
    _, at = prod.eval_spans.norm_map(r["hyp"])
    for e in r["edits"]:
        x, y = e["hyp_chars"]
        if e["kind"] != "homophone" or y <= x:
            continue
        s, t = prod.eval_spans.raw_span(at, x, y)
        rows = [w for w in full.get(r["path"], []) if w[0] < t and s < w[1]]
        if any(alt == e["ref"] for w in rows for alt, _ in w[3]):
            continue
        exact = any(w[0] == s and w[1] == t for w in rows)
        h, f = e["hyp"], e["ref"]
        k = ("ref hiragana" if KANA.match(f) else "ref katakana" if KATA.match(f) else "hyp kana" if KANA.match(h) or KATA.match(h) else "kanji-kanji")
        k += ", same span row" if exact else (", other rows" if rows else ", no row")
        shape[k] += 1
        ex.append((h, f, [w[2] for w in rows][:2]))
for k, v in shape.most_common():
    print(f"{v:4d} {k}")
import random; random.seed(0)
for x in random.sample(ex, 40):
    print(x)
