"""How much of the missing right words a wider reading lookup would offer (Kotobae #98, paper plan step 4).
The app's rows take IPADIC words read the same and of the same part of speech, on the segmenter's own
words. For the dev half's homophones with the right word in no row (nocand_cases.py): is the right word in
IPADIC at all, read the same as the written span (its IPADIC readings, or UniDic's of the span), and of
which part of speech; then the same with the lookup kept to nouns, as the app has it. As the cost, the
alternatives per 1,000 characters of the dev half's correct clips under each lookup, on UniDic's words.

usage (from asr/): python tools/cand_dict.py <IPADIC source dir> [name sb0s] [line 0.1] [w 1] [det 0.96] [score 9.5]
"""
import collections
import csv
import glob
import json
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import fugashi  # noqa: E402
import jaconv  # noqa: E402

import nocand_cases  # noqa: E402
import prod  # noqa: E402

tagger = fugashi.Tagger()


def kanji(t):
    return any("一" <= ch <= "鿿" for ch in t)


def ipadic(d):
    """reading -> {(surface, part of speech)}, and surface -> readings, over entries with a kanji."""
    by, of = collections.defaultdict(set), collections.defaultdict(set)
    for f in glob.glob(os.path.join(d, "*.csv")):
        for row in csv.reader(open(f, encoding="utf-8")):
            if len(row) < 12 or not kanji(row[0]) or row[11] == "*":
                continue
            pos = row[4] + ("・固有" if row[5] == "固有名詞" else "")
            by[row[11]].add((row[0], pos))
            of[row[0]].add(row[11])
    return by, of


def unidic_reading(t):
    return "".join((getattr(w.feature, "kana", None) or w.surface) for w in tagger(t))


def readings(t, of):
    return of.get(t, set()) | {jaconv.hira2kata(unidic_reading(t))}


def main():
    d = ARGS[0]
    name = ARGS[1] if len(ARGS) > 1 else "sb0s"
    line, w, det, score = (float(x) for x in (ARGS[2:6] + ["0.1", "1", "0.96", "9.5"][len(ARGS[2:6]):]))
    by, of = ipadic(d)
    c = collections.Counter()
    pos_hit = collections.Counter()
    for _, text, a, b, e in nocand_cases.cases(name, line, w, det, score):
        ref = e["ref"]
        c["cases"] += 1
        c["right word in IPADIC"] += ref in of
        offered = {(sf, p) for rd in readings(text[a:b], of) for sf, p in by.get(rd, ())}
        hit = [p for sf, p in offered if sf == ref]
        c["offered, any part of speech"] += bool(hit)
        c["offered, common nouns only"] += "名詞" in hit
        for p in set(hit):
            pos_hit[p] += 1
    print(f"{name}: missed homophones with no candidate {c['cases']}")
    for k in ("right word in IPADIC", "offered, any part of speech", "offered, common nouns only"):
        print(f"  {k}: {c[k]}")
    print("  right word's part of speech when offered: " + ", ".join(f"{p} {n}" for p, n in pos_hit.most_common()))
    half = json.load(open("split-calib.json"))
    cost, chars = collections.Counter(), 0
    for r in prod.VAL:
        if half[r["path"]] != "dev" or r["status"] != "clean":
            continue
        chars += len(r["hyp"])
        for wd in tagger(r["hyp"]):
            if not kanji(wd.surface):
                continue
            alts = {(sf, p) for rd in readings(wd.surface, of) for sf, p in by.get(rd, ()) if sf != wd.surface}
            cost["any"] += len({sf for sf, _ in alts})
            cost["nouns"] += len({sf for sf, p in alts if p == "名詞"})
    print(f"  alternatives per 1,000 chars of correct dev clips (UniDic words): nouns only "
          f"{1000 * cost['nouns'] / chars:.0f}, any part of speech {1000 * cost['any'] / chars:.0f}")


if __name__ == "__main__":
    main()
