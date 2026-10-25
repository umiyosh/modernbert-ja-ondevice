"""Synthetic homophone pairs for the on-device check's scorer (Kotobae #98): in correct sentences, a word is
swapped for an alternative the app itself would offer, and the pair (the right word against the swapped-in
one) goes to train_scorer.py in scorer_pairs.py's format. This is the confusion-set substitution that
Chinese spelling correction trains on; it reaches pairs that no transcript has shown.

- sentences: the reference text of Common Voice train clips (not quarantined), of Kokoro readings outside the
  held-out lines, and JWTD v2 train's corrected text. A sentence sharing a 15-character run with any
  evaluation text is left out, as in jwtd_pairs.py.
- alternatives, as the app offers them: the misconversion and voice tables' entries for the word, and IPADIC
  common nouns read the same. One whose UniDic lemma is the word's own (another spelling of it) is skipped.
- one pair per sentence, at most --per-group pairs per (wrong, right) group, --count pairs in all; the spoken
  sentences are used up first and JWTD fills the rest.

usage (from typo-study/asr): python synth_pairs.py <ipadic dir> <confusions.json> <voice.json> <jwtd train.jsonl>
                             <out.jsonl> --count N [--per-group 50] [--seed 0]
"""

import argparse
import collections
import csv
import glob
import json
import os
import random

import fugashi

from jwtd_pairs import TEXTS, runs, texts_of
from scorer_pairs import sentence

tagger = fugashi.Tagger()


def kanji(t: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in t)


def ipadic(d: str) -> tuple[dict, dict]:
    """reading -> common nouns read so, and common noun -> its readings, over entries with a kanji."""
    by, of = collections.defaultdict(set), collections.defaultdict(set)
    for f in glob.glob(os.path.join(d, "*.csv")):
        for row in csv.reader(open(f, encoding="utf-8")):
            if len(row) < 12 or row[4] != "名詞" or row[5] == "固有名詞" or not kanji(row[0]) or row[11] == "*":
                continue
            by[row[11]].add(row[0])
            of[row[0]].add(row[11])
    return by, of


def lemma(t: str) -> str:
    ws = list(tagger(t))
    return getattr(ws[0].feature, "lemma", None) or t if len(ws) == 1 else t


def voice_sentences(held: set[str]):
    for line in open("labels.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "train" and r["status"] != "quarantine":
            yield r["ref"]
    for f in sorted(glob.glob("synth-kokoro/asr-*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in held:
                yield r["ref"]


def jwtd_sentences(jwtd: str):
    for line in open(jwtd, encoding="utf-8"):
        yield json.loads(line)["post_text"]


def unique(texts) -> list[str]:
    return list(dict.fromkeys(t.strip() for t in texts if t and t.strip()))


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("ipadic", "confusions", "voice", "jwtd", "out"):
        ap.add_argument(k)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--per-group", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by, of = ipadic(a.ipadic)
    tables = [json.load(open(p, encoding="utf-8")) for p in (a.confusions, a.voice)]
    seen = set().union(*(runs(t) for p in TEXTS.split(",") for t in texts_of(p)))
    held = {json.loads(line)["path"] for line in open("eval-kokoro/labels.jsonl", encoding="utf-8")}
    # Spoken sentences first, written JWTD text after: JWTD alone would be nine in ten of the pool.
    voice, written = unique(voice_sentences(held)), unique(jwtd_sentences(a.jwtd))
    rng.shuffle(voice)
    rng.shuffle(written)
    pool = voice + written
    rows, groups, c = [], collections.Counter(), collections.Counter()
    for i, text in enumerate(pool):
        if i == len(voice):
            c["from voice"] = len(rows)
        if len(rows) >= a.count:
            break
        if runs(text) & seen:
            c["overlap"] += 1
            continue
        at, spans = 0, []
        for w in tagger(text):
            s = text.find(w.surface, at)
            if s < 0:
                break
            at = s + len(w.surface)
            alts = {x for t in tables for x in t.get(w.surface, {}).get("to", [])}
            alts |= {x for rd in of.get(w.surface, ()) for x in by.get(rd, ())}
            alts.discard(w.surface)
            if alts and kanji(w.surface):
                spans.append((s, at, w.surface, sorted(alts)))
        rng.shuffle(spans)
        for s, e, good, alts in spans:
            bad = rng.choice(alts)
            g = f"syn:{bad}>{good}"
            if groups[g] >= a.per_group or lemma(bad) == lemma(good):
                continue
            body = text[:s] + bad + text[e:]
            lo, hi = sentence(body, s, s + len(bad))
            rows.append({"text": body[lo:hi], "s": s - lo, "e": s + len(bad) - lo, "good": good, "bad": bad,
                         "kind": "fix", "group": g})
            groups[g] += 1
            break
        else:
            c["no swap"] += 1
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"sentences": len(pool), "written": len(rows), "groups": len(groups), "skipped": dict(c)}))


if __name__ == "__main__":
    main()
