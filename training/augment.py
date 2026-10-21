"""Synthesize misconversion examples from the corrected JWTD train sentences, so the detector sees
errors the edit history lacks: homophones outside the confusion table (精読→正読), voice-input
slips written in katakana (場当たり→バータリ), and hiragana words turned into a kanji noun with the
same reading (ます→鱒). Each correct sentence gets at most one error; its correct version is
already in build_data.py's rows. Readings come from UniDic (unidic-lite, used only to generate
the data).

Output: <out>.npz (flat token ids / labels with row offsets, the format train.py reads) and
<out>.jsonl (the text pairs, to read).

usage: python training/augment.py <jwtd train.jsonl> <pairs.json> <out prefix> [--exclude sentences.txt]
"""
import argparse
import json
import os
import random
import re
import resource
from array import array
from collections import Counter, defaultdict

import fugashi
import numpy as np

from build_data import encode, error_chars, excluded

KANJI = re.compile(r"[一-鿿々]")
HIRA = re.compile(r"^[ぁ-ゖ]+$")
CONTENT = {"名詞", "動詞", "形容詞", "形状詞", "副詞"}
# Shares of each error type when the sentence allows it. The table type mirrors the real edit
# history; the rest cover what it lacks. The one-kanji swap ("kanji") is left out: adding it at
# 0.2 lowered the recall in our evaluation from 54% to 46% and marked 次に, 訪ね, 朝 on correct text.
TRIES = 8
WEIGHTS = {"table": 0.4, "homophone": 0.35, "katakana": 0.125, "kanjiize": 0.125}
VOWEL = {c: v for v, cs in zip("アイウエオ", ["アカサタナハマヤラワガザダバパァャ", "イキシチニヒミリギジヂビピィ",
         "ウクスツヌフムユルグズヅブプゥュ", "エケセテネヘメレゲゼデベペェ", "オコソトノホモヨロヲゴゾドボポォョ"]) for c in cs}
LONG = {("ア", "ア"), ("イ", "イ"), ("ウ", "ウ"), ("エ", "エ"), ("オ", "オ"), ("オ", "ウ"), ("エ", "イ")}

tagger = fugashi.Tagger()


def kanji_set(s: str) -> set[str]:
    return set(KANJI.findall(s))


def spelling_variant(a: str, b: str) -> bool:
    # 子供/子ども, 現れ/現われ: same reading and no new kanji, so not a misconversion.
    ka, kb = kanji_set(a), kanji_set(b)
    return ka <= kb or kb <= ka


def lengthen(kana: str) -> str:
    # Voice input often writes the long vowel as ー (バアタリ→バータリ).
    out = []
    for c in kana:
        prev = out[-1] if out else ""
        out.append("ー" if (VOWEL.get(prev), c) in LONG and prev != "ー" else c)
    return "".join(out)


def to_kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) for c in s)


def build_vocab(sents: list[str]) -> Counter:
    vocab = Counter()
    for s in sents:
        for w in tagger(s):
            f = w.feature
            if f.kana and f.kana != "*" and KANJI.search(w.surface):
                vocab[(w.surface, f.kana, f.pos1, f.pos2)] += 1
    return vocab


def kanji_readings(vocab: Counter) -> dict[str, dict[str, int]]:
    # Readings per kanji, weighted by how many words use them. Kanji standing alone give the seed
    # (正/セイ); on-readings mostly live inside compounds, so a two-kanji word with one known
    # reading gives the other (精読/セイドク with 精/セイ → 読/ドク).
    def plain(s: str) -> bool:
        return all(KANJI.match(c) for c in s) and "々" not in s

    seed = defaultdict(Counter)
    pairs = []
    for (surf, kana, pos, pos2), n in vocab.items():
        if n < 5 or pos2 == "固有名詞" or not plain(surf):
            continue
        if len(surf) == 1:
            seed[surf][kana] += 1
        elif len(surf) == 2:
            pairs.append((surf, kana))
    known = {c: set(rs) for c, rs in seed.items()}
    for _ in range(3):
        found = defaultdict(Counter)
        for surf, kana in pairs:
            for r in known.get(surf[0], ()):
                if kana.startswith(r) and 0 < len(kana) - len(r) <= 4:
                    found[surf[1]][kana[len(r):]] += 1
            for r in known.get(surf[1], ()):
                if kana.endswith(r) and 0 < len(kana) - len(r) <= 4:
                    found[surf[0]][kana[:-len(r)]] += 1
        for c, rs in found.items():
            known.setdefault(c, set()).update(r for r, n in rs.items() if n >= 3)
    # Final weights: how many two-kanji words each reading explains, plus the seed.
    out = defaultdict(dict)
    for surf, kana in pairs:
        for i, c in enumerate(surf):
            other = surf[1 - i]
            for r in known.get(c, ()):
                rest = kana[len(r):] if i == 0 and kana.startswith(r) else kana[:-len(r)] if i == 1 and kana.endswith(r) else None
                if rest and rest in known.get(other, ()):
                    out[c][r] = out[c].get(r, 0) + 1
    for c, rs in seed.items():
        for r, n in rs.items():
            out[c][r] = out[c].get(r, 0) + n
    return out


def tables(vocab: Counter, pairs: dict[str, int]):
    homo = defaultdict(list)
    nouns = defaultdict(list)
    # Names are left out: a swapped name (千尋→千広) cannot be told from the context.
    for (surf, kana, pos, pos2), n in vocab.items():
        if n >= 10 and pos in CONTENT and pos2 != "固有名詞":
            homo[(kana, pos)].append((surf, n))
        if n >= 20 and pos2 == "普通名詞" and all(KANJI.match(c) for c in surf):
            nouns[kana].append((surf, n))
    readings = kanji_readings(vocab)
    by_reading = defaultdict(list)
    for c, rs in readings.items():
        # A slip lands on a kanji that is common for the reading, not on 魎 or 黌.
        for r, n in rs.items():
            if n >= 10:
                by_reading[r].append((c, n))
    rev = defaultdict(list)
    for key, n in pairs.items():
        wrong, right = key.split("\t")
        # One-off pairs are mostly noise (品→棺).
        if n >= 3 and KANJI.search(wrong) and KANJI.search(right):
            rev[right].append((wrong, n))
    return homo, nouns, rev, readings, by_reading


def pick(cands: list[tuple[str, int]], rng: random.Random, power: float = 0.5) -> str:
    return rng.choices([c for c, _ in cands], [n ** power for _, n in cands])[0]


def align(surf: str, kana: str, readings) -> list[str] | None:
    # Splits a compound's reading into one reading per kanji (精読/セイドク → セイ, ドク), or None.
    if not surf:
        return [] if not kana else None
    for r in readings.get(surf[0], {}):
        if kana.startswith(r):
            rest = align(surf[1:], kana[len(r):], readings)
            if rest is not None:
                return [r] + rest
    return None


def kanji_swaps(surf: str, kana: str, readings, by_reading) -> list[tuple[str, int]]:
    parts = align(surf, kana, readings)
    if not parts:
        return []
    return [(surf[:i] + k + surf[i + 1:], n) for i, r in enumerate(parts)
            for k, n in by_reading.get(r, []) if k != surf[i]]


def candidates(words, homo, nouns, rev, readings, by_reading):
    # Each candidate is (type, start, end, replacement builder); spans are character offsets.
    spans, pos = [], 0
    for w in words:
        pos += len(w.white_space)
        spans.append((pos, pos + len(w.surface)))
        pos += len(w.surface)
    out = defaultdict(list)
    for i, w in enumerate(words):
        f = w.feature
        for k in (1, 2, 3):
            if i + k <= len(words):
                surf = "".join(x.surface for x in words[i:i + k])
                if surf in rev:
                    out["table"].append((spans[i][0], spans[i + k - 1][1], rev[surf]))
        if f.pos1 in CONTENT and f.pos2 != "固有名詞" and KANJI.search(w.surface) and len(w.surface) >= 2:
            alts = [(s, n) for s, n in homo.get((f.kana, f.pos1), [])
                    if s != w.surface and not spelling_variant(s, w.surface)]
            if alts:
                out["homophone"].append((*spans[i], alts))
            if f.pos1 == "名詞" and f.kana and 2 <= len(w.surface) <= 4 and all(KANJI.match(c) for c in w.surface):
                swaps = kanji_swaps(w.surface, f.kana, readings, by_reading)
                if swaps:
                    out["kanji"].append((*spans[i], swaps))
            if f.kana and f.kana != "*":
                out["katakana"].append((*spans[i], f.kana))
        if HIRA.match(w.surface) and 2 <= len(w.surface) <= 3:
            # でき→出来, ため→為 are the same word in kanji, not a slip.
            alts = [(s, n) for s, n in nouns.get(to_kata(w.surface), []) if not kanji_set(s) & kanji_set(f.lemma)]
            if alts:
                out["kanjiize"].append((*spans[i], alts))
    return out


def draw_type(cands, rng: random.Random) -> str | None:
    types = [t for t in WEIGHTS if cands.get(t)]
    return rng.choices(types, [WEIGHTS[x] for x in types])[0] if types else None


def corrupt(sent: str, cands, t: str, rng: random.Random) -> str:
    a, b, arg = rng.choice(cands[t])
    if t == "katakana":
        rep = lengthen(arg) if rng.random() < 0.5 else arg
    else:
        rep = pick(arg, rng, 1.0 if t == "table" else 0.5)
    return sent[:a] + rep + sent[b:]


def labelled(sent: str, cands, t: str, rng: random.Random):
    # A draw can fail (no difference, or past the truncation); retry within the same type.
    for _ in range(TRIES):
        err = corrupt(sent, cands, t, rng)
        bad = error_chars(err, sent)
        if not bad or len(bad) > 12:
            continue
        ids, lab = encode(err, bad)
        if 1 not in lab:
            continue  # the error fell past the truncation
        return err, ids, lab
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--exclude")
    args = ap.parse_args()
    src, pairs_path, out = args.train, args.pairs, args.out
    # An evaluation sentence must not be trained on, even as the correct side of a synthetic pair.
    skip = excluded(args.exclude)
    sents = [s for s in dict.fromkeys(json.loads(l)["post_text"] for l in open(src)) if s not in skip]
    vocab_path = os.path.join(os.path.dirname(out) or ".", "vocab.tsv")
    if os.path.exists(vocab_path):
        vocab = Counter({tuple(r[:4]): int(r[4]) for r in (l.rstrip("\n").split("\t") for l in open(vocab_path))})
    else:
        vocab = build_vocab(sents)
        with open(vocab_path, "w") as fh:
            fh.writelines(f"{s}\t{k}\t{p}\t{p2}\t{n}\n" for (s, k, p, p2), n in vocab.items())
    tabs = tables(vocab, json.load(open(pairs_path)))
    homo, nouns, rev = tabs[:3]
    del vocab
    print(f"{len(sents)} sentences, homophone keys {len(homo)}, noun readings {len(nouns)}, table rights {len(rev)}", flush=True)

    rng = random.Random(20261014)
    order = list(range(len(sents)))
    rng.shuffle(order)
    ids_all, lab_all, offsets = array("i"), array("b"), array("q", [0])
    kinds, missed = Counter(), Counter()
    with open(out + ".jsonl", "w") as fh:
        for n, i in enumerate(order):
            sent = sents[i]
            cands = candidates(list(tagger(sent)), *tabs)
            t = draw_type(cands, rng)
            got = labelled(sent, cands, t, rng) if t else None
            if not got:
                missed[t] += 1
                continue
            err, ids, lab = got
            ids_all.extend(ids); lab_all.extend(lab); offsets.append(len(ids_all))
            kinds[t] += 1
            fh.write(json.dumps({"type": t, "text": err, "fixed": sent}, ensure_ascii=False) + "\n")
            if n % 100000 == 0:
                print(f"{n}/{len(order)} rows {len(offsets) - 1} rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f}GB", flush=True)
    np.savez(out + ".npz", ids=np.frombuffer(ids_all, np.int32), labels=np.frombuffer(lab_all, np.int8),
             offsets=np.frombuffer(offsets, np.int64))
    lab = np.frombuffer(lab_all, np.int8)
    print(f"{len(offsets) - 1} rows {dict(kinds)}, no error placed {dict(missed)}, error tokens {(lab == 1).sum()}/{(lab >= 0).sum()}")
    print(f"peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f}GB")


if __name__ == "__main__":
    main()
