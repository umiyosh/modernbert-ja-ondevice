"""Classify how each Whisper transcript (hyp) differs from the Common Voice script (ref).

- broken: the transcript is broken (cut off, or Latin letters / long repeats that the script lacks). Not used
- same: equal once punctuation and spaces are removed
- notation: the same lemmas (三時 / 3時, ください / 下さい), or only kana written as kanji or back
  (かいさん / 解散). Not an error
- error: the lemmas differ. Each differing part gets a kind:
  - homophone: same reading, different word (追求→追及, 誤→5)
  - mishear: different reading
  - drop / extra: a script word is missing / a word not in the script was added

usage: python asr/classify.py <transcripts.jsonl>
"""

import collections
import difflib
import json
import re
import sys
import unicodedata

import fugashi

DIGITS = "〇一二三四五六七八九"
PUNCT = re.compile(r"[\s、。，．,!?！？「」『』（）()・…―\-ー〜~：:；;\"'“”‘’]")
TAGGER = fugashi.Tagger()
KANA = re.compile(r"[ぁ-ゖァ-ヺー]+")
LATIN = re.compile(r"[A-Za-z]")
REPEAT = re.compile(r"(.)\1{3,}")


def small(n: int) -> str:
    out = ""
    for v, u in ((1000, "千"), (100, "百"), (10, "十")):
        q, n = divmod(n, v)
        if q:
            out += ("" if q == 1 else DIGITS[q]) + u
    return out + (DIGITS[n] if n else "")


def kansuji(m: re.Match) -> str:
    n = int(m.group(0))
    if n == 0:
        return "〇"
    if n >= 10**16:
        # A run this long is a code or a made-up string of digits, read one by one.
        return "".join(DIGITS[int(d)] for d in m.group(0))
    out = ""
    for v, u in ((10**12, "兆"), (10**8, "億"), (10**4, "万")):
        q, n = divmod(n, v)
        if q:
            out += small(q) + u
    return out + small(n)


def norm(s: str) -> str:
    # Digits become kanji numerals because UniDic gives digits no lemma.
    s = unicodedata.normalize("NFKC", s).replace("%", "パーセント")
    s = re.sub(r"\d+", kansuji, s)
    # The long-vowel mark is part of a word; protect it from the punctuation pattern.
    s = s.replace("ー", "\0")
    s = PUNCT.sub("", s)
    return s.replace("\0", "ー")


def words(s: str) -> list[tuple[str, str, str]]:
    # (lemma, reading, surface); symbols and unknown words have no lemma and use the surface.
    return [(w.feature.lemma or w.surface, w.feature.kana or w.surface, w.surface) for w in TAGGER(s)]


def broken(r: str, h: str) -> bool:
    # Whisper listens in 30-second windows and can drop the rest of a long clip; Latin letters and
    # repeats are what it makes up when it cannot hear.
    return (
        not 0.6 <= len(h) / max(len(r), 1) <= 1.5
        or (LATIN.search(h) is not None and LATIN.search(r) is None)
        or (REPEAT.search(h) is not None and REPEAT.search(r) is None)
    )


def diff(ref: str, hyp: str) -> tuple[str, list[dict]]:
    r, h = norm(ref), norm(hyp)
    if r == h:
        return "same", []
    if broken(r, h):
        return "broken", []
    rw, hw = words(r), words(h)
    if [w[0] for w in rw] == [w[0] for w in hw]:
        return "notation", []
    edits = []
    sm = difflib.SequenceMatcher(a=[w[0] for w in rw], b=[w[0] for w in hw], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        a, b = rw[i1:i2], hw[j1:j2]
        ka, kb = "".join(w[1] for w in a), "".join(w[1] for w in b)
        sa, sb = "".join(w[2] for w in a), "".join(w[2] for w in b)
        if op == "replace" and ka == kb and (KANA.fullmatch(sa) or KANA.fullmatch(sb)):
            continue
        kind = {"delete": "drop", "insert": "extra"}.get(op) or ("homophone" if ka == kb else "mishear")
        # at: the script words replaced; ref_kana / hyp_kana: readings, to compare what two recognizers heard
        edits.append({"kind": kind, "ref": sa, "hyp": sb, "at": [i1, i2], "ref_kana": ka, "hyp_kana": kb})
    return ("error", edits) if edits else ("notation", [])


def main() -> None:
    recs = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    verdicts, kinds, examples = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for rec in recs:
        v, edits = diff(rec["ref"], rec["hyp"])
        verdicts[v] += 1
        if v in ("notation", "broken") and len(examples[v]) < 15:
            examples[v].append(f"{rec['ref'][:60]} → {rec['hyp'][:60]}")
        for e in edits:
            kinds[e["kind"]] += 1
            if len(examples[e["kind"]]) < 15:
                examples[e["kind"]].append(f"{e['ref']}→{e['hyp']}　（{rec['ref']}）")
    n = len(recs)
    print(f"clips {n}")
    for v, c in verdicts.most_common():
        print(f"  {v}: {c} ({100 * c / n:.1f}%)")
    print("edits in error clips:", dict(kinds.most_common()))
    for k, xs in examples.items():
        print(f"\n## {k}")
        for x in xs:
            print("  " + x)


if __name__ == "__main__":
    main()
