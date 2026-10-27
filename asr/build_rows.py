"""Token-classification rows from the labelled train clips, in the npz format of
modernbert-ja-ondevice's training/train.py (flat ids / labels with row offsets).

Each clean or error clip of split train gives its Whisper transcript as the text, the string Kotobae's
detector reads. Characters of homophone, mishear and extra edits are labelled 1; a drop labels the
character before the gap, as JWTD rows label a missing character. Quarantined clips are left out.
Train repeats sentences (about seven readers each), so rows with the same tokens and labels are kept
once: a clean reading the recognizer got right adds nothing the first copy did not.

With a second recognizer's transcripts (second_asr.py), its transcript of each such clip gives a row
too, labelled from its own diff with the script: real errors of another family of recognizer, as
Kotobae reads AquaVoice rather than Whisper. A clip quarantined for either transcript is left out, and
a second transcript that classify.diff calls broken is skipped.

Only paths that trainval.tsv puts in train are read, and the builder asserts it, so no row can come
from val or test (PROTOCOL.md, leak controls). Writes <out>.npz and <out>.json (counts only).

usage: python asr/build_rows.py <split dir with trainval.tsv> <labels.jsonl> <out prefix> [second.jsonl]
"""

import collections
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

from classify import diff, norm_map
from near_dup import read

L = 64  # the training length of the published detector (training/build_data.py)
SPAN_KINDS = {"homophone", "mishear", "extra"}
tok = AutoTokenizer.from_pretrained("sbintuitions/modernbert-ja-130m")


def bad_chars(rec: dict) -> set[int] | None:
    m = norm_map(rec["hyp"])
    if m is None:
        return None
    _, at = m
    bad = set()
    for e in rec["edits"]:
        a, b = e["hyp_chars"]
        if e["kind"] in SPAN_KINDS and b > a:
            bad.update(range(at[a], at[b - 1] + 1))
        elif e["kind"] == "drop" and at:
            bad.add(at[max(a - 1, 0)] if a <= len(at) else at[-1])
    # A numeral of norm maps back to the last digit of its run (100 → 百 → the last 0), so an error
    # in a number marks the whole run of digits.
    s = rec["hyp"]
    for i in sorted(bad):
        if s[i].isdigit():
            j = i
            while j > 0 and s[j - 1].isdigit():
                j -= 1
            k = i
            while k + 1 < len(s) and s[k + 1].isdigit():
                k += 1
            bad.update(range(j, k + 1))
    return bad


def encode(text: str, bad: set[int]) -> tuple[list[int], list[int]]:
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=L)
    labels = [-100 if b == a else int(any(c in bad for c in range(a, b))) for a, b in enc["offset_mapping"]]
    return enc["input_ids"], labels


def save(rows: list[tuple[list[int], list[int]]], path: str) -> None:
    lens = np.fromiter((len(ids) for ids, _ in rows), np.int64, len(rows))
    offsets = np.concatenate([[0], np.cumsum(lens)])
    ids = np.fromiter((x for r, _ in rows for x in r), np.int32, offsets[-1])
    labels = np.fromiter((x for _, lab in rows for x in lab), np.int8, offsets[-1])
    np.savez(path, ids=ids, labels=labels, offsets=offsets)


def second_rec(rec: dict, hyp2: str) -> dict | None:
    verdict, edits = diff(rec["ref"], hyp2)
    if verdict == "broken":
        return None
    return {"hyp": hyp2, "status": "error" if verdict == "error" else "clean", "edits": edits}


def main() -> None:
    split_dir, labels, out = sys.argv[1:4]
    second = {}
    if len(sys.argv) > 4:
        with open(sys.argv[4], encoding="utf-8") as f:
            second = {r["path"]: r["hyp2"] for r in map(json.loads, f)}
    train = {r["path"] for r in read(os.path.join(split_dir, "trainval.tsv")) if r["split"] == "train"}
    seen, rows = set(), []
    n = collections.Counter()
    with open(labels, encoding="utf-8") as f:
        for rec in map(json.loads, f):
            if rec["split"] != "train" or rec["status"] == "quarantine":
                continue
            assert rec["path"] in train, "a row would come from outside train"
            n["clips"] += 1
            versions = [("", rec)]
            if rec["path"] in second:
                r2 = second_rec(rec, second[rec["path"]])
                if r2 is None:
                    n["second_broken"] += 1
                else:
                    versions.append(("second_", r2))
            for tag, v in versions:
                bad = bad_chars(v) if v["status"] == "error" else set()
                if bad is None:
                    n[tag + "no_char_map"] += 1
                    continue
                ids, lab = encode(v["hyp"], bad)
                if v["status"] == "error" and 1 not in lab:
                    n[tag + "error_past_truncation"] += 1
                    continue
                key = (tuple(ids), tuple(lab))
                if key in seen:
                    n[tag + "duplicates"] += 1
                    continue
                seen.add(key)
                rows.append((ids, lab))
                n[tag + v["status"]] += 1
    save(rows, out + ".npz")
    pos = sum(lab.count(1) for _, lab in rows)
    tot = sum(sum(x >= 0 for x in lab) for _, lab in rows)
    n.update(rows=len(rows), error_tokens=pos, tokens=tot)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(n, f, indent=1)
        f.write("\n")
    print(json.dumps(n))


if __name__ == "__main__":
    main()
