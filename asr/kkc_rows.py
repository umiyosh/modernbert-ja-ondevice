"""Token-classification rows from the IME-simulated misconversions (synth_kkc.py) for the detector's
pretraining (Kotobae #98, CSCD-NS recipe, arXiv 2211.08788), in build_rows.py's npz format.

Each kept swap gives its text with the swapped word labelled 1, and its sentence as written gives a row
labelled all 0 (once per sentence): the detector sees as many correct sentences as swapped ones, so it
learns which words are wrong rather than that a sentence holds one.

Leak control: a swap whose text or sentence shares a 15-character run with any clip outside train
(labels.jsonl: reference and transcript) is dropped, so no evaluation sentence reaches the detector.

usage (from typo-study/asr): python kkc_rows.py <pretrain.jsonl> <labels.jsonl> <out prefix> [--limit N] [--seed 0]
                                                [--clean-share 1.0]

--clean-share keeps each swapped sentence's original with that probability: mixed with real voice rows,
which carry their own correct clips, the plan halves them (0.5).
"""

import argparse
import collections
import json
import random
import unicodedata

from build_rows import encode, save

RUN = 15


def shingles(s: str) -> set[str]:
    s = unicodedata.normalize("NFKC", s)
    return {s[i:i + RUN] for i in range(len(s) - RUN + 1)}


def held(labels: str) -> set[str]:
    out = set()
    with open(labels, encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r["split"] != "train":
                out |= shingles(r["ref"]) | shingles(r["hyp"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-share", type=float, default=1.0)
    a = ap.parse_args()
    keep = random.Random(a.seed + 2)
    hold = held(a.labels)
    with open(a.data, encoding="utf-8") as f:
        swaps = [json.loads(line) for line in f]
    random.Random(a.seed).shuffle(swaps)
    n = collections.Counter()
    rows, seen = [], set()
    for x in swaps:
        if a.limit and n["swap"] >= a.limit:
            break
        t, s, e = x["text"], x["s"], x["e"]
        orig = t[:s] + x["orig"] + t[e:]
        if shingles(t) & hold or shingles(orig) & hold:
            n["held_out"] += 1
            continue
        ids, lab = encode(t, set(range(s, e)))
        if 1 not in lab:
            n["past_truncation"] += 1
            continue
        rows.append((ids, lab))
        n["swap"] += 1
        if orig not in seen:
            seen.add(orig)
            if keep.random() < a.clean_share:
                rows.append(encode(orig, set()))
                n["clean"] += 1
    random.Random(a.seed + 1).shuffle(rows)
    save(rows, a.out + ".npz")
    n.update(rows=len(rows))
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(n, f, indent=1)
        f.write("\n")
    print(json.dumps(n))


if __name__ == "__main__":
    main()
