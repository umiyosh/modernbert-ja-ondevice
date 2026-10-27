"""Remove sentences that nearly match a later split's sentence, and write split v1.

Two sentences are near duplicates when the Jaccard similarity of their character 3-grams, after
classify.norm, is at least 0.5; at that level the same frame with one word swapped matches
(木村さんに電話／本を貸してもらいました). Train loses sentences near val or test, and val loses
sentences near test, so test stays as fixed. Writes counts only.

usage: python asr/near_dup.py <cv ja dir> <split-v0 dir> <split-v1 dir> <report.json>
"""

import collections
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time

from classify import norm

NEAR = 0.5
LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)


def read(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def grams(s: str) -> frozenset[str]:
    return frozenset(s[i : i + 3] for i in range(max(len(s) - 2, 1)))


def near(a: list[str], b: list[str], t: float = NEAR) -> list[tuple[int, int, float]]:
    """Every pair (i, j, jaccard) with jaccard(a[i], b[j]) >= t.

    Prefix filtering: with 3-grams in one global order, two sets with Jaccard >= t share a 3-gram
    among the first |s| - ceil(t|s|) + 1 of each, so probing prefixes finds every such pair.
    """
    ga, gb = [grams(s) for s in a], [grams(s) for s in b]
    df = collections.Counter(x for g in itertools.chain(ga, gb) for x in g)

    def prefix(g: frozenset[str]) -> list[str]:
        return sorted(g, key=lambda x: (df[x], x))[: len(g) - math.ceil(t * len(g)) + 1]

    index = collections.defaultdict(list)
    for i, g in enumerate(ga):
        for x in prefix(g):
            index[x].append(i)
    out = []
    for j, g in enumerate(gb):
        for i in {i for x in prefix(g) for i in index.get(x, ())}:
            jac = len(g & ga[i]) / len(g | ga[i])
            if jac >= t:
                out.append((i, j, jac))
    return out


def main() -> None:
    root, v0, v1, out = sys.argv[1:5]
    split = {r["path"]: r["split"] for r in read(os.path.join(v0, "manifest.tsv"))}
    sent = {r["path"]: norm(r["sentence"]) for r in read(os.path.join(root, "validated.tsv")) if r["path"] in split}
    by = {s: sorted({sent[p] for p in split if split[p] == s}) for s in ("train", "val", "test")}

    found = {}
    drop = {"train": set(), "val": set()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        pairs = near(by[a], by[b])
        found[f"{a}-{b}"] = {str(t): len({j for _, j, x in pairs if x >= t}) for t in LEVELS}
        drop[a] |= {by[a][i] for i, _, _ in pairs}

    def kept(p: str) -> bool:
        return split[p] == "test" or sent[p] not in drop[split[p]]

    os.makedirs(v1, exist_ok=True)
    manifest = os.path.join(v1, "manifest.tsv")
    if os.path.exists(manifest):
        os.chmod(manifest, 0o600)  # leak_check.py seals it
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("path\tsplit\n" + "".join(f"{p}\t{split[p]}\n" for p in sorted(split) if kept(p)))
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": f"Jaccard of character 3-grams after classify.norm >= {NEAR}, all pairs by prefix filtering",
        "later_sentences_with_a_near_earlier_sentence": found,
        "sentences_removed": {s: len(drop[s]) for s in drop},
        "clips_removed": {s: sum(split[p] == s and not kept(p) for p in split) for s in drop},
        "v1_clips": dict(collections.Counter(split[p] for p in split if kept(p))),
        "v1_manifest_sha256": hashlib.sha256(open(manifest, "rb").read()).hexdigest(),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
