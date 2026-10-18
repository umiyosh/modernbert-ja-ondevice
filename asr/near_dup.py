"""Find train sentences that nearly match a val or test sentence, and write split v1 without them.

Two sentences are near duplicates when the Jaccard similarity of their character 3-grams, after
classify.norm, reaches the threshold. Only train loses clips, so val and test stay as fixed.
Writes counts only.

usage: python asr/near_dup.py <cv ja dir> <split-v0 dir> <split-v1 dir> <report.json> [threshold]
"""

import collections
import csv
import hashlib
import json
import os
import sys
import time

from classify import norm

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
COMMON = 3000  # 3-grams in more train sentences than this are skipped when finding candidates


def grams(s: str) -> frozenset[str]:
    return frozenset(s[i : i + 3] for i in range(max(len(s) - 2, 1)))


def read(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def main() -> None:
    root, v0, v1, out = sys.argv[1:5]
    cut = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
    split = {r["path"]: r["split"] for r in read(os.path.join(v0, "manifest.tsv"))}
    sent = {r["path"]: norm(r["sentence"]) for r in read(os.path.join(root, "validated.tsv")) if r["path"] in split}

    by_split = collections.defaultdict(set)
    for p, s in split.items():
        by_split[s].add(sent[p])
    train = sorted(by_split["train"])
    tg = [grams(s) for s in train]
    index = collections.defaultdict(list)
    for i, g in enumerate(tg):
        for x in g:
            index[x].append(i)

    # The train sentences each held-out sentence nearly matches, by threshold.
    near = {t: set() for t in THRESHOLDS + (cut,)}
    held = {"val": collections.Counter(), "test": collections.Counter()}
    for s in ("val", "test"):
        for h in by_split[s]:
            g = grams(h)
            shared = collections.Counter()
            for x in g:
                ids = index.get(x)
                if ids is not None and len(ids) <= COMMON:
                    shared.update(ids)
            for i, n in shared.most_common(50):
                j = n / len(g | tg[i])
                for t in near:
                    if j >= t:
                        near[t].add(train[i])
            top = max((n / len(g | tg[i]) for i, n in shared.most_common(50)), default=0.0)
            for t in THRESHOLDS:
                if top >= t:
                    held[s][t] += 1

    drop = near[cut]
    os.makedirs(v1, exist_ok=True)
    with open(os.path.join(v1, "manifest.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "split"])
        for p in sorted(split):
            if not (split[p] == "train" and sent[p] in drop):
                w.writerow([p, split[p]])
    kept = collections.Counter(s for p, s in split.items() if not (s == "train" and sent[p] in drop))
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "Jaccard of character 3-grams after classify.norm; top 50 candidates by shared 3-grams",
        "threshold": cut,
        "held_out_sentences_with_a_near_train_sentence": {s: {str(t): held[s][t] for t in THRESHOLDS} for s in held},
        "train_sentences_near_val_or_test": {str(t): len(near[t]) for t in THRESHOLDS},
        "train_clips_removed": sum(1 for p, s in split.items() if s == "train" and sent[p] in drop),
        "v1_clips": dict(kept),
        "v1_manifest_sha256": hashlib.sha256(open(os.path.join(v1, "manifest.tsv"), "rb").read()).hexdigest(),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
