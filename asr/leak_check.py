"""Check a split for leaks between train, val and test, then seal test away from training code.

Checks, each written as counts to the report:
- sentences shared across splits: raw, after classify.norm, and near (near_dup.near, all pairs at
  character 3-gram Jaccard >= 0.5)
- speakers (client_id, used only as a grouping key) and audio file hashes shared across splits
- test clips among the transcripts
- val / test sentences found in JWTD v2 train, the data the current detector was trained on
  (equal after norm, or contained in a JWTD sentence when at least 8 characters long)

Then writes <split dir>/trainval.tsv for training, jwtd_overlap_val.txt (val clips whose sentence is in
JWTD, kept out of the primary metric), and test.tsv with a jwtd column. test.tsv and manifest.tsv are
left with no read permission, so training code cannot open a test list by mistake; `chmod 600` them
for the final evaluation.

usage: python asr/leak_check.py <cv ja dir> <split dir> <transcripts.jsonl> <jwtd train.jsonl> <report.json>
"""

import collections
import hashlib
import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from classify import norm
from near_dup import NEAR, near, read

MIN_CONTAINED = 8


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def in_jwtd(jwtd: str, held: set[str]) -> tuple[set[str], set[str]]:
    # Sentences of 8+ characters are looked up by their first 8 characters at every position of JWTD text.
    heads = collections.defaultdict(set)
    for s in held:
        if len(s) >= MIN_CONTAINED:
            heads[s[:MIN_CONTAINED]].add(s)
    equal, contained = set(), set()
    with open(jwtd, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for text in (norm(r["pre_text"]), norm(r["post_text"])):
                if text in held:
                    equal.add(text)
                for i in range(len(text) - MIN_CONTAINED + 1):
                    for s in heads.get(text[i : i + MIN_CONTAINED], ()):
                        if text.startswith(s, i):
                            contained.add(s)
    return equal, contained


def main() -> None:
    root, split_dir, transcripts, jwtd, out = sys.argv[1:6]
    manifest = os.path.join(split_dir, "manifest.tsv")
    os.chmod(manifest, 0o600)
    split = {r["path"]: r["split"] for r in read(manifest)}
    rows = {r["path"]: r for r in read(os.path.join(root, "validated.tsv")) if r["path"] in split}
    names = ("train", "val", "test")
    pairs = list(itertools.combinations(names, 2))

    raw = {s: {rows[p]["sentence"] for p in split if split[p] == s} for s in names}
    normed = {s: {norm(x) for x in raw[s]} for s in names}
    speakers = {s: {rows[p]["client_id"] for p in split if split[p] == s} for s in names}
    paths = sorted(split)
    with ThreadPoolExecutor(8) as ex:
        hashes = dict(zip(paths, ex.map(lambda p: sha(os.path.join(root, "clips", p)), paths)))
    audio = {s: {hashes[p] for p in paths if split[p] == s} for s in names}

    with open(transcripts, encoding="utf-8") as f:
        done = [json.loads(line)["path"] for line in f]

    equal, contained = in_jwtd(jwtd, normed["val"] | normed["test"])

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest_sha256": hashlib.sha256(open(manifest, "rb").read()).hexdigest(),
        "clips": dict(collections.Counter(split.values())),
        "across_splits": {
            f"{a}-{b}": {
                "raw_sentences": len(raw[a] & raw[b]),
                "normalized_sentences": len(normed[a] & normed[b]),
                f"near_sentences_jaccard_{NEAR}": len({j for _, j, _ in near(sorted(normed[a]), sorted(normed[b]))}),
                "speakers": len(speakers[a] & speakers[b]),
                "audio_hashes": len(audio[a] & audio[b]),
            }
            for a, b in pairs
        },
        "transcripts": {"clips": len(done), "test_clips": sum(split.get(p) == "test" for p in done)},
        "jwtd_train": {
            s: {
                "sentences": len(normed[s]),
                "equal": len(normed[s] & equal),
                f"contained_{MIN_CONTAINED}_chars_or_more": len(normed[s] & contained),
            }
            for s in ("val", "test")
        },
    }

    in_j = equal | contained
    trainval = os.path.join(split_dir, "trainval.tsv")
    test = os.path.join(split_dir, "test.tsv")
    with open(trainval, "w", encoding="utf-8") as f:
        f.write("path\tsplit\n" + "".join(f"{p}\t{split[p]}\n" for p in paths if split[p] != "test"))
    with open(os.path.join(split_dir, "jwtd_overlap_val.txt"), "w", encoding="utf-8") as f:
        f.write("".join(p + "\n" for p in paths if split[p] == "val" and norm(rows[p]["sentence"]) in in_j))
    if os.path.exists(test):
        os.chmod(test, 0o600)
    with open(test, "w", encoding="utf-8") as f:
        f.write("path\tsplit\tjwtd\n")
        f.write("".join(f"{p}\ttest\t{int(norm(rows[p]['sentence']) in in_j)}\n" for p in paths if split[p] == "test"))
    os.chmod(test, 0)
    os.chmod(manifest, 0)
    report["sealed"] = {"trainval_rows": sum(split[p] != "test" for p in paths), "test_rows": report["clips"]["test"]}

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
