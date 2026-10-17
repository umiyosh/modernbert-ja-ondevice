"""Split Common Voice Japanese into train / val / test so that no clip, speaker or normalized sentence
crosses splits.

Linking speakers and sentences puts almost every clip into one connected component (split_check.py),
so the split cannot follow components. The official dev / test are the cores of val / test, and clips
that share a speaker or a sentence with val or test are left out of train. Test clips whose transcripts
were looked at in the trial (trial-test.jsonl) are left out of test.

Writes to <out>/:
- manifest.tsv: path, split. client_id only checks the split and is not written
- asr_order.txt: transcription order, val then train (one reader per sentence first, then a second).
  Test is not in it; it is transcribed only for the final evaluation
- report.json: counts and overlap checks

usage: python asr/make_split.py <cv ja dir> <dir with trial-test.jsonl> <out dir>
"""

import collections
import csv
import hashlib
import json
import os
import random
import sys

from classify import norm

SEED = 0


def read(root: str, name: str) -> list[dict[str, str]]:
    with open(os.path.join(root, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def keys(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    return {
        "path": {r["path"] for r in rows},
        "speaker": {r["client_id"] for r in rows},
        "sentence": {norm(r["sentence"]) for r in rows},
    }


def main() -> None:
    root, asr, out = sys.argv[1], sys.argv[2], sys.argv[3]
    validated = read(root, "validated.tsv")
    with open(os.path.join(asr, "trial-test.jsonl"), encoding="utf-8") as f:
        seen = {json.loads(line)["path"] for line in f}

    test = [r for r in read(root, "test.tsv") if r["path"] not in seen]
    test_sen = keys(test)["sentence"]
    # The official dev and test share 52 sentences once normalized; drop them from val to keep test whole.
    val = [r for r in read(root, "dev.tsv") if norm(r["sentence"]) not in test_sen]
    held = {k: keys(val)[k] | keys(test)[k] for k in ("speaker", "sentence")}
    train = [r for r in validated if r["client_id"] not in held["speaker"] and norm(r["sentence"]) not in held["sentence"]]
    splits = {"train": train, "val": val, "test": test}

    k = {n: keys(rs) for n, rs in splits.items()}
    overlap = {
        f"{a}-{b}": {key: len(k[a][key] & k[b][key]) for key in ("path", "speaker", "sentence")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    assert all(v == 0 for o in overlap.values() for v in o.values()), overlap

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write("path\tsplit\n")
        for n, rs in splits.items():
            for r in sorted(rs, key=lambda r: r["path"]):
                f.write(f"{r['path']}\t{n}\n")

    rng = random.Random(SEED)
    shuffled = train[:]
    rng.shuffle(shuffled)
    reads = collections.Counter()
    ranked = []
    for i, r in enumerate(shuffled):
        ranked.append((reads[norm(r["sentence"])], i, r["path"]))
        reads[norm(r["sentence"])] += 1
    order = [r["path"] for r in val] + [p for *_, p in sorted(ranked)]
    with open(os.path.join(out, "asr_order.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(order) + "\n")

    with open(os.path.join(out, "manifest.tsv"), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    report = {
        "seed": SEED,
        "manifest_sha256": digest,
        "counts": {n: {key: len(v) for key, v in k[n].items()} for n in splits},
        "overlap": overlap,
        "dropped": {
            # validated clips in no split because they share a speaker or a sentence with val or test
            "validated_unassigned": len(validated) - len(train) - len(val) - len(test),
            "test_trial_seen": len(seen),
            "val_sentence_in_test": len(read(root, "dev.tsv")) - len(val),
        },
    }
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
