"""Check the clips of a split for duplicate audio, bad lengths, down votes and reported sentences.

Writes counts only (no path, sentence or speaker), so the report can be committed.

usage: python asr/quality_check.py <cv ja dir> <manifest.tsv> <report.json>
"""

import collections
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SHORT_MS = 1000  # shorter than this rarely holds a whole sentence
LONG_MS = 30000  # Whisper hears 30 seconds at a time


def read(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    root, manifest, out = sys.argv[1:4]
    split = {r["path"]: r["split"] for r in read(manifest)}
    rows = {r["path"]: r for r in read(os.path.join(root, "validated.tsv")) if r["path"] in split}
    ms = {r["clip"]: int(r["duration[ms]"]) for r in read(os.path.join(root, "clip_durations.tsv"))}
    reported = {r["sentence_id"] for r in read(os.path.join(root, "reported.tsv")) if r.get("sentence_id")}

    paths = sorted(split)
    with ThreadPoolExecutor(8) as ex:
        hashes = dict(zip(paths, ex.map(lambda p: sha(os.path.join(root, "clips", p)), paths)))

    by_hash = collections.defaultdict(list)
    for p, h in hashes.items():
        by_hash[h].append(split[p])
    dup_within = collections.Counter()
    dup_across = collections.Counter()
    for splits in by_hash.values():
        if len(splits) < 2:
            continue
        kinds = set(splits)
        if len(kinds) > 1:
            for s in splits:
                dup_across[s] += 1
        else:
            dup_within[splits[0]] += len(splits)

    stats = {}
    for s in ("train", "val", "test"):
        ps = [p for p in paths if split[p] == s]
        d = sorted(ms.get(p, 0) for p in ps)
        stats[s] = {
            "clips": len(ps),
            "no_duration": sum(p not in ms for p in ps),
            "empty": sum(ms.get(p, 1) == 0 for p in ps),
            "shorter_than_1s": sum(0 < ms.get(p, SHORT_MS) < SHORT_MS for p in ps),
            "longer_than_30s": sum(ms.get(p, 0) > LONG_MS for p in ps),
            "duration_ms_p50_p95_max": [d[len(d) // 2], d[int(len(d) * 0.95)], d[-1]] if d else None,
            "down_votes_gt_0": sum(int(rows[p]["down_votes"] or 0) > 0 for p in ps),
            "up_votes_lt_2": sum(int(rows[p]["up_votes"] or 0) < 2 for p in ps),
            "reported_sentence": sum(rows[p]["sentence_id"] in reported for p in ps),
            "audio_duplicates_within_split": dup_within[s],
            "audio_duplicates_across_splits": dup_across[s],
        }
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest_sha256": hashlib.sha256(open(manifest, "rb").read()).hexdigest(),
        "hashed_clips": len(hashes),
        "splits": stats,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
