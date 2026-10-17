"""Count whether Common Voice Japanese can be split so that no speaker and no sentence crosses splits.

Exploration only; make_split.py writes the split. client_id is used only as a grouping key, never to
find out who a speaker is.

usage: python asr/split_check.py <cv ja dir> <dir with trial-test.jsonl and night1.jsonl>
"""

import collections
import csv
import json
import os
import sys

from classify import norm


def read(root: str, name: str) -> list[dict[str, str]]:
    with open(os.path.join(root, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def components(rows: list[dict[str, str]]) -> list[int]:
    # Connected components of the bipartite graph speakers–sentences with one edge per clip; sizes in clips.
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        a, b = find("s:" + r["client_id"]), find("t:" + norm(r["sentence"]))
        if a != b:
            parent[a] = b
    return sorted(collections.Counter(find("s:" + r["client_id"]) for r in rows).values(), reverse=True)


def main() -> None:
    root, data = sys.argv[1], sys.argv[2]
    v = read(root, "validated.tsv")
    split = {n: read(root, f"{n}.tsv") for n in ("train", "dev", "test")}
    spk = {n: {r["client_id"] for r in rs} for n, rs in split.items()}
    sen = {n: {norm(r["sentence"]) for r in rs} for n, rs in split.items()}
    out: dict = {"validated": {"clips": len(v), "speakers": len({r["client_id"] for r in v}), "sentences_norm": len({norm(r["sentence"]) for r in v})}}

    out["official_overlap"] = {
        f"{a}-{b}": {"speakers": len(spk[a] & spk[b]), "sentences_norm": len(sen[a] & sen[b])}
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    sizes = components(v)
    out["components"] = {"count": len(sizes), "largest_clips": sizes[0], "largest_share": round(sizes[0] / len(v), 3)}

    seen = {r["path"] for r in jsonl(os.path.join(data, "trial-test.jsonl"))}
    test = [r for r in split["test"] if r["path"] not in seen]
    dev = split["dev"]
    held_spk = spk["test"] | spk["dev"]
    held_sen = sen["test"] | sen["dev"]
    train = [r for r in v if r["client_id"] not in held_spk and norm(r["sentence"]) not in held_sen]
    bridging = [r for r in v if (r["client_id"] in held_spk) != (norm(r["sentence"]) in held_sen)]
    out["proposed"] = {
        n: {"clips": len(rs), "speakers": len({r["client_id"] for r in rs}), "sentences_norm": len({norm(r["sentence"]) for r in rs})}
        for n, rs in (("train", train), ("val", dev), ("test", test))
    }
    out["proposed"]["dropped_bridging_clips"] = len(bridging)
    out["proposed"]["test_dropped_trial_seen"] = len(split["test"]) - len(test)

    train_paths = {r["path"] for r in train}
    night = jsonl(os.path.join(data, "night1.jsonl"))
    out["night1"] = {"transcribed": len(night), "in_proposed_train": sum(r["path"] in train_paths for r in night)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
