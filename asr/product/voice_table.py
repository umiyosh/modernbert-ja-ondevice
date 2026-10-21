"""The voice-input confusion table: for each word Whisper wrote in place of a homophone, what the speaker
said, from the Common Voice train split and the Kokoro readings outside the held-out lines. Written in the
misconversion table's format ({word: {wrong, kept, to}}) so the Rust check reads it the same way, and with
the pair counts the noisy-channel bonus needs. Run from asr/.

usage: python voice_table.py <out dir> [kokoro asr jsonl ...]
"""

import collections
import glob
import json
import os
import re
import sys


def pairs_and_texts(kokoro_files):
    held = {json.loads(l)["path"] for l in open("eval-kokoro/labels.jsonl", encoding="utf-8")}
    pairs, texts = collections.Counter(), []
    for l in open("labels.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["split"] != "train" or r["jwtd"] or r["status"] == "quarantine":
            continue
        texts.append(r["hyp"])
        pairs.update((e["hyp"], e["ref"]) for e in r["edits"] or [] if e["kind"] == "homophone")
    for f in kokoro_files:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in held:
                continue
            texts.append(r["hyp"])
            pairs.update((e["hyp"], e["ref"]) for e in r["edits"] or [] if e.get("kind") == "homophone")
    return pairs, texts


def main():
    out = sys.argv[1]
    files = sys.argv[2:] or sorted(glob.glob("synth-kokoro/asr-*.jsonl"))
    pairs, texts = pairs_and_texts(files)
    wrong = collections.Counter()
    to = collections.defaultdict(collections.Counter)
    for (h, r), n in pairs.items():
        if len(h) >= 2 and h != r:
            wrong[h] += n
            to[h][r] += n
    words = sorted(wrong, key=len, reverse=True)
    seen = collections.Counter()
    finder = re.compile("|".join(map(re.escape, words)))
    for t in texts:
        seen.update(m.group(0) for m in finder.finditer(t))
    table = {w: {"wrong": wrong[w], "kept": max(0, seen[w] - wrong[w]), "to": [r for r, _ in to[w].most_common()]} for w in words}
    os.makedirs(out, exist_ok=True)
    json.dump(table, open(os.path.join(out, "voice.json"), "w", encoding="utf-8"), ensure_ascii=False)
    counts = {w: {"seen": seen[w], "to": dict(to[w])} for w in words}
    json.dump(counts, open(os.path.join(out, "voice-counts.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps({"files": len(files), "texts": len(texts), "pairs": len(pairs), "words": len(table),
                      "edits": sum(wrong.values())}))


if __name__ == "__main__":
    main()
