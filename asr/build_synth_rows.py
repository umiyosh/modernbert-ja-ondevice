"""Training rows and a held-out evaluation set from synth_asr.py's labelled readings.

Sentences are split by their line in the sentence file: lines at or past --eval-from are never
trained on and become <eval dir> (texts.json and labels.jsonl in the shape eval_spans.py reads, split
"val"), which measures misconversions on written text, the product's case. The rest become rows the
way build_rows.py makes them (Whisper's text, error characters labelled 1); broken readings are left
out of both.

usage: python asr/build_synth_rows.py <asr.jsonl> <out prefix> <eval dir> [--eval-from 15840]
"""

import collections
import json
import os
import sys

from build_rows import bad_chars, encode, save


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src, out, eval_dir = args[:3]
    cut = int(sys.argv[sys.argv.index("--eval-from") + 1]) if "--eval-from" in sys.argv else 15840
    rows, seen, n = [], set(), collections.Counter()
    texts, labels = [], []
    for r in map(json.loads, open(src, encoding="utf-8")):
        if r["verdict"] == "broken":
            n["broken"] += 1
            continue
        rec = {"path": r["id"], "split": "val", "status": "error" if r["verdict"] == "error" else "clean",
               "jwtd": False, "ref": r["ref"], "hyp": r["hyp"], "edits": r["edits"]}
        if int(r["id"].rsplit("-", 1)[1]) >= cut:
            texts.append({"id": r["id"], "text": r["hyp"]})
            labels.append(rec)
            n["eval_" + rec["status"]] += 1
            continue
        bad = bad_chars(rec) if rec["status"] == "error" else set()
        if bad is None:
            n["no_char_map"] += 1
            continue
        ids, lab = encode(rec["hyp"], bad)
        if rec["status"] == "error" and 1 not in lab:
            n["error_past_truncation"] += 1
            continue
        key = (tuple(ids), tuple(lab))
        if key in seen:
            n["duplicates"] += 1
            continue
        seen.add(key)
        rows.append((ids, lab))
        n[rec["status"]] += 1
    save(rows, out + ".npz")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    with open(os.path.join(eval_dir, "labels.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in labels)
    n.update(rows=len(rows))
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(n, f, indent=1)
        f.write("\n")
    print(json.dumps(n))


if __name__ == "__main__":
    main()
