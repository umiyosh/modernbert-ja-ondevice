"""Label each transcribed train / val clip from the script, and quarantine clips whose label may be wrong.

- clean: the transcript equals the script (same, or notation only); measures false marks
- error: Whisper changed script words (homophone, mishear, drop, extra edits of classify.diff)
- quarantine: kept out of training and of every metric, for any of
  - broken: the transcript is cut off or made up (classify.broken)
  - short: the clip is under 1 second
  - reported: the sentence was reported on Common Voice (spelling, other language, hard to read)
  - reader: at a mishear, drop or extra edit, ReazonSpeech heard the same reading as Whisper. Two
    recognizers of different families hearing the same non-script words suggests the reader said them.
    Homophone edits sound like the script, so they are never the reader's
  - no_second: ReazonSpeech has not transcribed the clip yet

Val clips whose sentence is in JWTD train are flagged (jwtd) for leaving the primary metric.
Writes labels.jsonl (local: holds transcripts) and a report of counts (committed).

usage: python asr/labels.py <cv ja dir> <split dir> <whisper.jsonl> <reazon.jsonl> <labels.jsonl> <report.json>
"""

import collections
import hashlib
import json
import os
import sys
import time

from classify import diff
from near_dup import read

SHORT_MS = 1000
READER_KINDS = {"mishear", "drop", "extra"}


def load(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["path"]: r for r in map(json.loads, f)}


def reader_slips(edits: list[dict], second: list[dict]) -> int:
    heard = {(tuple(e["at"]), e["hyp_kana"]) for e in second}
    return sum(e["kind"] in READER_KINDS and (tuple(e["at"]), e["hyp_kana"]) in heard for e in edits)


def main() -> None:
    root, split_dir, whisper, reazon, out, report_path = sys.argv[1:7]
    trainval = os.path.join(split_dir, "trainval.tsv")
    split = {r["path"]: r["split"] for r in read(trainval)}
    rows = {r["path"]: r for r in read(os.path.join(root, "validated.tsv")) if r["path"] in split}
    ms = {r["clip"]: int(r["duration[ms]"]) for r in read(os.path.join(root, "clip_durations.tsv"))}
    reported = {r["sentence_id"] for r in read(os.path.join(root, "reported.tsv")) if r.get("sentence_id")}
    with open(os.path.join(split_dir, "jwtd_overlap_val.txt"), encoding="utf-8") as f:
        jwtd = set(f.read().split())
    w, z = load(whisper), load(reazon)

    status, reasons, kinds, agree = (collections.defaultdict(collections.Counter) for _ in range(4))
    with open(out, "w", encoding="utf-8") as f:
        for p in sorted(set(w) & set(split)):
            s, ref, hyp = split[p], w[p]["ref"], w[p]["hyp"]
            verdict, edits = diff(ref, hyp)
            why = []
            if verdict == "broken":
                why.append("broken")
            if ms.get(p, 0) < SHORT_MS:
                why.append("short")
            if rows[p]["sentence_id"] in reported:
                why.append("reported")
            if p not in z:
                why.append("no_second")
            elif edits:
                slips = reader_slips(edits, diff(ref, z[p]["hyp2"])[1])
                if slips:
                    why.append("reader")
                agree[s]["reader_slip_edits"] += slips
                agree[s]["checked_edits"] += sum(e["kind"] in READER_KINDS for e in edits)
            st = "quarantine" if why else ("error" if verdict == "error" else "clean")
            status[s][st] += 1
            reasons[s].update(why)
            if st == "error":
                kinds[s].update(e["kind"] for e in edits)
            rec = {"path": p, "split": s, "status": st, "reasons": why, "jwtd": p in jwtd, "ref": ref, "hyp": hyp, "edits": edits}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "trainval_sha256": hashlib.sha256(open(trainval, "rb").read()).hexdigest(),
        "second_recognizer": "reazon-research/reazonspeech-k2-v2 (rev 291488c8, greedy search)",
        "splits": {
            s: {
                "clips": sum(status[s].values()),
                "status": dict(status[s]),
                "quarantine_reasons": dict(reasons[s]),
                "edit_kinds_in_error_clips": dict(kinds[s]),
                "mishear_drop_extra_edits_checked": agree[s]["checked_edits"],
                "of_which_reader_slips": agree[s]["reader_slip_edits"],
            }
            for s in ("train", "val")
        },
        "val_jwtd_flagged": len(jwtd),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
