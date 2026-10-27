"""The external holdout (PROTOCOL.md, amendment of 2026-10-27): FLEURS ja_jp, all three splits.

prep: lays FLEURS out as cv_asr.py and second_asr.py read Common Voice (clips/ and validated.tsv), so the
same scripts transcribe it. sentence_id is FLEURS's id (one sentence, up to three readers), the
sentence is its raw transcription.

labels: ReazonSpeech k2 v2 is the recognizer under test here and Whisper the second one, the reverse of
Common Voice, so the transcript is never Whisper's. Edits and statuses follow labels.py (broken and
reader-slip clips quarantined; FLEURS has no reported sentences, and every clip is over 1 second or
quarantined as short). Clips whose sentence is in JWTD train are flagged as on Common Voice. Writes
labels (local: holds transcripts), texts.json for Kotobae's typo_eval, and a report of counts.

usage: python asr/fleurs.py prep <fleurs dir> <out root>
       python asr/fleurs.py labels <out root> <reazon.jsonl> <whisper.jsonl> <jwtd train.jsonl> <eval dir> <report.json>
"""

import collections
import csv
import json
import os
import sys
import time

from classify import diff, norm
from labels import READER_KINDS, SHORT_MS, load, reader_slips
from leak_check import in_jwtd

SAMPLE_RATE = 16000
SPLITS = ("train", "dev", "test")
COLUMNS = ["id", "file", "raw", "transcription", "phonemes", "samples", "gender"]


def rows(fleurs: str):
    for s in SPLITS:
        with open(os.path.join(fleurs, f"{s}.tsv"), encoding="utf-8", newline="") as f:
            for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                yield s, dict(zip(COLUMNS, r))


def prep(fleurs: str, root: str) -> None:
    os.makedirs(os.path.join(root, "clips"), exist_ok=True)
    out = []
    for s, r in rows(fleurs):
        link = os.path.join(root, "clips", r["file"])
        if not os.path.lexists(link):
            os.symlink(os.path.join(fleurs, "audio", s, s, r["file"]), link)
        out.append({"path": r["file"], "sentence_id": r["id"], "sentence": r["raw"], "split": s,
                    "ms": int(r["samples"]) * 1000 // SAMPLE_RATE})
    with open(os.path.join(root, "validated.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(root, "order.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(r["path"] for r in out) + "\n")
    print(len(out), "clips,", len({r["sentence_id"] for r in out}), "sentences")


def labels(root: str, reazon: str, whisper: str, jwtd: str, eval_dir: str, report_path: str) -> None:
    with open(os.path.join(root, "validated.tsv"), encoding="utf-8", newline="") as f:
        meta = {r["path"]: r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")}
    z, w = load(reazon), load(whisper)
    normed = {p: norm(m["sentence"]) for p, m in meta.items()}
    equal, contained = in_jwtd(jwtd, set(normed.values()))
    in_train = equal | contained
    status, reasons, kinds = collections.Counter(), collections.Counter(), collections.Counter()
    texts = []
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "labels.jsonl"), "w", encoding="utf-8") as f:
        for p in sorted(meta):
            if p not in z:
                continue
            ref, hyp = meta[p]["sentence"], z[p]["hyp2"]
            verdict, edits = diff(ref, hyp)
            why = []
            if verdict == "broken":
                why.append("broken")
            if int(meta[p]["ms"]) < SHORT_MS:
                why.append("short")
            if p not in w:
                why.append("no_second")
            elif edits and reader_slips(edits, diff(ref, w[p]["hyp"])[1]):
                why.append("reader")
            st = "quarantine" if why else ("error" if verdict == "error" else "clean")
            status[st] += 1
            reasons.update(why)
            if st == "error":
                kinds.update(e["kind"] for e in edits)
            rec = {"path": p, "speaker": meta[p]["sentence_id"], "split": "external", "status": st, "reasons": why,
                   "jwtd": normed[p] in in_train, "ref": ref, "hyp": hyp, "edits": edits}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            texts.append({"id": p, "text": hyp})
    json.dump(texts, open(os.path.join(eval_dir, "texts.json"), "w", encoding="utf-8"), ensure_ascii=False)
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus": "google/fleurs ja_jp (CC BY 4.0), train + dev + test",
        "recognizer": "reazon-research/reazonspeech-k2-v2 (greedy search); second: whisper-large-v3-turbo (MLX)",
        "clips": sum(status.values()),
        "sentences": len({meta[p]["sentence_id"] for p in z if p in meta}),
        "status": dict(status),
        "quarantine_reasons": dict(reasons),
        "edit_kinds_in_error_clips": dict(kinds),
        "reader_kinds": sorted(READER_KINDS),
        "jwtd_train_sentences": len(in_train),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"prep": prep, "labels": labels}[sys.argv[1]](*sys.argv[2:])
