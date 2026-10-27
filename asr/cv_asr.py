"""Transcribe Common Voice clips with Whisper (MLX) and append one JSON line per clip.

Rerunning with the same output skips the clips already written, so a stopped run resumes.
client_id is not read: the transcripts are not used to tell who a speaker is.

A clip that takes longer than --limit seconds is written to <out>.hung and the process exits with
HUNG; rerunning skips it. Whisper can decode the same window again and again when a sampled
fallback ends on <|0.00|>, and MLX cannot be interrupted from Python, so only exiting frees it.

usage: python asr/cv_asr.py <cv ja dir> <out.jsonl> --order <asr_order.txt> --n 70000
       python asr/cv_asr.py <cv ja dir> <out.jsonl> --tsv train.tsv --n 500   (one clip per sentence)
       while :; do python asr/cv_asr.py ...; [ $? -eq 75 ] || break; done     (restart after a hang)
"""

import argparse
import collections
import csv
import json
import os
import random
import threading
import time

import mlx_whisper

from audio_io import prefetch

MODEL = "mlx-community/whisper-large-v3-turbo"
HUNG = 75  # exit status after a hang; EX_TEMPFAIL, so a wrapper knows to restart


def read(tsv: str) -> list[dict[str, str]]:
    with open(tsv, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def pick(root: str, tsv: str, n: int, seed: int, holdout: list[str]) -> list[dict[str, str]]:
    # One reader per sentence first, then a second reader, and so on, so the errors do not pile
    # up on a few sentences. Sentences of the holdout files are left for evaluation.
    held = {r["sentence"] for h in holdout for r in read(os.path.join(root, h))}
    rows = read(os.path.join(root, tsv))
    random.Random(seed).shuffle(rows)
    reads = collections.Counter()
    order = []
    for i, r in enumerate(rows):
        if r["sentence"] in held:
            continue
        order.append((reads[r["sentence"]], i, r))
        reads[r["sentence"]] += 1
    order.sort(key=lambda x: x[:2])
    return [r for *_, r in order[:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Common Voice ja directory (validated.tsv and clips/)")
    ap.add_argument("out", help="output JSONL")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tsv", default="validated.tsv")
    ap.add_argument("--holdout", nargs="*", default=[], help="tsv files whose sentences are not used")
    ap.add_argument("--order", help="paths in transcription order (asr_order.txt of make_split.py); overrides --holdout")
    ap.add_argument("--limit", type=float, default=120, help="seconds per clip before it counts as hung")
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {json.loads(line)["path"] for line in f}
    hung_file = a.out + ".hung"
    hung = set()
    if os.path.exists(hung_file):
        with open(hung_file, encoding="utf-8") as f:
            hung = set(f.read().split())
    done |= hung
    if a.order:
        by_path = {r["path"]: r for r in read(os.path.join(a.root, a.tsv))}
        with open(a.order, encoding="utf-8") as f:
            chosen = [by_path[p] for p in f.read().split()[: a.n]]
    else:
        chosen = pick(a.root, a.tsv, a.n, a.seed, a.holdout)
    rows = [r for r in chosen if r["path"] not in done]
    print(f"todo {len(rows)} (done {len(done) - len(hung)}, hung {len(hung)})", flush=True)

    t0, skipped = time.time(), 0
    audios = prefetch([os.path.join(a.root, "clips", r["path"]) for r in rows])
    with open(a.out, "a", encoding="utf-8") as f:
        for i, (r, audio) in enumerate(zip(rows, audios), 1):
            if audio is None:
                skipped += 1
                continue
            timer = threading.Timer(a.limit, give_up, (hung_file, r["path"]))
            timer.start()
            hyp = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, language="ja")["text"]
            timer.cancel()
            rec = {"path": r["path"], "sentence_id": r["sentence_id"], "ref": r["sentence"], "hyp": hyp}
            # Flushed per clip: a hang ends the process with os._exit, which drops buffered lines.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 500 == 0 or i == len(rows):
                print(f"{time.strftime('%H:%M')} {i}/{len(rows)} {(time.time() - t0) / i:.2f}s/clip skipped {skipped}", flush=True)


def give_up(hung_file: str, path: str) -> None:
    with open(hung_file, "a", encoding="utf-8") as f:
        f.write(path + "\n")
    print(f"{time.strftime('%H:%M')} hung on {path}; exiting to restart", flush=True)
    os._exit(HUNG)


if __name__ == "__main__":
    main()
