"""Transcribe synth_tts.py's readings with the Whisper of cv_asr.py and label them with classify.diff,
as Common Voice clips are labelled: the sentence read is the script, Whisper's text the transcript.
Readings already in <out> are skipped, so it can run while synthesis is still going.

usage: python asr/synth_asr.py <out.jsonl> <manifest.jsonl>...
"""

import json
import os
import sys

import mlx_whisper

from classify import diff
from cv_asr import MODEL


def main() -> None:
    out, manifests = sys.argv[1], sys.argv[2:]
    done = set()
    if os.path.exists(out):
        done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")}
    todo = [r for m in manifests for r in map(json.loads, open(m, encoding="utf-8")) if r["id"] not in done]
    n = 0
    with open(out, "a", encoding="utf-8") as f:
        for r in todo:
            hyp = mlx_whisper.transcribe(r["wav"], path_or_hf_repo=MODEL, language="ja")["text"].strip()
            verdict, edits = diff(r["text"], hyp)
            f.write(json.dumps({"id": r["id"], "engine": r["engine"], "voice": r["voice"], "ref": r["text"], "hyp": hyp,
                                "verdict": verdict, "edits": edits}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
