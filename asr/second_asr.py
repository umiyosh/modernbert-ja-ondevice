"""Transcribe the clips Whisper has transcribed again with ReazonSpeech k2 v2, a recognizer of another
family, so labels.py can tell a reader's slip (both recognizers hear the same non-script words) from a
Whisper error. Runs on the CPU, beside Whisper on the GPU. Appends one JSON line per clip and resumes.

usage: python asr/second_asr.py <cv ja dir> <whisper.jsonl> <out.jsonl> <reazonspeech-k2-v2 dir>
"""

import json
import os
import sys
import time

import sherpa_onnx

from audio_io import SAMPLE_RATE, prefetch


def main() -> None:
    root, whisper, out, model = sys.argv[1:5]
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["path"] for line in f}
    with open(whisper, encoding="utf-8") as f:
        todo = [p for p in (json.loads(line)["path"] for line in f) if p not in done]
    print(f"todo {len(todo)} (done {len(done)})", flush=True)

    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{model}/encoder-epoch-99-avg-1.onnx",
        decoder=f"{model}/decoder-epoch-99-avg-1.onnx",
        joiner=f"{model}/joiner-epoch-99-avg-1.onnx",
        tokens=f"{model}/tokens.txt",
        num_threads=4,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, (p, audio) in enumerate(zip(todo, prefetch([os.path.join(root, "clips", p) for p in todo])), 1):
            if audio is None:
                continue
            s = rec.create_stream()
            s.accept_waveform(SAMPLE_RATE, audio)
            rec.decode_stream(s)
            f.write(json.dumps({"path": p, "hyp2": s.result.text}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 2000 == 0 or i == len(todo):
                print(f"{time.strftime('%H:%M')} {i}/{len(todo)} {(time.time() - t0) / i:.2f}s/clip", flush=True)


if __name__ == "__main__":
    main()
