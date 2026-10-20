"""Read written sentences aloud with a synthetic voice, so Whisper's errors on written-style text can be
labelled the way Common Voice clips are (synth_asr.py). Common Voice scripts are short spoken
sentences with almost no Latin, digits or written phrasing, and a detector trained on them alone marked
those in Kotobae's texts.

Three engines: Irodori-TTS (natural, voice-cloned from the model's bundled samples; run with AtodeKiku's
MLX runtime python), macOS `say` (robotic but fast; its Japanese voices) and Kokoro 82M (light ONNX, its
five Japanese voices). Only Kokoro's readings may train a model that is published or sold: macOS licenses
its system voices for personal, non-commercial use only. Voices rotate by line.
Lines [start, end) of the sentence file are read; a line whose wav exists is skipped, so a stopped run
resumes. Appends {id, engine, voice, text, wav} to <out dir>/manifest-<engine>-<start>.jsonl (one file per
worker, so parallel readers never interleave lines).

usage: <python> asr/synth_tts.py irodori|say|kokoro <sentences.txt> <out dir> <start> <end>
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

IRODORI = Path.home() / "Library/Caches/com.umiyosh.atodekiku/local-tts-models/local-irodori-mlx-mlx-community-Irodori-TTS-v4.1-Small-8bit"
LAUNCHER = Path.home() / "dvlp/AtodeKiku/src-tauri/resources"
SAY_VOICES = ["Kyoko", "Eddy (日本語（日本）)", "Flo (日本語（日本）)", "Reed (日本語（日本）)", "Rocko (日本語（日本）)",
              "Sandy (日本語（日本）)", "Shelley (日本語（日本）)", "Grandma (日本語（日本）)", "Grandpa (日本語（日本）)"]
# Apache-2.0; the voices other than jf_alpha learnt from Koniwa's readings (CC BY 3.0). The quantized model is
# the one AtodeKiku ships.
KOKORO = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_REVISION = "1939ad2a8e416c0acfeecc08a694d14ef25f2231"
KOKORO_VOICES = ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"]


def irodori_reader():
    sys.path.insert(0, str(LAUNCHER))
    import irodori_mlx_launcher as launcher

    generate = launcher._load_generate_audio()
    model = launcher._load_irodori_model(str(IRODORI), None)
    voices = sorted(p.name for p in (IRODORI / "samples").iterdir())

    def read(i: int, text: str, wav: Path) -> str:
        voice = voices[i % len(voices)]
        generate(text=text, model=model, ref_audio=str(IRODORI / "samples" / voice), output_path=str(wav.parent),
                 file_prefix=wav.stem, audio_format="wav", join_audio=True, verbose=False, stt_model=None)
        return voice

    return read


def say_reader():
    def read(i: int, text: str, wav: Path) -> str:
        voice = SAY_VOICES[i % len(SAY_VOICES)]
        # The text goes on stdin: a sentence starting with "-" was taken for an option.
        subprocess.run(["say", "-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000", "-o", str(wav)],
                       input=text, text=True, check=True)
        return voice

    return read


def kokoro_reader():
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from misaki import ja

    def get(name: str) -> str:
        return hf_hub_download(KOKORO, name, revision=KOKORO_REVISION)

    vocab = json.load(open(get("tokenizer.json"), encoding="utf-8"))["model"]["vocab"]
    options = ort.SessionOptions()
    # Several readers share the CPU with Whisper.
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(get("onnx/model_quantized.onnx"), options)
    styles = {v: np.fromfile(get(f"voices/{v}.bin"), dtype=np.float32).reshape(-1, 1, 256) for v in KOKORO_VOICES}
    # The G2P the Japanese voices were trained with.
    g2p = ja.JAG2P()

    def read(i: int, text: str, wav: Path) -> str:
        voice = KOKORO_VOICES[i % len(KOKORO_VOICES)]
        ids = [vocab[c] for c in g2p(text)[0] if c in vocab][:510]
        if not ids:
            raise ValueError("no phonemes")
        # A voice holds one style per input length.
        feed = {"input_ids": np.array([[0, *ids, 0]], dtype=np.int64), "style": styles[voice][len(ids)],
                "speed": np.ones(1, dtype=np.float32)}
        sf.write(wav, session.run(None, feed)[0][0], 24000)
        return voice

    return read


READERS = {"irodori": irodori_reader, "say": say_reader, "kokoro": kokoro_reader}


def read_or_skip(read, i: int, text: str, wav: Path) -> str | None:
    """A line is tried three times, then skipped (None). Two readers died on one minute's file errors
    from `say` and left 1,756 lines unread; now a bad line or minute costs only itself."""
    for _ in range(3):
        # Irodori's library prints its settings for every call; keep the log to progress lines.
        with open(os.devnull, "w") as null:
            saved, sys.stdout = sys.stdout, null
            try:
                return read(i, text, wav)
            # misaki's Japanese G2P asserts on some digit runs ("10"); that line alone is skipped.
            except (subprocess.CalledProcessError, ValueError, AssertionError) as e:
                err = e
            finally:
                sys.stdout = saved
        wav.unlink(missing_ok=True)
        time.sleep(5)
    print(f"skipped line {i}: {err}", flush=True)
    return None


def main() -> None:
    engine, sentences, out, start, end = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    lines = open(sentences, encoding="utf-8").read().splitlines()[start:end]
    (out / engine).mkdir(parents=True, exist_ok=True)
    read = READERS[engine]()
    with open(out / f"manifest-{engine}-{start}.jsonl", "a", encoding="utf-8") as m:
        for i, text in enumerate(lines, start):
            # Kokoro's 24 kHz WAV took 490 KB a line, 18 GB for the sentence file; MP3 is the format
            # Common Voice's clips come in.
            wav = out / engine / f"{i:06d}.{'mp3' if engine == 'kokoro' else 'wav'}"
            if wav.exists() or wav.with_suffix(".wav").exists():
                continue
            voice = read_or_skip(read, i, text, wav)
            if voice is None:
                continue
            m.write(json.dumps({"id": f"{engine}-{i:06d}", "engine": engine, "voice": voice, "text": text, "wav": str(wav)}, ensure_ascii=False) + "\n")
            m.flush()
            if (i - start + 1) % 100 == 0:
                print(f"{engine} {i - start + 1}/{len(lines)}", flush=True)


if __name__ == "__main__":
    main()
