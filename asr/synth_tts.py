"""Read written sentences aloud with a synthetic voice, so Whisper's errors on written-style text can be
labelled the way Common Voice clips are (synth_asr.py). Common Voice scripts are short spoken
sentences with almost no Latin, digits or written phrasing, and a detector trained on them alone marked
those in Kotobae's texts.

Two engines: Irodori-TTS (natural, voice-cloned from the model's bundled samples; run with AtodeKiku's
MLX runtime python) and macOS `say` (robotic but fast; its Japanese voices). Voices rotate by line.
Lines [start, end) of the sentence file are read; a line whose wav exists is skipped, so a stopped run
resumes. Appends {id, engine, voice, text, wav} to <out dir>/manifest-<engine>-<start>.jsonl (one file per
worker, so parallel readers never interleave lines).

usage: <python> asr/synth_tts.py irodori|say <sentences.txt> <out dir> <start> <end>
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


def read_or_skip(read, i: int, text: str, wav: Path) -> str | None:
    """A line is tried three times, then skipped (None). Two readers died on one minute's file errors
    from `say` and left 1,756 lines unread; now a bad line or minute costs only itself."""
    for _ in range(3):
        # Irodori's library prints its settings for every call; keep the log to progress lines.
        with open(os.devnull, "w") as null:
            saved, sys.stdout = sys.stdout, null
            try:
                return read(i, text, wav)
            except subprocess.CalledProcessError as e:
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
    read = irodori_reader() if engine == "irodori" else say_reader()
    with open(out / f"manifest-{engine}-{start}.jsonl", "a", encoding="utf-8") as m:
        for i, text in enumerate(lines, start):
            wav = out / engine / f"{i:06d}.wav"
            if wav.exists():
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
