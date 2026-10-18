"""Decode Common Voice MP3 clips to 16 kHz mono float32, ahead of the recognizer that reads them."""

import collections
import itertools
import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SAMPLE_RATE = 16000  # both Whisper and ReazonSpeech read 16 kHz


def decode(path: str) -> np.ndarray | None:
    # mlx_whisper's load_audio returns an MLX array, which fails on worker threads without a GPU
    # stream. Decode to numpy instead.
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        return None  # counted as skipped in the progress line
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768


def prefetch(paths: list[str], depth: int = 16) -> Iterator[np.ndarray | None]:
    # Decode the next clips while the recognizer runs on the current one; at most `depth` ahead to bound memory.
    with ThreadPoolExecutor(4) as ex:
        it = iter(paths)
        q = collections.deque(ex.submit(decode, p) for p in itertools.islice(it, depth))
        for p in it:
            yield q.popleft().result()
            q.append(ex.submit(decode, p))
        while q:
            yield q.popleft().result()
