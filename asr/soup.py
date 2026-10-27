"""Average the weights of checkpoints fine-tuned from the same detector (a uniform model soup): one
model, no extra cost at inference, and less of the run-to-run spread a single run's threshold shows.
A checkpoint given as <dir>@<w> weighs w (default 1, normalised): with the published detector as one
side this interpolates back toward it (WiSE-FT), keeping some of what the fine-tuning forgot.

usage: python asr/soup.py <out dir> <checkpoint dir>[@w]...
"""

import os
import shutil
import sys

from safetensors.torch import load_file, save_file


def main() -> None:
    out = sys.argv[1]
    parts = [a.rsplit("@", 1) if "@" in a else (a, "1") for a in sys.argv[2:]]
    dirs, ws = [d for d, _ in parts], [float(w) for _, w in parts]
    states = [load_file(os.path.join(d, "model.safetensors")) for d in dirs]
    avg = {k: sum(w * s[k].float() for w, s in zip(ws, states)).div(sum(ws)).to(states[0][k].dtype) for k in states[0]}
    os.makedirs(out, exist_ok=True)
    save_file(avg, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(dirs[0]):
        if f != "model.safetensors" and os.path.isfile(os.path.join(dirs[0], f)):
            shutil.copy(os.path.join(dirs[0], f), out)
    print(out, "from", len(dirs), "checkpoints")


if __name__ == "__main__":
    main()
