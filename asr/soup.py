"""Average the weights of checkpoints fine-tuned from the same detector (a uniform model soup): one
model, no extra cost at inference, and less of the run-to-run spread a single run's threshold shows.

usage: python asr/soup.py <out dir> <checkpoint dir>...
"""

import os
import shutil
import sys

from safetensors.torch import load_file, save_file


def main() -> None:
    out, dirs = sys.argv[1], sys.argv[2:]
    states = [load_file(os.path.join(d, "model.safetensors")) for d in dirs]
    avg = {k: sum(s[k].float() for s in states).div(len(states)).to(states[0][k].dtype) for k in states[0]}
    os.makedirs(out, exist_ok=True)
    save_file(avg, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(dirs[0]):
        if f != "model.safetensors" and os.path.isfile(os.path.join(dirs[0], f)):
            shutil.copy(os.path.join(dirs[0], f), out)
    print(out, "from", len(dirs), "checkpoints")


if __name__ == "__main__":
    main()
