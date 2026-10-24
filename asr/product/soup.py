"""Weight interpolation between two detectors (Kotobae #98; WiSE-FT, Wortsman+ 2022): the night's detector
raised P on correct words so far that the detector mark had to go up to 0.995 and the ordinary typos it
alone used to mark fell under their floors. A mix keeps part of the old detector's calibration.

usage: python product/soup.py <old run> <new run> <share of new> <out dir>
"""
import os
import shutil
import sys

from safetensors.torch import load_file, save_file


def main():
    old, new, share, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    a = load_file(os.path.join(old, "model.safetensors"))
    b = load_file(os.path.join(new, "model.safetensors"))
    if a.keys() != b.keys():
        raise SystemExit("the two runs do not share their weights' names")
    os.makedirs(out, exist_ok=True)
    mix = {k: (a[k].float() * (1 - share) + b[k].float() * share).to(a[k].dtype) for k in a}
    save_file(mix, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(new, f), out)
    print(f"{out}: {share} of {new}, {1 - share:.2f} of {old}")


if __name__ == "__main__":
    main()
