"""Experiment 3 of the 60% plan (Kotobae #98): re-run the deployed detector on each candidate-substituted
sentence (2110.01857: an ELECTRA-style discriminator checks a replacement better than masked-LM scoring).
For every alternative a run scored with a lead above --min, writes the detector's P(typo) over the span in
the sentence as written and with the alternative in, as the app splits sentences. No new model: the
detector is the one the app ships (runs/kaw03, exported to Core ML as the app's typo-detector).

usage (from asr/, under guard.py): python tools/verify_det.py <eval dir> <name> [--min -3] [--run DIR]
writes <eval dir>/verify-<name>.json: {key: [[s, e, alt, p_written, p_alt], ...]}
--run checks with another detector (a retrained one); the file must then not be a link to another run's.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

RUN = os.path.expanduser("~/dvlp/kotobae/.jev-eval/typo-study/asr/runs/kaw03")
ENDS = "。！？\n"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
B = 64


def sentence(text, s, e):
    """The app's detection unit around s..e: from after the previous end mark to the next one."""
    a = max((i + 1 for i in range(s) if text[i] in ENDS), default=0)
    b = next((i + 1 for i in range(e, len(text)) if text[i] in ENDS), len(text))
    return a, b


@torch.no_grad()
def span_p(model, tok, items):
    """items: (sentence, s, e) in sentence offsets -> max P(TYPO) over tokens overlapping s..e."""
    enc = tok([x[0] for x in items], return_offsets_mapping=True, padding=True, truncation=True,
              max_length=128, return_tensors="pt")
    offs = enc.pop("offset_mapping")
    p = model(**{k: v.to(DEV) for k, v in enc.items()}).logits.float().softmax(-1)[..., 1].cpu()
    out = []
    for i, (_, s, e) in enumerate(items):
        vals = [float(p[i, j]) for j, (a, b) in enumerate(offs[i].tolist()) if b > a and a < e and s < b]
        out.append(max(vals, default=0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("name")
    ap.add_argument("--min", type=float, default=-3.0)
    ap.add_argument("--run", default=RUN)
    a = ap.parse_args()
    target = os.path.join(a.dir, f"verify-{a.name}.json")
    if os.path.islink(target):
        raise SystemExit(f"{target} links to another run's checks; remove the link first")
    tok = AutoTokenizer.from_pretrained(a.run)
    model = AutoModelForTokenClassification.from_pretrained(a.run).to(DEV).eval()
    texts = {t["id"]: t["text"] for t in json.load(open(os.path.join(a.dir, "texts.json"), encoding="utf-8"))}
    deltas = json.load(open(os.path.join(a.dir, f"deltas-{a.name}.json"), encoding="utf-8"))
    jobs = []
    for key, rows in deltas.items():
        text = texts[key]
        for s, e, written, alts in rows:
            x, y = sentence(text, s, e)
            for alt, lead in alts:
                if lead > a.min:
                    jobs.append((key, s, e, alt, (text[x:y], s - x, e - x), (text[x:s] + alt + text[e:y], s - x, s - x + len(alt))))
    print(len(jobs), "alternatives", flush=True)
    out = {}
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        pw = span_p(model, tok, [j[4] for j in chunk])
        pa = span_p(model, tok, [j[5] for j in chunk])
        for (key, s, e, alt, _, _), x, y in zip(chunk, pw, pa):
            out.setdefault(key, []).append([s, e, alt, round(x, 5), round(y, 5)])
        if i % (B * 100) == 0:
            print(i, flush=True)
    json.dump(out, open(target, "w", encoding="utf-8"), ensure_ascii=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
