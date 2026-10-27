"""Run a detector checkpoint (PyTorch) over the evaluation texts the way Kotobae's `typo::pipeline`
does, and write the files `eval_spans.py` reads, so a fine-tuned detector is scored without a Core ML
export. Each sentence (cut after 。！？ or a newline) is encoded without special tokens, cut into
windows of 62 tokens and wrapped in <cls> / <sep>; every token with characters gets its P(TYPO).

Writes res-<name>det-r0.json (probes) and, unless it exists, deltas-<name>.json with no candidates,
so the marks are the detector's alone (C-det / D-det of PROTOCOL.md).

usage: python asr/detect.py <model or dir> <eval dir with texts.json> <name> [--revision REV]
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ENDS = "。！？\n"
DETECT_LEN, CLS, SEP, PAD = 64, 6, 4, 3  # src-tauri/src/services/typo/{pipeline,tokens}.rs
BATCH = 64


def windows(tok, text: str) -> list[tuple[int, list[int], list[tuple[int, int]]]]:
    out, start = [], 0
    for i, c in enumerate(text):
        if c not in ENDS and i + 1 != len(text):
            continue
        sent = text[start : i + 1]
        if sent.strip():
            enc = tok(sent, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            for k in range(0, len(ids), DETECT_LEN - 2):
                out.append((start, [CLS] + ids[k : k + DETECT_LEN - 2] + [SEP], offs[k : k + DETECT_LEN - 2]))
        start = i + 1
    return out


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def probes(model, tok, texts: dict[str, str]) -> dict[str, list]:
    dev = next(model.parameters()).device
    flat = [(path, w) for path, t in texts.items() for w in windows(tok, t)]
    out = {path: [] for path in texts}
    for k in range(0, len(flat), BATCH):
        chunk = flat[k : k + BATCH]
        n = max(len(w[1]) for _, w in chunk)
        x = torch.full((len(chunk), n), PAD)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for i, (_, (_, ids, _)) in enumerate(chunk):
            x[i, : len(ids)] = torch.tensor(ids)
            att[i, : len(ids)] = 1
        p = torch.softmax(model(input_ids=x.to(dev), attention_mask=att.to(dev)).logits.float(), -1)[..., 1].cpu().numpy()
        for i, (path, (off, _, offs)) in enumerate(chunk):
            for j, (a, b) in enumerate(offs):
                if b > a:
                    out[path].append([off + a, off + b, round(float(p[i, j + 1]), 4), "det"])
    return out


def load(name: str, revision: str | None):
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForTokenClassification.from_pretrained(name, revision=revision).to(device()).eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("eval_dir")
    ap.add_argument("name")
    ap.add_argument("--revision")
    a = ap.parse_args()
    texts = {r["id"]: r["text"] for r in json.load(open(os.path.join(a.eval_dir, "texts.json"), encoding="utf-8"))}
    model, tok = load(a.model, a.revision)
    res = probes(model, tok, texts)
    with open(os.path.join(a.eval_dir, f"res-{a.name}det-r0.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    deltas = os.path.join(a.eval_dir, f"deltas-{a.name}.json")
    if not os.path.exists(deltas):
        with open(deltas, "w", encoding="utf-8") as f:
            json.dump({p: [] for p in texts}, f)
    print(f"{len(res)} texts, {sum(map(len, res.values()))} probes, max P {max((x[2] for v in res.values() for x in v), default=0):.4f}")


if __name__ == "__main__":
    main()
