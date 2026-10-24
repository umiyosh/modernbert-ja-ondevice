"""The detector's probes of an evaluation run computed again in Python, as the app computes them (one
sentence at a time, up to 62 tokens a window), but with a chosen pair of sentence marks (Kotobae #98).
The app wraps each window in <cls> <sep>; the detector was fine-tuned on rows tokenised with <s> </s>
(build_rows.py), and gives the same tokens very different P under the two. The scorer's leads do not
depend on the detector, so a run that scored every candidate (screen 0) can be re-screened with these
probes by screen_sweep.py and verify_sweep.py.

usage (from asr/): python tools/reprobe.py <eval dir> <src name> <new name> [--marks s|cls] [--run DIR] [--cpu]
writes <eval dir>/res-<new>det-r0.json and links the source's deltas and verify files under <new>.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ENDS = "。！？\n"
LEN = 64
MARKS = {"s": (1, 2), "cls": (6, 4)}
B = 64


def windows(tok, text):
    """(char offset, ids, offsets) per detector window, as pipeline.rs::detect cuts them."""
    out, start = [], 0
    for i, c in enumerate(text):
        if c not in ENDS and i + 1 != len(text):
            continue
        sent = text[start:i + 1]
        if sent.strip():
            enc = tok(sent, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            for k in range(0, len(ids), LEN - 2):
                out.append((start, ids[k:k + LEN - 2], offs[k:k + LEN - 2]))
        start = i + 1
    return out


@torch.no_grad()
def probes(model, tok, texts, marks, dev):
    first, last = marks
    jobs = [(key, w) for key, text in texts.items() for w in windows(tok, text)]
    out = {key: [] for key in texts}
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        n = max(len(w[1]) for _, w in chunk) + 2
        x = torch.full((len(chunk), n), tok.pad_token_id)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for r, (_, (_, ids, _)) in enumerate(chunk):
            x[r, :len(ids) + 2] = torch.tensor([first, *ids, last])
            att[r, :len(ids) + 2] = 1
        p = model(input_ids=x.to(dev), attention_mask=att.to(dev)).logits.float().softmax(-1)[..., 1].cpu().tolist()
        for r, (key, (start, _, offs)) in enumerate(chunk):
            out[key] += [[start + a, start + b, round(p[r][j + 1], 4), "det"] for j, (a, b) in enumerate(offs) if b > a]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("src")
    ap.add_argument("new")
    ap.add_argument("--marks", default="s", choices=MARKS)
    ap.add_argument("--run", default=os.path.expanduser("~/dvlp/kotobae/.jev-eval/typo-study/asr/runs/kaw03"))
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = "cpu" if a.cpu or not torch.backends.mps.is_available() else "mps"
    tok = AutoTokenizer.from_pretrained(a.run)
    model = AutoModelForTokenClassification.from_pretrained(a.run).to(dev).eval()
    src = json.load(open(os.path.join(a.dir, f"res-{a.src}det-r0.json"), encoding="utf-8"))
    texts = {t["id"]: t["text"] for t in json.load(open(os.path.join(a.dir, "texts.json"), encoding="utf-8")) if t["id"] in src}
    out = probes(model, tok, texts, MARKS[a.marks], dev)
    json.dump(out, open(os.path.join(a.dir, f"res-{a.new}det-r0.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for kind in ("deltas", "verify", "cands"):
        s, t = os.path.join(a.dir, f"{kind}-{a.src}.json"), os.path.join(a.dir, f"{kind}-{a.new}.json")
        if os.path.exists(s) and not os.path.lexists(t):
            os.symlink(os.path.basename(s), t)
    print(f"{a.dir}: {len(out)} texts, {sum(map(len, out.values()))} probes", flush=True)


if __name__ == "__main__":
    main()
