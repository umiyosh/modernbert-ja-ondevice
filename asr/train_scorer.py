"""Fine-tune the on-device check's scorer (modernbert-ja-130m as a masked LM) on scorer_pairs.py's pairs
(Kotobae #98). The score is the app's own (typo::pipeline::score): the sentence cut to 32 tokens around
the word, and the summed log P of the word's tokens and two on each side, each masked in turn. A pair's
lead is the good word's score minus the bad word's; the loss pushes it past a margin (softplus), and a
masked-LM loss on the good sentences keeps the model a language model (Masked-FT, arXiv 2305.17721).

A share of the pair groups is held out; its ranking accuracy and mean lead are printed at each save, so
a run that only memorises its pairs shows. Checkpoints go to <out>/step<N>.

usage: python asr/train_scorer.py <pairs.jsonl> <out dir> [--lr 2e-5] [--steps 3000] [--batch 16]
                                  [--margin 2.0] [--mlm 1.0] [--save-every 1000] [--seed 0]
"""

import argparse
import json
import os
import random
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.24")
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = "sbintuitions/modernbert-ja-130m"
LEN, NEIGHBOURS = 32, 2
HELD = 0.05
# Scored positions per window at most (a longer word is left out), and masked tokens per sentence in
# the masked-LM loss. Every batch is padded to the same shapes: MPS keeps buffers per input shape, and
# varying shapes let them pile up (11 GB within 20 steps).
MAX_AT, MLM_K = 9, 3
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def around(tok, text, s, e):
    """The app's window: token ids cut to LEN around the word, and the positions scored."""
    enc = tok(text, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    inside = [i for i, (a, b) in enumerate(offs) if b > a and a < e and s < b]
    if not inside:
        return None
    if len(ids) > LEN:
        body = LEN - 2
        mid = (inside[0] + inside[-1]) // 2
        lo = max(1, min(max(0, mid - body // 2), len(ids) - 1 - body))
        ids = [ids[0]] + ids[lo:lo + body] + [ids[-1]]
        inside = [i - lo + 1 for i in inside if i >= lo and i - lo + 1 < LEN - 1]
        if not inside:
            return None
    lo = max(1, inside[0] - NEIGHBOURS)
    hi = min(inside[-1] + NEIGHBOURS, len(ids) - 2)
    return ids, list(range(lo, hi + 1))


def variants(tok, r):
    """The good and the bad sentence's windows, or None when either has no token for the word."""
    t, s, e = r["text"], r["s"], r["e"]
    out = []
    for w in (r["good"], r["bad"]):
        v = around(tok, t[:s] + w + t[e:], s, s + len(w))
        if v is None or len(v[1]) > MAX_AT or len(v[0]) - 2 < MLM_K:
            return None
        out.append(v)
    return out


def scores(model, tok, wins):
    """Each window's summed log P over its scored positions, each masked in turn (one row per position)."""
    rows = [(ids, at, k) for k, (ids, ats) in enumerate(wins) for at in ats]
    # Padding rows score into an extra slot that is dropped.
    rows += [([tok.cls_token_id, tok.sep_token_id], 1, len(wins))] * (len(wins) * MAX_AT - len(rows))
    x = torch.full((len(rows), LEN), tok.pad_token_id)
    att = torch.zeros((len(rows), LEN), dtype=torch.long)
    for j, (ids, at, _) in enumerate(rows):
        x[j, : len(ids)] = torch.tensor(ids)
        x[j, at] = tok.mask_token_id
        att[j, : len(ids)] = 1
    pos = torch.tensor([at for _, at, _ in rows], device=DEV)
    tgt = torch.tensor([ids[at] for ids, at, _ in rows], device=DEV)
    hidden = model.model(input_ids=x.to(DEV), attention_mask=att.to(DEV)).last_hidden_state
    h = hidden[torch.arange(len(rows), device=DEV), pos]
    lp = model.decoder(model.head(h)).float().log_softmax(-1)
    got = lp[torch.arange(len(rows), device=DEV), tgt]
    owner = torch.tensor([k for _, _, k in rows], device=DEV)
    return torch.zeros(len(wins) + 1, device=DEV).index_add(0, owner, got)[:-1]


def mlm_loss(model, tok, wins, gen):
    """Masked-LM loss on the good sentences, MLM_K tokens of each hidden (Masked-FT keeps the LM)."""
    x = torch.full((len(wins), LEN), tok.pad_token_id)
    att = torch.zeros((len(wins), LEN), dtype=torch.long)
    for j, (ids, _) in enumerate(wins):
        x[j, : len(ids)] = torch.tensor(ids)
        att[j, : len(ids)] = 1
    y = x.clone()
    hide = torch.zeros(x.shape, dtype=torch.bool)
    for j, (ids, _) in enumerate(wins):
        picked = torch.randperm(len(ids) - 2, generator=gen)[:MLM_K] + 1
        hide[j, picked] = True
    x[hide] = tok.mask_token_id
    hidden = model.model(input_ids=x.to(DEV), attention_mask=att.to(DEV)).last_hidden_state
    h = hidden[hide.to(DEV)]
    logits = model.decoder(model.head(h)).float()
    return F.cross_entropy(logits, y[hide].to(DEV))


def evaluate(model, tok, held, batch):
    model.eval()
    leads = []
    with torch.no_grad():
        for i in range(0, len(held), batch):
            wins = [w for v in held[i:i + batch] for w in v[1]]
            s = scores(model, tok, wins).view(-1, 2)
            leads += (s[:, 0] - s[:, 1]).tolist()
    model.train()
    kinds = {}
    for (r, _), d in zip(held, leads):
        kinds.setdefault(r["kind"], []).append(d)
    return {k: {"n": len(v), "right": round(sum(d > 0 for d in v) / len(v), 3), "lead": round(sum(v) / len(v), 2)}
            for k, v in kinds.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--mlm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--held-max", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForMaskedLM.from_pretrained(BASE).to(DEV).train()
    rows = [json.loads(line) for line in open(a.pairs, encoding="utf-8")]
    groups = sorted({r["group"] for r in rows})
    random.Random(a.seed).shuffle(groups)
    held_groups = set(groups[: int(len(groups) * HELD)])
    train, held = [], []
    for r in rows:
        v = variants(tok, r)
        if v is not None:
            (held if r["group"] in held_groups else train).append((r, v))
    held = held[: a.held_max]
    print(json.dumps({"train": len(train), "held": len(held), "held_groups": len(held_groups)}), flush=True)
    print("step 0", json.dumps(evaluate(model, tok, held, a.batch), ensure_ascii=False), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = max(1, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (a.steps - s) / (a.steps - warm))))
    started, run = time.time(), [0.0, 0.0]
    for step in range(1, a.steps + 1):
        chunk = random.sample(train, a.batch)
        wins = [w for _, v in chunk for w in v]
        s = scores(model, tok, wins).view(-1, 2)
        rank = F.softplus(a.margin - (s[:, 0] - s[:, 1])).mean()
        mlm = mlm_loss(model, tok, [v[0] for _, v in chunk], gen) if a.mlm > 0 else torch.zeros((), device=DEV)
        loss = rank + a.mlm * mlm
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run[0] += rank.item()
        run[1] += mlm.item()
        if step % 100 == 0:
            print(f"step {step} rank {run[0] / 100:.3f} mlm {run[1] / 100:.3f} {time.time() - started:.0f}s", flush=True)
            run = [0.0, 0.0]
        if step % a.save_every == 0 or step == a.steps:
            path = os.path.join(a.out, f"step{step}")
            model.save_pretrained(path)
            tok.save_pretrained(path)
            print(f"step {step}", json.dumps(evaluate(model, tok, held, a.batch), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
