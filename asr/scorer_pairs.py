"""Pairs for teaching the on-device check's scorer (the masked LM that compares a written word with each
alternative by pseudo-log-likelihood) which word belongs in the sentence (Kotobae #98).

- fix: a homophone Whisper wrote in place of what was said; the said word should outscore it.
- keep: a voice-table word used correctly; it should outscore the word voice input confuses it with,
  so that scoring every often-confused word does not mark correct text.

Both come from the training side only: Common Voice train clips and the Kokoro readings outside the
held-out lines (eval-kokoro). Each line: {"text", "s", "e", "good", "bad", "kind", "group"}, where the
sentence holds `text[s:e]` as written and `group` names the pair, so a held-out share can be cut by pair.

usage (from typo-study/asr): python scorer_pairs.py <out.jsonl> [--keep-per-fix 1.0] [--seed 0]
"""

import argparse
import collections
import glob
import json
import random
import re

import eval_spans

ENDS = "。！？\n"
FORCE_RATE = 0.2


def sentence(text: str, s: int, e: int) -> tuple[int, int]:
    lo = max((i + 1 for i in range(s) if text[i] in ENDS), default=0)
    hi = next((i + 1 for i in range(e, len(text)) if text[i] in ENDS), len(text))
    return lo, hi


def records(held: set[str]):
    for line in open("labels.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "train" and not r["jwtd"] and r["status"] != "quarantine":
            yield r["hyp"], r["edits"] or []
    for f in sorted(glob.glob("synth-kokoro/asr-*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in held:
                yield r["hyp"], r["edits"] or []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--keep-per-fix", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    held = {json.loads(line)["path"] for line in open("eval-kokoro/labels.jsonl", encoding="utf-8")}
    voice = json.load(open("voice-table/voice.json", encoding="utf-8"))
    forced = {w for w, e in voice.items() if e["wrong"] / max(1, e["wrong"] + e["kept"]) >= FORCE_RATE}
    finder = re.compile("|".join(map(re.escape, sorted(forced, key=len, reverse=True))))
    fixes, keeps, skipped = [], [], collections.Counter()
    for hyp, edits in records(held):
        _, at = eval_spans.norm_map(hyp)
        spans = []
        for e in edits:
            if e["hyp_chars"][1] <= e["hyp_chars"][0]:
                continue
            s, t = eval_spans.raw_span(at, *e["hyp_chars"])
            spans.append((s, t))
            if e["kind"] != "homophone":
                continue
            if hyp[s:t] != e["hyp"]:
                skipped["span"] += 1
                continue
            lo, hi = sentence(hyp, s, t)
            fixes.append({"text": hyp[lo:hi], "s": s - lo, "e": t - lo, "good": e["ref"], "bad": e["hyp"],
                          "kind": "fix", "group": f"{e['hyp']}>{e['ref']}"})
        # A forced word away from every error is taken as used correctly.
        for m in finder.finditer(hyp):
            s, t = m.span()
            if any(x < t and s < y for x, y in spans):
                continue
            lo, hi = sentence(hyp, s, t)
            w = m.group(0)
            keeps.append({"text": hyp[lo:hi], "s": s - lo, "e": t - lo, "good": w, "bad": voice[w]["to"][0],
                          "kind": "keep", "group": f"{w}>{voice[w]['to'][0]}"})
    rng.shuffle(keeps)
    keeps = keeps[: int(len(fixes) * a.keep_per_fix)]
    rows = fixes + keeps
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"fix": len(fixes), "keep": len(keeps), "skipped": dict(skipped),
                      "groups": len({r["group"] for r in rows})}))


if __name__ == "__main__":
    main()
