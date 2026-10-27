"""Where each real voice-input homophone on val is lost (Kotobae #98, step 2a): every error falls in one
stage, in pipeline order, for one run's outputs (cands-/deltas-/res- of <name>) at its thresholds.

  found by detector / found by score    marked (the gated marks prod.py counts)
  no candidate                          no candidate row turns the written text into the right one
  not scored                            the right candidate exists, but the row was screened out
  scorer prefers written                its lead <= 0
  lead short of the mark                0 < lead < score threshold
  other                                 lead clears the mark yet nothing is marked (the gate)

Pairs are split into seen (the written>right pair occurs anywhere the models learned from: Common Voice
train, every Kokoro reading outside the held-out lines, the JWTD misconversion table) and unseen.

usage (from asr/): python tools/stages.py <name> <det> <score>
"""

import collections
import glob
import json
import os
import sys

sys.argv, ARGS = [sys.argv[0], "--gate"], sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402

es = prod.eval_spans
MODELS = os.path.expanduser("~/dvlp/kotobae/src-tauri/models")


def learned_pairs() -> set[tuple[str, str]]:
    pairs = set()
    for line in open("labels.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "train":
            pairs.update((e["hyp"], e["ref"]) for e in r["edits"] or [] if e["kind"] == "homophone")
    held = {json.loads(line)["path"] for line in open("eval-kokoro/labels.jsonl", encoding="utf-8")}
    for f in glob.glob("synth-kokoro/asr-*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in held:
                pairs.update((e["hyp"], e["ref"]) for e in r["edits"] or [] if e.get("kind") == "homophone")
    for w, e in json.load(open(os.path.join(MODELS, "confusions.json"), encoding="utf-8")).items():
        pairs.update((w, t) for t in e["to"])
    return pairs


def main() -> None:
    name, det, score = ARGS[0], float(ARGS[1]), float(ARGS[2])
    src = prod.load("eval-val", name)
    cands = json.load(open(f"eval-val/cands-{name}.json", encoding="utf-8"))
    deltas = src[1]
    seen_pairs = learned_pairs()
    out = collections.Counter()
    for r in prod.VAL:
        if r["status"] == "clean":
            continue
        text = r["hyp"]
        _, at = es.norm_map(text)
        marks = prod.mk(src, r["path"], det, score)
        probes = src[0].get(r["path"], [])
        for e in r["edits"] or []:
            x, y = e["hyp_chars"]
            if e["kind"] != "homophone" or y <= x:
                continue
            a, b = es.raw_span(at, x, y)
            right = text[:a] + e["ref"] + text[b:]
            group = "seen" if (e["hyp"], e["ref"]) in seen_pairs else "unseen"
            out[(group, "all")] += 1
            if any(s < b and a < t for s, t in marks):
                by_det = any(s < b and a < t and p >= det for s, t, p, _ in probes)
                out[(group, "found by detector" if by_det else "found by score")] += 1
                continue
            fits = lambda s, t, alt: text[:s] + alt + text[t:] == right  # noqa: E731
            cand = any(fits(s, t, alt) for s, t, alts in cands.get(r["path"], []) for alt in alts)
            leads = [v for s, t, _, alts in deltas.get(r["path"], []) for alt, v in alts if fits(s, t, alt)]
            if not cand:
                stage = "no candidate"
            elif not leads:
                stage = "not scored"
            elif max(leads) <= 0:
                stage = "scorer prefers written"
            elif max(leads) < score:
                stage = "lead short of the mark"
            else:
                stage = "other"
            out[(group, stage)] += 1
    stages = ["found by detector", "found by score", "no candidate", "not scored", "scorer prefers written",
              "lead short of the mark", "other"]
    for group in ("seen", "unseen"):
        n = out[(group, "all")]
        print(f"{group}: {n}")
        for st in stages:
            print(f"  {st:24s} {out[(group, st)]:4d}  {out[(group, st)] / max(1, n):.1%}")
    n = out[("seen", "all")] + out[("unseen", "all")]
    print(f"all: {n}")
    for st in stages:
        k = out[("seen", st)] + out[("unseen", st)]
        print(f"  {st:24s} {k:4d}  {k / n:.1%}")


if __name__ == "__main__":
    main()
