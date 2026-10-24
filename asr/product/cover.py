"""Experiment 2 of the 60% plan (Kotobae #98): how often the right word of a real-voice homophone is among
the candidates, per candidate switch, on the dev half only, and what each switch costs in candidates on
correct clips (more alternatives to score, more chances of a false mark).

usage (from asr/): python tools/cover.py <base name> <name> ...   (runs in eval-dev)
"""
import json
import os
import sys

sys.argv, ARGS = [sys.argv[0], "--gate"], sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402

es = prod.eval_spans
half = json.load(open("split-calib.json"))


def offered(cands, r):
    """The dev clip's homophone errors, each with whether some candidate row turns it right."""
    text = r["hyp"]
    _, at = es.norm_map(text)
    out = []
    for e in r["edits"] or []:
        x, y = e["hyp_chars"]
        if e["kind"] != "homophone" or y <= x:
            continue
        a, b = es.raw_span(at, x, y)
        right = text[:a] + e["ref"] + text[b:]
        out.append(any(text[:s] + alt + text[t:] == right for s, t, alts in cands.get(r["path"], []) for alt in alts))
    return out


def main():
    rs = [r for r in prod.VAL if half[r["path"]] == "dev"]
    clean = [r for r in rs if r["status"] == "clean"]
    chars = sum(len(r["hyp"]) for r in clean)
    load = lambda n: json.load(open(f"eval-dev/cands-{n}.json", encoding="utf-8"))  # noqa: E731
    base = load(ARGS[0])
    base_hits = [h for r in rs for h in offered(base, r)]
    for name in ARGS:
        c = load(name)
        hits = [h for r in rs for h in offered(c, r)]
        gained = sum(1 for b0, h in zip(base_hits, hits) if h and not b0)
        lost = sum(1 for b0, h in zip(base_hits, hits) if b0 and not h)
        rows = sum(len(c.get(r["path"], [])) for r in clean)
        alts = sum(len(a) for r in clean for _, _, a in c.get(r["path"], []))
        print(f"{name:8s} right word offered {sum(hits)}/{len(hits)} ({100 * sum(hits) / len(hits):.1f}%) "
              f"gained {gained} lost {lost} | clean: rows {1000 * rows / chars:.1f}, alternatives {1000 * alts / chars:.1f} per 1,000 chars")


if __name__ == "__main__":
    main()
