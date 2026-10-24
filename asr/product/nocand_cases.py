"""The dev half's real-voice homophones that one system misses with the right word in no candidate row
(as tools/nocand.py finds them), shared by the candidate-stage probes (Kotobae #98, paper plan step 4)."""
import json

import prod
import screen_sweep as s
import verify_sweep as v

es = prod.eval_spans


def cases(name="sb0s", line=0.1, w=1.0, det=0.96, score=9.5):
    """(record, written text, start, end, edit) for each such homophone."""
    half = json.load(open("split-calib.json"))
    src = prod.load("eval-dev", name)
    if w:
        src = v.verified(src, json.load(open(f"eval-dev/verify-{name}.json", encoding="utf-8")), w)
    src = s.screened(src, line)
    cands = json.load(open(f"eval-dev/cands-{name}.json", encoding="utf-8"))
    for r in prod.VAL:
        if half[r["path"]] != "dev" or r["status"] == "clean":
            continue
        text = r["hyp"]
        _, at = es.norm_map(text)
        ms = prod.mk(src, r["path"], det, score)
        for e in r["edits"] or []:
            x, y = e["hyp_chars"]
            if e["kind"] != "homophone" or y <= x:
                continue
            a, b = es.raw_span(at, x, y)
            if any(p < b and a < q for p, q in ms):
                continue
            right = text[:a] + e["ref"] + text[b:]
            if any(text[:p] + alt + text[q:] == right for p, q, alts in cands.get(r["path"], []) for alt in alts):
                continue
            yield r, text, a, b, e
