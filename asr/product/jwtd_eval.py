"""The public comparison set of the paper plan (Kotobae #98, step 2): the JWTD v2 test sentences with one
kanji misconversion, measured with the app's pipeline next to JWTD's own results (Tanaka+ 2020, 2021).
Each erroneous sentence is a text with its gold span; its corrected sentence is a correct text, for the
false marks per 1,000 characters (JWTD's test has no correct sentences of its own).

A sentence is left out when it shares a 15-character run with Kotobae's evaluation texts (the typo sets
were drawn from this test split and chose our thresholds). Sharing a run with JWTD train, which every
training set is built from, or with an evaluation folder is flagged, not dropped, and counted, so the
paper can report both. A pair (written, right) that JWTD train also corrects is flagged as seen.

usage (from asr/): python tools/jwtd_eval.py <jwtd dir> <out dir>
writes <out dir>/texts.json (the app's evaluation input) and <out dir>/labels.json
"""
import glob
import json
import os
import sys

CATS = {"kanji-conversion_a": "homophone", "kanji-conversion_b": "other-kanji"}
K = 15


def runs(text):
    return {text[i:i + K] for i in range(len(text) - K + 1)}


def span(pre, post, wrong, right):
    """Where the recorded word sits: the occurrence of wrong whose replacement by right gives post. JWTD
    records the edit by word (解放→開放), so a character diff would mark only the changed character."""
    at = [k for k in range(len(pre)) if pre.startswith(wrong, k) and pre[:k] + right + pre[k + len(wrong):] == post]
    return (at[0], at[0] + len(wrong)) if len(at) == 1 and wrong else None


def texts_of(path):
    data = json.load(open(path, encoding="utf-8"))
    return [t["text"] for t in data] if isinstance(data, list) else list(data.values())


def main():
    src, out = sys.argv[1], sys.argv[2]
    cases = []
    for i, line in enumerate(open(os.path.join(src, "test.jsonl"), encoding="utf-8")):
        d = json.loads(line)
        if len(d["diffs"]) != 1 or d["diffs"][0]["category"] not in CATS:
            continue
        x = d["diffs"][0]
        sp = span(d["pre_text"], d["post_text"], x["pre_str"], x["post_str"])
        if sp is None:
            continue
        cases.append({"i": i, "kind": CATS[x["category"]], "pre": d["pre_text"], "post": d["post_text"],
                      "span": sp, "pair": (x["pre_str"], x["post_str"])})
    print(f"single-block cases: {len(cases)}", flush=True)

    here = os.path.dirname(os.path.abspath("labels.jsonl"))
    kotobae = set().union(*(runs(t) for t in texts_of(os.path.join(here, "..", "texts.json"))))
    evals = {}
    for p in glob.glob(os.path.join(here, "eval-*", "texts.json")):
        if os.path.abspath(os.path.dirname(p)) != os.path.abspath(out):
            evals[os.path.basename(os.path.dirname(p))] = set().union(*(runs(t) for t in texts_of(p)))
    need = set().union(*(runs(c["pre"]) | runs(c["post"]) for c in cases))
    in_train, pairs = set(), set()
    for line in open(os.path.join(src, "train.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        for t in (d["pre_text"], d["post_text"]):
            in_train |= runs(t) & need
        pairs.update((x["pre_str"], x["post_str"]) for x in d["diffs"])

    texts, labels, dropped = [], {}, 0
    counts = {"train_text": 0, "seen_pair": 0, **{f"eval:{k}": 0 for k in evals}}
    for c in cases:
        r = runs(c["pre"]) | runs(c["post"])
        if r & kotobae:
            dropped += 1
            continue
        flags = {"train_text": bool(r & in_train), "seen_pair": c["pair"] in pairs,
                 "evals": sorted(k for k, s in evals.items() if r & s)}
        counts["train_text"] += flags["train_text"]
        counts["seen_pair"] += flags["seen_pair"]
        for k in flags["evals"]:
            counts[f"eval:{k}"] += 1
        err, ok = f"jwtd{c['i']}~err", f"jwtd{c['i']}~ok"
        texts += [{"id": err, "text": c["pre"]}, {"id": ok, "text": c["post"]}]
        labels[err] = {"kind": c["kind"], "span": c["span"], "ok": ok, **flags}
    os.makedirs(out, exist_ok=True)
    json.dump(texts, open(os.path.join(out, "texts.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(labels, open(os.path.join(out, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False)
    kinds = {k: sum(v["kind"] == k for v in labels.values()) for k in CATS.values()}
    chars = sum(len(t["text"]) for t in texts)
    print(f"dropped (Kotobae texts): {dropped}; kept {len(labels)} {kinds}, {chars} chars; flagged {counts}", flush=True)


if __name__ == "__main__":
    main()
