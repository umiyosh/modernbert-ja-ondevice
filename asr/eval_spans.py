"""Score a system's marks on labelled clips with the metrics of PROTOCOL.md.

Marks come from the files Kotobae's `typo_eval` test writes (res-<name>det-r0.json: the detector's
P(typo) per token; deltas-<name>.json: each candidate's lead over the written word) and are made as
Kotobae's `typo::report` makes them, at any thresholds. Error spans are the Whisper characters of
homophone, mishear and extra edits; a span is found when a mark overlaps it. A drop is found when a
mark touches a character next to the gap. False marks are counted on clean clips per 1,000
characters. The 95% CI of each recall is a cluster bootstrap over speakers.

usage: python asr/eval_spans.py <cv ja dir> <labels.jsonl> <split> <eval dir> <name> <report.json>
                                [detector threshold] [score threshold] [--with-jwtd]
"""

import collections
import json
import os
import sys
import time

import numpy as np

from classify import norm_map
from near_dup import read

DETECTOR_MARK = 0.4707  # Kotobae's shipped thresholds (src-tauri/src/services/typo.rs)
SCORE_MARK = 5.13
SPAN_KINDS = {"homophone", "mishear", "extra"}
RESAMPLES = 10000


def marks(probes: list, deltas: list, det: float, score: float) -> list[tuple[int, int]]:
    # Same as typo::report: candidates with a lead >= score, then runs of neighbouring detector tokens
    # >= det that no candidate mark overlaps.
    found = [(s, e) for s, e, _, alts in deltas if alts and max(v for _, v in alts) >= score]
    runs: list[list[int]] = []
    for s, e, p, _ in sorted(probes, key=lambda x: x[0]):
        if p < det:
            continue
        if runs and s <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])
    found += [(s, e) for s, e in runs if not any(a < e and s < b for a, b in found)]
    return sorted(found)


def raw_span(at: list[int], a: int, b: int) -> tuple[int, int]:
    return at[a], at[b - 1] + 1


def clip_counts(rec: dict, ms: list[tuple[int, int]]) -> dict:
    hits = lambda s, e: any(a < e and s < b for a, b in ms)  # noqa: E731
    c = collections.Counter()
    if rec["status"] == "clean":
        c["clean_chars"] += len(rec["hyp"])
        c["false_marks"] += len(ms)
        return c
    _, at = norm_map(rec["hyp"])
    spans = []
    for e in rec["edits"]:
        a, b = e["hyp_chars"]
        if e["kind"] in SPAN_KINDS and b > a:
            s, t = raw_span(at, a, b)
            spans.append((s, t))
            c["spans"] += 1
            c["found"] += hits(s, t)
            if e["kind"] == "homophone":
                c["homophone_spans"] += 1
                c["homophone_found"] += hits(s, t)
        elif e["kind"] == "drop":
            near = [at[i] for i in (a - 1, a) if 0 <= i < len(at)]
            c["drops"] += 1
            c["drops_found"] += any(hits(x, x + 1) for x in near)
    c["marks_on_error_clips"] += len(ms)
    c["marks_on_spans"] += sum(any(a < t and s < b for s, t in spans) for a, b in ms)
    c["error_clips"] += 1
    c["error_clips_marked"] += bool(ms)
    return c


def rate(c: collections.Counter, num: str, den: str) -> float | None:
    return c[num] / c[den] if c[den] else None


def summary(c: collections.Counter) -> dict:
    return {
        "span_recall": rate(c, "found", "spans"),
        "homophone_recall": rate(c, "homophone_found", "homophone_spans"),
        "drop_recall": rate(c, "drops_found", "drops"),
        "span_precision": rate(c, "marks_on_spans", "marks_on_error_clips"),
        "false_marks_per_1k_chars": 1000 * c["false_marks"] / c["clean_chars"] if c["clean_chars"] else None,
        "error_clips_marked": rate(c, "error_clips_marked", "error_clips"),
    }


def bootstrap(per_speaker: dict[str, collections.Counter], keys: list[tuple[str, str, str]]) -> dict:
    fields = sorted({x for _, num, den in keys for x in (num, den)})
    m = np.array([[g[x] for x in fields] for g in per_speaker.values()], dtype=np.float64)
    rng = np.random.default_rng(0)
    sums = np.stack([m[rng.integers(0, len(m), len(m))].sum(0) for _ in range(RESAMPLES)])
    out = {}
    for k, num, den in keys:
        n, d = sums[:, fields.index(num)], sums[:, fields.index(den)]
        r = n[d > 0] / d[d > 0]
        out[k] = [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))] if len(r) else None
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root, labels, split, eval_dir, name, out = args[:6]
    det = float(args[6]) if len(args) > 6 else DETECTOR_MARK
    score = float(args[7]) if len(args) > 7 else SCORE_MARK
    with_jwtd = "--with-jwtd" in sys.argv
    speaker = {r["path"]: r["client_id"] for r in read(os.path.join(root, "validated.tsv"))}
    probes = json.load(open(os.path.join(eval_dir, f"res-{name}det-r0.json"), encoding="utf-8"))
    deltas = json.load(open(os.path.join(eval_dir, f"deltas-{name}.json"), encoding="utf-8"))

    total, jw = collections.Counter(), collections.Counter()
    per_speaker = collections.defaultdict(collections.Counter)
    with open(labels, encoding="utf-8") as f:
        for rec in map(json.loads, f):
            if rec["split"] != split or rec["status"] == "quarantine" or rec["path"] not in probes:
                continue
            c = clip_counts(rec, marks(probes[rec["path"]], deltas[rec["path"]], det, score))
            if rec["jwtd"] and not with_jwtd:
                jw.update(c)
                continue
            total.update(c)
            per_speaker[speaker[rec["path"]]].update(c)

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": name,
        "split": split,
        "thresholds": {"detector": det, "score": score},
        "jwtd_overlap": "included" if with_jwtd else "excluded (reported separately)",
        "counts": dict(total),
        "speakers": len(per_speaker),
        "metrics": summary(total),
        "ci95_speaker_bootstrap": bootstrap(
            per_speaker,
            [("span_recall", "found", "spans"), ("homophone_recall", "homophone_found", "homophone_spans"),
             ("false_marks_per_char", "false_marks", "clean_chars")],
        ),
        "jwtd_overlap_only": {"counts": dict(jw), "metrics": summary(jw)},
    }
    ci = report["ci95_speaker_bootstrap"]
    if ci.get("false_marks_per_char"):
        ci["false_marks_per_1k_chars"] = [1000 * x for x in ci.pop("false_marks_per_char")]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
