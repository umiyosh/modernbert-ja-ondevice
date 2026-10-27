"""The final evaluation on test (PROTOCOL.md, amendment of 2026-10-27), run once after every choice was
fixed. Each system at its fixed point: C (the shipped check, screen 0.05, kana/kanji gate, detector
0.995 / score 7.0), D (Kotobae before #98, 0.4707 / 5.13, as shipped), A (the rule spans), B (Jev's
errors measure), and F when its run exists. Metrics of eval_spans on clips neither quarantined nor in
JWTD train; JWTD clips reported apart. 95% intervals: cluster bootstrap over speakers (10,000, seed 0);
C against each other system: sign-flip permutation within speakers (10,000, seed 0). The primary test
is C − D in span recall; the secondary ones (C − D in homophone recall and false marks) are Holm-adjusted.

usage (from asr/): python tools/final_eval.py <labels-test.jsonl> <eval dir> <report.json>
"""
import collections
import csv
import json
import os
import sys

import numpy as np

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import learned  # noqa: E402
import prod  # noqa: E402
import screen_sweep as ss  # noqa: E402

es = prod.eval_spans
N = 10_000
# The runs of one set are named <system><tag>: tag "test" for Common Voice test, "fl" for the FLEURS
# holdout (split "external", resampled by its sentence id). FINAL_C / FINAL_D only check this script on val.
TAG = os.environ.get("FINAL_TAG", "test")
SPLIT = os.environ.get("FINAL_SPLIT", "test")
C_RUN, D_RUN = os.environ.get("FINAL_C", f"m45{TAG}"), os.environ.get("FINAL_D", f"d{TAG}")
KEYS = [("span_recall", "found", "spans"), ("homophone_recall", "homophone_found", "homophone_spans"),
        ("false_marks_per_1k_chars", "false_marks", "clean_chars"), ("span_precision", "marks_on_spans", "marks_on_error_clips")]


def ratio(sums, num, den, fields, scale):
    """sums: (..., fields) totals over speakers"""
    return scale * sums[..., fields.index(num)] / np.maximum(sums[..., fields.index(den)], 1e-12)


def resamples(n_spk):
    """How many times each speaker is drawn in each bootstrap resample (N x speakers), and the sign-flip
    masks (N x speakers), so totals are one matrix product instead of an N x speakers x fields array."""
    rng = np.random.default_rng(0)
    draws = np.stack([np.bincount(rng.integers(0, n_spk, n_spk), minlength=n_spk) for _ in range(N)]).astype(float)
    flips = rng.integers(0, 2, (N, n_spk)).astype(float)
    return draws, flips


def systems(d):
    """name -> path -> marks"""
    out = {}
    c = prod.load(d, C_RUN)
    if c:
        c = ss.screened(c, 0.05)
        out["C"] = lambda p: prod.mk(c, p, 0.995, 7.0)
    dd = prod.load(d, D_RUN)
    if dd:
        out["D"] = lambda p: es.marks(dd[0].get(p, []), dd[1].get(p, []), 0.4707, 5.13)
    rules = os.path.join(d, f"rules-{D_RUN}.json")
    if os.path.exists(rules):
        a = json.load(open(rules, encoding="utf-8"))
        out["A"] = lambda p: [tuple(x) for x in a.get(p, [])]
    jev = os.path.join(d, f"jev-{TAG}.jsonl")
    if os.path.exists(jev):
        b = {r["id"]: r for r in map(json.loads, open(jev, encoding="utf-8"))}
        out["B"] = lambda p: [tuple(x) for x in b[p]["marks"]] if p in b and "marks" in b[p] else None
    f = prod.load(d, f"fs{TAG}")
    if f:
        f = ss.screened(f, 0.05)
        point = json.load(open(os.path.join(d, f"fs{TAG}-point.json")))
        out["F"] = lambda p: prod.mk(f, p, point["det"], point["score"])
    return out


def main():
    labels, d, report_path = ARGS[:3]
    cv = {r["path"]: r["client_id"] for r in csv.DictReader(open(learned.CV, encoding="utf-8"), delimiter="\t")}
    recs = [r for r in map(json.loads, open(labels, encoding="utf-8")) if r["split"] == SPLIT and r["status"] != "quarantine"]
    prod.TEXT.update({r["path"]: r["hyp"] for r in recs})
    who = {r["path"]: r.get("speaker") or cv[r["path"]] for r in recs}
    main_recs = [r for r in recs if not r["jwtd"]]
    jw_recs = [r for r in recs if r["jwtd"]]
    sysm = systems(d)
    fields = sorted({x for _, n, dd in KEYS for x in (n, dd)})
    spk = sorted({who[r["path"]] for r in main_recs})
    at = {s: i for i, s in enumerate(spk)}
    mats, missing = {}, {}
    draws, flips = resamples(len(spk))
    report = {"systems": {}, "clips": len(main_recs), "speakers": len(spk), "jwtd_clips": len(jw_recs)}
    for name, mk in sysm.items():
        m = np.zeros((len(spk), len(fields)))
        miss = 0
        jw = collections.Counter()
        for r in main_recs:
            ms = mk(r["path"])
            if ms is None:
                miss += 1
                continue
            c = es.clip_counts(r, ms)
            m[at[who[r["path"]]]] += [c[x] for x in fields]
        for r in jw_recs:
            ms = mk(r["path"])
            if ms is not None:
                jw.update(es.clip_counts(r, ms))
        mats[name], missing[name] = m, miss
        boot = draws @ m
        row = {"unscored_clips": miss, "counts": {x: int(m[:, i].sum()) for i, x in enumerate(fields)}}
        for k, num, den in KEYS:
            scale = 1000 if k.startswith("false") else 1
            row[k] = float(ratio(m.sum(0), num, den, fields, scale))
            row[k + "_ci95"] = [float(x) for x in np.percentile(ratio(boot, num, den, fields, scale), [2.5, 97.5])]
        row["jwtd_only"] = es.summary(jw)
        report["systems"][name] = row

    def paired(a, b, num, den, scale):
        A, B = mats[a], mats[b]
        obs = ratio(B.sum(0), num, den, fields, scale) - ratio(A.sum(0), num, den, fields, scale)
        diff = ratio(draws @ B, num, den, fields, scale) - ratio(draws @ A, num, den, fields, scale)
        swapped_b = flips @ A + (1 - flips) @ B
        swapped_a = flips @ B + (1 - flips) @ A
        perm = ratio(swapped_b, num, den, fields, scale) - ratio(swapped_a, num, den, fields, scale)
        p = ((np.abs(perm) >= abs(obs) - 1e-12).sum() + 1) / (N + 1)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        return {"diff": float(obs), "ci95": [float(lo), float(hi)], "p": float(p)}

    tests = {}
    if "C" in mats and "D" in mats:
        tests["primary C-D span_recall"] = paired("D", "C", "found", "spans", 1)
        sec = {"C-D homophone_recall": paired("D", "C", "homophone_found", "homophone_spans", 1),
               "C-D false_marks_per_1k_chars": paired("D", "C", "false_marks", "clean_chars", 1000)}
        order = sorted(sec, key=lambda k: sec[k]["p"])
        run = 0.0
        for i, k in enumerate(order):
            run = max(run, min(1.0, sec[k]["p"] * (len(order) - i)))
            sec[k]["p_holm"] = run
        tests.update(sec)
    for other in ("A", "B", "F"):
        if "C" in mats and other in mats:
            tests[f"C-{other} homophone_recall"] = paired(other, "C", "homophone_found", "homophone_spans", 1)
    report["tests"] = tests
    if "C" in report["systems"]:
        c = report["systems"]["C"]
        ok_fm = c["false_marks_per_1k_chars"] <= 1.0
        report["kotobae98_goal"] = {
            "homophone_recall": c["homophone_recall"],
            "ci95": c["homophone_recall_ci95"],
            "false_marks_per_1k_chars": c["false_marks_per_1k_chars"],
            "reached": bool(c["homophone_recall"] >= 0.60 and ok_fm),
            "shown": bool(c["homophone_recall_ci95"][0] >= 0.60 and ok_fm),
        }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
