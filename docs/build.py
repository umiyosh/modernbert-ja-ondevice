"""Data for the progress page (docs/index.html) of the voice-input work (asr/): the candidates measured along
the way, the cross-fitted ledger of scorer versions, each scorer training run's curves, and the final
evaluation. Reads the local measurement outputs that asr/product/progress.py and dash_rows.py write, which
stay outside this repository, and the reports in asr/reports. Counts and rates only, never text.

usage: python docs/build.py <dash-candidates.json> <dash-crossfit.json> <progress-curves.csv> <progress.md>
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "asr", "reports")
SYSTEMS = {"C": "公開した検出器と採点役", "D": "#98 の前の版", "F": "学習なしの東北大 BERT で採点（藤井・新納 2021）",
           "B": "Jev（文節ごとに誤りを問う商用 API）", "A": "規則だけ（取り違え表と未知語）"}
KEEP = ("span_recall", "span_recall_ci95", "homophone_recall", "homophone_recall_ci95",
        "false_marks_per_1k_chars", "false_marks_per_1k_chars_ci95")


def final(name):
    r = json.load(open(os.path.join(REPORTS, name), encoding="utf-8"))
    rows = [{"key": k, "label": SYSTEMS[k], **{f: r["systems"][k][f] for f in KEEP},
             "homophones": r["systems"][k]["counts"]["homophone_spans"]} for k in SYSTEMS if k in r["systems"]]
    tests = {k: {f: v[f] for f in ("diff", "ci95", "p")} for k, v in r["tests"].items()}
    return {"clips": r["clips"], "groups": r["speakers"], "systems": rows, "tests": tests}


def curves(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        run = out.setdefault(r["run"], {"loss": [], "held": []})
        num = {k: float(v) for k, v in r.items() if k not in ("run", "kind") and v not in ("", None)}
        run["loss" if r["kind"] == "loss" else "held"].append(num)
    return out


def runs(path):
    """The per-run table of progress.md: steps, held-out keep and fix rates first and last, last rank loss."""
    text = open(path, encoding="utf-8").read()
    part = text[text.index("## 学習ごとの取り分けた例"):]
    rows = []
    for m in re.finditer(r"^\| (run\d+) \| (\d+) \| ([\d.]+) → ([\d.]+) \| ([\d.]+) → ([\d.]+) \| ([\d.]+) \|$", part, re.M):
        run, steps, k0, k1, f0, f1, loss = m.groups()
        rows.append({"run": run, "steps": int(steps), "keep": [float(k0), float(k1)],
                     "fix": [float(f0), float(f1)], "rank_loss": float(loss)})
    return rows


def main():
    cand, cross, curve, md = sys.argv[1:5]
    data = {
        "final_test": final("final-test.json"),
        "final_fleurs": final("final-fleurs.json"),
        "candidates": json.load(open(cand, encoding="utf-8")),
        "crossfit": json.load(open(cross, encoding="utf-8")),
        "curves": curves(curve),
        "runs": runs(md),
    }
    with open(os.path.join(HERE, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print({k: len(v) if isinstance(v, (list, dict)) else v for k, v in data.items()})


if __name__ == "__main__":
    main()
