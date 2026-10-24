"""Experiment 1 of the 60% plan (Kotobae #98): lower the screening line (which words the scorer sees)
while the marks stay as strict as before. One run scored every candidate (coreml/screen-all.sh); for each
screening line the rows the app would not have scored are dropped, the two marks are re-chosen as
recalibrate.py chooses them (every false-mark limit of the check before #98, typo sets not below it,
right words marked at most 2), and the dev half's homophone recall is printed. The calibration half is
not read. At the shipped line 0.018 the run must give the shipped run's numbers.

usage (from asr/): python tools/screen_sweep.py <name> [lines, default 0.018,0.01,0.005,0.002,0]
"""
import collections
import json
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import clean_fm  # noqa: E402
import prod  # noqa: E402

half = json.load(open("split-calib.json"))
MODELS = os.path.expanduser("~/dvlp/kotobae/.wt/exp/screen-sweep/src-tauri/models")
FORCE_RATE = 0.2
voice = json.load(open(os.path.join(MODELS, "voice-confusions.json"), encoding="utf-8"))
FORCED = {w for w, e in voice.items() if e["wrong"] / max(e["wrong"] + e["kept"], 1) >= FORCE_RATE}


def screened(src, line):
    """The run as if words below the line had not been scored (forced words always are)."""
    P, D = src
    out = {}
    for key, rows in D.items():
        probes = P.get(key, [])
        keep = []
        for row in rows:
            s, e, written = row[0], row[1], row[2]
            sus = max((p for a, b, p, _ in probes if a < e and s < b), default=0.0)
            if written in FORCED or sus >= line:
                keep.append(row)
        out[key] = keep
    return P, out


def hom(src, det, score):
    c = collections.Counter()
    for r in prod.VAL:
        if half[r["path"]] == "dev":
            c.update(prod.clip_counts(r, prod.mk(src, r["path"], det, score)))
    s = prod.summary(c)
    return s["homophone_recall"], s["false_marks_per_1k_chars"]


def clean(src, det, score):
    return sum(len(prod.mk(src, k, det, score)) for k in clean_fm.TEXTS)


def scored_per_1k(src):
    P, D = src
    rs = [r for r in prod.VAL_CLEAN if half[r["path"]] == "dev"]
    return 1000 * sum(len(D.get(r["path"], [])) for r in rs) / sum(len(r["hyp"]) for r in rs)


def limits():
    """The false-mark limits and typo-set floors of the check before #98, as recalibrate.py sets them."""
    prod.GATE = False
    D0 = prod.load("eval-val", "d")
    base_sets = prod.sets(prod.load("..", "rust"), 0.4707, 5.13)
    base = {"kotobae": prod.kotobae_fm(prod.load("..", "rust"), 0.4707, 5.13),
            "clean": clean(prod.load("eval-clean", "d"), 0.4707, 5.13)}
    c = collections.Counter()
    for r in prod.VAL:
        if half[r["path"]] == "dev":
            c.update(prod.clip_counts(r, prod.mk(D0, r["path"], 0.4707, 5.13)))
    base["dev_fm"] = prod.summary(c)["false_marks_per_1k_chars"]
    prod.GATE = True
    return base, base_sets


DETS = [round(0.5 + 0.02 * i, 2) for i in range(21)]


def search(V, K, C, base, base_sets, scores=None, dets=None):
    """The two marks with the most dev-half homophones within every limit, as (recall, det, score, fm, clean, sets)."""
    best = None
    for det in dets or DETS:
        for score in scores or [round(5.0 + 0.25 * i, 2) for i in range(29)]:
            sets = prod.sets(K, det, score)
            rec, fm = hom(V, det, score)
            ok = (prod.kotobae_fm(K, det, score) <= max(1.0, base["kotobae"]) and clean(C, det, score) <= base["clean"]
                  and fm <= base["dev_fm"] and sets["right_marked"] <= 2
                  and all(sets[k] >= v for k, v in base_sets.items() if k != "right_marked"))
            if ok and (best is None or rec > best[0]):
                best = (rec, det, score, fm, clean(C, det, score), sets)
    return best


def show(best):
    return (f"best det {best[1]} score {best[2]}: dev {best[0]:.3f} fm {best[3]:.3f} clean {best[4]} {best[5]}"
            if best else "nothing within limits")


def main():
    name = ARGS[0]
    lines = [float(x) for x in (ARGS[1] if len(ARGS) > 1 else "0.018,0.01,0.005,0.002,0").split(",")]
    runs = {d: prod.load(d, name) for d in ("eval-dev", "..", "eval-clean")}
    base, base_sets = limits()
    print("limits:", {k: round(v, 3) for k, v in base.items()}, base_sets)
    ship = prod.load("eval-val", "ship3")
    print(f"shipped run at 0.68/6.75: dev {hom(ship, 0.68, 6.75)[0]:.3f}")
    for line in lines:
        V, K, C = (screened(runs[d], line) for d in ("eval-dev", "..", "eval-clean"))
        print(f"line {line}: scored words {scored_per_1k(V):.1f} per 1,000 chars (dev clean) | at 0.68/6.75 dev "
              f"{hom(V, 0.68, 6.75)[0]:.3f} | {show(search(V, K, C, base, base_sets))}", flush=True)


if __name__ == "__main__":
    main()
