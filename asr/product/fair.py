"""The dev half measured the way prior work reports (Kotobae #98, comparing with the literature in
lit/2026-10-23-prior-results.md). Prior work picks its threshold for best F1 or at a sentence false-positive
rate several times ours, and counts sentences, not spans. So for one system the two marks are swept and,
at each matched false-alarm level, the operating point with the most homophones found is reported with:
span recall and precision, sentence-level detection (CSC style) on all error clips and on clips whose only
errors are homophones, the sentence false-positive rate, balanced accuracy (what a set with as many correct
as erroneous sentences would give), and homophone recall split by whether the pair was seen in training.
Only Kotobae's product limits are dropped here; the false-alarm level is the only constraint.

usage (from asr/): python tools/fair.py [system ...]   (systems: D ship3 free marks trial)
"""
import collections
import json
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import marks_sweep as m  # noqa: E402
import prod  # noqa: E402
import screen_sweep as s  # noqa: E402
import verify_sweep as v  # noqa: E402

es = prod.eval_spans
half = json.load(open("split-calib.json"))
RS = [r for r in prod.VAL if half[r["path"]] == "dev"]
SEEN = {(e["hyp"], e["ref"]) for r in map(json.loads, open("labels.jsonl", encoding="utf-8"))
        if r["split"] == "train" for e in r["edits"] or [] if e["kind"] == "homophone"}
# Matched levels: ours, LEMON's average (ReLM+DISC), about RED-ACE, about SoftCorrect (per 1,000 chars),
# and the sentence FPRs of LEMON's average, Yu+ on SIGHAN15 and the small/large mix on CSCD-NS.
FM = (1.17, 1.27, 2.0, 3.0, 4.6)
FPR = (0.019, 0.045, 0.077, 0.109)


def run(d, name, line=None, w=0.0):
    src = prod.load(d, name)
    if w:
        src = v.verified(src, json.load(open(os.path.join(d, f"verify-{name}.json"), encoding="utf-8")), w)
    return s.screened(src, line) if line is not None else src


SYSTEMS = {
    "D": (lambda: run("eval-val", "d"), False),
    "ship3": (lambda: run("eval-val", "ship3"), True),
    "free": (lambda: run("eval-dev", "sb0", 0.005, 1.0), True),
    "marks": (lambda: run("eval-dev", "sb0s", 0.1, 1.0), True),
    "trial": (lambda: run("eval-dev", "sb0kk", 0.05), True),
    "night": (lambda: run("eval-dev", "sb0kn", 0.1, 1.0), True),
    **{f"soup{p}": ((lambda p=p: run("eval-dev", f"sb0m{p}", 0.05, 1.0)), True) for p in (2, 3, 4)},
}


def gold(r):
    _, at = es.norm_map(r["hyp"])
    return [(*es.raw_span(at, *e["hyp_chars"]), e) for e in r["edits"]
            if e["kind"] in es.SPAN_KINDS and e["hyp_chars"][1] > e["hyp_chars"][0]]


GOLD = {r["path"]: gold(r) for r in RS if r["status"] != "clean"}
HOM_ONLY = {k for k, g in GOLD.items() if g and all(e["kind"] == "homophone" for *_, e in g)}


def measure(src, det, score):
    c = collections.Counter()
    for r in RS:
        ms = prod.mk(src, r["path"], det, score)
        if r["status"] == "clean":
            c["clean"] += 1
            c["chars"] += len(r["hyp"])
            c["fm"] += len(ms)
            c["fp_clips"] += bool(ms)
            continue
        g = GOLD[r["path"]]
        hit = lambda a, b: any(x < b and a < y for x, y in ms)  # noqa: E731
        on = [any(a < t and x < b for x, t, _ in g) for a, b in ms]
        c["spans"] += len(g)
        c["found"] += sum(hit(a, b) for a, b, _ in g)
        for a, b, e in g:
            if e["kind"] == "homophone":
                k = "seen" if (e["hyp"], e["ref"]) in SEEN else "unseen"
                c["hom"], c["hom_" + k] = c["hom"] + 1, c["hom_" + k] + 1
                c["homf"] += hit(a, b)
                c["homf_" + k] += hit(a, b)
        c["marks"] += len(ms)
        c["marks_on"] += sum(on)
        loose = any(on)
        strict = bool(ms) and all(on) and all(hit(a, b) for a, b, _ in g)
        for tag in ("all", "hom") if r["path"] in HOM_ONLY else ("all",):
            c[f"{tag}_err"] += 1
            c[f"{tag}_flag"] += bool(ms)
            c[f"{tag}_loose"] += loose
            c[f"{tag}_strict"] += strict
    return c


def report(c):
    fpr = c["fp_clips"] / c["clean"]
    out = {"fm": 1000 * c["fm"] / c["chars"], "fpr": fpr, "hom": c["homf"] / c["hom"], "all": c["found"] / c["spans"],
           "prec": c["marks_on"] / (c["marks"] + c["fm"]),
           "hom_seen": c["homf_seen"] / c["hom_seen"], "hom_unseen": c["homf_unseen"] / c["hom_unseen"]}
    for tag in ("all", "hom"):
        for kind in ("loose", "strict"):
            rec = c[f"{tag}_{kind}"] / c[f"{tag}_err"]
            prec = c[f"{tag}_{kind}"] / (c[f"{tag}_flag"] + c["fp_clips"])
            out[f"s_{tag}_{kind}"] = (prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0, (rec + 1 - fpr) / 2)
    return out


def best(points, key, limit, value="hom"):
    ok = [p for p in points if p[key] <= limit]
    return max(ok, key=lambda p: (p[value], -p[key])) if ok else None


def pct(x):
    return f"{100 * x:5.1f}"


def line(tag, p):
    if p is None:
        return f"  {tag}: nothing"
    sa, sh = p["s_all_strict"], p["s_hom_strict"]
    la, lh = p["s_all_loose"], p["s_hom_loose"]
    return (f"  {tag} det {p['det']} score {p['score']}: fm {p['fm']:.2f}/1k fpr {pct(p['fpr'])} | hom {pct(p['hom'])} "
            f"(seen {pct(p['hom_seen'])} unseen {pct(p['hom_unseen'])}) all {pct(p['all'])} prec {pct(p['prec'])} | "
            f"sent-hom strict P/R/F1/bal {'/'.join(pct(x) for x in sh)} loose {'/'.join(pct(x) for x in lh)} | "
            f"sent-all strict {'/'.join(pct(x) for x in sa)} loose {'/'.join(pct(x) for x in la)}")


def main():
    names = ARGS or ["marks"]
    n_hom = sum(1 for k in HOM_ONLY)
    print(f"dev clips {len(RS)}, error clips {len(GOLD)}, homophone-only error clips {n_hom}, "
          f"clean {sum(r['status'] == 'clean' for r in RS)}", flush=True)
    for name in names:
        load, gate = SYSTEMS[name]
        src = load()
        prod.GATE = gate
        points = []
        for det in m.DETS:
            for score in m.SCORES:
                points.append({"det": det, "score": score, **report(measure(src, det, score))})
        prod.GATE = True
        print(name, flush=True)
        for lim in FM:
            print(line(f"fm<={lim}", best(points, "fm", lim)), flush=True)
        for lim in FPR:
            print(line(f"fpr<={lim}", best(points, "fpr", lim)), flush=True)
        f1 = max(points, key=lambda p: p["s_hom_strict"][2])
        print(line("best sentence F1 (homophone set)", f1), flush=True)
        front = sorted((p for p in points), key=lambda p: p["fm"])
        curve, top = [], -1.0
        for p in front:
            if p["hom"] > top:
                top = p["hom"]
                curve.append(f"{p['fm']:.2f}:{pct(p['hom']).strip()}/{pct(p['prec']).strip()}")
        print("  curve fm:hom/prec " + " ".join(curve[:: max(1, len(curve) // 25)]), flush=True)


if __name__ == "__main__":
    main()
