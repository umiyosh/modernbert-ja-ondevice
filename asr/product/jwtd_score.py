"""Scores the JWTD test set (tools/jwtd_eval.py) at operating points chosen on the dev half, never on JWTD
(Kotobae #98, paper plan step 2). Reported per point: misconversion recall (all; without texts that share a
run with JWTD train or a scorer training set; pairs JWTD train does and does not correct), recall of kanji
misconversions with another reading, false marks per 1,000 characters of the corrected sentences, sentence
detection (strict: every mark on the error; loose: some mark on it), and precision both over every mark and
over the erroneous sentences only, as JWTD reports it (its test has no correct sentences).

usage (from asr/): python tools/jwtd_score.py [name, default sb0s] [line, default 0.1] [w, default 1] [det score]
With det and score, only that point (another system's product point on the dev half) is scored.
"""
import json
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402
import screen_sweep as s  # noqa: E402
import verify_sweep as v  # noqa: E402

D = "eval-jwtd"
# (det, score) from the dev half: the product point of the marks fix, then fair.py's best point at each
# false-mark level (1.17, 1.27, 2.0, 3.0, 4.6 per 1,000 characters).
POINTS = [("product", 0.96, 9.5), ("fm1.17", 0.98, 9.5), ("fm1.27", 0.99, 9.0), ("fm2.0", 0.98, 6.0),
          ("fm3.0", 0.98, 4.0), ("fm4.6", 0.84, 4.0)]


def pct(a, b):
    return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "  -  "


def score(src, labels, det, score_):
    c = {}
    add = lambda k, n=1: c.__setitem__(k, c.get(k, 0) + n)  # noqa: E731
    for err, lab in labels.items():
        ms = prod.mk(src, err, det, score_)
        a, b = lab["span"]
        hit = any(x < b and a < y for x, y in ms)
        on = [x < b and a < y for x, y in ms]
        kind = lab["kind"]
        clean_src = not lab["train_text"] and not any(e.startswith("eval-mine") for e in lab["evals"])
        groups = [kind] + ([f"{kind}:main"] if clean_src else []) + [f"{kind}:{'seen' if lab['seen_pair'] else 'unseen'}"]
        for g in groups:
            add(g)
            add(g + ":hit", hit)
        add("err_marks", len(ms))
        add("err_marks_on", sum(on))
        if kind == "homophone":
            add("sent_err")
            add("sent_flag", bool(ms))
            add("sent_loose", hit)
            add("sent_strict", hit and all(on))
        ok = prod.mk(src, lab["ok"], det, score_)
        n = len(prod.TEXT[lab["ok"]])
        for g in ["ok"] + (["ok:unused"] if "eval-clean" not in lab["evals"] else []):
            add(g + ":chars", n)
            add(g + ":fm", len(ok))
            add(g + ":texts")
            add(g + ":flag", bool(ok))
    return c


def main():
    name = ARGS[0] if ARGS else "sb0s"
    line = float(ARGS[1]) if len(ARGS) > 1 else 0.1
    w = float(ARGS[2]) if len(ARGS) > 2 else 1.0
    labels = json.load(open(os.path.join(D, "labels.json"), encoding="utf-8"))
    prod.TEXT.update({t["id"]: t["text"] for t in json.load(open(os.path.join(D, "texts.json"), encoding="utf-8"))})
    src = prod.load(D, name)
    if w:
        src = v.verified(src, json.load(open(os.path.join(D, f"verify-{name}.json"), encoding="utf-8")), w)
    src = s.screened(src, line)
    points = [("product", float(ARGS[3]), float(ARGS[4]))] if len(ARGS) > 4 else POINTS
    print(f"JWTD test: {len(labels)} erroneous sentences; {name}, line {line}, w {w}", flush=True)
    for tag, det, sc in points:
        c = score(src, labels, det, sc)
        g = lambda k: (c.get(k + ":hit", 0), c.get(k, 0))  # noqa: E731
        fm = 1000 * c["ok:fm"] / c["ok:chars"]
        fm_u = 1000 * c.get("ok:unused:fm", 0) / max(c.get("ok:unused:chars", 1), 1)
        prec_all = c["err_marks_on"] / max(c["err_marks"] + c["ok:fm"], 1)
        prec_err = c["err_marks_on"] / max(c["err_marks"], 1)
        print(f"{tag} (det {det}, score {sc})\n"
              f"  homophone recall {pct(*g('homophone'))} | main {pct(*g('homophone:main'))} | "
              f"seen pair {pct(*g('homophone:seen'))} | unseen pair {pct(*g('homophone:unseen'))}\n"
              f"  other-reading kanji recall {pct(*g('other-kanji'))}\n"
              f"  false marks {fm:.2f}/1k chars (texts not used for the limit {fm_u:.2f}); corrected sentences marked "
              f"{pct(c['ok:flag'], c['ok:texts'])}\n"
              f"  sentences (homophone): strict {pct(c['sent_strict'], c['sent_err'])} loose {pct(c['sent_loose'], c['sent_err'])}\n"
              f"  precision: every mark {100 * prec_all:.1f}% | erroneous sentences only {100 * prec_err:.1f}%", flush=True)


if __name__ == "__main__":
    main()
