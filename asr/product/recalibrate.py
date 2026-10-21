"""Thresholds for one run chosen as step 2a set out (Kotobae #98): every false-mark limit is the check
before #98 on the same text (Kotobae's correct texts, the 150,000 characters of eval-clean, clean Common
Voice on the dev half), the typo sets may not fall below it in any subset, and right words marked stay at
most 2; among those, the highest homophone recall on the dev half. The calibration half then reports.

usage (from asr/): python tools/recalibrate.py <name>
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


def hom(src, h, det, score):
    c = collections.Counter()
    for r in prod.VAL:
        if half[r["path"]] == h:
            c.update(prod.clip_counts(r, prod.mk(src, r["path"], det, score)))
    s = prod.summary(c)
    return s["homophone_recall"], s["false_marks_per_1k_chars"]


def clean(src, det, score):
    return sum(len(prod.mk(src, k, det, score)) for k in clean_fm.TEXTS)


name = ARGS[0]
V, K, S, C = (prod.load(d, n) for d, n in (("eval-val", name), ("..", name), ("eval-kokoro", name), ("eval-clean", name)))
prod.GATE = False
D = prod.load("eval-val", "d")
base_sets = prod.sets(prod.load("..", "rust"), 0.4707, 5.13)
base = {"kotobae": prod.kotobae_fm(prod.load("..", "rust"), 0.4707, 5.13), "clean": clean(prod.load("eval-clean", "d"), 0.4707, 5.13),
        "dev_fm": hom(D, "dev", 0.4707, 5.13)[1], "calib_hom": hom(D, "calib", 0.4707, 5.13)[0]}
prod.GATE = True
print("before #98:", {k: round(v, 3) for k, v in base.items()}, base_sets)
rows = []
# A scorer trained with a wider margin spreads every lead, so its mark can sit well above 5.13.
for det in [round(0.5 + 0.02 * i, 2) for i in range(21)]:
    for score in [4.5, 4.75, 5.0, 5.13] + [round(5.25 + 0.25 * i, 2) for i in range(20)]:
        sets = prod.sets(K, det, score)
        ok = (prod.kotobae_fm(K, det, score) <= max(1.0, base["kotobae"]) and clean(C, det, score) <= base["clean"]
              and hom(V, "dev", det, score)[1] <= base["dev_fm"] and sets["right_marked"] <= 2
              and all(sets[k] >= v for k, v in base_sets.items() if k != "right_marked"))
        if ok:
            rows.append((hom(V, "dev", det, score)[0], det, score, sets))
rows.sort(reverse=True)
for dev_hom, det, score, sets in rows[:5]:
    calib_hom, calib_fm = hom(V, "calib", det, score)
    print(f"det {det} score {score}: dev {dev_hom:.3f} | calib {calib_hom:.3f} (fm {calib_fm:.3f}) | clean {clean(C, det, score)} "
          f"| kotobae fm {prod.kotobae_fm(K, det, score):.2f} | kokoro {prod.found(S, prod.SYN, det, score)['homophone_recall']:.3f} | {sets}")
