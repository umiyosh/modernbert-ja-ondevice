"""Rows for the progress dashboard: every candidate measured in the app's form, at the setting prod.py
would pick (meets all typo subsets first, then most homophones found on real voice). Run from asr/."""

import datetime
import json
import os
import sys

sys.argv = [sys.argv[0], "--gate"]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "dash")
SUBSETS = ["dev", "holdout", "fresh", "dev-seen", "real"]

REAL = "本物の音声だけ"
MAC = "macOS の声（公開不可）"
MACW = "macOS の声＋今の検出器と混ぜる（公開不可）"
KOK = "Kokoro の声＋今の検出器と混ぜる"
C = [
    ("csoup", REAL, "本物の音声・複数本の平均"),
    ("cclean1500", REAL, "本物の音声＋誤りのない文"),
    ("csm2500", MAC, "macOS の声・2,500 歩"),
    ("csm3000", MAC, "macOS の声・3,000 歩"),
    ("cs12", MAC, "macOS の声・2 本の平均"),
    ("crp3000", MAC, "macOS の声・復習つき"),
    ("cw07", MACW, "macOS の声 1・新 70%"),
    ("cw05", MACW, "macOS の声 1・新 50%"),
    ("cw04", MACW, "macOS の声 1・新 40%"),
    ("cw03", MACW, "macOS の声 1・新 30%"),
    ("c4w05", MACW, "macOS の声 2・新 50%"),
    ("c4w04", MACW, "macOS の声 2・新 40%"),
    ("c4w03", MACW, "macOS の声 2・新 30%"),
    ("ch04", MACW, "macOS の声 3・新 40%"),
    ("ch05", MACW, "macOS の声 3・新 50%"),
    ("c5w04", MACW, "macOS の声 4・新 40%"),
    ("k3w03", KOK, "Kokoro 3,000 歩・新 30%"),
    ("k3w04", KOK, "Kokoro 3,000 歩・新 40%"),
    ("k3w05", KOK, "Kokoro 3,000 歩・新 50%"),
    ("k3w02", KOK, "Kokoro 3,000 歩・新 20%"),
    ("k5w03", KOK, "Kokoro 5,000 歩・新 30%"),
    ("k5w04", KOK, "Kokoro 5,000 歩・新 40%"),
    ("kaw03", KOK, "Kokoro 全 12 個・新 30%"),
    ("kaw04", KOK, "Kokoro 全 12 個・新 40%"),
]


def when(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).astimezone().isoformat(timespec="seconds")


def out(name, stage, label, at, r, base, ok):
    s = r["sets"]
    return {
        "label": label, "stage": stage, "at": at, "ok": ok,
        "below_d": sum(s.get(k, 0) < base["sets"].get(k, 0) for k in SUBSETS) if base else 0,
        "hom_real": r["val_hom"], "span_real": r["val"],
        "hom_read": r["syn_hom"] if r["syn_hom"] else None,
        "typos": sum(s.get(k, 0) for k in SUBSETS), "subsets": "・".join(str(s.get(k, 0)) for k in SUBSETS),
        "fm_kotobae": r["kotobae_fm"], "fm_real": r["val_fm"], "det": r["det"], "score": r["score"],
    }


def main():
    prod.GATE = False
    base = prod.row(prod.load("eval-val", "d"), prod.load("..", "rust"), prod.load("eval-kokoro", "rustd"), 0.4707, 5.13)
    prod.GATE = True
    kb, vb = max(1.0, base["kotobae_fm"]), max(1.0, base["val_fm"])
    rows = [out("d", "今の検出器", "今の検出器（公開中）", "2026-10-18T23:45:15+09:00", base, None, True)]
    for name, stage, label in C:
        V, K = prod.load("eval-val", name), prod.load("..", name)
        S = prod.load("eval-kokoro", name)
        best = []
        for score in prod.SCORES:
            det = next((d for d in prod.DETS if prod.kotobae_fm(K, d, score) <= kb and prod.val_fm(V, d, score) <= vb), None)
            if det is None:
                continue
            r = prod.row(V, K, S, det, score)
            ok = all(r["sets"].get(k, 0) >= base["sets"].get(k, 0) for k in SUBSETS)
            best.append((ok, r))
        ok, r = sorted(best, key=lambda x: (not x[0], -x[1]["val_hom"]))[0]
        rows.append(out(name, stage, label, when(f"eval-val/res-{name}det-r0.json"), r, base, ok))
        print(label, rows[-1], flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "candidates.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
