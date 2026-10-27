"""Rows for the progress dashboard (Kotobae #98): the candidate metrics at each sweep's dev point, in the
dashboard's existing schema, and the crossfit ledger from progress.py. Counts and rates only, never text.

usage (from asr/): python tools/dash_rows.py <old candidates.json> -> ../dash-candidates.json, ../dash-crossfit.json
"""
import datetime
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import progress  # noqa: E402

DETECTOR = "検出器の作り直し（候補・文脈の幅・重みの混合）"
SCORER = "採点役の学び直しと混合"
NAMES = {
    "sb0run3step3000": ("見比べ役の学び直し 3 回目・3,000 歩", "見比べ役の学び直し"),
    "sb0run3step6000": ("見比べ役の学び直し 3 回目・6,000 歩", "見比べ役の学び直し"),
    "sb0s": ("検出器 sb0（候補をすべて採点）", DETECTOR),
    "sa0s": ("検出器 sa0（候補をすべて採点）", DETECTOR),
    "sb0m2": ("下地学習した検出器を 2 割混ぜる", DETECTOR),
    "sb0m3": ("下地学習した検出器を 3 割混ぜる", DETECTOR),
    "sb0m4": ("下地学習した検出器を 4 割混ぜる", DETECTOR),
    "sb1s": ("語ごとの読みで候補を出す・前後 2 トークン", DETECTOR),
    "sb2s": ("語ごとの読みで候補を出す・前後 3 トークン", DETECTOR),
    "sb3s": ("語ごとの読みで候補を出す・前後 4 トークン", DETECTOR),
    "sb2m3": ("前後 3 トークン＋下地学習 3 割", DETECTOR),
    "sb2k3": ("前後 3 トークン＋蒸留した検出器 3 割", DETECTOR),
    "sb2k4": ("前後 3 トークン＋蒸留した検出器 4 割", DETECTOR),
    "sb2k5": ("前後 3 トークン＋蒸留した検出器 5 割", DETECTOR),
    "sb2k6": ("前後 3 トークン＋蒸留した検出器 6 割", DETECTOR),
    "sb4s": ("地名の候補を足す", DETECTOR),
    "sb4k5": ("地名の候補＋蒸留した検出器 5 割", DETECTOR),
    "sb2k5q0": ("採点役 run2（混ぜない対照）", SCORER),
    "sb2k5q3": ("採点役 run3 を 3 割混ぜる", SCORER),
    "sb2k5q5": ("採点役 run3 を 5 割混ぜる", SCORER),
    "sb2k5q7": ("採点役 run3 を 7 割混ぜる（q7）", SCORER),
    "sb2k5q9": ("採点役 run3 を 9 割混ぜる", SCORER),
}
LINE = re.compile(r"^line ([\d.]+) .*best det ([\d.]+) score ([\d.]+): dev ([\d.]+) fm ([\d.]+) clean (\d+) (\{.*\})")


def label_of(name: str, line: str) -> tuple[str, str] | None:
    if name in NAMES:
        label, stage = NAMES[name]
    elif name in progress.NOTES:
        label, stage = progress.NOTES[name][1], SCORER
    else:
        return None
    return (label + ("（線 0.05）" if line == "0.05" else ""), stage)


def iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def candidates(old: list[dict]) -> list[dict]:
    rows = [r for r in old]
    seen = {r["label"] for r in rows}
    for path in sorted(glob.glob("scorer/sweep-*.txt"), key=os.path.getmtime):
        name = re.match(r"sweep-([a-z0-9]+)", os.path.basename(path)).group(1)
        hits = [m for m in map(LINE.match, open(path)) if m]
        if not hits:
            continue
        m = hits[-1]
        got = label_of(name, m[1])
        if got is None or got[0] in seen:
            continue
        sub = json.loads(m[7].replace("'", '"'))
        parts = [sub[k] for k in ("dev", "holdout", "fresh", "dev-seen", "real")]
        rows.append({"label": got[0], "stage": got[1], "at": iso(os.path.getmtime(path)), "ok": True, "below_d": 0,
                     "hom_real": float(m[4]), "typos": sum(parts), "subsets": "・".join(map(str, parts)),
                     "fm_real": float(m[5]), "det": float(m[2]), "score": float(m[3])})
        seen.add(got[0])
    return rows


def crossfit() -> list[dict]:
    out = []
    for r in progress.versions():
        short = r["version"].replace("sb2k5", "")
        out.append({"label": f"{short}（線 {r['line']}）", "change": r["change"], "at": iso(r["mtime"]),
                    "recall85": r["recall_0.85"], "fm85": r["fm_0.85"], "recall10": r["recall_1.0"],
                    "fm10": r["fm_1.0"], "vs_q7": r["vs_q7"], "verdict": r["verdict"]})
    return out


def main() -> None:
    old = json.load(open(sys.argv[1], encoding="utf-8"))
    json.dump(candidates(old), open("../dash-candidates.json", "w"), ensure_ascii=False, indent=1)
    json.dump(crossfit(), open("../dash-crossfit.json", "w"), ensure_ascii=False, indent=1)
    print(len(old), "->", len(json.load(open("../dash-candidates.json"))), "candidates;",
          len(json.load(open("../dash-crossfit.json"))), "crossfit rows")


if __name__ == "__main__":
    main()
