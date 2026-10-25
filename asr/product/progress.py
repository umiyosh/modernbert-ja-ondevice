"""The progress ledger of the scorer retraining (Kotobae #98): every measured version in one table, read from
crossfit.py and crossfit_pair.py's outputs, and every training run's curve, read from train_scorer.py's logs.
Counts and rates only, never text. Run after each chain; it rewrites both files.

usage (from asr/): python tools/progress.py   -> ../progress.md, ../progress-curves.csv
"""
import csv
import glob
import json
import os
import re

# What each version changed, and the verdict; the numbers come from the measurement files.
NOTES = {
    "sb2k5q7": ("10-24", "基準。採点役 soup-run3-7（run3 を 7 割混ぜた版）", "基準"),
    "sb2k5r45": ("10-25", "run4（難しい負例 hard4 を足した学習）と run2 を半分ずつ", "不採用"),
    "sb2k5r47": ("10-25", "run4 と run2 を 7:3", "不採用"),
    "sb2k5m45": ("10-25", "soup-run3-7 と run4 を半分ずつ", "暫定の最良（p = 0.065 で有意ではない）"),
    "sb2k5n55": ("10-25", "soup-m45 と run5（難しい負例 hard5 を追加）を半分ずつ", "不採用"),
    "sb2k5n35": ("10-25", "soup-run3-7 と run5 を半分ずつ", "不採用"),
    "sb2k5p45": ("10-25", "soup-run3-7 と run6（アプリと同じ前後 3 トークンで学習）を半分ずつ", "不採用"),
    "sb2k5p55": ("10-25", "soup-m45 と run6 を半分ずつ", "不採用"),
    "sb2k5s45": ("10-25", "soup-run3-7 と run7（前後 3 トークン、直す例の余白 8）を半分ずつ", "不採用（0.85 倍に収まる点が無い）"),
    "sb2k5s55": ("10-25", "soup-m45 と run7 を半分ずつ", "不採用"),
    "sb2k5j45": ("10-25", "soup-run3-7 と run8（JWTD 学習分の同音 10,896 組を追加）を半分ずつ", ""),
    "sb2k5j55": ("10-25", "soup-m45 と run8 を半分ずつ", ""),
    "sb2k5t0": ("10-26", "run9（人工の同音の取り違え 12 万組を追加、15,000 歩）をそのまま", ""),
    "sb2k5t45": ("10-26", "soup-run3-7 と run9 を半分ずつ", ""),
    "sb2k5t55": ("10-26", "soup-m45 と run9 を半分ずつ", ""),
}
POOLED = re.compile(r"pooled ([\d.]+)x D: homophones ([\d.]+)% \((\d+)/(\d+)\) fm ([\d.]+)/1k")
PAIR = re.compile(r"(\S+) (\d+)/(\d+) vs (\S+) (\d+)/(\d+): only \S+ (\d+), only \S+ (\d+), exact McNemar p=([\d.e-]+)")
STEP = re.compile(r"^step (\d+) rank ([\d.]+) mlm ([\d.]+)")
HELD = re.compile(r"^step (\d+) (\{.*\})$")


def line_of(path, name):
    tail = os.path.basename(path)[len(f"crossfit-{name}"):-4]
    return {"": "0.1", "-l05": "0.05"}.get(tail, tail.lstrip("-l"))


def versions():
    rows = []
    for path in glob.glob("scorer/crossfit-*.txt"):
        name = re.match(r"crossfit-(sb2k5[a-z]*\d*)", os.path.basename(path)).group(1)
        line = line_of(path, name)
        got = {m[0]: m for m in POOLED.findall(open(path).read())}
        pair = f"scorer/pair-{name}-l{line}.txt"
        p = PAIR.search(open(pair).read()) if os.path.exists(pair) else None
        date, change, verdict = NOTES.get(name, ("", "", ""))
        full, k85 = got.get("1.0"), got.get("0.85")
        rows.append({
            "version": name, "line": line, "date": date, "change": change,
            "recall_0.85": f"{k85[1]}% ({k85[2]}/{k85[3]})" if k85 and k85[3] == "566" else "収まらない",
            "fm_0.85": k85[4] if k85 and k85[3] == "566" else "",
            "recall_1.0": f"{full[1]}% ({full[2]}/{full[3]})" if full else "",
            "fm_1.0": full[4] if full else "",
            "vs_q7": f"失う {p[7]} / 得る {p[8]}、p = {float(p[9]):.2g}" if p else "",
            "verdict": verdict, "mtime": os.path.getmtime(path)})
    return sorted(rows, key=lambda r: (r["mtime"], r["line"]))


def curves():
    out = []
    for path in sorted(glob.glob("scorer/run[0-9]*.log")):
        run = os.path.basename(path)[:-4]
        if not run[3:].isdigit():
            continue
        loss = {}
        for ln in open(path, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if m := STEP.match(ln):
                loss[int(m[1])] = (float(m[2]), float(m[3]))
            elif m := HELD.match(ln):
                h = json.loads(m[2])
                out.append({"run": run, "step": int(m[1]), "kind": "held",
                            "keep_right": h.get("keep", {}).get("right"), "fix_right": h.get("fix", {}).get("right"),
                            "keep_lead": h.get("keep", {}).get("lead"), "fix_lead": h.get("fix", {}).get("lead")})
        out += [{"run": run, "step": s, "kind": "loss", "rank": r, "mlm": l} for s, (r, l) in sorted(loss.items())]
    return out


def main():
    vs, cs = versions(), curves()
    cols = ["run", "step", "kind", "rank", "mlm", "keep_right", "fix_right", "keep_lead", "fix_lead"]
    with open("../progress-curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(sorted(cs, key=lambda r: (r["run"], r["step"], r["kind"])))
    md = ["# 採点役の学び直しの推移（Kotobae #98）", "",
          "tools/progress.py が測定の出力から作る。同音の検出率は CV の検証 566 件で、2 つの半分で互いに閾値を選んで測った値（crossfit）。"
          "「0.85 倍」は誤った印の上限を前の版 D の 0.85 倍にしたとき。採否は 0.85 倍の値と基準 q7 との対の検定で決める。", "",
          "## 版ごとの成績", "",
          "| 版 | 線 | 日 | 変えたこと | 検出率（0.85 倍） | 誤った印/1k | 検出率（1.0 倍） | 誤った印/1k | 基準 q7 との比較 | 採否 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['version']} | {r['line']} | {r['date']} | {r['change']} | {r['recall_0.85']} | {r['fm_0.85']} | "
           f"{r['recall_1.0']} | {r['fm_1.0']} | {r['vs_q7']} | {r['verdict']} |" for r in vs]
    md += ["", "## 学習ごとの取り分けた例での正答率（最初 → 最後）", "",
           "| 学習 | 歩数 | 正しい語を残す（keep） | 直す（fix） | 最後の順位の損失 |", "|---|---|---|---|---|"]
    for run in sorted({c["run"] for c in cs}, key=lambda x: int(x[3:])):
        held = [c for c in cs if c["run"] == run and c["kind"] == "held"]
        loss = [c for c in cs if c["run"] == run and c["kind"] == "loss"]
        if held:
            a, b = held[0], held[-1]
            md.append(f"| {run} | {b['step']} | {a['keep_right']} → {b['keep_right']} | {a['fix_right']} → {b['fix_right']} | "
                      f"{loss[-1]['rank'] if loss else ''} |")
    md += ["", "取り分けた例は学習ごとに違う（run1・run2 は難しい負例を含まず、run6・run7 は前後 3 トークンで採点する）ので、"
           "学習どうしは比べず、同じ学習の最初と最後だけを比べる。版の成績は上の表で比べる。",
           "q7 との対の検定の p は、crossfit_pair.py を作る前の版（r45・r47・m45）では欄が空になる。"
           "m45 は線 0.05 で q7 の線 0.1 に対し失う 12 / 得る 24、p = 0.065（HANDOFF 11 節）。", "",
           "100 歩ごとの損失と、保存のたびの正答率は progress-curves.csv にある。"]
    open("../progress.md", "w").write("\n".join(md) + "\n")
    print(f"{len(vs)} measurements, {len(cs)} curve points")


if __name__ == "__main__":
    main()
