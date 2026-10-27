"""Summarize the transcripts per split into a JSON report and one PROGRESS.md line.

The report holds counts only, never Common Voice sentences, so it can be committed. It fails if any
test clip has been transcribed: test is transcribed only for the final evaluation.

usage: python asr/report_asr.py <transcripts.jsonl> <manifest.tsv> <report.json> <PROGRESS.md> <title>
"""

import collections
import csv
import hashlib
import json
import sys
import time

from classify import diff


def main() -> None:
    transcripts, manifest, report, progress, title = sys.argv[1:6]
    with open(manifest, encoding="utf-8", newline="") as f:
        split = {r["path"]: r["split"] for r in csv.DictReader(f, delimiter="\t")}
    recs = [json.loads(line) for line in open(transcripts, encoding="utf-8")]

    leaked = [r["path"] for r in recs if split.get(r["path"]) == "test"]
    if leaked:
        sys.exit(f"{len(leaked)} test clips were transcribed before the final evaluation, e.g. {leaked[0]}")

    verdicts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    kinds: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in recs:
        s = split.get(r["path"], "unassigned")
        v, edits = diff(r["ref"], r["hyp"])
        verdicts[s][v] += 1
        kinds[s].update(e["kind"] for e in edits)

    with open(manifest, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    out = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest_sha256": digest,
        "transcripts": len(recs),
        "test_transcripts": 0,
        "splits": {
            s: {"clips": sum(verdicts[s].values()), "verdicts": dict(verdicts[s]), "edit_kinds": dict(kinds[s])}
            for s in sorted(verdicts)
        },
    }
    with open(report, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    parts = []
    for s in ("train", "val"):
        c = out["splits"].get(s)
        if c:
            n = c["clips"]
            homophone = c["edit_kinds"].get("homophone", 0)
            parts.append(f"{s} {n:,} clips ({100 * c['verdicts'].get('error', 0) / n:.1f}% with errors, {homophone:,} homophone edits)")
    with open(progress, "a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} {title}: {'; '.join(parts)}. Test transcripts: 0.\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    n = {s: out["splits"].get(s, {}).get("clips", 0) for s in ("train", "val")}
    # nightly.sh commits with this subject
    print(f"COMMIT: feat(asr): {title} (train {n['train']:,}, val {n['val']:,} clips transcribed)")


if __name__ == "__main__":
    main()
