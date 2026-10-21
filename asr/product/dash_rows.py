"""Turn report_asr.py reports into the growth rows of the training dashboard (not committed)."""
import json, sys
rows = []
for p in sys.argv[2:]:
    r = json.load(open(p))
    for split in ("train", "val"):
        c = r["splits"].get(split)
        if not c:
            continue
        rows.append({"at": r["created"], "split": split, "clips": c["clips"],
                     "error_clips": c["verdicts"].get("error", 0), "broken_clips": c["verdicts"].get("broken", 0),
                     "homophone_edits": c["edit_kinds"].get("homophone", 0), "mishear_edits": c["edit_kinds"].get("mishear", 0)})
json.dump(rows, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(rows)
