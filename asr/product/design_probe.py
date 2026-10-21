# Facts for the redesign: (1) BART likelihood gain of JWTD edits by category, and its value on
# known pairs; (2) how often the erroneous and corrected sentences align token by token, which a
# correction-style objective (predict the right token at each position) needs.
import difflib, json, os, random, sys
from collections import Counter, defaultdict
os.environ.setdefault("HF_HOME", "../hf"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sbintuitions/modernbert-ja-30m")
rows = [json.loads(l) for l in open("../jwtd/jwtd_v2.0/train.jsonl")]
gain = defaultdict(list); pair_gain = defaultdict(list)
for d in rows:
    for x in d["diffs"]:
        g = x["post_bart_likelihood"] - x["pre_bart_likelihood"]
        gain[x["category"]].append(g); pair_gain[(x["pre_str"], x["post_str"])].append(g)
print("BART gain (post - pre) by category: share <= 0, median")
for c, v in sorted(gain.items(), key=lambda kv: -len(kv[1])):
    v = np.array(v); print(f"  {c:20s} n={len(v):6d}  <=0 {np.mean(v <= 0):.1%}  median {np.median(v):+.1f}")
for w in [("製作", "制作"), ("制作", "製作"), ("若干", "弱冠"), ("始めて", "初めて"), ("勤めた", "務めた"), ("固体", "個体"),
          ("型", "形"), ("形", "型"), ("の", "に"), ("自信", "自身"), ("開放", "解放")]:
    v = np.array(pair_gain.get(w, [0])); print(f"  {w[0]}→{w[1]}: n={len(v)} <=0 {np.mean(v <= 0):.0%} median {np.median(v):+.1f}")
random.seed(0); sample = random.sample(rows, 50000)
ok = Counter(); tot = Counter()
for d in sample:
    a = tok(d["pre_text"], add_special_tokens=False)["input_ids"]; b = tok(d["post_text"], add_special_tokens=False)["input_ids"]
    cats = {x["category"] for x in d["diffs"]}
    same = all(op == "equal" or (i2 - i1 == j2 - j1) for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes())
    for c in cats:
        tot[c] += 1; ok[c] += same
print("token-aligned pairs (every changed span keeps its token count):")
for c in sorted(tot, key=lambda c: -tot[c]):
    print(f"  {c:20s} {ok[c]}/{tot[c]} ({ok[c] / tot[c]:.0%})")
