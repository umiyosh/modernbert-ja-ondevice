# Wu et al. (ACL 2023) split: recall on items whose wrong->right pair occurs in the JWTD train
# table (INC) vs not (EXC), crossed thresholds at <=1 clean mark per 1k chars (judge.py protocol).
import json, os, sys
D = sys.argv[1]; sys.path.insert(0, D); sys.argv = ["judge.py"]
import judge
pairs = json.load(open(os.path.join(D, "pairs.json")))
st = judge.st
for spec in ["ft30maug", "ft30mbound", "mlm130m", "cm130a3c5", "ft30maug+base", "ft30mbound+base", "mlm130m+base", "cm130a3c5+base"]:
    cr = judge.run(judge.System(spec))[1.0]["crossed"]
    g = {"INC": [0, 0], "EXC": [0, 0], "voice": [0, 0]}
    for c, h in cr.items():
        k = "voice" if st[c]["source"] != "jwtd" else ("INC" if f"{st[c]['wrong']}\t{st[c]['right']}" in pairs else "EXC")
        g[k][0] += h; g[k][1] += 1
    print(f"{spec:18s} " + "  ".join(f"{k} {a}/{n} ({a / n:.0%})" for k, (a, n) in g.items()))
