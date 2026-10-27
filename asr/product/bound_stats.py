# How many error rows a per-token bound on the training error rate would drop, by clean count range.
import numpy as np
V = 120000
parts = {n: np.load(f"{n}.npz") for n in ("base", "aug")}
C = np.zeros(V, np.int64); E = np.zeros(V, np.int64)
row_err = {}
for n, d in parts.items():
    ids, lab, off = d["ids"], d["labels"], d["offsets"]
    C += np.bincount(ids[lab == 0], minlength=V); E += np.bincount(ids[lab == 1], minlength=V)
    pos = np.nonzero(lab == 1)[0]
    row_err[n] = (np.searchsorted(off, pos, side="right") - 1, ids[pos])
tot_err_rows = {n: len(np.unique(r)) for n, (r, _) in row_err.items()}
print("error rows", tot_err_rows, "global error token rate %.4f" % (E.sum() / (C.sum() + E.sum())))
rate = E / np.maximum(C + E, 1)
for b in (0.03, 0.05, 0.1, 0.2):
    for nmin in (20, 100, 500):
        over = (C >= nmin) & (rate > b)
        # excess error occurrences beyond the bound, a lower bound on what has to go
        allow = np.floor(b / (1 - b) * C)
        excess = np.where(over, E - allow, 0).clip(0)
        print(f"b={b} nmin={nmin}: tokens over {over.sum()}, error occurrences over bound {int(excess.sum())} of {int(E.sum())} ({excess.sum() / E.sum():.0%})")
