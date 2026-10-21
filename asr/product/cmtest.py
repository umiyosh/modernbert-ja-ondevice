# Isolates the segfault: runs one Core ML model many times with random inputs on chosen compute units.
import sys, numpy as np, coremltools as ct
path, cu, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, cu))
spec = {i.name: i for i in m.get_spec().description.input}
B = 32; L = 64 if "l64" in path else 32
rng = np.random.default_rng(0)
for k in range(n):
    x = {"input_ids": rng.integers(5, 1000, (B, L)).astype(np.int32), "attention_mask": np.ones((B, L), np.int32)}
    if "position" in spec:
        x["position"] = np.zeros(B, np.int32); x["target"] = np.zeros(B, np.int32)
    m.predict(x)
print("ok", path.rsplit("/", 1)[-1], cu, n)
