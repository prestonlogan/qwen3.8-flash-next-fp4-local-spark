import numpy as np, time
m = worker.model_runner.model
o = [mod for _, mod in m.named_modules() if getattr(mod, "disk_table", None) is not None][0].disk_table._cache
rng = np.random.default_rng(0)
out = []
W = np.empty((4096, o.weight_row_bytes), np.uint8); S = np.empty((4096, o.scale_row_bytes), np.uint8)
for n in (8, 16, 32, 64, 128, 256):
    ts = []
    for r in range(30):
        ids = rng.integers(0, 320000000, n, dtype=np.int64)
        t = time.perf_counter(); o._native.ple_reader_run(o._reader, memoryview(ids), memoryview(W), memoryview(S), n); ts.append(time.perf_counter() - t)
    out.append((n, round(float(np.median(ts)) * 1e6), round(float(np.min(ts)) * 1e6)))
RESULT = out
