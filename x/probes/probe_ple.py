# Instrument DiskRowCache: wall time of _read_staged (incl. ids sync) and native exec time.
import time, os
from flashinfer.experimental.b12x.sequence._shared import disk_table as dt
C = dt.DiskRowCache
mode = "install"
if mode == "install":
    if not hasattr(C, "_px_orig"):
        C._px_orig = C._read_staged
        C._px = {"n": 0, "wall": 0.0, "sync": 0.0, "exec": 0.0, "rows": 0, "bytes": 0}
        def _rs(self, count):
            import torch
            t0 = time.perf_counter()
            with torch.cuda.device(self.device):
                self._ids_ready.synchronize()
            t1 = time.perf_counter()
            C._px_orig(self, count)
            t2 = time.perf_counter()
            s = self._native.ple_reader_stats(self._reader) if self._native else {}
            p = C._px; p["n"] += 1; p["sync"] += t1 - t0; p["wall"] += t2 - t0
            p["exec"] += s.get("execution_seconds", 0); p["rows"] += count; p["bytes"] += s.get("read_bytes", 0)
        C._read_staged = _rs
    for k in C._px: C._px[k] = 0
    RESULT = "installed/reset"
else:
    p = dict(C._px); n = max(p["n"], 1)
    RESULT = {k: (v / n * 1e3 if k in ("wall", "sync", "exec") else v / n) for k, v in p.items()} | {"calls": p["n"]}
