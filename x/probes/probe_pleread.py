# Split DiskRowCache._read_staged into ids sync wait vs native disk read (CPU wall), per decode step.
import time, builtins, statistics as stt, torch
from b12x.sequence._shared import disk_table as dt
C = dt.DiskRowCache
if not hasattr(builtins, "_fx_pr"):
    builtins._fx_pr = []; builtins._fx_pr_orig = C._read_staged
    def _read_staged(self, count):
        with torch.cuda.device(self.device):
            t0 = time.perf_counter(); self._ids_ready.synchronize(); t1 = time.perf_counter()
            if self._gds is not None:
                self._gds.read(self._ids_buffer, count, self._transaction_stream); return
            self._native.ple_reader_run(self._reader, self._ids_buffer, self._weight_buffer, self._scale_buffer, count)
            t2 = time.perf_counter()
            builtins._fx_pr.append((t1 - t0, t2 - t1, count, t2))
    C._read_staged = _read_staged
    cg = worker.model_runner.cudagraph_manager
    builtins._fx_rf = []; builtins._fx_rf_orig = cg.run_fullgraph
    def rf(*a, _o=cg.run_fullgraph, **k):
        builtins._fx_rf.append(time.perf_counter()); return _o(*a, **k)
    cg.run_fullgraph = rf
    RESULT = "wrapped"
else:
    C._read_staged = builtins._fx_pr_orig
    worker.model_runner.cudagraph_manager.run_fullgraph = builtins._fx_rf_orig
    L = builtins._fx_pr[20:]; del builtins._fx_pr
    import bisect
    R = builtins._fx_rf
    gaps = sorted(R[i] - x[3] for x in L for i in [bisect.bisect_left(R, x[3])] if i < len(R) and R[i] - x[3] < 0.01)
    w = sorted(x[0] for x in L); rd = sorted(x[1] for x in L)
    RESULT = (f"n={len(L)} count median {stt.median(x[2] for x in L)}; sync wait median {stt.median(w)*1e3:.2f} ms; "
              f"read median {stt.median(rd)*1e3:.3f} ms p10 {rd[len(rd)//10]*1e3:.3f} p90 {rd[9*len(rd)//10]*1e3:.3f}; read end -> run_fullgraph call median {stt.median(gaps)*1e3:.3f} ms")
