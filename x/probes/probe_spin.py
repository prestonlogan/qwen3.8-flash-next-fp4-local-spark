# Variant of the PLE read path: spin-poll the ids event (GIL released each iteration) instead of
# blocking, so the CPU core is awake and clocked up when the io_uring batch is issued.
import time, torch
from flashinfer.experimental.b12x.sequence._shared import disk_table as dt
C = dt.DiskRowCache
if not hasattr(C, "_sp_orig"):
    C._sp_orig = C._read_staged
C._sp = {"n": 0, "exec": 0.0, "wall": 0.0}
def _rs(self, count):
    ev = self._ids_ready
    while not ev.query():
        time.sleep(0)
    t0 = time.perf_counter()
    if self._gds is not None:
        self._gds.read(self._ids_buffer, count, self._transaction_stream)
    else:
        self._native.ple_reader_run(self._reader, self._ids_buffer, self._weight_buffer, self._scale_buffer, count)
    p = C._sp; p["n"] += 1; p["wall"] += time.perf_counter() - t0
C._read_staged = _rs
RESULT = "spin installed"
