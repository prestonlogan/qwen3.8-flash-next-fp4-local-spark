from flashinfer.experimental.b12x.sequence._shared import disk_table as dt
p = dt.DiskRowCache._sp; RESULT = {"calls": p["n"], "io_ms": p["wall"] / max(p["n"], 1) * 1e3}
