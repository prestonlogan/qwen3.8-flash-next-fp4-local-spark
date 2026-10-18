from flashinfer.experimental.b12x.sequence._shared import disk_table as dt
C = dt.DiskRowCache
if not hasattr(C, "_hit_orig"):
    C._hit_orig = C._read_staged
    C._hit = {"seen": set(), "calls": 0, "lk": 0, "hits": 0, "allhit": 0}
    def _rs(self, count):
        C._hit_orig(self, count)
        h = C._hit; ids = self.ids_host[:count].tolist()
        s = h["seen"]; nh = sum(1 for i in ids if i in s)
        h["calls"] += 1; h["lk"] += count; h["hits"] += nh; h["allhit"] += (nh == count); s.update(ids)
    C._read_staged = _rs
h = C._hit
RESULT = {k: (len(v) if isinstance(v, set) else v) for k, v in h.items()}
