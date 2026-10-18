out = []
m = worker.model_runner.model
for name, mod in m.named_modules():
    dtab = getattr(mod, "disk_table", None)
    if dtab is not None:
        o = dtab._cache
        s = o._native.ple_reader_stats(o._reader)
        out.append((name, dict(s) | {"max_lookups": o.max_lookups, "table_rows": o.table_rows, "shard_rows": o.shard_rows,
                    "wrb": o.weight_row_bytes, "srb": o.scale_row_bytes, "shards": len(o._sources), "cls": str(type(o))}))
RESULT = out
