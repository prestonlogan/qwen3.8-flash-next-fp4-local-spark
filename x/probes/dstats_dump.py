# Dump FP4X_DENSE_STATS accumulators (mean x^2 per input channel of each MXFP8 dense linear, keyed by first-call
# order) to /x/dense_stats.pt, with module names for cross-checking against another server (dstats_check.py).
import traceback
try:
    import torch, fp4patch
    names = {id(getattr(d, "b12x_mxfp8_packed_weight", None)): n for n, d in worker.model_runner.model.named_modules()
             if hasattr(d, "b12x_mxfp8_packed_weight")}
    out = {}
    for idx, shp, pid in fp4patch._DORDER.values():
        st = fp4patch._DSTATS.get(idx)
        if st is not None:
            out[idx] = {"shape": shp, "name": names.get(pid), "h": (st[0] / st[1]).cpu(), "rows": st[1]}
    torch.save(out, "/x/dense_stats.pt")
    RESULT = f"{len(out)} layers dumped; rows {sorted({v['rows'] for v in out.values()})}; unnamed {sum(v['name'] is None for v in out.values())}"

except Exception:
    RESULT = traceback.format_exc()[-1500:]
