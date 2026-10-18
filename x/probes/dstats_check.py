# Check that this server's first-call order maps to the same module names as /x/dense_stats.pt, and report the AWQ
# alpha choices made so far.
import traceback
try:
    import torch, fp4patch, fp4x_kernels as F
    from collections import Counter
    names = {id(getattr(d, "b12x_mxfp8_packed_weight", None)): n for n, d in worker.model_runner.model.named_modules()
             if hasattr(d, "b12x_mxfp8_packed_weight")}
    ref = torch.load("/x/dense_stats.pt")
    here = {idx: names.get(pid) for idx, shp, pid in fp4patch._DORDER.values()}
    bad = [(i, v["name"], here.get(i)) for i, v in ref.items() if here.get(i) != v["name"]]
    RESULT = f"checked {len(ref)}; mismatches {len(bad)} {bad[:3]}; alphas {sorted(Counter(F._AWQ_LOG).items())}"

except Exception:
    RESULT = traceback.format_exc()[-1500:]
try:
    from collections import defaultdict
    by = defaultdict(int)
    for v in ref.values(): by[(v["shape"], v["name"] is None, v["h"].numel(), v["rows"])] += 1
    RESULT += f"; dump groups {dict(by)}"
except Exception:
    RESULT += traceback.format_exc()[-500:]
