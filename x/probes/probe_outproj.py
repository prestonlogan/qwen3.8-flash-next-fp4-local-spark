# out_proj: module forward vs raw W6 kernel on the plugin's cached weights (graph-timed, M=3).
import torch, sys
sys.path.insert(0, "/x")
import fp4x_kernels as F
r = worker.model_runner; m = r.model; mods = dict(m.named_modules())
def gtime(fn, it=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
from vllm.model_executor.kernels.linear import b12x_blockscaled as BB
def find_cache(fn, depth=0):
    for c in (getattr(fn, "__closure__", None) or ()):
        v = c.cell_contents
        if isinstance(v, dict) and v and isinstance(next(iter(v.values())), tuple): return v
        if callable(v) and depth < 2:
            r_ = find_cache(v, depth + 1)
            if r_ is not None: return r_
cache = find_cache(BB.B12xBlockscaledLinear.run)
ds = [d for n, d in mods.items() if n.endswith("linear_attn.out_proj") and "mtp" not in n]
X = torch.randn(3, 6144, device="cuda", dtype=torch.bfloat16)
Q = [q for q in cache.values() if q[0].shape == (2560, 3072)]
out = [f"{len(ds)} out_proj modules, {len(Q)} cached W6", f"module fwd: {gtime(lambda: [d(X) for d in ds]):.3f} ms",
       f"raw w6: {gtime(lambda: [F.w6_linear(X, *q, (32, 256, 1, 4)) for q in Q]):.3f} ms",
       f"module fwd again: {gtime(lambda: [d(X) for d in ds]):.3f} ms", f"out type {type(ds[0](X))}"]
RESULT = "\n".join(out)
