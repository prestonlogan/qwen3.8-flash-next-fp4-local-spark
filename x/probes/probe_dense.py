# Graph-timed dense linears grouped by shape at verify M; compares to bandwidth floor.
import torch, traceback, sys, os
from collections import defaultdict
r = worker.model_runner; m = r.model; mods = dict(m.named_modules())
out = []
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
M = int(os.environ.get("PM", "3"))
from vllm.model_executor.layers.linear import LinearBase
nv4 = {tuple(int(v) for v in t.split("x")) for t in os.environ.get("FP4X_NV4_SHAPES", "").split(",") if t}
groups = defaultdict(list)
for n, d in mods.items():
    if isinstance(d, LinearBase) and hasattr(d, "b12x_mxfp8_packed_weight") and "mtp" not in n and "visual" not in n:
        w = d.b12x_mxfp8_packed_weight.weight.values
        groups[tuple(w.shape)].append((n, d))
tot = 0.0
for (nn, k), ds in sorted(groups.items()):
    X = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
    try:
        t = gtime(lambda: [d(X) for _, d in ds])
    except Exception:
        out.append(f"{(nn,k)} err " + traceback.format_exc()[-300:]); continue
    bpe = 0.5 + 1/16 if (nn, k) in nv4 else 0.75 + 2/32
    mb = len(ds) * nn * k * bpe / 1e6
    tot += t
    out.append(f"{nn}x{k} x{len(ds)} ({ds[0][0].split('.')[-1]}): {t:.2f} ms, {t/len(ds)*1000:.1f} us each, {mb:.0f} MB -> {mb/t/1e3:.0f} GB/s")
out.append(f"total {tot:.2f} ms")
RESULT = "\n".join(out)
