# Tune Triton BF16 / W8 GEMVs on the 48 router gates (512x2560) at verify M.
import torch, itertools, sys
sys.path.insert(0, "/x")
import fp4x_kernels as F
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
M = 3
Ws = [mods[f"language_model.model.layers.{i}.mlp.gate"].weight for i in range(48)]
x = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
out.append(f"cuBLAS: {gtime(lambda: [torch.nn.functional.linear(x, w) for w in Ws]):.3f} ms")
Q8 = [F.quantize_w8(w) for w in Ws]
for name, fn in (("bf16", lambda c: [F.bf16_linear(x, w, c) for w in Ws]), ("w8", lambda c: [F.w8_linear(x, *q, c) for q in Q8])):
    res = []
    for c in itertools.product([16, 32, 64], [64, 128, 256, 512], [1, 2, 4, 5], [2, 4, 8]):
        if 2560 % (c[1] * c[2]): continue
        try: res.append((gtime(lambda c=c: fn(c), it=10), c))
        except Exception: pass
    res.sort(); out.append(f"{name} best {[(round(a, 3), b) for a, b in res[:3]]}")
RESULT = "\n".join(out)
