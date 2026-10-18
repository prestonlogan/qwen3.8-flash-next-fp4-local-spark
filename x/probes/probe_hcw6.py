# Tune W6 for the HC mixer shapes on the real weights (97 mixers, verify M) and report error vs BF16.
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
hcs = [h for n, h in mods.items() if type(h).__name__ == "GatedResidual" and "mtp" not in n]
for which in ("down", "up"):
    Ws = [((h.input_mix_weight_down_block_inject if h.use_combine else h.input_mix_weight_down) if which == "down" else h.input_mix_weight_up).weight for h in hcs]
    n, k = Ws[0].shape
    x = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
    Q6 = [F.quantize_w6(w.float()) for w in Ws]
    Q8 = [F.quantize_w8(w) for w in Ws]
    e6 = e8 = 0.0
    for w, q6, q8 in list(zip(Ws, Q6, Q8))[::8]:
        ref = x.float() @ w.float().T
        e6 = max(e6, ((F.w6_linear(x, *q6, (32, 64, 1, 4)).float() - ref).norm() / ref.norm()).item())
        e8 = max(e8, ((F.w8_linear(x, *q8, (32, 64, 1, 4)).float() - ref).norm() / ref.norm()).item())
    out.append(f"{which} {n}x{k}: rel err w6 {e6:.4f} w8 {e8:.4f}")
    t8 = gtime(lambda: [F.w8_linear(x, *q, (32, 512, 4, 4) if which == "down" else (32, 64, 1, 8)) for q in Q8])
    res = []
    for c in itertools.product([16, 32, 64], [64, 128, 256, 512], [1, 2, 4, 5, 8], [2, 4, 8]):
        if k % (c[1] * c[2]): continue
        try: res.append((gtime(lambda c=c: [F.w6_linear(x, *q, c) for q in Q6], it=10), c))
        except Exception: pass
    res.sort()
    out.append(f"  w8 cur {t8:.2f} ms; w6 best {res[0][0]:.2f} ms {res[0][1]}; next {[(round(a,2), b) for a, b in res[1:4]]}")
    del Q6, Q8
RESULT = "\n".join(out)
