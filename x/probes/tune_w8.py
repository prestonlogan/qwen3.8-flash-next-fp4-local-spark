import itertools, sys, torch
sys.path.insert(0, "/x"); import fp4x_kernels as F
def bench(fn, it=40):
    for _ in range(3): fn()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it * 1e3
for (n, k) in [(336, 10240), (10240, 320)]:
    ncp = 40; W = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(ncp)]
    Q = [F.quantize_w8(w) for w in W]
    x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ W[0].float().T
    res = []
    for c in itertools.product([16, 32, 64], [64, 128, 256, 512], [1, 2, 4, 8], [2, 4, 8]):
        if c[1] * c[2] > k or k % (c[1] * c[2]): continue
        try:
            o = F.w8_linear(x, *Q[0], c).float()
            if ((o - ref).abs().max() / ref.abs().max()).item() > 3e-2: continue
            res.append((bench(lambda c=c: [F.w8_linear(x, q, s, c) for q, s in Q]) / ncp, c))
        except Exception: pass
    res.sort(); tb = bench(lambda: [F.bf16_linear(x, w, (16, 512, 1, 4) if k > 1000 else (32, 64, 1, 4)) for w in W]) / ncp
    print(f"w8 N={n} K={k}: {res[0][0]:.1f}us cfg={res[0][1]}  (bf16 triton {tb:.1f}us)", flush=True)
