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
for (n, k) in [(16384, 2560), (13312, 2560), (1280, 2560), (640, 2560)]:
    w = torch.randn(n, k, device="cuda") * 0.02
    q0 = F.quantize_nv4(w); ncp = max(1, int(200e6 // (n * k * 0.5625))); Q = [tuple(t.clone() if torch.is_tensor(t) else t for t in q0) for _ in range(ncp)]
    x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16); ref = x.float() @ w.T
    res = []
    for c in itertools.product([16, 32, 64, 128], [128, 256, 512], [2, 4, 8]):
        if k % c[1]: continue
        try:
            o = F.nv4_linear(x, *Q[0], c).float()
            if ((o - ref).abs().max() / ref.abs().max()).item() > 0.2: continue
            res.append((bench(lambda c=c: [F.nv4_linear(x, *q, c) for q in Q]) / ncp, c))
        except Exception: pass
    res.sort(); print(f"nv4 N={n} K={k}: {res[0][0]:.1f}us cfg={res[0][1]} GB/s {n*k*0.5625/res[0][0]/1e3:.0f}", flush=True)
