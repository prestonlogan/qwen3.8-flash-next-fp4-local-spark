import itertools, json, sys, torch
sys.path.insert(0, "/x"); import fp4x_kernels as F
torch.manual_seed(0); dev = "cuda"
MX = [(16384, 2560), (96, 2560), (13312, 2560), (640, 2560), (1280, 2560), (2560, 6144), (2560, 640)]
BG = [(336, 10240), (10240, 320)]
Ms = [int(a) for a in (sys.argv[1:] or ["4"])]
def bench(fn, it=200):
    for _ in range(5): fn()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g):
            for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(it // 2): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it * 1e3  # us
best = {}
cands = [c for c in itertools.product([16, 32, 64], [64, 128, 256, 512], [1, 2, 4, 8], [2, 4, 8])]
for kind, shapes in (("mx8", MX), ("bg", BG)):
    for (n, k) in shapes:
        if kind == "mx8":
            w = (torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
            s = torch.randint(118, 126, (n, k // 32), device=dev, dtype=torch.uint8)
            wf = (w.float().view(n, k // 32, 32) * torch.exp2(s.float() - 127)[..., None]).view(n, k)
            nbytes = n * k + n * k // 32
        else:
            w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02; wf = w.float(); nbytes = 2 * n * k
        for m in Ms:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            ref = x.float() @ wf.T
            ncp = max(1, int(256e6 // nbytes)); W = [w.clone() for _ in range(ncp)]; S = [s.clone() for _ in range(ncp)] if kind == "mx8" else None
            res = []
            for c in cands:
                if c[1] * c[2] > k or k % (c[1] * c[2]): continue
                try:
                    o = (F.mx8_linear(x, w, s, c) if kind == "mx8" else F.bf16_linear(x, w, c)).float()
                    f = (lambda c=c: [F.mx8_linear(x, W[i], S[i], c) for i in range(ncp)]) if kind == "mx8" else (lambda c=c: [F.bf16_linear(x, W[i], c) for i in range(ncp)])
                    err = ((o - ref).abs().max() / ref.abs().max()).item()
                    if err > 2e-2: continue
                    res.append((bench(f, 40) / ncp, c))
                except Exception as ex:
                    pass
            res.sort()
            # cuBLAS / torch reference time for BF16 dequantized weight
            WB = [wf.to(torch.bfloat16) for _ in range(max(1, int(256e6 // (2 * n * k))))]; t_ref = bench(lambda: [x @ b.T for b in WB], 40) / len(WB); del WB
            t, c = res[0]
            print(f"{kind} N={n} K={k} M={m}: best {t:.1f}us {nbytes/t/1e3:.0f} GB/s cfg={c}  (torch bf16 mm {t_ref:.1f}us)", flush=True)
            best[f"{kind}:{n}:{k}:{m}"] = c
json.dump(best, open("/x/tuned_%s.json" % "_".join(map(str, Ms)), "w"))
