import itertools, sys, torch
sys.path.insert(0, "/x"); import fp4x_kernels as F
dev = "cuda"
def bench(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it * 1e3
for (n, k) in [(16384, 2560), (13312, 2560), (1280, 2560)]:
    w = (torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn); s = torch.randint(118, 126, (n, k // 32), device=dev, dtype=torch.uint8)
    wb = (w.float().view(n, k // 32, 32) * torch.exp2(s.float() - 127)[..., None]).view(n, k).to(torch.bfloat16)
    for m in (512, 4096):
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        res = []
        for c in itertools.product([64, 128, 256], [32, 64], [1], [4, 8], [64, 128, 256]):
            try: res.append((bench(lambda c=c: F.mx8_linear(x, w, s, c)), c))
            except Exception as e: pass
        res.sort(); t_ref = bench(lambda: x @ wb.T)
        print(f"N={n} K={k} M={m}: triton {res[0][0]:.0f}us cfg={res[0][1]}  cublas-bf16 {t_ref:.0f}us  TFLOPS {2*m*n*k/res[0][0]/1e6:.1f}", flush=True)
