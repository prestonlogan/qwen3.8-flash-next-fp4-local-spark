import torch, time, fp4x_kernels as F
RES = []
for (n, k), cfgs in {(336, 10240): [(16,128,2,16),(16,128,2,20),(16,64,2,32),(16,128,4,16),(16,128,1,16),(32,128,2,16),(16,256,2,20),(16,64,1,40),(16,128,2,40)],
                     (10240, 320): [(32,64,4,1),(128,64,4,1),(64,64,2,1),(32,32,2,1),(16,64,1,1)]}.items():
    W = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    qs = F.quantize_nv4(W)
    x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    ref = F.nv4_linear(x, *qs, (16, 64, 4, 1)).float()
    copies = [F.quantize_nv4(torch.randn(n, k, device="cuda", dtype=torch.bfloat16)*0.02) for _ in range(24)]
    for cfg in cfgs:
        try:
            out = F.nv4_linear(x, *qs, cfg).float()
            err = ((out-ref).abs().max() / ref.abs().max()).item()
            torch.cuda.synchronize(); t = time.perf_counter()
            for r in range(10):
                for c in copies: F.nv4_linear(x, *c, cfg)
            torch.cuda.synchronize(); us = (time.perf_counter()-t)/240*1e6
            RES.append(((n,k), cfg, round(err,4), round(us,1)))
        except Exception as e:
            RES.append(((n,k), cfg, repr(e)[:80]))
RESULT = "\n".join(map(str, RES))
