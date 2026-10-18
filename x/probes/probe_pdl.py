# Microbench: does programmatic dependent launch (PDL) shorten chains of small GEMVs inside a CUDA graph?
import torch, sys, triton, triton.language as tl
from triton.language.extra.cuda import gdc_wait, gdc_launch_dependents
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

@triton.jit
def _w8p(X, W, S, O, M, N, K, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    # MODE 0: no PDL calls; 1: trigger+wait at start; 2: trigger at start, first weight tile loaded before wait
    pid_n = tl.program_id(0)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    if MODE == 1 or MODE == 2:
        gdc_launch_dependents()
    rk = tl.arange(0, BK)
    w = tl.load(W + rn[:, None] * K + rk[None, :], rn[:, None] < N, other=0).to(tl.float32)
    rg = tl.arange(0, BK // 64)
    s = tl.load(S + rn[:, None] * (K // 64) + rg[None, :], rn[:, None] < N, other=0.0)
    if MODE >= 1:
        gdc_wait()
    x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
    w = tl.reshape(tl.reshape(w, (BN, BK // 64, 64)) * s[:, :, None], (BN, BK)).to(tl.bfloat16)
    acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    for k0 in tl.range(BK, K, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], rn[:, None] < N, other=0).to(tl.float32)
        rg = k0 // 64 + tl.arange(0, BK // 64)
        s = tl.load(S + rn[:, None] * (K // 64) + rg[None, :], rn[:, None] < N, other=0.0)
        w = tl.reshape(tl.reshape(w, (BN, BK // 64, 64)) * s[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(O + rm[:, None] * N + rn[None, :], acc.to(O.dtype.element_ty), msk)

def w8p(x, q, s, o, BN, BK, nw, mode):
    n, k = q.shape
    _w8p[(triton.cdiv(n, BN),)](x, q, s, o, x.shape[0], n, k, x.stride(0), BM=16, BN=BN, BK=BK, MODE=mode, num_warps=nw, launch_pdl=mode >= 1)

M = 3
hcs = [h for n, h in mods.items() if type(h).__name__ == "GatedResidual" and "mtp" not in n]
gates = [mods[f"language_model.model.layers.{i}.mlp.gate"].weight for i in range(48)]
for label, Ws, BN, BK, nw in (("HC up 10240x320", [h.input_mix_weight_up.weight for h in hcs], 32, 64, 8),
                               ("router 512x2560", gates, 16, 512, 4),
                               )[:2]:
    Q = [F.quantize_w8(w) for w in Ws]
    n, k = Ws[0].shape
    # chain: each GEMV reads the previous output's first k columns (real data dependency)
    bufs = [torch.randn(M, max(n, k), device="cuda", dtype=torch.bfloat16) for _ in range(len(Q) + 1)]
    ref = None
    res = []
    for inter in (False, True):
      for mode in (0, 1, 2, 3):
        def chain(mode=mode, inter=inter):
            for i, (q, s) in enumerate(Q):
                if inter: bufs[0][:, :8].mul_(1.0)
                w8p(bufs[0][:, :k], q, s, bufs[1 + i][:, :n], BN, BK, nw, mode)
        t = gtime(chain)
        res.append(f"{'+torch op ' if inter else ''}mode{mode} {t/len(Q)*1000:.1f} us")
    out.append(f"{label} x{len(Q)}: " + "; ".join(res))
    del Q, bufs
RESULT = "\n".join(out)
