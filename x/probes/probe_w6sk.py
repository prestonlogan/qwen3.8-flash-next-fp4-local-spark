# W6 GEMV with semaphore split-K (last CTA casts + re-zeroes the fp32 tile) vs the served W6 kernel, real weights.
import torch, sys, itertools, triton, triton.language as tl
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
def _w6s(X, L, H, S, ACC, CNT, O, M, N, K, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per = K // KSPLIT; k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        rl = k0 // 2 + tl.arange(0, BK // 2)
        p = tl.load(L + rn[:, None] * (K // 2) + rl[None, :], rn[:, None] < N, other=0)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        lo = tl.reshape(tl.join(p & 0xF, p >> 4), (BN, BK)).to(tl.int32)
        rh = k0 // 4 + tl.arange(0, BK // 4)
        h = tl.load(H + rn[:, None] * (K // 4) + rh[None, :], rn[:, None] < N, other=0)
        hi = tl.reshape(tl.join(tl.join(h & 3, (h >> 2) & 3), tl.join((h >> 4) & 3, h >> 6)), (BN, BK)).to(tl.int32)
        q = (lo | (hi << 4)) - 32
        rg = k0 // 32 + tl.arange(0, BK // 32)
        s = tl.load(S + rn[:, None] * (K // 32) + rg[None, :], rn[:, None] < N, other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(q.to(tl.float32), (BN, BK // 32, 32)) * s[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    aptr = ACC + rm[:, None] * N + rn[None, :]
    tl.atomic_add(aptr, acc, msk, sem="relaxed")
    tl.debug_barrier()
    old = tl.atomic_add(CNT + pid_n, 1, sem="acq_rel")
    if old == KSPLIT - 1:
        tot = tl.load(aptr, msk, other=0.0, cache_modifier=".cg")
        tl.store(O + rm[:, None] * N + rn[None, :], tot.to(tl.bfloat16), msk)
        tl.store(aptr, tl.zeros((BM, BN), tl.float32), msk)
        tl.debug_barrier()
        tl.atomic_xchg(CNT + pid_n, 0, sem="relaxed")

ACC = torch.zeros((16, 16384), dtype=torch.float32, device="cuda")
CNT = torch.zeros((2048,), dtype=torch.int32, device="cuda")
def w6s(x, L, H, S, cfg):
    BN, BK, KS, nw = cfg
    m, k = x.shape; n = L.shape[0]
    o = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _w6s[(triton.cdiv(n, BN), KS)](x, L, H, S, ACC, CNT, o, m, n, k, x.stride(0), BM=16, BN=BN, BK=BK, KSPLIT=KS, num_warps=nw)
    return o

from vllm.model_executor.kernels.linear import b12x_blockscaled as BB
def find_cache(fn, depth=0):
    for c in (getattr(fn, "__closure__", None) or ()):
        v = c.cell_contents
        if isinstance(v, dict) and v and isinstance(next(iter(v.values())), tuple): return v
        if callable(v) and depth < 2:
            r_ = find_cache(v, depth + 1)
            if r_ is not None: return r_
    return None
cache = find_cache(BB.B12xBlockscaledLinear.run)
# the plugin's lowbit cache is keyed by id(B12xBlockscaledLinear runner); find runners per layer
out.append(f"lowbit cache entries: {len(cache)}")
qs_by_shape = {}
for key, q in cache.items():
    if len(q) == 3 and q[0].dtype == torch.uint8 and q[1].dtype == torch.uint8:
        n = q[0].shape[0]; k = q[0].shape[1] * 2
        qs_by_shape.setdefault((n, k), []).append(q)
M = 3
CFG_W6 = {(16384, 2560): (32, 256, 1, 4), (96, 2560): (16, 256, 2, 4), (13312, 2560): (16, 256, 1, 4),
          (640, 2560): (16, 256, 1, 2), (1280, 2560): (16, 256, 1, 2), (2560, 6144): (32, 256, 1, 4), (2560, 640): (32, 128, 1, 2)}
for (n, k), Q in sorted(qs_by_shape.items()):
    x = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
    c0 = CFG_W6.get((n, k), (32, 256, 1, 4))
    ref = F.w6_linear(x, *Q[0], c0)
    t0 = gtime(lambda: [F.w6_linear(x, *q, c0) for q in Q])
    res = []
    for c in itertools.product([16, 32, 64], [128, 256, 512], [1, 2, 3, 4, 6, 8], [2, 4, 8]):
        if k % (c[1] * c[2]) or (n // c[0]) > 2048: continue
        try:
            o = w6s(x, *Q[0], c)
            if (o.float() - ref.float()).abs().max().item() > 1e-2 * ref.float().abs().max().item(): out.append(f"mismatch {c}"); continue
            res.append((gtime(lambda c=c: [w6s(x, *q, c) for q in Q], it=10), c))
        except Exception as e: pass
    res.sort()
    out.append(f"{n}x{k} x{len(Q)}: served {t0:.3f} ms {c0}; semaphore best {[(round(a, 3), b) for a, b in res[:3]]}")
RESULT = "\n".join(out)
