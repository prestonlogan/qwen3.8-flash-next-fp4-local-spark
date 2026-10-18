# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of qwen3.8-flash-next-fp4-local-spark.
# Triton decode GEMVs for FP4-Local on GB10 (ported from the E46 exp_mx8 / exp_bf16gemv kernels).
# mx8: exact MXFP8 weights (fp8 e4m3 + e8m0 per-32 scale) x BF16 activations, fp32 accumulate.
# bg:  BF16 weights x BF16 activations, fp32 accumulate.
import torch, triton, triton.language as tl


@triton.jit
def _mx8(X, W, S, O, M, N, K, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         KSPLIT: tl.constexpr):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); pid_k = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN); rm = pid_m * BM + tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per = K // KSPLIT; k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], rn[:, None] < N, other=0.0).to(tl.float32)
        rg = k0 // 32 + tl.arange(0, BK // 32)
        e = tl.load(S + rn[:, None] * (K // 32) + rg[None, :], rn[:, None] < N, other=127).to(tl.int32)
        sc = tl.exp2((e - 127).to(tl.float32))
        w = tl.reshape(tl.reshape(w, (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    if KSPLIT == 1:
        tl.store(O + rm[:, None] * N + rn[None, :], acc.to(O.dtype.element_ty), msk)
    else:
        tl.atomic_add(O + rm[:, None] * N + rn[None, :], acc, msk, sem="relaxed")


@triton.jit
def _bg(X, W, O, M, N, K, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr):
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); pid_k = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN); rm = pid_m * BM + tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per = K // KSPLIT; k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], rn[:, None] < N, other=0.0)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    if KSPLIT == 1:
        tl.store(O + rm[:, None] * N + rn[None, :], acc.to(O.dtype.element_ty), msk)
    else:
        tl.atomic_add(O + rm[:, None] * N + rn[None, :], acc, msk, sem="relaxed")


def _run(kern, x2, ws, n, k, cfg, out_dtype):
    m = x2.shape[0]
    BN, BK, KS, nw = cfg[:4]
    BM = cfg[4] if len(cfg) > 4 else (16 if m <= 16 else 64)
    assert k % KS == 0 and (k // KS) % BK == 0, (k, cfg)
    grid = (triton.cdiv(n, BN), triton.cdiv(m, BM), KS)
    if KS == 1:
        out = torch.empty((m, n), dtype=out_dtype, device=x2.device)
        kern[grid](x2, *ws, out, m, n, k, x2.stride(0), BM=BM, BN=BN, BK=BK, KSPLIT=1, num_warps=nw)
        return out
    acc = torch.zeros((m, n), dtype=torch.float32, device=x2.device)
    kern[grid](x2, *ws, acc, m, n, k, x2.stride(0), BM=BM, BN=BN, BK=BK, KSPLIT=KS, num_warps=nw)
    return acc if out_dtype == torch.float32 else acc.to(out_dtype)


def mx8_linear(x2, w8, s, cfg, out_dtype=torch.bfloat16):
    return _run(_mx8, x2, (w8, s), w8.shape[0], w8.shape[1], cfg, out_dtype)


def bf16_linear(x2, w, cfg, out_dtype=torch.bfloat16):
    return _run(_bg, x2, (w,), w.shape[0], w.shape[1], cfg, out_dtype)


@triton.jit
def _nv4(X, W, S, O, gs, M, N, K, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr):
    # W: [N, K/2] uint8 (low nibble = even k), S: [N, K/16] fp8 e4m3 block scales, gs: fp32 global scale
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per = K // KSPLIT; k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        rb = k0 // 2 + tl.arange(0, BK // 2)
        q = tl.load(W + rn[:, None] * (K // 2) + rb[None, :], rn[:, None] < N, other=0).to(tl.int32)
        lo = q & 0xF; hi = (q >> 4) & 0xF
        qq = tl.reshape(tl.join(lo, hi), (BN, BK))
        e = (qq >> 1) & 3; mt = (qq & 1).to(tl.float32)
        mag = tl.where(e == 0, mt * 0.5, (1.0 + mt * 0.5) * tl.exp2((e - 1).to(tl.float32)))
        v = tl.where((qq & 8) != 0, -mag, mag)
        rs = k0 // 16 + tl.arange(0, BK // 16)
        sc = tl.load(S + rn[:, None] * (K // 16) + rs[None, :], rn[:, None] < N, other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(v, (BN, BK // 16, 16)) * sc[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    acc = acc * gs
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    if KSPLIT == 1:
        tl.store(O + rm[:, None] * N + rn[None, :], acc.to(O.dtype.element_ty), msk)
    else:
        tl.atomic_add(O + rm[:, None] * N + rn[None, :], acc, msk, sem="relaxed")


def nv4_linear(x2, wp, s, gs, cfg=(32, 256, 4), out_dtype=torch.bfloat16):
    m, k = x2.shape; n = wp.shape[0]
    BN, BK, nw = cfg[:3]
    KS = cfg[3] if len(cfg) > 3 else 1
    assert k % KS == 0 and (k // KS) % BK == 0, (k, cfg)
    if KS == 1:
        out = torch.empty((m, n), dtype=out_dtype, device=x2.device)
        _nv4[(triton.cdiv(n, BN), 1)](x2, wp, s, out, gs, m, n, k, x2.stride(0), BM=16, BN=BN, BK=BK, KSPLIT=1, num_warps=nw)
        return out
    acc = torch.zeros((m, n), dtype=torch.float32, device=x2.device)
    _nv4[(triton.cdiv(n, BN), KS)](x2, wp, s, acc, gs, m, n, k, x2.stride(0), BM=16, BN=BN, BK=BK, KSPLIT=KS, num_warps=nw)
    return acc.to(out_dtype)


@triton.jit
def _w8(X, W, S, O, M, N, K, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr):
    # W: [N, K] int8 symmetric, S: [N, K/64] fp32 per-group scales
    pid_n = tl.program_id(0); pid_k = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per = K // KSPLIT; k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], rn[:, None] < N, other=0).to(tl.float32)
        rg = k0 // 64 + tl.arange(0, BK // 64)
        s = tl.load(S + rn[:, None] * (K // 64) + rg[None, :], rn[:, None] < N, other=0.0)
        w = tl.reshape(tl.reshape(w, (BN, BK // 64, 64)) * s[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    if KSPLIT == 1:
        tl.store(O + rm[:, None] * N + rn[None, :], acc.to(O.dtype.element_ty), msk)
    else:
        tl.atomic_add(O + rm[:, None] * N + rn[None, :], acc, msk, sem="relaxed")


@torch.no_grad()
def quantize_w8(w):
    n, k = w.shape
    f = w.float().view(n, k // 64, 64)
    s = f.abs().amax(-1).clamp_min(1e-12) / 127.0
    q = torch.round(f / s[..., None]).clamp(-127, 127).to(torch.int8).view(n, k)
    return q.contiguous(), s.contiguous()


def w8_linear(x2, q, s, cfg, out_dtype=torch.bfloat16):
    return _run(_w8, x2, (q, s), q.shape[0], q.shape[1], cfg, out_dtype)


@triton.jit
def _w6(X, L, H, S, O, M, N, K, XS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr):
    # W6A16 g32 (E46 layout): q in [-32,31] = lo-nibble plane [N,K/2] + hi-2bit plane [N,K/4], fp16 scale per 32
    pid_n = tl.program_id(0); pid_m = tl.program_id(1); pid_k = tl.program_id(2)
    rn = pid_n * BN + tl.arange(0, BN); rm = pid_m * BM + tl.arange(0, BM)
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
    if KSPLIT == 1:
        tl.store(O + rm[:, None] * N + rn[None, :], acc.to(O.dtype.element_ty), msk)
    else:
        tl.atomic_add(O + rm[:, None] * N + rn[None, :], acc, msk, sem="relaxed")


@torch.no_grad()
def quantize_w6(w):
    """w: [N,K] float -> (lo [N,K/2] u8, hi [N,K/4] u8, s [N,K/32] f16), RTN g32."""
    n, k = w.shape
    L = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
    H = torch.empty((n, k // 4), dtype=torch.uint8, device=w.device)
    S = torch.empty((n, k // 32), dtype=torch.float16, device=w.device)
    for a in range(0, n, 2048):
        f = w[a:a + 2048].float().view(-1, k // 32, 32)
        s = (f.abs().amax(-1, keepdim=True) / 31.0).to(torch.float16).float().clamp_min(6.2e-5)
        q = (torch.round(f / s).clamp(-32, 31).to(torch.int32) + 32).view(-1, k)
        lo = q & 15; hi = q >> 4
        L[a:a + 2048] = (lo[:, 0::2] | (lo[:, 1::2] << 4)).to(torch.uint8)
        H[a:a + 2048] = (hi[:, 0::4] | (hi[:, 2::4] << 2) | (hi[:, 1::4] << 4) | (hi[:, 3::4] << 6)).to(torch.uint8)
        S[a:a + 2048] = s.squeeze(-1).to(torch.float16)
    return L, H, S


@torch.no_grad()
def mx8_dequant(w8, s_u8, rows=2048):
    n, k = w8.shape
    out = torch.empty((n, k), dtype=torch.float32, device=w8.device)
    for a in range(0, n, rows):
        out[a:a + rows] = (w8[a:a + rows].float().view(-1, k // 32, 32) * torch.exp2(s_u8[a:a + rows].float() - 127)[..., None]).view(-1, k)
    return out


def w6_linear(x2, L, H, S, cfg, out_dtype=torch.bfloat16):
    return _run(_w6, x2, (L, H, S), L.shape[0], L.shape[1] * 2, cfg, out_dtype)


_E2M1 = None


@torch.no_grad()
def quantize_nv4(w, rows=2048):
    """RTN NVFP4: e2m1 values (lo nibble = even k), e4m3 scale per 16, fp32 global scale. Returns (packed, s, gs)."""
    global _E2M1
    n, k = w.shape
    if _E2M1 is None or _E2M1.device != w.device:
        _E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=w.device)
    amax_all = 0.0
    for a in range(0, n, rows):
        amax_all = max(amax_all, w[a:a + rows].float().abs().max().item())
    gs = max(amax_all / 6.0 / 448.0, 1e-30)
    P = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
    S = torch.empty((n, k // 16), dtype=torch.float8_e4m3fn, device=w.device)
    mids = (_E2M1[1:] + _E2M1[:-1]) / 2
    for a in range(0, n, rows):
        f = w[a:a + rows].float().view(-1, k // 16, 16)
        sc = (f.abs().amax(-1, keepdim=True) / 6.0 / gs).clamp(min=2 ** -9, max=448.0).to(torch.float8_e4m3fn)
        sf = sc.float() * gs
        v = (f / sf).clamp(-6, 6)
        mag = torch.bucketize(v.abs(), mids)  # 0..7
        q = (mag | ((v < 0).to(torch.int64) << 3)).view(-1, k)
        P[a:a + rows] = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
        S[a:a + rows] = sc.view(-1, k // 16)
    return P, S, gs


# --- INT2 coarse head (from E46 exp_head2.py) ---
@triton.jit
def _int2_head(X, W, S, O, RS, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               XM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ROW_SCALE: tl.constexpr):
    # S: per-16 group scales (fp16, or e4m3 relative to the fp32 per-row scale RS when ROW_SCALE)
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XM + rk[None, :], (rm[:, None] < M), other=0.0)
        rkq = k0 // 4 + tl.arange(0, BK // 4)
        p = tl.load(W + rn[:, None] * (K // 4) + rkq[None, :], (rn[:, None] < N), other=0)
        w = tl.reshape(tl.join(tl.join(p & 3, (p >> 2) & 3), tl.join((p >> 4) & 3, p >> 6)), (BN, BK)).to(tl.float32) - 1.5
        rg = k0 // 16 + tl.arange(0, BK // 16)
        s = tl.load(S + rn[:, None] * (K // 16) + rg[None, :], (rn[:, None] < N), other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(w, (BN, BK // 16, 16)) * s[:, :, None], (BN, BK))
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    if ROW_SCALE:
        acc = acc * tl.load(RS + rn, rn < N, other=0.0)[None, :]
    tl.store(O + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))

# kernel element order within one byte after join/reshape: [b0-1, b4-5, b2-3, b6-7]
ORDER = (0, 2, 1, 3)   # element j of each 4-group lives in bit-pair ORDER[j]

@torch.no_grad()
def quantize_int2(weight: torch.Tensor, scale_fmt: str = "fp16"):
    """INT2 g16 (levels -1.5..1.5 x s). scale_fmt fp16 -> (packed, s); e4m3 -> (packed, s_e4m3, row_scale_fp32)."""
    n, k = weight.shape
    packed = torch.empty((n, k // 4), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((n, k // 16), dtype=torch.float16 if scale_fmt == "fp16" else torch.float8_e4m3fn, device=weight.device)
    rows = torch.empty((n,), dtype=torch.float32, device=weight.device)
    for a in range(0, n, 8192):
        f = weight[a:a + 8192].float().view(-1, k // 16, 16)
        s = f.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 1.5
        if scale_fmt == "fp16":
            s = s.half(); sf = s.float()
        else:
            rs = s.amax(1, keepdim=True) / 448.0
            s = (s / rs).to(torch.float8_e4m3fn); sf = s.float() * rs
            rows[a:a + 8192] = rs.view(-1)
        idx = (torch.floor(f / sf) + 2).clamp(0, 3).to(torch.uint8).view(-1, k // 4, 4)
        b = torch.zeros(idx.shape[:2], dtype=torch.uint8, device=weight.device)
        for j in range(4):
            b |= idx[..., j] << (2 * ORDER[j])
        packed[a:a + 8192] = b; scales[a:a + 8192] = s.squeeze(-1)
    return (packed, scales) if scale_fmt == "fp16" else (packed, scales, rows)

def dequant_int2(packed, scales):
    n = packed.shape[0]; k = packed.shape[1] * 4
    idx = torch.stack([(packed >> (2 * ORDER[j])) & 3 for j in range(4)], -1).view(n, k).float() - 1.5
    return (idx.view(n, k // 16, 16) * scales.float()[..., None]).view(n, k)

def int2_logits(x, P, S, N, BN=64, BK=512, nw=4, RS=None):
    x = x.to(torch.bfloat16).contiguous(); m, k = x.shape
    out = torch.empty((m, N), dtype=torch.float32, device=x.device)
    _int2_head[(triton.cdiv(N, BN),)](x, P, S, out, out if RS is None else RS, m, N, k, x.stride(0),
                                      max(16, triton.next_power_of_2(m)), BN, BK, RS is not None, num_warps=nw)
    return out


@triton.jit
def _refine_mx8(X, IDX, W, S, O, C, K: tl.constexpr, V, BC: tl.constexpr, BK: tl.constexpr):
    # Exact logits for C candidate rows per token from MXFP8 rows (fp8 e4m3 + e8m0 per 32), scattered into O [M, V].
    m = tl.program_id(0); cb = tl.program_id(1)
    rc = cb * BC + tl.arange(0, BC)
    rows = tl.load(IDX + m * C + rc, rc < C, other=0).to(tl.int64)
    acc = tl.zeros((BC,), tl.float32)
    for k0 in tl.range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + m * K + rk).to(tl.float32)
        w = tl.load(W + rows[:, None] * K + rk[None, :], (rc < C)[:, None], other=0.0).to(tl.float32)
        e = tl.load(S + rows[:, None] * (K // 32) + (k0 // 32 + tl.arange(0, BK // 32))[None, :], (rc < C)[:, None], other=127).to(tl.int32)
        w = tl.reshape(tl.reshape(w, (BC, BK // 32, 32)) * tl.exp2((e - 127).to(tl.float32))[:, :, None], (BC, BK))
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(O + m * V + rows, acc.to(O.dtype.element_ty), rc < C)


def refine_mx8_(out, x, idx, w8, s_u8, BC=16, BK=256, nw=4):
    """In place: out[m, idx[m, c]] = exact MXFP8 row idx[m, c] . x[m] (fp32 accumulate)."""
    m, C = idx.shape; K = x.shape[1]
    _refine_mx8[(m, triton.cdiv(C, BC))](x.contiguous(), idx.contiguous(), w8, s_u8, out, C, K, out.shape[1], BC=BC, BK=BK, num_warps=nw)
    return out


@triton.jit
def _refine_nv4(X, IDX, W, S, O, gs, C, K: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr):
    # Exact logits of C candidate rows per token from NVFP4 rows (same decode as _nv4); O: [M, C] fp32.
    m = tl.program_id(0); cb = tl.program_id(1)
    rc = cb * BC + tl.arange(0, BC)
    rows = tl.load(IDX + m * C + rc, rc < C, other=0).to(tl.int64)
    acc = tl.zeros((BC,), tl.float32)
    for k0 in tl.range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + m * K + rk).to(tl.float32)
        q = tl.load(W + rows[:, None] * (K // 2) + (k0 // 2 + tl.arange(0, BK // 2))[None, :], (rc < C)[:, None], other=0).to(tl.int32)
        qq = tl.reshape(tl.join(q & 0xF, (q >> 4) & 0xF), (BC, BK))
        e = (qq >> 1) & 3; mt = (qq & 1).to(tl.float32)
        mag = tl.where(e == 0, mt * 0.5, (1.0 + mt * 0.5) * tl.exp2((e - 1).to(tl.float32)))
        v = tl.where((qq & 8) != 0, -mag, mag)
        sc = tl.load(S + rows[:, None] * (K // 16) + (k0 // 16 + tl.arange(0, BK // 16))[None, :], (rc < C)[:, None], other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(v, (BC, BK // 16, 16)) * sc[:, :, None], (BC, BK)).to(tl.bfloat16).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(O + m * C + rc, acc * gs, rc < C)


def refine_nv4(x, idx, wp, s, gs, BC=16, BK=256, nw=4):
    """[M, C] exact NVFP4 logits for candidate rows idx[m, c] (fp32 accumulate)."""
    m, C = idx.shape; K = x.shape[1]
    out = torch.empty((m, C), dtype=torch.float32, device=x.device)
    _refine_nv4[(m, triton.cdiv(C, BC))](x.contiguous(), idx.contiguous(), wp, s, out, gs, C, K, BC=BC, BK=BK, num_warps=nw)
    return out


@torch.no_grad()
def dequant_nv4(P, S, gs):
    """[N, K/2] e2m1 pairs + [N, K/16] e4m3 scales -> fp32 [N, K]."""
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=P.device)
    n = P.shape[0]
    v = torch.stack([lut[(P & 15).long()], lut[(P >> 4).long()]], -1).view(n, -1)
    return (v.view(n, -1, 16) * S.float()[..., None] * gs).view(n, -1)


# --- Fused hyper-connection mixer kernels (decode, M <= 16, W8 weights):
#   hc_down: split-K W8 GEMV whose last CTA per column tile casts the fp32 sum to BF16 (down/injection output)
#            and applies the bottleneck activation silu(bf16(x / HC)) in its epilogue; replaces
#            zero-fill + GEMV + cast + scaled_silu.
#   hc_up:   W8 GEMV over stream-interleaved up rows whose epilogue applies the sigmoid gate and the stream
#            mean against the normalized input; replaces GEMV + gate_mean.
# Rounding points follow b12x's _ScaledSilu / _GateMean kernels.


@triton.jit
def _hc_down(X, W, S, ACC, CNT, DI, BOT, M, XS, N: tl.constexpr, K: tl.constexpr, LR: tl.constexpr,
             HC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    k_per: tl.constexpr = K // KSPLIT
    k_lo = pid_k * k_per
    for k0 in tl.range(k_lo, k_lo + k_per, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], rn[:, None] < N, other=0).to(tl.float32)
        rg = k0 // 64 + tl.arange(0, BK // 64)
        s = tl.load(S + rn[:, None] * (K // 64) + rg[None, :], rn[:, None] < N, other=0.0)
        w = tl.reshape(tl.reshape(w, (BN, BK // 64, 64)) * s[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    aptr = ACC + rm[:, None] * N + rn[None, :]
    tl.atomic_add(aptr, acc, msk, sem="relaxed")
    tl.debug_barrier()
    old = tl.atomic_add(CNT + pid_n, 1, sem="acq_rel")
    if old == KSPLIT - 1:
        tot = tl.load(aptr, msk, other=0.0, cache_modifier=".cg")
        v = tot.to(tl.bfloat16)
        tl.store(DI + rm[:, None] * N + rn[None, :], v, msk)
        sc = (v.to(tl.float32) / HC).to(tl.bfloat16).to(tl.float32)
        tl.store(BOT + rm[:, None] * LR + rn[None, :], (sc * tl.sigmoid(sc)).to(tl.bfloat16), msk & (rn[None, :] < LR))
        tl.store(aptr, tl.zeros((BM, BN), tl.float32), msk)
        tl.debug_barrier()
        tl.atomic_xchg(CNT + pid_n, 0, sem="relaxed")


@triton.jit
def _hc_up(X, W, S, NRM, O, M, XS, NS, K: tl.constexpr, H: tl.constexpr, HC: tl.constexpr,
           BM: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
    # W rows are stream-interleaved: tile j covers rows [s * H + j * BJ + t] for s < HC, t < BJ, stored contiguously.
    BN: tl.constexpr = HC * BJ
    pid = tl.program_id(0)
    rr = pid * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XS + rk[None, :], rm[:, None] < M, other=0.0)
        w = tl.load(W + rr[:, None] * K + rk[None, :]).to(tl.float32)
        rg = k0 // 64 + tl.arange(0, BK // 64)
        s = tl.load(S + rr[:, None] * (K // 64) + rg[None, :])
        w = tl.reshape(tl.reshape(w, (BN, BK // 64, 64)) * s[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    g = tl.reshape(acc.to(tl.bfloat16).to(tl.float32), (BM, HC, BJ))
    gate = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    rs = tl.arange(0, HC); rj = pid * BJ + tl.arange(0, BJ)
    col = rs[:, None] * H + rj[None, :]
    mm = (rm < M)[:, None, None]
    val = tl.load(NRM + rm[:, None, None] * NS + col[None, :, :], mm, other=0.0).to(tl.float32)
    tot = tl.sum((val * gate).to(tl.bfloat16).to(tl.float32), axis=1)
    tl.store(O + rm[:, None] * H + rj[None, :], (tot / HC).to(tl.bfloat16), (rm < M)[:, None])


class HCFused:
    """Per-process scratch (one split-K accumulator + tile counters shared by every mixer: calls are stream-ordered)."""

    def __init__(self, device, n_down=336, max_m=16, bn=32):
        self.acc = torch.zeros((max_m, n_down), dtype=torch.float32, device=device)
        self.cnt = torch.zeros((triton.cdiv(n_down, bn),), dtype=torch.int32, device=device)

    def down(self, x, q, s, lr, hc, cfg=(32, 512, 4, 4)):
        BN, BK, KS, nw = cfg
        m, k = x.shape; n = q.shape[0]
        di = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        bot = torch.empty((m, lr), dtype=torch.bfloat16, device=x.device)
        _hc_down[(triton.cdiv(n, BN), KS)](x, q, s, self.acc, self.cnt, di, bot, m, x.stride(0), N=n, K=k, LR=lr, HC=hc,
                                           BM=16, BN=BN, BK=BK, KSPLIT=KS, num_warps=nw)
        return di, bot

    @staticmethod
    def interleave_up(q, s, hc, bj):
        """Reorder up-projection rows [HC*H, K] so each BJ-column tile holds its HC stream rows contiguously."""
        n, k = q.shape; h = n // hc
        idx = torch.arange(n, device=q.device).view(hc, h // bj, bj).permute(1, 0, 2).reshape(-1)
        return q[idx].contiguous(), s[idx].contiguous()

    @staticmethod
    def up(bot, qi, si, normalized, hc, cfg=(8, 64, 8)):
        BJ, BK, nw = cfg
        m, k = bot.shape; n = qi.shape[0]; h = n // hc
        out = torch.empty((m, h), dtype=torch.bfloat16, device=bot.device)
        _hc_up[(h // BJ,)](bot, qi, si, normalized, out, m, bot.stride(0), normalized.stride(0), K=k, H=h, HC=hc,
                           BM=16, BJ=BJ, BK=BK, num_warps=nw)
        return out
