# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of qwen3.8-flash-next-fp4-local-spark.
# Not used by the plugin: a dead end kept for reference (b12x's routed GEMMs already run at the bandwidth floor).
# Decode (M <= 4) routed-expert kernels reading b12x's prepared W4A16 NVFP4 expert tensors in place.
# Layout (verified against the checkpoint): w[e, kt, nt*128 + p] int32 holds a 16(k) x 64(n) tile;
# p = tc_col*16 + r*4 + w, nibble slot s = a*4 + b*2 + c  ->  k = 2r + a + 8c, n = 16w + tc_col + 8b.
# w1 rows 0..I-1 are gate, I..2I-1 are up. Scales: byte per (k-tile, n) at a permuted column, e5m3
# (fp16 bits = byte << 7); real weight = e2m1 * fp16(byte << 7) * alpha[e] * 2^-126.
import torch, triton, triton.language as tl


@triton.jit
def _e2m1(q):
    # true e2m1 value of nibble q (int32), fp32
    sgn = (q & 8) << 28
    v = (((q & 7) << 22) + (126 << 23)) | sgn
    v = v.to(tl.float32, bitcast=True)
    sub = tl.where((q & 1) != 0, 0.5, 0.0)
    sub = tl.where((q & 8) != 0, -sub, sub)
    return tl.where((q & 6) == 0, sub, v)


@triton.jit
def _scale_col(n):
    # position of logical column n (within its 64-block) in b12x's permuted scale row
    c = n % 64
    j1 = (c % 8) * 8 + c // 8
    j2 = (j1 & ~3) | ((j1 & 1) << 1) | ((j1 >> 1) & 1)
    return (n - c) + j2


@triton.jit
def _tile_dot(W, S, x_ptr, xs, m_ok, kt0, KT: tl.constexpr, nt, NROW, M_PAD: tl.constexpr):
    # returns acc[M_PAD, 8(tc_col), 4(w), 2(b)] for k-tiles kt0..kt0+KT of n-tile nt (scales applied, alpha not)
    p = tl.arange(0, 128)
    ktr = kt0 + tl.arange(0, KT)
    q = tl.load(W + ktr[:, None] * (NROW // 64 * 128) + nt * 128 + p[None, :])          # [KT, 128] int32
    q = tl.reshape(q, (KT, 8, 4, 4))                                                    # kt, tc_col, r, w
    sh = tl.arange(0, 8) * 4
    nib = (q[:, :, :, :, None] >> sh[None, None, None, None, :]) & 0xF                  # kt, tc, r, w, slot
    nib = tl.reshape(nib, (KT, 8, 4, 4, 2, 2, 2))                                       # kt, tc, r, w, a, b, c
    v = _e2m1(nib)
    # scale per (kt, n): n = 16w + tc + 8b
    tc = tl.arange(0, 8); wv = tl.arange(0, 4); bv = tl.arange(0, 2)
    nl = nt * 64 + 16 * wv[None, :, None] + tc[:, None, None] + 8 * bv[None, None, :]    # [8, 4, 2]
    sc_b = tl.load(S + ktr[:, None, None, None] * NROW + _scale_col(nl)[None, :, :, :])  # [KT, 8, 4, 2] uint8
    sc = (sc_b.to(tl.int16) << 7).to(tl.float16, bitcast=True).to(tl.float32)
    v = v * tl.reshape(sc, (KT, 8, 1, 4, 1, 2, 1))
    # x at k = kt*16 + 2r + a + 8c
    rv = tl.arange(0, 4); av = tl.arange(0, 2); cv = tl.arange(0, 2)
    kk = ktr[:, None, None, None] * 16 + 2 * rv[None, :, None, None] + av[None, None, :, None] + 8 * cv[None, None, None, :]  # [KT,4,2,2]
    mm = tl.arange(0, M_PAD)
    xv = tl.load(x_ptr + mm[:, None, None, None, None] * xs + kk[None], m_ok[:, None, None, None, None], other=0.0).to(tl.float32)  # [M,KT,4,2,2]
    prod = v[None] * tl.reshape(xv, (M_PAD, KT, 1, 4, 1, 2, 1, 2))                    # M, kt, tc, r, w, a, b, c
    prod = tl.sum(prod, axis=7)
    prod = tl.sum(prod, axis=5)
    prod = tl.sum(prod, axis=3)
    prod = tl.sum(prod, axis=1)                                                         # M, tc, w, b
    return prod


@triton.jit
def _moe_up(X, IDS, W1, S1, AL1, ACT, M, XS, TOPK: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
            M_PAD: tl.constexpr, KT: tl.constexpr):
    # grid (M*TOPK slots, I//64 n-tiles of the gate half). Leader slot per unique expert computes gate & up tiles for
    # every token routed to it: ACT[slot, m, n] = silu(gate) * up (bf16), for n in this tile.
    s = tl.program_id(0); nt = tl.program_id(1)
    NS = M * TOPK
    sl = tl.arange(0, 64)
    ids = tl.load(IDS + sl, sl < NS, other=-1)
    e = tl.load(IDS + s)
    if (e < 0) | (tl.sum(tl.where((sl < s) & (ids == e), 1, 0)) > 0):
        return
    mm = tl.arange(0, M_PAD)
    tok_hit = tl.sum(tl.where((ids[None, :] == e) & ((sl[None, :] // TOPK) == mm[:, None]), 1, 0), axis=1) > 0
    m_ok = tok_hit & (mm < M)
    W = W1 + e.to(tl.int64) * (K // 16) * (2 * I // 64 * 128)
    S = S1 + e.to(tl.int64) * (K // 16) * (2 * I)
    g = tl.zeros((M_PAD, 8, 4, 2), tl.float32)
    u = tl.zeros((M_PAD, 8, 4, 2), tl.float32)
    for kt0 in range(0, K // 16, KT):
        g += _tile_dot(W, S, X, XS, m_ok, kt0, KT, nt, 2 * I, M_PAD)
        u += _tile_dot(W, S, X, XS, m_ok, kt0, KT, nt + I // 64, 2 * I, M_PAD)
    al = tl.load(AL1 + e) * 1.1754943508222875e-38  # alpha * 2^-126
    g = (g * al).to(tl.bfloat16).to(tl.float32)
    u = (u * al).to(tl.bfloat16).to(tl.float32)
    sg = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    a = (sg * u).to(tl.bfloat16)
    tc = tl.arange(0, 8); wv = tl.arange(0, 4); bv = tl.arange(0, 2)
    nl = nt * 64 + 16 * wv[None, :, None] + tc[:, None, None] + 8 * bv[None, None, :]
    ptr = ACT + (s * M_PAD + mm[:, None, None, None]) * I + nl[None]
    tl.store(ptr, a, m_ok[:, None, None, None])


@triton.jit
def _moe_down(ACT, IDS, WTS, W2, S2, AL2, OUT, M, TOPK: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
              M_PAD: tl.constexpr, KT: tl.constexpr):
    # grid (M*TOPK slots, H//64). Leader slot: OUT[m, n] += w[m, e] * (W2_e @ ACT[slot, m]) (fp32 atomics).
    s = tl.program_id(0); nt = tl.program_id(1)
    NS = M * TOPK
    sl = tl.arange(0, 64)
    ids = tl.load(IDS + sl, sl < NS, other=-1)
    e = tl.load(IDS + s)
    if (e < 0) | (tl.sum(tl.where((sl < s) & (ids == e), 1, 0)) > 0):
        return
    mm = tl.arange(0, M_PAD)
    hit = (ids[None, :] == e) & ((sl[None, :] // TOPK) == mm[:, None])
    m_ok = (tl.sum(tl.where(hit, 1, 0), axis=1) > 0) & (mm < M)
    wts = tl.load(WTS + sl, sl < NS, other=0.0).to(tl.float32)
    wm = tl.sum(tl.where(hit, wts[None, :], 0.0), axis=1)                               # [M_PAD]
    W = W2 + e.to(tl.int64) * (I // 16) * (H // 64 * 128)
    S = S2 + e.to(tl.int64) * (I // 16) * H
    acc = tl.zeros((M_PAD, 8, 4, 2), tl.float32)
    for kt0 in range(0, I // 16, KT):
        acc += _tile_dot(W, S, ACT + s * M_PAD * I, I, m_ok, kt0, KT, nt, H, M_PAD)
    al = tl.load(AL2 + e)
    f = (wm * al).to(tl.bfloat16).to(tl.float32) * 1.1754943508222875e-38           # bf16(tw * alpha2) * 2^-126
    acc = (acc.to(tl.bfloat16).to(tl.float32) * f[:, None, None, None]).to(tl.bfloat16).to(tl.float32)
    tc = tl.arange(0, 8); wv = tl.arange(0, 4); bv = tl.arange(0, 2)
    nl = nt * 64 + 16 * wv[None, :, None] + tc[:, None, None] + 8 * bv[None, None, :]
    tl.atomic_add(OUT + mm[:, None, None, None] * H + nl[None], acc, m_ok[:, None, None, None], sem="relaxed")


def moe_decode(x, ids, wts, w1, s1, al1, w2, s2, al2, I=640, KT1=8, KT2=8, nw1=4, nw2=4, out_bf16=None):
    """x [M, H] bf16, ids [M, topk] int, wts [M, topk] fp32 -> [M, H] fp32 (sum of routed experts); also cast
    into out_bf16 if given."""
    m, h = x.shape; topk = ids.shape[1]
    mp = max(4, triton.next_power_of_2(m))
    ids_f = ids.reshape(-1).to(torch.int32).contiguous()
    act = torch.empty((m * topk, mp, I), dtype=torch.bfloat16, device=x.device)
    out = torch.zeros((m, h), dtype=torch.float32, device=x.device)
    _moe_up[(m * topk, I // 64)](x, ids_f, w1, s1.view(torch.uint8), al1, act, m, x.stride(0), TOPK=topk, I=I, K=h,
                                 M_PAD=mp, KT=KT1, num_warps=nw1)
    _moe_down[(m * topk, h // 64)](act, ids_f, wts.reshape(-1).float().contiguous(), w2, s2.view(torch.uint8), al2, out,
                                   m, TOPK=topk, I=I, H=h, M_PAD=mp, KT=KT2, num_warps=nw2)
    if out_bf16 is not None:
        out_bf16.copy_(out)
    return out


# ---- v2: dequantize a [KT*16, 64] bf16 tile (exact: e2m1 x e4m3 fits bf16) and use tensor-core dots (M padded to 16).
@triton.jit
def _deq_tile(W, S, kt0, KT: tl.constexpr, nt, NROW):
    p = tl.arange(0, 128)
    ktr = kt0 + tl.arange(0, KT)
    q = tl.load(W + ktr[:, None] * (NROW // 64 * 128) + nt * 128 + p[None, :])          # [KT, 128] int32
    sh = tl.arange(0, 8) * 4
    nib = (q[:, :, None] >> sh[None, None, :]) & 0xF                                   # kt, (tc, r, w), (a, b, c)
    nib = tl.reshape(nib, (KT, 8, 4, 4, 2, 2, 2))
    nib = tl.permute(nib, (0, 6, 2, 4, 3, 5, 1))                                       # kt, c, r, a | w, b, tc
    v = tl.reshape(_e2m1(nib), (KT, 16, 64))
    n = nt * 64 + tl.arange(0, 64)
    sc_b = tl.load(S + ktr[:, None] * NROW + _scale_col(n)[None, :])                   # [KT, 64] uint8
    sc = (sc_b.to(tl.int16) << 7).to(tl.float16, bitcast=True).to(tl.float32)
    return tl.reshape(v * sc[:, None, :], (KT * 16, 64)).to(tl.bfloat16)


@triton.jit
def _moe_up2(X, IDS, W1, S1, AL1, ACT, M, XS, TOPK: tl.constexpr, I: tl.constexpr, K: tl.constexpr, KT: tl.constexpr):
    s = tl.program_id(0); nt = tl.program_id(1)
    NS = M * TOPK
    sl = tl.arange(0, 64)
    ids = tl.load(IDS + sl, sl < NS, other=-1)
    e = tl.load(IDS + s)
    if (e < 0) | (tl.sum(tl.where((sl < s) & (ids == e), 1, 0)) > 0):
        return
    mm = tl.arange(0, 16)
    tok_hit = tl.sum(tl.where((ids[None, :] == e) & ((sl[None, :] // TOPK) == mm[:, None]), 1, 0), axis=1) > 0
    m_ok = tok_hit & (mm < M)
    W = W1 + e.to(tl.int64) * (K // 16) * (2 * I // 64 * 128)
    S = S1 + e.to(tl.int64) * (K // 16) * (2 * I)
    g = tl.zeros((16, 64), tl.float32)
    u = tl.zeros((16, 64), tl.float32)
    kk = tl.arange(0, KT * 16)
    for kt0 in range(0, K // 16, KT):
        xv = tl.load(X + mm[:, None] * XS + kt0 * 16 + kk[None, :], m_ok[:, None], other=0.0)
        g = tl.dot(xv, _deq_tile(W, S, kt0, KT, nt, 2 * I), g)
        u = tl.dot(xv, _deq_tile(W, S, kt0, KT, nt + I // 64, 2 * I), u)
    al = tl.load(AL1 + e) * 1.1754943508222875e-38
    g = (g * al).to(tl.bfloat16).to(tl.float32)
    u = (u * al).to(tl.bfloat16).to(tl.float32)
    sg = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    a = (sg * u).to(tl.bfloat16)
    n = nt * 64 + tl.arange(0, 64)
    tl.store(ACT + (s * 16 + mm[:, None]) * I + n[None, :], a, m_ok[:, None])


@triton.jit
def _moe_down2(ACT, IDS, WTS, W2, S2, AL2, OUT, M, TOPK: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
               KT: tl.constexpr):
    s = tl.program_id(0); nt = tl.program_id(1)
    NS = M * TOPK
    sl = tl.arange(0, 64)
    ids = tl.load(IDS + sl, sl < NS, other=-1)
    e = tl.load(IDS + s)
    if (e < 0) | (tl.sum(tl.where((sl < s) & (ids == e), 1, 0)) > 0):
        return
    mm = tl.arange(0, 16)
    hit = (ids[None, :] == e) & ((sl[None, :] // TOPK) == mm[:, None])
    m_ok = (tl.sum(tl.where(hit, 1, 0), axis=1) > 0) & (mm < M)
    wts = tl.load(WTS + sl, sl < NS, other=0.0).to(tl.float32)
    wm = tl.sum(tl.where(hit, wts[None, :], 0.0), axis=1)
    W = W2 + e.to(tl.int64) * (I // 16) * (H // 64 * 128)
    S = S2 + e.to(tl.int64) * (I // 16) * H
    acc = tl.zeros((16, 64), tl.float32)
    kk = tl.arange(0, KT * 16)
    for kt0 in range(0, I // 16, KT):
        xv = tl.load(ACT + (s * 16 + mm[:, None]) * I + kt0 * 16 + kk[None, :], m_ok[:, None], other=0.0)
        acc = tl.dot(xv, _deq_tile(W, S, kt0, KT, nt, H), acc)
    al = tl.load(AL2 + e)
    f = (wm * al).to(tl.bfloat16).to(tl.float32) * 1.1754943508222875e-38
    acc = (acc.to(tl.bfloat16).to(tl.float32) * f[:, None]).to(tl.bfloat16).to(tl.float32)
    n = nt * 64 + tl.arange(0, 64)
    tl.atomic_add(OUT + mm[:, None] * H + n[None, :], acc, m_ok[:, None], sem="relaxed")


def moe_decode2(x, ids, wts, w1, s1, al1, w2, s2, al2, I=640, KT1=4, KT2=4, nw1=4, nw2=4, ns1=3, ns2=3, out_bf16=None):
    m, h = x.shape; topk = ids.shape[1]
    ids_f = ids.reshape(-1).to(torch.int32).contiguous()
    act = torch.empty((m * topk, 16, I), dtype=torch.bfloat16, device=x.device)
    out = torch.zeros((m, h), dtype=torch.float32, device=x.device)
    _moe_up2[(m * topk, I // 64)](x, ids_f, w1, s1.view(torch.uint8), al1, act, m, x.stride(0), TOPK=topk, I=I, K=h,
                                  KT=KT1, num_warps=nw1, num_stages=ns1)
    _moe_down2[(m * topk, h // 64)](act, ids_f, wts.reshape(-1).float().contiguous(), w2, s2.view(torch.uint8), al2,
                                    out, m, TOPK=topk, I=I, H=h, KT=KT2, num_warps=nw2, num_stages=ns2)
    if out_bf16 is not None:
        out_bf16.copy_(out)
    return out


# ---- v3: gate and up tiles in separate programs (2x the CTAs); the down kernel applies silu(g)*u while loading.
@triton.jit
def _moe_up3(X, IDS, W1, S1, AL1, GU, M, XS, TOPK: tl.constexpr, I: tl.constexpr, K: tl.constexpr, KT: tl.constexpr):
    s = tl.program_id(0); nt = tl.program_id(1)
    NS = M * TOPK
    sl = tl.arange(0, 64)
    ids = tl.load(IDS + sl, sl < NS, other=-1)
    e = tl.load(IDS + s)
    if (e < 0) | (tl.sum(tl.where((sl < s) & (ids == e), 1, 0)) > 0):
        return
    mm = tl.arange(0, 16)
    tok_hit = tl.sum(tl.where((ids[None, :] == e) & ((sl[None, :] // TOPK) == mm[:, None]), 1, 0), axis=1) > 0
    m_ok = tok_hit & (mm < M)
    W = W1 + e.to(tl.int64) * (K // 16) * (2 * I // 64 * 128)
    S = S1 + e.to(tl.int64) * (K // 16) * (2 * I)
    g = tl.zeros((16, 64), tl.float32)
    kk = tl.arange(0, KT * 16)
    for kt0 in range(0, K // 16, KT):
        xv = tl.load(X + mm[:, None] * XS + kt0 * 16 + kk[None, :], m_ok[:, None], other=0.0)
        g = tl.dot(xv, _deq_tile(W, S, kt0, KT, nt, 2 * I), g)
    al = tl.load(AL1 + e) * 1.1754943508222875e-38
    n = nt * 64 + tl.arange(0, 64)
    tl.store(GU + (s * 16 + mm[:, None]) * (2 * I) + n[None, :], (g * al).to(tl.bfloat16), m_ok[:, None])


@triton.jit
def _moe_down3(GU, IDS, WTS, W2, S2, AL2, OUT, M, TOPK: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
               KT: tl.constexpr):
    s = tl.program_id(0); nt = tl.program_id(1)
    NS = M * TOPK
    sl = tl.arange(0, 64)
    ids = tl.load(IDS + sl, sl < NS, other=-1)
    e = tl.load(IDS + s)
    if (e < 0) | (tl.sum(tl.where((sl < s) & (ids == e), 1, 0)) > 0):
        return
    mm = tl.arange(0, 16)
    hit = (ids[None, :] == e) & ((sl[None, :] // TOPK) == mm[:, None])
    m_ok = (tl.sum(tl.where(hit, 1, 0), axis=1) > 0) & (mm < M)
    wts = tl.load(WTS + sl, sl < NS, other=0.0).to(tl.float32)
    wm = tl.sum(tl.where(hit, wts[None, :], 0.0), axis=1)
    W = W2 + e.to(tl.int64) * (I // 16) * (H // 64 * 128)
    S = S2 + e.to(tl.int64) * (I // 16) * H
    acc = tl.zeros((16, 64), tl.float32)
    kk = tl.arange(0, KT * 16)
    for kt0 in range(0, I // 16, KT):
        ptr = GU + (s * 16 + mm[:, None]) * (2 * I) + kt0 * 16 + kk[None, :]
        g = tl.load(ptr, m_ok[:, None], other=0.0).to(tl.float32)
        u = tl.load(ptr + I, m_ok[:, None], other=0.0).to(tl.float32)
        sg = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        acc = tl.dot((sg * u).to(tl.bfloat16), _deq_tile(W, S, kt0, KT, nt, H), acc)
    al = tl.load(AL2 + e)
    f = (wm * al).to(tl.bfloat16).to(tl.float32) * 1.1754943508222875e-38
    acc = (acc.to(tl.bfloat16).to(tl.float32) * f[:, None]).to(tl.bfloat16).to(tl.float32)
    n = nt * 64 + tl.arange(0, 64)
    tl.atomic_add(OUT + mm[:, None] * H + n[None, :], acc, m_ok[:, None], sem="relaxed")


def moe_decode3(x, ids, wts, w1, s1, al1, w2, s2, al2, I=640, KT1=4, KT2=4, nw1=4, nw2=4, ns1=3, ns2=3, out_bf16=None):
    m, h = x.shape; topk = ids.shape[1]
    ids_f = ids.reshape(-1).to(torch.int32).contiguous()
    gu = torch.empty((m * topk, 16, 2 * I), dtype=torch.bfloat16, device=x.device)
    out = torch.zeros((m, h), dtype=torch.float32, device=x.device)
    _moe_up3[(m * topk, 2 * I // 64)](x, ids_f, w1, s1.view(torch.uint8), al1, gu, m, x.stride(0), TOPK=topk, I=I, K=h,
                                      KT=KT1, num_warps=nw1, num_stages=ns1)
    _moe_down3[(m * topk, h // 64)](gu, ids_f, wts.reshape(-1).float().contiguous(), w2, s2.view(torch.uint8), al2,
                                    out, m, TOPK=topk, I=I, H=h, KT=KT2, num_warps=nw2, num_stages=ns2)
    if out_bf16 is not None:
        out_bf16.copy_(out)
    return out
