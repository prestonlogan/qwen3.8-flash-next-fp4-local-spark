#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of qwen3.8-flash-next-fp4-local-spark (ported from the companion repo's T8).
"""T8: offline FastMTP-style LoRA self-distillation of the Qwen3.8-Flash-Next MTP drafter, against the FP4-Local target.
The FP4-Local MTP/lm_head/embedding tensors are bit-identical to the pre-QAD checkpoint; the target's final mixer differs.
Target checkpoint read-only; only a small LoRA delta is written to /t8 (tmpfs).
Data: /t8/{ho,tr}_NNN.{pt,json} (target multi-stream hidden mh[T,10240] per position, token ids).
"""
import argparse, glob, json, math, os, time, random
import torch, torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="eval")          # eval | train
ap.add_argument("--tag", default="base")
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--alpha", type=float, default=64)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--ntrain", type=int, default=88)
ap.add_argument("--ce", type=float, default=0.3)  # weight of CE(argmax) vs KL
ap.add_argument("--beta", type=float, default=0.6)
ap.add_argument("--train_norms", type=int, default=1)
ap.add_argument("--load", default="")
ap.add_argument("--evalevery", type=int, default=44)
ap.add_argument("--fp8sim", type=int, default=0)
ap.add_argument("--full", type=int, default=0)   # full fine-tune of all dense MTP matrices (fp32 master)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--glob_tr", default="/t8/tr_*.pt")
ap.add_argument("--nval", type=int, default=12)
ap.add_argument("--tv", type=float, default=0.0)   # weight of sampled-acceptance loss 1 - sum min(p_T1(top20,p.95), q)
ap.add_argument("--pw", type=float, default=0.0)   # extra loss weight on anchors visited by the greedy verify walk
ap.add_argument("--dpace", type=int, default=0)   # D-PACE: per-anchor per-depth weights = d S~/d log q_k, S~ = sum_k prod_{i<=k} q_i (detached)
ap.add_argument("--sel", default="g")   # model selection on val: g (greedy tok/step) | s (sampled E tok/step) | gs (sum)
a = ap.parse_args()
torch.manual_seed(0); random.seed(0)
dev = "cuda"; bf = torch.bfloat16
LOG = open(f"/t8/log_{a.tag}.txt", "a")
def log(*s):
    m = " ".join(str(x) for x in s); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()
def memguard():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            kb = int(l.split()[1])
            if kb < 8 * 1024 * 1024: raise SystemExit(f"memguard: MemAvailable {kb/1e6:.1f} GB < 8")
            return kb / 1e6

M = glob.glob("/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/*/")[0]
wm = json.load(open(M + "model.safetensors.index.json"))["weight_map"]
_fh = {}
def g(k):
    f = wm[k]
    if f not in _fh: _fh[f] = safe_open(M + f, "pt", device="cpu")
    return _fh[f].get_tensor(k)
t0 = time.time()
P = "mtp."; L0 = "mtp.layers.0."
W = {}
for k in wm:
    if k.startswith("mtp.") and ".experts." not in k: W[k[4:]] = g(k).to(dev)
for k in ["hc_norm", "input_mix_weight_down", "input_mix_weight_up"]:
    W["tmix." + k] = g(f"model.language_model.hyper_connection_mixer.{k}.weight").to(dev)
dv = torch.load("/t8/dv.pt").to(dev)
lm = g("lm_head.weight").to(dev)[dv].contiguous()          # [Vd, 2560] bf16
emb = g("model.language_model.embed_tokens.weight").to(dev)  # [248320, 2560]
tgt2d = torch.full((248320,), -1, dtype=torch.long, device=dev); tgt2d[dv] = torch.arange(dv.numel(), device=dev)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
def deq(k):
    q = g(k + ".weight").to(dev); s = g(k + ".weight_scale").to(dev).float(); s2 = g(k + ".weight_scale_2").float().item()
    v = torch.stack((E2M1[(q & 15).long()], E2M1[(q >> 4).long()]), -1).reshape(q.shape[0], -1)
    return ((v.view(v.shape[0], -1, 16) * s[..., None]).reshape(v.shape) * s2).to(bf)
EG = torch.empty(512, 640, 2560, dtype=bf, device=dev); EU = torch.empty_like(EG); ED = torch.empty(512, 2560, 640, dtype=bf, device=dev)
for e in range(512):
    p = f"{L0}mlp.experts.{e}."
    EG[e] = deq(p + "gate_proj"); EU[e] = deq(p + "up_proj"); ED[e] = deq(p + "down_proj")
log(f"loaded in {time.time()-t0:.0f}s  dense={len(W)}  memavail={memguard():.1f}GB")

# ---------------- LoRA-wrapped linears ----------------
LORA = {}
LIN = ["fc_embedding", "fc_hidden", "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj",
       "layers.0.self_attn.o_proj", "layers.0.mlp.shared_expert.gate_proj", "layers.0.mlp.shared_expert.up_proj",
       "layers.0.mlp.shared_expert.down_proj"]
for hc in ["layers.0.attn_hyper_connection", "layers.0.mlp_hyper_connection", "hyper_connection_mixer"]:
    LIN += [hc + ".input_mix_weight_down", hc + ".input_mix_weight_up"]
scale = a.alpha / a.rank
params = []
FULL = {}
if a.full:
    for k in list(W):
        if k.startswith("tmix.") or k.endswith("norm.weight") or W[k].dim() != 2: continue
        FULL[k] = torch.nn.Parameter(W[k].float()); params.append(FULL[k])
    LIN = []
for n in LIN:
    w = W[n + ".weight"]
    A = torch.nn.Parameter(torch.randn(a.rank, w.shape[1], device=dev) / math.sqrt(w.shape[1]))
    B = torch.nn.Parameter(torch.zeros(w.shape[0], a.rank, device=dev))
    LORA[n] = (A, B); params += [A, B]
NORMS = [k[:-7] for k in W if k.endswith("norm.weight") and not k.startswith("tmix")]
ND = {}
if a.train_norms:
    for n in NORMS:
        d = torch.nn.Parameter(torch.zeros_like(W[n + ".weight"], dtype=torch.float32)); ND[n] = d; params.append(d)
if a.load:
    sd = torch.load(a.load)
    for n, (A, B) in LORA.items():
        if n + ".A" in sd: A.data.copy_(sd[n + ".A"]); B.data.copy_(sd[n + ".B"])
    for n, d in ND.items():
        if n + ".d" in sd: d.data.copy_(sd[n + ".d"])
    log("loaded", a.load)

def wt(k):
    return FULL[k].to(bf) if k in FULL else W[k]
def lin(x, n):
    y = F.linear(x, wt(n + ".weight"))
    if n in LORA:
        A, B = LORA[n]; y = y + F.linear(F.linear(x, A.to(bf)), B.to(bf)) * scale
    return y
def nw(n):
    w = W[n + ".weight"].float()
    return w + ND[n] if n in ND else w
def rms(x, n, group=None, eps=1e-6):
    xf = x.float()
    if group: xf = xf.unflatten(-1, (-1, group))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group: xf = xf.flatten(-2)
    return (xf * (1 + nw(n))).to(bf)
def rms_t(x, w, group=None, eps=1e-6):  # target mixer (frozen)
    xf = x.float()
    if group: xf = xf.unflatten(-1, (-1, group))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group: xf = xf.flatten(-2)
    return (xf * (1 + w.float())).to(bf)
def hc_mix(h, pre):
    hn = rms(h, pre + ".hc_norm", 2560)
    gt = torch.sigmoid(lin(F.silu(lin(hn, pre + ".input_mix_weight_down") / 4), pre + ".input_mix_weight_up")).unflatten(-1, (4, 2560))
    return (gt * hn.unflatten(-1, (4, 2560))).mean(-2), hn
def hc_combine(h, hn, out, pre):
    inj = 2 * torch.sigmoid(F.linear(hn, wt(pre + ".block_inject_weight.weight")) / 4)
    return (h.unflatten(-1, (4, 2560)) + out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
@torch.no_grad()
def target_mix(mh):
    hn = rms_t(mh, W["tmix.hc_norm"], 2560)
    gt = torch.sigmoid(F.linear(F.silu(F.linear(hn, W["tmix.input_mix_weight_down"]) / 4), W["tmix.input_mix_weight_up"])).unflatten(-1, (4, 2560))
    return (gt * hn.unflatten(-1, (4, 2560))).mean(-2)

inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, device=dev).float() / 64))
def rope(x, pos):  # x [N, H, 256], neox on first 64 dims
    f = pos.float()[:, None] * inv_freq[None]
    cos = torch.cat([f.cos(), f.cos()], -1)[:, None].to(x.dtype); sin = torch.cat([f.sin(), f.sin()], -1)[:, None].to(x.dtype)
    r, p = x[..., :64], x[..., 64:]
    r2 = torch.cat([-r[..., 32:], r[..., :32]], -1)
    return torch.cat([r * cos + r2 * sin, p], -1)

A_ = "layers.0.self_attn."
def attention(x, pos, kvs, k):
    """x [N,2560] for step k; kvs list of (K,V) [N,2,256] from previous steps (0..k-1). Returns out, (K,V)."""
    N = x.shape[0]
    qg = lin(x, A_ + "q_proj").view(N, 24, 512)
    q, gate = qg[..., :256], qg[..., 256:].reshape(N, 6144)
    q = rope(rms(q, A_ + "q_norm"), pos)
    kk = rope(rms(lin(x, A_ + "k_proj").view(N, 2, 256), A_ + "k_norm"), pos)
    vv = lin(x, A_ + "v_proj").view(N, 2, 256)
    allkv = kvs + [(kk, vv)]
    K0, V0 = allkv[0]
    qh = q.view(N, 2, 12, 256).float() * 0.0625                 # [N, kvh, grp, d]
    S = torch.einsum("nhgd,mhd->hgnm", qh, K0.float())  # [2,12,N,N]
    mask = torch.ones(N, N, dtype=torch.bool, device=dev).tril()
    S = S.masked_fill(~mask, float("-inf"))
    extra = [torch.einsum("nhgd,nhd->hgn", qh, Kj.float()) for Kj, _ in allkv[1:]]
    if extra:
        S = torch.cat([S, torch.stack(extra, -1)], -1)
    Pm = torch.softmax(S, -1)
    out = torch.einsum("hgnm,mhd->nhgd", Pm[..., :N], V0.float())
    for j, (_, Vj) in enumerate(allkv[1:]):
        out = out + Pm[..., N + j].permute(2, 0, 1)[..., None] * Vj.float()[:, :, None]
    out = out.reshape(N, 6144).to(bf) * torch.sigmoid(gate)
    return lin(out, A_ + "o_proj"), (kk, vv)

def moe(x):
    N = x.shape[0]
    rl = F.linear(x, wt("layers.0.mlp.gate.weight")).float().softmax(-1)
    tw, ti = rl.topk(10, -1); tw = (tw / tw.sum(-1, keepdim=True)).to(bf)
    fe = ti.reshape(-1); order = fe.argsort(); tok = order // 10
    cnt = torch.bincount(fe, minlength=512).tolist()
    xs = x[tok]; outs = []; s = 0
    for e, c in enumerate(cnt):
        if c == 0: continue
        seg = xs[s:s + c]; s += c
        outs.append(F.linear(F.silu(F.linear(seg, EG[e])) * F.linear(seg, EU[e]), ED[e]))
    y = torch.cat(outs) * tw.reshape(-1)[order][:, None]
    routed = torch.zeros_like(x).index_add(0, tok, y)
    sp = "layers.0.mlp.shared_expert."
    sh = lin(F.silu(lin(x, sp + "gate_proj")) * lin(x, sp + "up_proj"), sp + "down_proj")
    sh = sh * torch.sigmoid(F.linear(x, wt("layers.0.mlp.shared_expert_gate.weight")))
    return routed + sh

def mtp_step(h_multi, tok_ids, pos, kvs, k):
    e = lin(rms(emb[tok_ids], "pre_fc_norm_embedding"), "fc_embedding")
    N = h_multi.shape[0]
    hh = rms(h_multi, "pre_fc_norm_hidden").view(N, 4, 2560)
    hh = lin(hh, "fc_hidden")
    h = (e.unsqueeze(-2) + hh).flatten(-2)
    x, hn = hc_mix(h, "layers.0.attn_hyper_connection")
    o, kv = attention(x, pos, kvs, k)
    h = hc_combine(h, hn, o, "layers.0.attn_hyper_connection")
    x, hn = hc_mix(h, "layers.0.mlp_hyper_connection")
    h = hc_combine(h, hn, moe(x), "layers.0.mlp_hyper_connection")
    s, _ = hc_mix(h, "hyper_connection_mixer")
    return F.linear(s, lm).float(), h, kv          # draft logits over draft vocab

def load_seq(path):
    d = torch.load(path + ".pt"); j = json.load(open(path + ".json"))
    ids = torch.tensor(j["ids"], device=dev)
    if "mh" in d: mh = d["mh"].to(dev)
    else: mh = (d["mq"].to(dev).float().view(-1, 4, 2560) * d["ms"].to(dev).float()[..., None]).view(-1, 10240).to(bf)
    if a.fp8sim:
        x = mh.float().view(-1, 4, 2560); sc = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 448.0
        mh = ((x / sc).to(torch.float8_e4m3fn).float() * sc).view(-1, 10240).to(bf)
    T = mh.shape[0]
    assert T == ids.numel(), (T, ids.numel())
    return ids, mh, j["np"], d.get("sh")

def run_seq(ids, mh, np_, K=3, train=False, teacher=None):
    T = ids.numel(); N = T - 1                        # anchors t = 0..T-2
    t = torch.arange(N, device=dev)
    h = mh[:N]; kvs = []; logits = []
    for k in range(K):
        ti = (t + k + 1).clamp(max=T - 1)
        lg, h, kv = mtp_step(h, ids[ti], t + k + 1, kvs, k)
        kvs.append(kv); logits.append(lg)
    return logits

def teacher_logits(mh):
    with torch.no_grad():
        return F.linear(target_mix(mh), lm).float()     # [T, Vd]; row p predicts token p+1

def walk_mask(ids, lg, np_):
    """anchors visited by the greedy verify walk of the CURRENT drafter predictions (no grad)."""
    T = ids.numel(); t = torch.arange(T - 1, device=dev); c = torch.ones(T - 1, dtype=torch.bool, device=dev); ok = []
    for k in range(3):
        c = c & (dv[lg[k].detach().argmax(-1)] == ids[(t + k + 2).clamp(max=T - 1)]) & ((t + k + 2) <= T - 1); ok.append(c)
    L = (ok[0].long() + ok[1].long() + ok[2].long()).tolist(); m = torch.zeros(T - 1, dtype=torch.bool); a0 = max(np_ - 1, 0)
    while a0 + 4 <= T - 1:
        m[a0] = True; a0 += L[a0] + 1
    return m.to(dev)

@torch.no_grad()
def procwalk(paths, name):
    steps = 0; toks = 0
    for pth in paths:
        ids, mh, np_, _ = load_seq(pth); lg = run_seq(ids, mh, np_); m = walk_mask(ids, lg, np_)
        T = ids.numel(); t = torch.arange(T - 1, device=dev); c = torch.ones(T - 1, dtype=torch.bool, device=dev); L = torch.zeros(T - 1, device=dev)
        for k in range(3):
            c = c & (dv[lg[k].argmax(-1)] == ids[(t + k + 2).clamp(max=T - 1)]) & ((t + k + 2) <= T - 1); L += c.float()
        steps += int(m.sum()); toks += int((L[m] + 1).sum())
    log(f"[pwalk {name}] steps={steps} tok/step={toks/steps:.4f}"); return toks / steps

@torch.no_grad()
def evaluate(paths, name):
    acc = torch.zeros(3); n = 0; tfix = 0; tn = 0; shd = []
    for p in paths:
        ids, mh, np_, sh = load_seq(p); T = ids.numel()
        if sh is not None:
            shd.append(((target_mix(mh).float() - sh.to(dev).float()).norm() / sh.to(dev).float().norm()).item())
        tl = teacher_logits(mh)
        # teacher agreement with greedy continuation (sanity): positions p >= np_-1 predict ids[p+1]
        pr = torch.arange(max(np_ - 1, 0), T - 1, device=dev)
        tfix += (dv[tl[pr].argmax(-1)] == ids[pr + 1]).sum().item(); tn += pr.numel()
        lg = run_seq(ids, mh, np_)
        t = torch.arange(T - 1, device=dev)
        sel = (t + 2 >= np_) & (t + 4 <= T - 1)
        ok = torch.ones(int(sel.sum()), dtype=torch.bool, device=dev)
        for k in range(3):
            pred = dv[lg[k][sel].argmax(-1)]
            ok = ok & (pred == ids[t[sel] + k + 2]); acc[k] += ok.sum().item()
        n += int(sel.sum())
    c = acc / n; tps = 1 + c.sum().item()
    cond = [c[0].item(), (c[1] / c[0]).item(), (c[2] / c[1]).item()]
    log(f"[{name}] anchors={n} chained={[round(x,4) for x in c.tolist()]} cond={[round(x,3) for x in cond]} tok/step={tps:.4f} teacher_match={tfix/tn:.4f} shdiff={[round(x,4) for x in shd]}")
    return tps

ho = sorted(p[:-3] for p in glob.glob("/t8/ho_*.pt"))
tr = sorted(p[:-3] for p in glob.glob("/t8/tr_*.pt"))
val = tr[88:88 + a.nval]                       # fixed val set (tr_088..tr_099) across runs
extra = sorted(p[:-3] for p in glob.glob(a.glob_tr)) if a.glob_tr != "/t8/tr_*.pt" else []
trn = tr[:min(a.ntrain, 88)] + extra[:max(0, a.ntrain - 88)]
log(f"args {vars(a)} ho={len(ho)} train={len(trn)} val={len(val)}")
if a.mode == "procsim":
    # Selection-effect test: simulate the served greedy verify process (anchor -> accept chained L -> next anchor t+L+1)
    # using the same offline draft predictions, and compare with the all-anchor teacher-forced average.
    import collections
    with torch.no_grad():
        for name, paths in (("ho", ho), ("val", val)):
            allc = torch.zeros(3); na = 0; steps = 0; toks = 0; pc = torch.zeros(3); after = collections.Counter(); afterok = collections.Counter()
            for pth in paths:
                ids, mh, np_, _ = load_seq(pth); T = ids.numel(); lg = run_seq(ids, mh, np_)
                t = torch.arange(T - 1, device=dev); ok = torch.zeros(3, T - 1, dtype=torch.bool, device=dev); c = torch.ones(T - 1, dtype=torch.bool, device=dev)
                for k in range(3):
                    tgt = ids[(t + k + 2).clamp(max=T - 1)]; valid = (t + k + 2) <= T - 1
                    c = c & (dv[lg[k].argmax(-1)] == tgt) & valid; ok[k] = c
                sel = (t + 2 >= np_) & (t + 4 <= T - 1)
                allc += ok[:, sel].sum(-1).float().cpu(); na += int(sel.sum())
                okc = ok.cpu(); a0 = np_ - 1; prevL = None
                while a0 + 4 <= T - 1:
                    L = int(okc[0, a0]) + int(okc[1, a0]) + int(okc[2, a0])
                    for k in range(3): pc[k] += int(okc[k, a0])
                    if prevL is not None: after[prevL] += 1; afterok[prevL] += int(okc[0, a0])
                    steps += 1; toks += L + 1; a0 += L + 1; prevL = L
            ca = allc / na
            log(f"[procsim {name}] all-anchor chained={[round(x,4) for x in ca.tolist()]} tok/step={1+ca.sum().item():.4f} | process steps={steps} chained={[round(x,4) for x in (pc/steps).tolist()]} tok/step={toks/steps:.4f} | p1 after prev L=0..3: {[round(afterok[l]/max(after[l],1),3) for l in range(4)]} n={[after[l] for l in range(4)]}")
    raise SystemExit(0)
if a.mode == "depthsim":
    # Per-anchor depth-1..5 data for draft-depth / confidence-stopping / draft-shaping economics (offline).
    # greedy: ok[k] = draft argmax == realised token; conf[k] = draft max-softmax (T=1).
    # sampled (target T=1, top-k 20, top-p 0.95): acc[v][k] = sum min(p, q_v) with q_v = draft top-K/top-p at temp mult m.
    lmfull = g("lm_head.weight").to(dev); KD = 5
    def trunc(z, topk, topp):
        kv, ki = z.topk(topk, -1); pk = kv.softmax(-1)
        if topp < 1:
            cs = pk.cumsum(-1); pk = torch.where((cs - pk) < topp, pk, torch.zeros_like(pk)); pk = pk / pk.sum(-1, keepdim=True)
        return ki, pk
    VAR = {"cur": (20, 0.90, 0.9), "m0.8": (20, 0.90, 0.8), "m0.67": (20, 0.90, 0.667), "m0.5": (20, 0.90, 0.5), "m0.4": (20, 0.90, 0.4), "m0.5p1": (20, 1.0, 0.5)}
    out = {"var": list(VAR), "seqs": []}
    with torch.no_grad():
        for pth in val + ho:
            ids, mh, np_, _ = load_seq(pth); L = ids.numel()
            tm = target_mix(mh); lg = run_seq(ids, mh, np_, K=KD)
            t = torch.arange(L - 1, device=dev); sel = (t + 2 >= np_) & (t + KD + 1 <= L - 1); ts = t[sel]
            ok = []; conf = []; acc = {v: [] for v in VAR}
            for k in range(KD):
                dl = lg[k][sel].float()
                ok.append(dv[dl.argmax(-1)] == ids[ts + k + 2]); conf.append(dl.softmax(-1).amax(-1))
                ki, pk = trunc(F.linear(tm[ts + k + 1], lmfull).float(), 20, 0.95)
                di = tgt2d[ki]; valid = di >= 0
                for nm, (K, tp, mlt) in VAR.items():
                    qi, qp = trunc(dl / mlt, K, tp)
                    qt = torch.zeros_like(dl).scatter_(-1, qi, qp)
                    qk = torch.where(valid, qt.gather(-1, di.clamp(min=0)), torch.zeros_like(pk))
                    acc[nm].append(torch.minimum(pk, qk).sum(-1))
            out["seqs"].append({"name": os.path.basename(pth), "a0": int(ts[0]) if len(ts) else 0, "ok": torch.stack(ok).cpu(), "conf": torch.stack(conf).half().cpu(),
                                "acc": torch.stack([torch.stack(acc[v]) for v in VAR]).half().cpu()})
            log(f"[depthsim] {os.path.basename(pth)} anchors={int(sel.sum())} greedy chained d1..5={[round(x,3) for x in torch.stack(ok).float().cumprod(0).mean(1).tolist()]}")
    torch.save(out, "/t8/depthsim.pt"); log("saved /t8/depthsim.pt")
    raise SystemExit(0)
if a.mode == "probest":
    # Offline E[tok/step] under target sampling (T, top-k 20, top-p 0.95): greedy draft vs probabilistic q variants.
    lmfull = g("lm_head.weight").to(dev)
    def trunc(z, topk, topp):   # z: scaled logits [N, V]; returns (ids, probs) sparse
        kv, ki = z.topk(topk, -1); pk = kv.softmax(-1)
        if topp < 1:
            cs = pk.cumsum(-1); pk = torch.where((cs - pk) < topp, pk, torch.zeros_like(pk)); pk = pk / pk.sum(-1, keepdim=True)
        return ki, pk
    # variant: (K, draft top-p, draft temp multiplier)
    VAR = {"K20": (20, 1.0, 1.0), "K20p95": (20, 0.95, 1.0), "K20p90": (20, 0.90, 1.0), "K20p80": (20, 0.80, 1.0),
           "K10": (10, 1.0, 1.0), "K5": (5, 1.0, 1.0), "K64": (64, 1.0, 1.0),
           "K20m0.8": (20, 1.0, 0.8), "K20m0.9": (20, 1.0, 0.9), "K20m1.1": (20, 1.0, 1.1), "K20p95m0.9": (20, 0.95, 0.9)}
    with torch.no_grad():
        for T in (0.7, 1.0):
            names = ["g"] + list(VAR); S = {k: torch.zeros(3) for k in names}; E = {k: 0.0 for k in names}; n = 0
            for pth in val + ho:
                ids, mh, np_, _ = load_seq(pth); L = ids.numel()
                tm = target_mix(mh); lg = run_seq(ids, mh, np_)
                t = torch.arange(L - 1, device=dev); sel = (t + 2 >= np_) & (t + 4 <= L - 1); ts = t[sel]
                A = {k: [] for k in names}
                for k in range(3):
                    ki, pk = trunc(F.linear(tm[ts + k + 1], lmfull).float() / T, 20, 0.95)
                    di = tgt2d[ki]; valid = di >= 0
                    dl = lg[k][sel].float()
                    for nm, (K, tp, mlt) in VAR.items():
                        qi, qp = trunc(dl / (T * mlt), K, tp)
                        qt = torch.zeros_like(dl).scatter_(-1, qi, qp)
                        qk = torch.where(valid, qt.gather(-1, di.clamp(min=0)), torch.zeros_like(pk))
                        A[nm].append(torch.minimum(pk, qk).sum(-1))
                    dg = dv[dl.argmax(-1)].long()
                    A["g"].append((pk * (ki == dg[:, None])).sum(-1))
                for nm, X in A.items():
                    S[nm] += torch.stack([x.sum() for x in X]).cpu()
                    E[nm] += (1 + X[0] + X[0] * X[1] + X[0] * X[1] * X[2]).sum().item()
                n += int(sel.sum())
            for nm in names:
                log(f"[probest2 T={T}] {nm:12s} acc={[round(x,4) for x in (S[nm]/n).tolist()]} E[tok/step]={E[nm]/n:.4f} rel_g={E[nm]/E['g']-1:+.2%}")
    raise SystemExit(0)
def trunc_p(z, topk=20, topp=0.95):
    kv, ki = z.topk(topk, -1); pk = kv.softmax(-1)
    cs = pk.cumsum(-1); pk = torch.where((cs - pk) < topp, pk, torch.zeros_like(pk)); pk = pk / pk.sum(-1, keepdim=True)
    return ki, pk
@torch.no_grad()
def evaluate_s(paths, name, T=1.0, K=20, TP=0.9, M=0.9):
    """Served-equivalent sampled E[tok/step]: target T/top-k20/top-p.95 vs E41 draft q (top-K, top-p TP, temp x M)."""
    E = 0.0; n = 0; acc = torch.zeros(3)
    for pth in paths:
        ids, mh, np_, _ = load_seq(pth); L = ids.numel()
        tl = teacher_logits(mh); lg = run_seq(ids, mh, np_)
        t = torch.arange(L - 1, device=dev); sel = (t + 2 >= np_) & (t + 4 <= L - 1); ts = t[sel]; X = []
        for k in range(3):
            ki, pk = trunc_p(tl[ts + k + 1] / T)            # ids index the draft vocab (teacher is dv-restricted)
            dl = lg[k][sel].float(); qi, qp = trunc_p(dl / (T * M), K, TP)
            qt = torch.zeros_like(dl).scatter_(-1, qi, qp)
            X.append(torch.minimum(pk, qt.gather(-1, ki)).sum(-1))
        acc += torch.stack([x.sum() for x in X]).cpu(); E += (1 + X[0] + X[0] * X[1] + X[0] * X[1] * X[2]).sum().item(); n += int(sel.sum())
    log(f"[{name}] sampled T={T} acc={[round(x,4) for x in (acc/n).tolist()]} E[tok/step]={E/n:.4f}")
    return E / n
evaluate(ho, "ho@start"); v0 = evaluate(val, "val@start")
if a.sel == "p": procwalk(val, "val@start"); procwalk(ho, "ho@start"); s0 = 1.0
else: s0 = evaluate_s(val, "val@start"); evaluate_s(ho, "ho@start")
if a.mode == "train":
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    wts = torch.tensor([a.beta ** k for k in range(3)]); wts = wts / wts.sum()
    steps = int(a.epochs * len(trn)); sched = lambda i: min(1.0, (i + 1) / 20) * 0.5 * (1 + math.cos(math.pi * i / steps))
    best = -1; i = 0; t0 = time.time(); run = 0.0
    while i < steps:
        order = trn[:]; random.shuffle(order)
        for p in order:
            if i >= steps: break
            ids, mh, np_, _ = load_seq(p); T = ids.numel()
            tl = teacher_logits(mh)
            lg = run_seq(ids, mh, np_, train=True)
            loss = 0.0
            t = torch.arange(T - 1, device=dev)
            wa = (1.0 + a.pw * walk_mask(ids, lg, np_).float()) if a.pw > 0 else None
            if a.dpace:
                with torch.no_grad():
                    N_ = T - 1; qd = []
                    for k in range(3):
                        tgt = tl[(t + k + 1).clamp(max=T - 1)].argmax(-1)
                        qd.append(lg[k].float().softmax(-1).gather(-1, tgt[:, None]).squeeze(-1))   # tl is draft-vocab teacher logits
                    q_ = torch.stack(qd)                                  # [3, N]
                    pp = q_.clamp(min=0.05).cumprod(0)                    # prefix products (asymmetric smoothing: floor 0.05)
                    dw = pp.flip(0).cumsum(0).flip(0)                     # suffix sums: weight_k = sum_{j>=k} prod_{i<=j} q_i
                    dw = dw / dw.mean(1, keepdim=True).sum(0, keepdim=True).clamp(min=1e-6) * 3
            for k in range(3):
                valid = t + k + 1 <= T - 1
                tt = tl[(t + k + 1).clamp(max=T - 1)][valid]; ss = lg[k][valid]
                if a.dpace:
                    w_ = dw[k][valid]
                    kl = (F.kl_div(ss.log_softmax(-1), tt.log_softmax(-1), log_target=True, reduction="none").sum(-1) * w_).mean()
                    ce = (F.cross_entropy(ss, tt.argmax(-1), reduction="none") * w_).mean()
                elif wa is not None:
                    w_ = wa[valid]; w_ = w_ / w_.mean()
                    kl = (F.kl_div(ss.log_softmax(-1), tt.log_softmax(-1), log_target=True, reduction="none").sum(-1) * w_).mean()
                    ce = (F.cross_entropy(ss, tt.argmax(-1), reduction="none") * w_).mean()
                else:
                    kl = F.kl_div(ss.log_softmax(-1), tt.log_softmax(-1), log_target=True, reduction="batchmean")
                    ce = F.cross_entropy(ss, tt.argmax(-1))
                loss = loss + (1.0 / 3 if a.dpace else wts[k]) * ((1 - a.ce) * kl + a.ce * ce)
                if a.tv > 0:
                    with torch.no_grad(): ki, pk = trunc_p(tt.float())
                    q = (ss.float() / 0.9).softmax(-1)
                    loss = loss + a.tv * wts[k] * (1 - torch.minimum(pk, q.gather(-1, ki)).sum(-1)).mean()
            for gi in opt.param_groups: gi["lr"] = a.lr * sched(i)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            run = 0.9 * run + 0.1 * loss.item() if i else loss.item(); i += 1
            if i % 10 == 0: log(f"step {i}/{steps} loss {run:.4f} {time.time()-t0:.0f}s mem {memguard():.1f}GB")
            if i % a.evalevery == 0 or i == steps:
                vg = evaluate(val, f"val@{i}"); h = evaluate(ho, f"ho@{i}")
                if a.sel == "p":
                    v = procwalk(val, f"val@{i}"); procwalk(ho, f"ho@{i}")
                else:
                    vs = evaluate_s(val, f"val@{i}"); evaluate_s(ho, f"ho@{i}")
                    v = {"g": vg, "s": vs, "gs": vg / v0 + vs / s0}[a.sel]
                if v > best:
                    best = v
                    sd = {}
                    for n, (A, B) in LORA.items(): sd[n + ".A"] = A.detach().cpu(); sd[n + ".B"] = B.detach().cpu()
                    for n, d in ND.items(): sd[n + ".d"] = d.detach().cpu()
                    for k, fv in FULL.items(): sd[k[:-7] + ".full"] = fv.detach().to(bf).cpu()
                    torch.save(sd, f"/t8/lora_{a.tag}.pt"); log(f"saved best val {v:.4f}")
log("done")
