# Validate + time the new fast-head pieces: INT2 e4m3 two-level scales (BK 512) and the fused MXFP8 refine.
import torch, sys, importlib
sys.path.insert(0, "/x")
import fp4x_kernels as F
importlib.reload(F)
r = worker.model_runner; m = r.model
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
from vllm.models.qwen4_exp.nvidia import model as MD
cl = MD.Qwen4ExpForCausalLM.compute_logits
st = [c.cell_contents for c in cl.__closure__ if isinstance(c.cell_contents, dict) and "coarse" in c.cell_contents][0]
lm = [mm for n, mm in m.named_modules() if type(mm).__name__ == "Qwen4ExpForCausalLM"][0]
w8, s8 = st["w"], st["s"]; V = w8.shape[0]
out = []
P2 = torch.empty((V, 640), dtype=torch.uint8, device="cuda"); S2 = torch.empty((V, 160), dtype=torch.float8_e4m3fn, device="cuda"); R2 = torch.empty((V,), dtype=torch.float32, device="cuda")
for a in range(0, V, 16384):
    p, s_, rr = F.quantize_int2(F.mx8_dequant(w8[a:a + 16384], s8[a:a + 16384]), "e4m3")
    P2[a:a + 16384], S2[a:a + 16384], R2[a:a + 16384] = p, s_, rr
torch.cuda.synchronize()
P, S = st["coarse"]
def new_fast(h, C=512):
    co = F.int2_logits(h, P2, S2, V, RS=R2).to(torch.bfloat16)
    idx = torch.topk(co, C, dim=-1).indices
    return F.refine_mx8_(co, h, idx, w8, s8)
def mid_fast(h, C=512):  # fp16 scales, BK 512, fused refine
    co = F.int2_logits(h, P, S, V).to(torch.bfloat16)
    idx = torch.topk(co, C, dim=-1).indices
    return F.refine_mx8_(co, h, idx, w8, s8)
fast_old = [c.cell_contents for c in cl.__closure__ if callable(c.cell_contents) and getattr(c.cell_contents, "__name__", "") == "fast"][0]
torch.manual_seed(1)
# reference: the served fast head (validated against the MXFP8 head at build time)
for M in (1, 3):
    agree_new = agree_mid = 0; n = 0; dmax = 0.0; rank_new = []
    for t in range(40):
        h = (torch.randn(M, 2560, device="cuda") * (0.5 + t / 20)).to(torch.bfloat16)
        old = fast_old(lm, h); new = new_fast(h); mid = mid_fast(h)
        agree_new += (new.argmax(-1) == old.argmax(-1)).sum().item(); agree_mid += (mid.argmax(-1) == old.argmax(-1)).sum().item(); n += M
        top = torch.topk(old.float(), 20, dim=-1).indices
        dmax = max(dmax, (new.float().gather(1, top) - old.float().gather(1, top)).abs().max().item())
        co = F.int2_logits(h, P2, S2, V, RS=R2)
        tgt = old.float().argmax(-1)
        rank_new.append(int((co > co.gather(1, tgt[:, None])).sum(-1).max().item()))
    out.append(f"M={M}: argmax agree new {agree_new}/{n}, mid {agree_mid}/{n}; top20 max|d| new vs old {dmax:.3g}; worst coarse rank of argmax (e4m3) {max(rank_new)}")
    h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
    out.append(f"  old fast {gtime(lambda: fast_old(lm, h)):.3f} ms | mid (BK512+fused refine) {gtime(lambda: mid_fast(h)):.3f} | new (e4m3 too) {gtime(lambda: new_fast(h)):.3f} | coarse e4m3 {gtime(lambda: F.int2_logits(h, P2, S2, V, RS=R2)):.3f}")
    co = F.int2_logits(h, P2, S2, V, RS=R2).to(torch.bfloat16); idx = torch.topk(co, 512, dim=-1).indices
    out.append(f"  fused refine {gtime(lambda: F.refine_mx8_(co, h, idx, w8, s8)):.3f} ms")
RESULT = "\n".join(out)
