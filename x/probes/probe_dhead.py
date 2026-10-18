# Draft head: INT2 coarse + exact NVFP4 refine of the top C vs the served NVFP4 65k head (argmax, timing).
import torch, sys, importlib
sys.path.insert(0, "/x")
import fp4x_kernels as F
importlib.reload(F)
r = worker.model_runner
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
d = [mm for n, mm in r.speculator.model.named_modules() if hasattr(mm, "_fx_w")][0]
W, S, gs, idx = d._fx_w, d._fx_s, d._fx_gs, d._fx_idx; N = idx.numel()
q = [F.quantize_int2(F.dequant_nv4(W[a:a + 8192], S[a:a + 8192], gs)) for a in range(0, N, 8192)]
P2, S2 = torch.cat([t[0] for t in q]), torch.cat([t[1] for t in q])
out = [f"coarse {(P2.numel() + S2.numel() * 2) / 1e6:.0f} MB vs nv4 {(W.numel() + S.numel()) / 1e6:.0f} MB"]
# real draft hidden states: the speculator keeps its last inputs in sp.hidden_states; use random + those rows
H = [torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16) * (0.3 + 0.1 * t) for t in range(30)]
for C in (32, 64, 128):
    agree = 0; worst = 0
    for h in H:
        ref = F.nv4_linear(h, W, S, gs, (32, 256, 4)).float()
        co = F.int2_logits(h, P2, S2, N)
        cand = torch.topk(co, C, dim=-1).indices
        vals = F.refine_nv4(h, cand, W, S, gs)
        am = cand.gather(1, vals.argmax(-1, keepdim=True))
        agree += int((am.view(-1) == ref.argmax(-1)).sum())
        worst = max(worst, (vals - ref.gather(1, cand)).abs().max().item())
    out.append(f"C={C}: argmax agree {agree}/{len(H)}; refine max|d| vs nv4 GEMV {worst:.3g}")
h = H[0]
def new(C=64):
    co = F.int2_logits(h, P2, S2, N); cand = torch.topk(co, C, dim=-1).indices
    vals = F.refine_nv4(h, cand, W, S, gs)
    full = torch.full((1, d._fx_vocab), float("-inf"), dtype=torch.bfloat16, device="cuda")
    return full.scatter_(1, idx[cand], vals.to(torch.bfloat16))
out.append(f"served compute_logits {gtime(lambda: d.compute_logits(h)):.3f} ms | new C=64 {gtime(new):.3f} ms | coarse only {gtime(lambda: F.int2_logits(h, P2, S2, N)):.3f} ms")
RESULT = "\n".join(out)
