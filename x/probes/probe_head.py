# Break the target fast head and the draft head into graph-timed pieces at verify M.
import torch, sys
sys.path.insert(0, "/x")
import fp4x_kernels as F
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
st = None
for c in cl.__closure__:
    v = c.cell_contents
    if isinstance(v, dict) and "coarse" in v: st = v
fast = [c.cell_contents for c in cl.__closure__ if callable(c.cell_contents) and getattr(c.cell_contents, "__name__", "") == "fast"][0]
out = []
lm = [mm for n, mm in m.named_modules() if type(mm).__name__ == "Qwen4ExpForCausalLM"][0]
for M in (3, 1):
    h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
    P, S = st["coarse"]; V = st["w"].shape[0]
    out.append(f"M={M}: full fast head {gtime(lambda: fast(lm, h)):.3f} ms")
    out.append(f"  coarse int2 {gtime(lambda: F.int2_logits(h, P, S, V)):.3f} ms ({(P.numel() + S.numel() * S.element_size()) / 1e6:.0f} MB)")
    co = F.int2_logits(h, P, S, V).to(torch.bfloat16)
    out.append(f"  .to(bf16) {gtime(lambda: F.int2_logits(h, P, S, V).to(torch.bfloat16)) - gtime(lambda: F.int2_logits(h, P, S, V)):.3f} ms")
    out.append(f"  topk512 {gtime(lambda: torch.topk(co, 512, dim=-1)):.3f} ms")
    idx = torch.topk(co, 512, dim=-1).indices
    def refine():
        rows = idx.reshape(-1)
        wr = st["w"].index_select(0, rows).float().view(M * 512, -1, 32)
        sr = torch.exp2(st["s"].index_select(0, rows).float() - 127)[..., None]
        wd = (wr * sr).view(M, 512, -1)
        ex = torch.einsum("mck,mk->mc", wd, h.float())
        return co.scatter(1, idx, ex.to(co.dtype))
    out.append(f"  refine+scatter {gtime(refine):.3f} ms")
mtp = r.speculator.model
dh = [mm for n, mm in mtp.named_modules() if hasattr(mm, "_fx_w")]
if dh:
    d = dh[0]
    for M in (1,):
        h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
        out.append(f"draft head M={M}: compute_logits {gtime(lambda: d.compute_logits(h)):.3f} ms; nv4 only {gtime(lambda: F.nv4_linear(h, d._fx_w, d._fx_s, d._fx_gs, (32, 256, 4))):.3f} ms")
RESULT = "\n".join(out)
import itertools
P, S = st["coarse"]; V = st["w"].shape[0]
for M in (3, 1):
    h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
    res = []
    for bn, bk, nw in itertools.product([32, 64, 128, 256], [128, 256, 512], [2, 4, 8]):
        try: res.append((gtime(lambda: F.int2_logits(h, P, S, V, bn, bk, nw), it=10), (bn, bk, nw)))
        except Exception: pass
    res.sort(); out.append(f"int2 tune M={M}: " + str([(round(a, 3), b) for a, b in res[:4]]))
RESULT = "\n".join(out)
