import torch, sys, importlib
sys.path.insert(0, "/x")
import fp4x_kernels as F
importlib.reload(F)
def gtime(fn, it=30):
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
P, S = st["coarse"]; w8, s8 = st["w"], st["s"]; V = w8.shape[0]
out = []
h = torch.randn(3, 2560, device="cuda", dtype=torch.bfloat16)
for C in (512, 1024, 2048):
    def f(C=C):
        co = F.int2_logits(h, P, S, V).to(torch.bfloat16)
        idx = torch.topk(co, C, dim=-1).indices
        return F.refine_mx8_(co, h, idx, w8, s8)
    out.append(f"C={C}: {gtime(f):.3f} ms")
RESULT = "\n".join(out)
