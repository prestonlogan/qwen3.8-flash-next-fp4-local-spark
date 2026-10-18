# Fast-head candidate recall on real decode hidden states. 1st call: hook compute_logits to keep inputs.
# 2nd call (after some traffic): exact MXFP8 logits vs coarse top-C; reports top-20 containment and argmax rank.
import torch, sys, builtins
sys.path.insert(0, "/x")
import fp4x_kernels as F
from vllm.models.qwen4_exp.nvidia import model as MD
cl = MD.Qwen4ExpForCausalLM.compute_logits
st = [c.cell_contents for c in (cl.__closure__ or ()) if isinstance(c.cell_contents, dict) and "coarse" in c.cell_contents]
if not hasattr(builtins, "_fx_rec"):
    builtins._fx_rec = []
    builtins._fx_cl = cl
    def hooked(self, h, _cl=cl):
        if h.dim() == 2 and 0 < h.shape[0] <= 16 and len(builtins._fx_rec) < 600 and not torch.cuda.is_current_stream_capturing():
            builtins._fx_rec.append(h.detach().clone())
        return _cl(self, h)
    MD.Qwen4ExpForCausalLM.compute_logits = hooked
    RESULT = "hooked"
else:
    MD.Qwen4ExpForCausalLM.compute_logits = getattr(builtins, "_fx_cl", None) or MD.Qwen4ExpForCausalLM.compute_logits.__defaults__[0]
    st = [c.cell_contents for c in MD.Qwen4ExpForCausalLM.compute_logits.__closure__ if isinstance(c.cell_contents, dict) and "coarse" in c.cell_contents][0]
    w8, s8 = st["w"], st["s"]; V = w8.shape[0]; P, S = st["coarse"]
    H = torch.cat(builtins._fx_rec)[:1500]
    del builtins._fx_rec, builtins._fx_cl
    worst_rank = 0; miss = {64: 0, 128: 0, 256: 0, 512: 0}; n = 0
    for a in range(0, H.shape[0], 16):
        h = H[a:a + 16]
        exact = torch.empty((h.shape[0], V), dtype=torch.float32, device="cuda")
        for b in range(0, V, 32768):
            exact[:, b:b + 32768] = h.float() @ F.mx8_dequant(w8[b:b + 32768], s8[b:b + 32768]).T
        co = F.int2_logits(h, P, S, V)
        top20 = torch.topk(exact, 20, dim=-1).indices
        am = top20[:, :1]
        rank = (co > co.gather(1, am)).sum(-1)
        worst_rank = max(worst_rank, int(rank.max()))
        for C in miss:
            cand = torch.topk(co, C, dim=-1).indices
            hit = (top20[:, :, None] == cand[:, None, :]).any(-1).all(-1)
            miss[C] += int((~hit).sum())
        n += h.shape[0]
    RESULT = f"{n} real rows: worst coarse rank of true argmax {worst_rank}; rows whose true top-20 is not inside coarse top-C: " + ", ".join(f"C={C}: {v}" for C, v in miss.items())
