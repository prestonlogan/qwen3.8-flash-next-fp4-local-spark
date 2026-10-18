# Candidate recall of cheaper coarse heads on real decode rows: INT1 (sign x per-16 mean|w|), INT2 (served).
# 1st call hooks compute_logits to keep rows; 2nd call (after traffic) evaluates.
import torch, sys, builtins
sys.path.insert(0, "/x")
import fp4x_kernels as F
from vllm.models.qwen4_exp.nvidia import model as MD
if not hasattr(builtins, "_fx_rec"):
    builtins._fx_rec = []; builtins._fx_cl = cl = MD.Qwen4ExpForCausalLM.compute_logits
    def hooked(self, h, _cl=cl):
        if h.dim() == 2 and 0 < h.shape[0] <= 16 and len(builtins._fx_rec) < 600 and not torch.cuda.is_current_stream_capturing():
            builtins._fx_rec.append(h.detach().clone())
        return _cl(self, h)
    MD.Qwen4ExpForCausalLM.compute_logits = hooked
    RESULT = "hooked"
else:
    MD.Qwen4ExpForCausalLM.compute_logits = builtins._fx_cl
    st = [c.cell_contents for c in builtins._fx_cl.__closure__ if isinstance(c.cell_contents, dict) and "coarse" in c.cell_contents][0]
    w8, s8 = st["w"], st["s"]; V = w8.shape[0]; P, S = st["coarse"]
    H = torch.cat(builtins._fx_rec)[:1200]
    del builtins._fx_rec, builtins._fx_cl
    Cs = (256, 512, 1024, 2048, 4096)
    res = {k: {"rank": 0, "miss": {C: 0 for C in Cs}} for k in ("int2", "int1")}
    n = 0
    for a in range(0, H.shape[0], 16):
        h = H[a:a + 16]; hf = h.float()
        exact = torch.empty((h.shape[0], V), device="cuda"); c1 = torch.empty_like(exact)
        for b in range(0, V, 16384):
            wf = F.mx8_dequant(w8[b:b + 16384], s8[b:b + 16384])
            exact[:, b:b + 16384] = hf @ wf.T
            g = wf.view(wf.shape[0], -1, 16)
            w1 = (torch.sign(g) * g.abs().mean(-1, keepdim=True)).view_as(wf)
            c1[:, b:b + 16384] = hf @ w1.T
        co = {"int2": F.int2_logits(h, P, S, V), "int1": c1}
        top20 = torch.topk(exact, 20, dim=-1).indices
        for k, c in co.items():
            rank = (c > c.gather(1, top20[:, :1])).sum(-1)
            res[k]["rank"] = max(res[k]["rank"], int(rank.max()))
            for C in Cs:
                cand = torch.topk(c, C, dim=-1).indices
                res[k]["miss"][C] += int((~(top20[:, :, None] == cand[:, None, :]).any(-1).all(-1)).sum())
        n += h.shape[0]
    RESULT = f"{n} real rows\n" + "\n".join(f"{k}: worst argmax rank {v['rank']}; top-20 not in top-C: {v['miss']}" for k, v in res.items())
