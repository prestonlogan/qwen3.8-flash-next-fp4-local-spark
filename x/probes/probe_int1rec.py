# Two-stage head feasibility: recall of a 1-bit coarse pass (sign x per-group mean |w|) on real decode rows.
# Reports the true argmax's coarse rank and top-20 coverage inside the coarse top-C. Call 1 hooks (kick2.py x2), call 2 runs.
import torch, builtins, importlib.util
spec = importlib.util.spec_from_file_location("fp4x_k4", "/x/fp4x_kernels.py"); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
from vllm.models.qwen4_exp.nvidia import model as MD
if not hasattr(builtins, "_fx_i1"):
    st = builtins._fx_i1 = {"h": []}
    o = MD.Qwen4ExpForCausalLM.compute_logits; st["o"] = o
    def cl(self, hs, _o=o):
        if hs.dim() == 2 and hs.shape[0] <= 16 and len(st["h"]) < 400 and not torch.cuda.is_current_stream_capturing():
            st["h"].append(hs.detach().clone()); st["self"] = self
        return _o(self, hs)
    MD.Qwen4ExpForCausalLM.compute_logits = cl
    RESULT = "hooked"
else:
    st = builtins._fx_i1; MD.Qwen4ExpForCausalLM.compute_logits = st["o"]
    self = st["self"]; H = torch.cat(st["h"])[:1000].contiguous()
    pk = self.lm_head.b12x_mxfp8_packed_weight
    w = pk.weight.values; sc = pk.weight.scale_rows.view(torch.uint8).view(w.shape[0], -1)
    out = [f"{H.shape[0]} rows"]
    with torch.no_grad():
        ref = torch.cat([F.mx8_linear(H[i:i + 16], w, sc, (32, 512, 1, 4)).float() for i in range(0, H.shape[0], 16)])
        top1 = ref.argmax(-1); top20 = torch.topk(ref, 20, -1).indices
        N = w.shape[0]
        for g in (16, 32, 64):
            co = torch.empty_like(ref)
            for a in range(0, N, 16384):
                wf = F.mx8_dequant(w[a:a + 16384], sc[a:a + 16384])
                f = wf.view(-1, 2560 // g, g)
                q = (torch.sign(f) * f.abs().mean(-1, keepdim=True)).view(-1, 2560).to(torch.bfloat16)
                co[:, a:a + 16384] = (H @ q.T).float(); del wf, f, q
            rank = (co > co.gather(1, top1[:, None])).sum(1)
            res = [f"g{g}: argmax rank max {rank.max().item()} p99.9 {rank.float().quantile(0.999).item():.0f}"]
            for C in (1024, 2048, 4096, 8192):
                cc = torch.topk(co, C, -1).indices
                cov = torch.stack([torch.isin(top20[i], cc[i]).all() for i in range(H.shape[0])]).float().mean().item()
                res.append(f"C{C}: argmax in {(rank < C).float().mean().item():.4f} top20 in {cov:.4f}")
            out.append("; ".join(res))
    del builtins._fx_i1
    RESULT = "\n".join(out)
