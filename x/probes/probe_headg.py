# Target fast head: INT2 coarse with g16 vs g32/g64 fp16 scales. Graph time at M=3, plus recall on real decode rows
# (true argmax rank and top-20 coverage inside the coarse top-512). Call 1 hooks compute_logits (send kick2.py), call 2 runs.
import torch, builtins, importlib.util
spec = importlib.util.spec_from_file_location("fp4x_k3", "/x/fp4x_kernels.py"); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
from vllm.models.qwen4_exp.nvidia import model as MD
r = worker.model_runner
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
if not hasattr(builtins, "_fx_hg"):
    st = builtins._fx_hg = {"h": []}
    o = MD.Qwen4ExpForCausalLM.compute_logits; st["o"] = o
    def cl(self, hs, _o=o):
        if hs.dim() == 2 and hs.shape[0] <= 16 and len(st["h"]) < 400 and not torch.cuda.is_current_stream_capturing():
            st["h"].append(hs.detach().clone()); st["self"] = self
        return _o(self, hs)
    MD.Qwen4ExpForCausalLM.compute_logits = cl
    RESULT = "hooked"
else:
    st = builtins._fx_hg; MD.Qwen4ExpForCausalLM.compute_logits = st["o"]
    self = st["self"]; H = torch.cat(st["h"])[:1200]
    pk = self.lm_head.b12x_mxfp8_packed_weight
    w = pk.weight.values; sc = pk.weight.scale_rows.view(torch.uint8).view(w.shape[0], -1)
    wf = F.mx8_dequant(w, sc); N = w.shape[0]
    Q = {g: F.quantize_int2(wf, group=g) for g in (16, 32, 64)}
    out = [f"{H.shape[0]} real rows"]
    with torch.no_grad():
        ref = torch.cat([F.mx8_linear(H[i:i + 16].contiguous(), w, sc, (32, 512, 1, 4)).float() for i in range(0, H.shape[0], 16)])
        top1 = ref.argmax(-1); top20 = torch.topk(ref, 20, -1).indices
        h3 = H[:3].contiguous()
        for g, (P, S) in Q.items():
            co = torch.cat([F.int2_logits(H[i:i + 16].contiguous(), P, S, N) for i in range(0, H.shape[0], 16)])
            rank = (co > co.gather(1, top1[:, None])).sum(1)
            covs = []
            for C in (512, 1024):
                cc = torch.topk(co, C, -1).indices
                covs.append(torch.stack([torch.isin(top20[i], cc[i]).all() for i in range(H.shape[0])]).float().mean().item())
            cov = "/".join(f"{c:.4f}" for c in covs)
            t32 = gtime(lambda: F.int2_logits(h3, P, S, N))
            tb = gtime(lambda: F.int2_logits(h3, P, S, N, out_dtype=torch.bfloat16))
            out.append(f"g{g}: scales {S.numel()*2/1e6:.0f} MB; argmax rank max {rank.max().item()} p99 {rank.float().quantile(0.99).item():.0f}; "
                       f"top20 in top512/1024 {cov}; int2 {t32:.3f} ms (fp32 out) {tb:.3f} ms (bf16 out)")
        co = F.int2_logits(h3, *Q[16], N).to(torch.bfloat16)
        out.append(f"topk512 bf16 [3,{N}]: {gtime(lambda: torch.topk(co, 512, dim=-1)):.3f} ms; to(bf16) {gtime(lambda: F.int2_logits(h3, *Q[16], N).to(torch.bfloat16)) - gtime(lambda: F.int2_logits(h3, *Q[16], N)):.3f} ms")
        out.append(f"topk1024: {gtime(lambda: torch.topk(co, 1024, dim=-1)):.3f} ms")
        idx = torch.topk(co, 512, dim=-1).indices
        out.append(f"refine: {gtime(lambda: F.refine_mx8_(co, h3, idx, w, sc)):.3f} ms; whole current fast head: {gtime(lambda: self.compute_logits(h3)):.3f} ms")
    del builtins._fx_hg
    RESULT = "\n".join(out)
